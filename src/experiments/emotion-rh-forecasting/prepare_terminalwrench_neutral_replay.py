"""Freeze native-template token replay for neutral v7 GPT-OSS/Gemma 4B traces."""
from __future__ import annotations

# Resolve repository imports when this file is executed directly.
from pathlib import Path as _Path
import sys as _sys
_REPO = next((p for p in _Path(__file__).resolve().parents
              if (p / "src/experiments/shared").is_dir()), None)
if _REPO is not None:
    _sys.path[:0] = [str(_Path(__file__).resolve().parent), str(_REPO), str(_REPO / "src/environments"),
                    str(_REPO / "src/experiments")] + [
        str(_REPO / "src/experiments" / p) for p in
        ("emotion-probes", "emotion-rh-forecasting", "emotion-steering", "emotion-self-report")]


import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from shared.io import atomic, digest, read, require, sha  # noqa: E402
from emotion_rh_forecasting.inputs import Sources, load_probes, vector_identity  # noqa: E402
from emotion_rh_forecasting.terminalwrench_replay import sentence_cuts  # noqa: E402
from preflight_terminalwrench_neutral_v7_20261026 import OUT, PROBES, SOURCES  # noqa: E402

MODEL_KEYS = {"GPT-OSS 120B": "gptoss120b", "Gemma 4B": "gemma4b"}
MARKER_OPEN = "\ue000TW_PROBE_START_{}\ue001"
MARKER_CLOSE = "\ue000TW_PROBE_END_{}\ue001"


def rendered_offsets(messages, bodies, tokenizer):
    """Remove all markers after native rendering and locate original assistant text."""
    marked = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    positions = {}
    parts = []
    position = 0
    cursor = 0
    for step_id, body in bodies:
        start_mark = MARKER_OPEN.format(step_id)
        end_mark = MARKER_CLOSE.format(step_id)
        lo = marked.index(start_mark, cursor)
        parts.append(marked[cursor:lo])
        position += lo - cursor
        body_start = lo + len(start_mark)
        body_end = marked.index(end_mark, body_start)
        require(marked[body_start:body_end] == body, f"Native template changed assistant body: {step_id}")
        positions[step_id] = (position, position + len(body))
        parts.append(body)
        position += len(body)
        cursor = body_end + len(end_mark)
    parts.append(marked[cursor:])
    result = "".join(parts)
    require(all(result[a:b] == body for step_id, body in bodies
                for a, b in [positions[step_id]]), "Marker removal changed body")
    return result, positions


def prepare(model):
    key = MODEL_KEYS[model]
    audit = read(OUT / "readiness.json")
    require(audit["models"][model]["readiness"] == "needs_activation_replay", "Model lacks two positive folds")
    bundle = load_probes(PROBES[model], Sources())
    tokenizer = AutoTokenizer.from_pretrained(bundle["model"].get("tokenizer_model", bundle["model"]["name"]),
                                                revision=bundle["model"].get("tokenizer_revision", bundle["model"]["revision"]),
                                                use_fast=True, trust_remote_code=False)
    require(tokenizer.is_fast, "Fast offsets required")
    root = ROOT / "runs/emotion-rh-replay" / f"{key}-terminalwrench-neutral-v7-best12-20261026"
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    labels_in = []
    for folder in SOURCES:
        source = folder / "labels.json"
        require(any(p["path"] == str(source.relative_to(ROOT)) and p["sha256"] == sha(source)
                    for p in audit["label_sources"]), "Label source hash differs")
        labels_in += [t for t in read(source)["traces"] if t["model"] == model and
                      t["task"]["condition"] == "baseline" and t["task"]["emotion"] is None and
                      t["task"]["strength"] == 0]
    labels_in.sort(key=lambda t: t["id"])
    dataset, labels_out, jobs, fold_map = [], [], [], {}
    for trace in labels_in:
        source = ROOT / trace["source_path"]
        require(sha(source) == trace["source_sha256"], f"Source episode changed: {source}")
        trajectory = json.loads(read(source)["trajectory_text"])
        named = hashlib.sha256(trace["id"].encode()).hexdigest()[:24]
        units_by_step = {}
        for u in trace["units"]:
            require(u["kind"] == "sentence_or_fragment" and u["field"] == "message", "Unexpected unit")
            units_by_step.setdefault(str(u["step_id"]), []).append(u)
        messages, bodies, agent_spans = [], [], {}
        for step in trajectory["steps"]:
            if step["source"] == "user":
                messages.append({"role": "user", "content": step["message"]})
            elif step["source"] == "agent":
                step_id = str(step["step_id"])
                prose = step.get("message", "")
                calls = step.get("tool_calls") or []
                body = prose + ("\n\n# Submitted commands\n```json\n" +
                                json.dumps(calls, ensure_ascii=False, indent=2) + "\n```" if calls else "")
                require(MARKER_OPEN.format(step_id) not in body and MARKER_CLOSE.format(step_id) not in body,
                        "Reserved marker in source")
                messages.append({"role": "assistant", "content": MARKER_OPEN.format(step_id) +
                                 body + MARKER_CLOSE.format(step_id)})
                bodies.append((step_id, body))
                agent_spans[step_id] = prose
                outputs = [r.get("content", "") for r in (step.get("observation") or {}).get("results", [])]
                if outputs:
                    messages.append({"role": "user", "content": "\n\n".join(outputs)})
        rendered, positions = rendered_offsets(messages, bodies, tokenizer)
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        token_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
        offsets = np.asarray(encoded["offset_mapping"], dtype=np.int64)
        kind = np.zeros(len(token_ids), dtype=np.int8)
        for start, end in positions.values():
            kind[(offsets[:, 1] > start) & (offsets[:, 0] < end)] = 1
        units, original = [], []
        for source_unit in trace["units"]:
            step_id = str(source_unit["step_id"])
            start, _ = positions[step_id]
            lo, hi = source_unit["char_start"], source_unit["char_end"]
            require(agent_spans[step_id][lo:hi] == source_unit["text"], "ATIF sentence changed")
            global_lo, global_hi = start + lo, start + hi
            require(rendered[global_lo:global_hi] == source_unit["text"], "Rendered sentence changed")
            base = {k: source_unit[k] for k in ("unit_id", "kind", "char_start", "char_end",
                                                "text", "hacking", "needs_review")}
            original.append(base)
            units.append({**base, "source_char_start": lo, "source_char_end": hi,
                          "char_start": global_lo, "char_end": global_hi, "turn_id": named})
        require(units, "Empty labeled trace")
        sentence_cuts(offsets, units)
        input_path = root / "inputs" / f"{named}.npz"
        np.savez_compressed(input_path, token_ids=token_ids, offsets=offsets, kind=kind)
        job = {"id": named, "source_sha256": trace["source_sha256"], "input_sha256": sha(input_path),
               "rendered": rendered, "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
               "source_boundaries": [{"unit_id": u["unit_id"], "char_start": u["char_start"],
                                      "char_end": u["char_end"]} for u in units],
               "native_context_tokens": len(token_ids)}
        atomic(root / "inputs" / f"{named}.json", job)
        jobs.append({k: v for k, v in job.items() if k not in ("rendered", "source_boundaries")})
        task = trace["task"]["scenario"]
        fold = audit["models"][model]["task_folds"][task]
        fold_map[named] = fold
        dataset.append({"id": named, "task": task, "group_id": task, "fold": fold,
                        "turn_ids": [named], "units": units})
        labels_out.append({"id": named, "task": task, "group_id": task,
                           "source_sha256": trace["source_sha256"], "units": original})
    atomic(root / "probes.json", bundle)
    atomic(root / "dataset-traces.json", dataset)
    label_root = root / "labels"
    atomic(label_root / "labels.json", {"traces": labels_out})
    atomic(label_root / "folds.json", {"trace_folds": fold_map,
                                       "labels_sha256": sha(label_root / "labels.json")})
    spec = {"schema_version": 1, "config": {"labels": str(label_root)}, "model": bundle["model"],
            "probe_identity": vector_identity(bundle), "probes_sha256": sha(root / "probes.json"),
            "jobs": jobs, "labels_sha256": sha(label_root / "labels.json"),
            "folds_sha256": sha(label_root / "folds.json"),
            "dataset_traces_sha256": sha(root / "dataset-traces.json"),
            "renderer": f"{key}-native-ATIF-visible-v1",
            "interpretation": "Native-template replay of neutral v7 ATIF visible message/command text; "
                              "sentence labels from GPT-6.1 Sol with Astra review on new traces. "
                              "Reconstructed replay, not original generation activations.",
            "chunk_tokens": 512, "max_tokens": 131072,
            "source_hashes": {str(Path(__file__).name): sha(Path(__file__))}}
    if key == "gptoss120b":
        spec.update(validation_protocol="gptoss-causal-schedule-v1", attention="eager",
                    kernel_revision="b26c936cb02fb91920cbccfc7454284b47d86b0d")
    spec["identity"] = digest(spec)
    atomic(root / "replay-spec.json", spec)
    with zipfile.ZipFile(root / "inputs.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((root / "inputs").iterdir()):
            archive.write(path, path.name)
    print(json.dumps({"model": model, "traces": len(dataset),
                      "sentences": sum(len(t["units"]) for t in dataset),
                      "tokens": sum(j["native_context_tokens"] for j in jobs),
                      "max_tokens": max(j["native_context_tokens"] for j in jobs), "out": str(root)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=list(MODEL_KEYS))
    prepare(parser.parse_args().model)
