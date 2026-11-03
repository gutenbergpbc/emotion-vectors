"""Freeze Haiku-labeled Gemma 27B coding/sales traces for causal token replay."""
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
CANONICAL = HERE.parent
ROOT = CANONICAL.parents[1]
sys.path[:0] = [str(CANONICAL), str(CANONICAL / "emotion-probes")]
from shared.io import atomic, digest, read, require, sha  # noqa: E402
from emotion_probes.gemma_rendering import render_case  # noqa: E402
from emotion_rh_forecasting.inputs import load_probes, Sources, vector_identity  # noqa: E402
from emotion_rh_forecasting.terminalwrench_replay import sentence_cuts  # noqa: E402

PROBE_SOURCE = ROOT / "runs/emotion-rh-replay/gemma-3-27b-terminalwrench-best12-pilot50-v2/probes.json"


def messages_for(source: dict, dataset: str) -> tuple[list[dict], dict[int, int]]:
    if dataset == "coding":
        return ([{"role": "user", "content": source["prompt"]},
                 {"role": "assistant", "content": source["response"].strip(),
                  "reasoning": (source.get("reasoning_text") or "").strip()}], {0: 1})
    messages = [dict(m) for m in source["initial_messages"]]
    positions = {}
    for index, turn in enumerate(source["turns"]):
        positions[turn["turn"]] = len(messages)
        messages.append({"role": "assistant", "content": turn["response"].strip(),
                         "reasoning": (turn.get("reasoning_text") or "").strip()})
        if index < len(source["turns"]) - 1:
            messages.append({"role": "user", "content": turn["feedback"]})
    return messages, positions


def render_span(messages: list[dict], position: int, stream: str, tokenizer) -> tuple[str, int, int]:
    body = messages[position][stream]
    require(body, "Empty assistant stream")
    case = {"id": "gemma27b-replay", "render_mode": "native", "messages": messages[:position+1],
            "target": {"message_index": position, "stream": stream,
                       "char_start": 0, "char_end": len(body), "text": body}}
    return render_case(case, tokenizer)


def make_job(trace: dict, tokenizer) -> tuple[dict, dict, dict]:
    source_path = ROOT / trace["source_path"]
    require(sha(source_path) == trace["source_sha256"], "Source changed")
    source = read(source_path)
    original_path = ROOT / source["original_source_path"]
    require(sha(original_path) == source["original_source_sha256"], "Original source changed")
    messages, positions = messages_for(source, trace["dataset"])
    # Rendering each assistant prefix maps exact source offsets into one final
    # Gemma chat-template transcript without exposing later turns at readout.
    spans = {}
    rendered = None
    for turn, position in positions.items():
        for field, stream in (("reasoning_text", "reasoning"), ("response", "content")):
            if not messages[position][stream]:
                continue
            prefix, start, end = render_span(messages, position, stream, tokenizer)
            if rendered is not None:
                require(prefix.startswith(rendered) or rendered.startswith(prefix), "Prefix serialization changed")
            if rendered is None or len(prefix) > len(rendered):
                rendered = prefix
            spans[(turn, field)] = (start, end)
    require(rendered, "No assistant text")
    units = []
    for unit in trace["units"]:
        if unit["kind"] != "sentence_or_fragment":
            continue
        key = (unit["turn"], unit["source_field"])
        require(key in spans, "Sentence stream missing")
        source_body = (source["response"] if trace["dataset"] == "coding" else
                       next(t["response"] if unit["source_field"] == "response" else t.get("reasoning_text") or ""
                            for t in source["turns"] if t["turn"] == unit["turn"]))
        if trace["dataset"] == "coding" and unit["source_field"] == "reasoning_text":
            source_body = source.get("reasoning_text") or ""
        leading = len(source_body) - len(source_body.lstrip())
        start = spans[key][0] + unit["char_start"] - leading
        end = spans[key][0] + unit["char_end"] - leading
        require(rendered[start:end] == unit["text"], "Rendered sentence differs from label")
        units.append({**unit, "source_char_start": unit["char_start"],
                      "source_char_end": unit["char_end"], "char_start": start,
                      "char_end": end, "turn_id": trace["id"]})
    require(units, "No prose units")
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = np.asarray(encoded["input_ids"], dtype=np.int64)
    offsets = np.asarray(encoded["offset_mapping"], dtype=np.int64)
    require(len(ids) == len(offsets), "Tokenizer offsets differ")
    kind = np.zeros(len(ids), dtype=np.int8)
    for start, end in spans.values():
        kind[(offsets[:, 1] > start) & (offsets[:, 0] < end)] = 1
    sentence_cuts(offsets, units)
    job = {"id": trace["id"], "source_sha256": trace["source_sha256"],
           "rendered": rendered, "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
           "source_boundaries": [{"unit_id": u["unit_id"], "char_start": u["char_start"],
                                  "char_end": u["char_end"]} for u in units],
           "native_context_tokens": len(ids)}
    arrays = {"token_ids": ids, "offsets": offsets, "kind": kind}
    dataset_trace = {"id": trace["id"], "task": trace["task"], "group_id": trace["group_id"],
                     "turn_ids": [trace["id"]], "units": units}
    return job, arrays, dataset_trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("coding", "sales"), required=True)
    args = parser.parse_args()
    dataset = args.dataset
    labels_root = ROOT / f"runs/emotion-rh-sentence-labels/gemma3-27b-{dataset}-haiku55-v1"
    labels_path = labels_root / "labels.json"
    labels = read(labels_path)
    require(labels["summary"]["status"] == "complete", "Incomplete Haiku labels")
    require(len(labels["traces"]) == labels["summary"]["expected_traces"], "Missing traces")
    folds = {"labels_sha256": sha(labels_path), "method": "held_out_scenario",
             "trace_folds": {t["id"]: sorted({r["scenario"] for r in labels["traces"]}).index(t["scenario"])
                             for t in labels["traces"]}}
    folds_path = labels_root / "folds.json"
    if folds_path.exists() and read(folds_path) != folds:
        old = read(folds_path)
        require(all(key.replace("/", "__") in folds["trace_folds"] and
                    folds["trace_folds"][key.replace("/", "__")] == value
                    for key, value in old["trace_folds"].items()), "Fold assignment changed")
    atomic(folds_path, folds)
    bundle = load_probes(PROBE_SOURCE, Sources())
    require(bundle["model"]["name"] == "google/gemma-3-27b-it", "Wrong probes")
    tokenizer = AutoTokenizer.from_pretrained(bundle["model"]["name"],
                                              revision=bundle["model"]["revision"], use_fast=True)
    require(tokenizer.is_fast, "Fast tokenizer required")
    version = 'v3' if dataset == 'coding' else 'v4'
    out = ROOT / f"runs/emotion-rh-replay/gemma3-27b-{dataset}-haiku55-{version}"
    out.mkdir(parents=True, exist_ok=True)
    atomic(out / "probes.json", bundle)
    jobs, traces = [], []
    for trace in labels["traces"]:
        job, arrays, dataset_trace = make_job(trace, tokenizer)
        dataset_trace["fold"] = folds["trace_folds"][trace["id"]]
        array_path = out / "inputs" / f"{trace['id']}.npz"
        array_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(array_path, **arrays)
        job["input_sha256"] = sha(array_path)
        atomic(out / "inputs" / f"{trace['id']}.json", job)
        jobs.append({k: v for k, v in job.items() if k not in ("rendered", "source_boundaries")})
        traces.append(dataset_trace)
    atomic(out / "dataset-traces.json", traces)
    spec = {"schema_version": 1, "config": {"labels": str(labels_root)}, "model": bundle["model"],
            "probe_identity": vector_identity(bundle), "probes_sha256": sha(out / "probes.json"),
            "jobs": jobs, "labels_sha256": sha(labels_path), "folds_sha256": sha(folds_path),
            "dataset_traces_sha256": sha(out / "dataset-traces.json"),
            "renderer": "gemma-native-preserved-streams-v2",
            "interpretation": "Pinned Gemma 27B native-template replay of archived outputs; retrospective Haiku 5.5 sentence labels. Coding has a verbose instruction; sales has its original prompt. Replay activations are reconstructed, not original serving activations.",
            "chunk_tokens": 512, "max_tokens": 131072,
            **({"independent_prefix_tolerance": .05} if dataset == "sales" else {}),
            "source_hashes": {p.name: sha(p) for p in (Path(__file__), HERE / "modal_terminalwrench_replay.py", HERE / "emotion_rh_forecasting/terminalwrench_replay.py")}}
    spec["identity"] = digest(spec)
    spec_path = out / "replay-spec.json"
    if spec_path.exists():
        require(read(spec_path)["identity"] == spec["identity"], "Replay identity changed")
    atomic(spec_path, spec)
    with zipfile.ZipFile(out / "inputs.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((out / "inputs").glob("*")):
            archive.write(path, path.name)
    print(json.dumps({"dataset": dataset, "traces": len(traces),
                      "sentences": sum(len(t["units"]) for t in traces),
                      "tokens": sum(j["native_context_tokens"] for j in jobs),
                      "max_tokens": max(j["native_context_tokens"] for j in jobs),
                      "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
