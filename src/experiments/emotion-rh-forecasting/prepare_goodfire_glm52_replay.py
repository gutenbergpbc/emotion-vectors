"""Freeze native GLM 5.2 NVFP4 replay of Goodfire FP8 archive text."""
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


from bisect import bisect_left, bisect_right
import hashlib
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
CANONICAL = HERE.parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(CANONICAL), str(HERE)]
from shared.io import atomic, digest, read, require, sha  # noqa: E402
from emotion_rh_forecasting.inputs import Sources, load_probes, vector_identity  # noqa: E402
import label_goodfire_glm52_deepseek_v2 as labels  # noqa: E402

PROBES = ROOT / "runs/emotion-probes/glm-5.2-nvfp4-diversity-v2/probes"
OUT = ROOT / "runs/emotion-rh-replay/goodfire-glm52-deepswe-nvfp4-best12-pilot-v2"


def messages_and_spans(trace, units, tokenizer, node_indices):
    nodes = trace["nodes"]
    fields = {(u["node_index"], u["channel"]) for u in units}
    markers = {}
    unmarked, marked = [], []
    for node_index in node_indices:
        node = nodes[node_index]
        raw = node.get("message") or {}
        role = raw.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            continue
        plain = {"role": role, "content": raw.get("content") or ""}
        if role == "assistant":
            plain["reasoning_content"] = raw.get("reasoning_content") or ""
            calls = []
            for call in raw.get("tool_calls") or []:
                arguments = call.get("arguments") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                require(isinstance(arguments, dict), "Non-object tool arguments")
                calls.append({"id": call.get("id"), "type": "function",
                              "function": {"name": call["name"], "arguments": arguments}})
            if calls:
                plain["tool_calls"] = calls
        elif role == "tool":
            if raw.get("name"):
                plain["name"] = raw["name"]
            if raw.get("tool_call_id"):
                plain["tool_call_id"] = raw["tool_call_id"]
        flagged = dict(plain)
        if role == "assistant":
            for channel, key in (("reasoning", "reasoning_content"), ("content", "content")):
                if (node_index, channel) not in fields:
                    continue
                body = plain[key]
                require(isinstance(body, str) and body.strip(), "Missing marked assistant text")
                begin = f"\ue000GF_{node_index}_{channel}_START\ue001"
                end = f"\ue000GF_{node_index}_{channel}_END\ue001"
                require(begin not in body and end not in body, "Marker in source text")
                flagged[key] = begin + body + end
                markers[(node_index, channel)] = (begin, end, body)
        unmarked.append(plain)
        marked.append(flagged)
    kwargs = {"tokenize": False, "add_generation_prompt": False,
              "clear_thinking": False, "reasoning_effort": "max"}
    rendered = tokenizer.apply_chat_template(unmarked, **kwargs)
    rendered_marked = tokenizer.apply_chat_template(marked, **kwargs)
    stripped = rendered_marked
    for begin, end, _ in markers.values():
        require(stripped.count(begin) == stripped.count(end) == 1, "Template lost marker")
        stripped = stripped.replace(begin, "").replace(end, "")
    require(stripped == rendered, "Marker round trip changed GLM rendering")
    spans = {}
    removed = 0
    events = sorted((rendered_marked.index(marker), key, kind, marker)
                    for key, (begin, end, _) in markers.items()
                    for kind, marker in (("begin", begin), ("end", end)))
    for position, key, kind, marker in events:
        if kind == "begin":
            spans[key] = [position - removed, None]
        else:
            spans[key][1] = position - removed
        removed += len(marker)
    for key, (begin, end) in spans.items():
        require(rendered[begin:end] == markers[key][2], f"Source span changed in template: {key}")
    return rendered, spans


def job_for(trace, case, tokenizer, units, node_indices, job_id):
    rendered, spans = messages_and_spans(trace, units, tokenizer, node_indices)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    require(len(ids) == len(offsets) and len(ids) <= 131072, "Trace exceeds GLM context")
    starts = [a for a, _ in offsets]
    ends = [b for _, b in offsets]
    assistant_tokens = sorted({i for a, b in spans.values()
                               for i in range(bisect_right(ends, a), bisect_left(starts, b))
                               if offsets[i][1] > offsets[i][0] and offsets[i][1] > a and offsets[i][0] < b})
    positions = {i: n for n, i in enumerate(assistant_tokens)}
    replay_units = []
    for unit in units:
        field_start, _ = spans[(unit["node_index"], unit["channel"])]
        a, b = field_start + unit["char_start"], field_start + unit["char_end"]
        require(rendered[a:b] == unit["text"], "Sentence offset changed")
        selected = [(i, min(offsets[i][1], b) - max(offsets[i][0], a))
                    for i in range(bisect_right(ends, a), bisect_left(starts, b))
                    if offsets[i][1] > offsets[i][0] and offsets[i][1] <= b
                    and i in positions and min(offsets[i][1], b) > max(offsets[i][0], a)]
        fallback = not selected
        if fallback:
            prior = [i for i in assistant_tokens if offsets[i][1] <= b]
            require(prior, "No prior causal assistant token")
            selected = [(prior[-1], 1)]
        replay_units.append({"unit_id": unit["unit_id"], "node_index": unit["node_index"],
                             "channel": unit["channel"], "char_start": unit["char_start"],
                             "char_end": unit["char_end"], "text": unit["text"],
                             "token_indices": [i for i, _ in selected],
                             "token_weights": [w for _, w in selected],
                             "output_positions": [positions[i] for i, _ in selected],
                             "prior_token_fallback": fallback})
    return {"id": job_id, "trace_id": trace["id"], "row_index": case["row_index"], "task": case["task"],
            "source_line_sha256": case["sha256"], "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "token_ids": ids, "token_ids_sha256": digest(ids), "tokens": len(ids),
            "output_tokens": len(assistant_tokens), "units": replay_units}


def main():
    selection = labels.select()
    require(read(labels.OUT / "selection.json") == selection, "Selection changed")
    cases = labels.prepare(selection)
    bundle = load_probes(PROBES, Sources())
    require(bundle["model"]["name"] == "nvidia/GLM-5.2-NVFP4", "Wrong probe model")
    tokenizer = AutoTokenizer.from_pretrained(bundle["model"]["tokenizer_model"],
                                               revision=bundle["model"]["tokenizer_revision"])
    require(tokenizer.is_fast, "Fast offset tokenizer required")
    OUT.mkdir(parents=True, exist_ok=True)
    probes_path = OUT / "probes.json"
    if probes_path.exists():
        require(read(probes_path) == bundle, "Probes changed")
    else:
        atomic(probes_path, bundle)
    by_row = {c["row_index"]: c for c in cases}
    jobs = []
    with (labels.SOURCE / "traces.jsonl").open("rb") as stream:
        for row_index, raw in enumerate(stream):
            if row_index not in by_row:
                continue
            case = by_row[row_index]
            require(hashlib.sha256(raw).hexdigest() == case["sha256"], "Trace changed")
            trace = json.loads(raw)["traces"][0]
            all_nodes = list(range(len(trace["nodes"])))
            try:
                trace_jobs = [job_for(trace, case, tokenizer, case["units"], all_nodes, trace["id"])]
            except ValueError as exc:
                if "Trace exceeds GLM context" not in str(exc):
                    raise
                lengths = [len(str((node.get("message") or {}).get("content") or "")) +
                           len(str((node.get("message") or {}).get("reasoning_content") or ""))
                           for node in trace["nodes"]]
                total = sum(lengths)
                running = 0
                cut = 2
                for index, length in enumerate(lengths):
                    running += length
                    if (running >= total / 2 and index > 2 and
                            (trace["nodes"][index].get("message") or {}).get("role") == "assistant"):
                        cut = index
                        break
                left = [u for u in case["units"] if u["node_index"] < cut]
                right = [u for u in case["units"] if u["node_index"] >= cut]
                require(left and right, "Context split lost sentences")
                trace_jobs = [job_for(trace, case, tokenizer, left, list(range(cut)), trace["id"] + "__part0"),
                              job_for(trace, case, tokenizer, right, [0, 1, *range(cut, len(all_nodes))],
                                      trace["id"] + "__part1")]
            for part, job in enumerate(trace_jobs):
                path = OUT / "inputs" / f"{job['id']}.json"
                if path.exists():
                    require(read(path) == job, "Replay job changed")
                else:
                    atomic(path, job)
                jobs.append({"id": job["id"], "trace_id": trace["id"], "part": part,
                             "sha256": sha(path), "row_index": row_index,
                             "task": case["task"], "tokens": job["tokens"],
                             "output_tokens": job["output_tokens"], "units": len(job["units"])})
                print(f"Prepared {job['id']}: {job['tokens']} tokens", flush=True)
    require(len(jobs) >= 12 and sum(j["units"] for j in jobs) == 12143, "Sentence count changed")
    spec = {"schema_version": 1, "model": bundle["model"], "probe_identity": vector_identity(bundle),
            "probes_sha256": sha(probes_path), "source_probe_manifest_sha256": sha(PROBES / "manifest.json"),
            "selection_sha256": sha(labels.OUT / "selection.json"),
            "label_request_manifest_sha256": sha(labels.OUT / "manifest.json"),
            "tokenizer_template_sha256": digest(tokenizer.get_chat_template()),
            "interpretation": "Native NVFP4 GLM replay of archived FP8 DeepSWE message/tool text. Traces over 131072 tokens use two non-overlapping temporal windows with system/user context repeated and earlier tool history omitted from the second window. Exploratory quantization transfer; these are not original FP8 activations.",
            "jobs": jobs, "total_tokens": sum(j["tokens"] for j in jobs),
            "total_sentences": sum(j["units"] for j in jobs)}
    spec["identity"] = digest(spec)
    path = OUT / "replay-spec.json"
    if path.exists():
        require(read(path) == spec, "Replay spec changed")
    else:
        atomic(path, spec)
    print(json.dumps({"jobs": len(jobs), "tokens": spec["total_tokens"],
                      "max_tokens": max(j["tokens"] for j in jobs), "sentences": spec["total_sentences"]}, indent=2))


if __name__ == "__main__":
    main()
