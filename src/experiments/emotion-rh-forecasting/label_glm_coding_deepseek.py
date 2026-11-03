"""Freeze and label ~1,000 GLM coding prose units with DeepSeek V4.1 Flash 1017.

Run from model-forensics:
  uv run --with pysbd python src/experiments/emotion-rh-forecasting/label_glm_coding_deepseek.py --prepare-only
  uv run --with pysbd python src/experiments/emotion-rh-forecasting/label_glm_coding_deepseek.py --limit-requests 2
  uv run --with pysbd python src/experiments/emotion-rh-forecasting/label_glm_coding_deepseek.py
"""
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
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

import label_glm_hack_traces as base


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "runs/emotion-rh-sentence-labels/glm53-coding-deepseek-v41-flash-1017-v1"
NEW = ROOT / "runs/emotion-rh-traces/glm53-coding-diverse-openrouter-v1/traces"
PREVIOUS = [
    ROOT / "runs/emotion-rh-sentence-labels/glm53-hack-traces-gpt55-v1/labels.json",
    ROOT / "runs/emotion-rh-sentence-labels/glm53-nonhack-traces-gpt55-v1/labels.json",
]
MODEL = "deepseek/deepseek-v4.1-flash-1017"
TARGET_PROSE = 1000


def source_item(path: Path) -> dict:
    raw = path.read_bytes()
    trace = json.loads(raw)
    assert trace["status"] == "complete" and trace["model_requested"] == "z-ai/glm-5.3"
    return {"id": trace["id"], "dataset": "coding", "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(raw).hexdigest(), "scenario": trace.get("scenario", "original")}


def prose_count(item: dict) -> int:
    trace = base.read_source(ROOT / item["path"], item["sha256"])
    return sum(unit["kind"] == "sentence_or_fragment"
               for field in ("reasoning_text", "response")
               for unit in base.segment(trace.get(field) or ""))


def selection() -> dict:
    items = []
    seen = set()
    for labels_path in PREVIOUS:
        labels = base.read_source(labels_path)
        for trace in labels["traces"]:
            if trace["dataset"] != "coding":
                continue
            item = source_item(ROOT / trace["source_path"])
            assert item["sha256"] == trace["source_sha256"]
            assert item["id"] not in seen
            seen.add(item["id"])
            items.append(item)
    assert len(items) == 23
    prose = sum(prose_count(item) for item in items)
    assert prose == 306
    grouped = defaultdict(list)
    for path in sorted(NEW.glob("*/*.json")):
        item = source_item(path)
        assert item["id"] not in seen
        grouped[item["scenario"]].append(item)
    assert len(grouped) == 6 and all(len(group) == 12 for group in grouped.values())
    # The order depends only on frozen IDs, never the generated text or judgments.
    for sample in range(12):
        for scenario in sorted(grouped):
            item = grouped[scenario][sample]
            item["prose_units"] = prose_count(item)
            items.append(item)
            prose += item["prose_units"]
            if prose >= TARGET_PROSE:
                break
        if prose >= TARGET_PROSE:
            break
    assert prose >= TARGET_PROSE
    for item in items:
        item.setdefault("prose_units", prose_count(item))
    assert len({item["id"] for item in items}) == len(items)
    return {"model": MODEL, "target_prose": TARGET_PROSE, "selected_prose": prose,
            "selection_rule": "All 23 previously labeled coding traces, then new scenarios in sorted round-robin order by sample index until at least 1,000 prose units; no new labels used.",
            "selected": items}


def label_with_recovery(case: dict, chunk: dict) -> dict:
    try:
        return base.label_chunk(case, chunk)
    except RuntimeError as exc:
        if "Incomplete: length" not in str(exc):
            raise
    # A rare DeepSeek response spends its whole completion budget reasoning.
    # Split only that request's target units; retain the original full context.
    key = case["id"].replace("/", "__") + f"__{chunk['start']:04d}"
    recovered = []
    recovery_paths = []
    for offset in range(0, len(chunk["units"]), 10):
        units = chunk["units"][offset:offset + 10]
        body = json.loads(json.dumps(chunk["body"]))
        body["reasoning"] = {"effort": "low", "exclude": True}
        body["messages"][1]["content"] = json.dumps(
            {"context": json.loads(chunk["body"]["messages"][1]["content"])["context"],
             "units_to_label": units}, ensure_ascii=False)
        body_sha = base.sha(json.dumps(body, ensure_ascii=False, sort_keys=True).encode())
        path = OUT / "recovery" / f"{key}__{offset:02d}.json"
        errors = []
        for attempt in range(4):
            try:
                if path.exists():
                    saved = base.read_source(path)
                    assert saved["request_sha256"] == body_sha
                    raw = saved["response"]
                else:
                    request = urllib.request.Request(
                        "https://openrouter.ai/api/v1/chat/completions",
                        data=json.dumps(body, ensure_ascii=False).encode(),
                        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
                                 "Content-Type": "application/json"})
                    with urllib.request.urlopen(request, timeout=300) as response:
                        raw = json.load(response)
                    base.write(path, {"request_sha256": body_sha, "original_request_sha256": chunk["request_sha256"],
                                      "request_body": body, "response": raw})
                assert raw["model"] == MODEL
                choice = raw["choices"][0]
                assert choice["finish_reason"] == "stop", f"Incomplete: {choice['finish_reason']}"
                labels = base.validate_labels(base.parse_output(choice["message"]["content"]), units)
                recovered.extend(labels)
                recovery_paths.append(str(path.relative_to(OUT)))
                break
            except Exception as sub_exc:
                errors.append(str(sub_exc))
                if path.exists():
                    path.rename(path.with_name(path.stem + f".failed{attempt + 1}.json"))
                time.sleep(min(2 ** attempt, 8))
        else:
            raise RuntimeError(f"Recovery failed for {key} offset {offset}: {errors}")
    base.validate_labels({"labels": recovered}, chunk["units"])
    result = {"trace_id": case["id"], "source_sha256": case["sha256"],
              "request_sha256": chunk["request_sha256"], "rubric_sha256": base.sha(base.RUBRIC.encode()),
              "model_requested": MODEL, "model_returned": MODEL,
              "created_at": datetime.now(timezone.utc).isoformat(),
              "usage": {"cost": sum((base.read_source(OUT / p)["response"].get("usage") or {}).get("cost", 0) or 0
                                     for p in recovery_paths)},
              "recovery_request_paths": recovery_paths, "labels": recovered}
    base.write(OUT / "annotations" / f"{key}.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--limit-requests", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    base.OUT = OUT
    base.MODEL = MODEL
    selected = selection()
    selection_path = OUT / "selection.json"
    if selection_path.exists():
        assert base.read_source(selection_path) == selected, "Frozen selection changed"
    else:
        base.write(selection_path, selected)
    rubric_path = OUT / "rubric.txt"
    if rubric_path.exists():
        assert rubric_path.read_text() == base.RUBRIC + "\n"
    else:
        rubric_path.write_text(base.RUBRIC + "\n")
    cases = base.prepare(selected["selected"])
    manifest = {"model": MODEL, "rubric_sha256": base.sha(base.RUBRIC.encode()),
                "selection_sha256": base.sha(selection_path.read_bytes()),
                "traces": len(cases), "prose_units": selected["selected_prose"],
                "all_units": sum(len(case["units"]) for case in cases),
                "requests": [{"trace_id": case["id"], "start": chunk["start"],
                              "request_sha256": chunk["request_sha256"]}
                             for case in cases for chunk in case["chunks"]]}
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        assert base.read_source(manifest_path) == manifest, "Frozen requests changed"
    else:
        base.write(manifest_path, manifest)
    print(json.dumps({key: manifest[key] for key in ("model", "traces", "prose_units", "all_units")}
                     | {"requests": len(manifest["requests"])}, indent=2), flush=True)
    if args.prepare_only:
        return
    assert os.environ.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY missing"
    jobs = [(case, chunk) for case in cases for chunk in case["chunks"]]
    if args.limit_requests is not None:
        jobs = jobs[:args.limit_requests]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(label_with_recovery, case, chunk): (case["id"], chunk["start"])
                   for case, chunk in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                failures.append({"job": futures[future], "error": str(exc)})
                print(f"FAILED {futures[future]}: {exc}", flush=True)
            if index % 10 == 0 or index == len(jobs):
                print(f"{index}/{len(jobs)} requests finished; {len(failures)} failed", flush=True)
    traces = []
    usage_cost = 0.0
    for case in cases:
        labels = []
        for chunk in case["chunks"]:
            key = case["id"].replace("/", "__") + f"__{chunk['start']:04d}"
            path = OUT / "annotations" / f"{key}.json"
            if not path.exists():
                break
            result = base.read_source(path)
            assert result["request_sha256"] == chunk["request_sha256"]
            labels.extend(base.validate_labels({"labels": result["labels"]}, chunk["units"]))
            usage_cost += (result.get("usage") or {}).get("cost", 0) or 0
        if len(labels) != len(case["units"]):
            continue
        traces.append({"id": case["id"], "dataset": "coding", "scenario": case.get("scenario", "original"),
                       "source_path": case["path"], "source_sha256": case["sha256"],
                       "units": [{**unit, **{key: labels[i][key] for key in ("hacking", "needs_review", "reason")}}
                                 for i, unit in enumerate(case["units"])]})
    counts = Counter()
    for trace in traces:
        counts[(trace["scenario"], "traces")] += 1
        for unit in trace["units"]:
            counts[(trace["scenario"], "all_units")] += 1
            if unit["kind"] == "sentence_or_fragment":
                counts[(trace["scenario"], "prose_units")] += 1
                counts[(trace["scenario"], "rh_prose_units")] += unit["hacking"]
                counts[(trace["scenario"], "review_prose_units")] += unit["needs_review"]
    by_scenario = {scenario: {key: counts[(scenario, key)] for key in
                             ("traces", "all_units", "prose_units", "rh_prose_units", "review_prose_units")}
                   for scenario in sorted({trace["scenario"] for trace in traces})}
    billed = {}
    for path in list((OUT / "attempts").rglob("*.json")) + list((OUT / "recovery").glob("*.json")):
        response = base.read_source(path).get("response") or {}
        billed[response.get("id") or str(path)] = (response.get("usage") or {}).get("cost", 0) or 0
    summary = {"status": "complete" if len(traces) == len(cases) else "partial",
               "created_at": datetime.now(timezone.utc).isoformat(), "judge_model": MODEL,
               "expected_traces": len(cases), "labeled_traces": len(traces),
               "expected_prose_units": selected["selected_prose"],
               "labeled_prose_units": sum(row["prose_units"] for row in by_scenario.values()),
               "rh_prose_units": sum(row["rh_prose_units"] for row in by_scenario.values()),
               "review_prose_units": sum(row["review_prose_units"] for row in by_scenario.values()),
               "by_scenario": by_scenario,
               "reported_api_cost_usd": round(sum(billed.values()), 6),
               "successful_response_cost_usd": round(usage_cost, 6),
               "rubric_sha256": manifest["rubric_sha256"],
               "method": "Full-trace post-hoc model labels for every assistant output unit. Prose count excludes code blocks and headings. Original GPT-5.5 labels are not reused.",
               "caveat": "Model judgments, not human ground truth; selected coding traces are not prevalence-representative."}
    base.write(OUT / "summary.json", summary)
    base.write(OUT / "labels.json", {"summary": summary, "traces": traces})
    base.write(OUT / "failures.json", failures)
    print(json.dumps(summary, indent=2), flush=True)
    if failures or (args.limit_requests is None and summary["status"] != "complete"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
