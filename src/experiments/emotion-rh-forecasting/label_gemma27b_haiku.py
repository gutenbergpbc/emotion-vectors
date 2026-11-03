"""Label archived Gemma 27B verbose coding and sales traces with Haiku.

Run from the repository root with ``uv run --with pysbd python ...``.  The
frozen input bridge keeps the original source path and hash for auditability.
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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import label_glm_hack_traces as base

ROOT = Path(__file__).resolve().parents[3]
MODEL = "anthropic/claude-haiku-5.5"
TRACE_MODEL = "google/gemma-3-27b-it"


def freeze_sources(dataset: str, out: Path) -> list[dict]:
    if dataset == "coding":
        root = ROOT / "runs/emotion-rh-traces/gemma3-27b-coding-verbose-openrouter-v1"
        paths = sorted((root / "traces").glob("*/*.json"))
        assert len(paths) == 66 and base.read_source(root / "stopped.json")["saved_traces"] == 66
    else:
        root = ROOT / "runs/emotion-rh-traces/gemma3-27b-sales-extension-openrouter-v1"
        cohort = base.read_source(root / "cohort-100.json")
        assert len(cohort["traces"]) == 100
        paths = [Path(item["path"]) for item in cohort["traces"]]
        for item, path in zip(cohort["traces"], paths):
            assert base.sha(path.read_bytes()) == item["sha256"]
    selected = []
    for path in paths:
        source = base.read_source(path)
        assert source.get("status", "complete") == "complete"
        original_sha = base.sha(path.read_bytes())
        if dataset == "coding":
            assert source["condition"] == "verbose_prompt"
            normalized = {k: source[k] for k in ("id", "scenario", "prompt", "response", "reasoning_text")}
            normalized["reasoning_text"] = normalized["reasoning_text"] or ""
        else:
            initial = source.get("initial_messages") or source["messages"][:2]
            assert [m["role"] for m in initial] == ["system", "user"]
            normalized = {"id": source["id"], "scenario": source["scenario"],
                          "initial_messages": initial,
                          "turns": [{"turn": t["turn"], "response": t["response"],
                                     "reasoning_text": t.get("reasoning_text") or "",
                                     "feedback": t.get("feedback") or ""} for t in source["turns"]]}
        normalized.update(dataset=dataset, model_requested=TRACE_MODEL,
                          original_source_path=str(path.relative_to(ROOT)), original_source_sha256=original_sha)
        dest = out / "sources" / dataset / source["scenario"] / (path.stem + ("-local" if "emotion-steering" in str(path) else "-openrouter") + ".json")
        if dest.exists():
            assert base.read_source(dest) == normalized, f"Frozen source changed: {dest}"
        else:
            base.write(dest, normalized)
        selected.append({"id": normalized["id"], "dataset": dataset,
                         "scenario": normalized["scenario"], "path": str(dest.relative_to(ROOT)),
                         "sha256": base.sha(dest.read_bytes())})
    assert len({item["id"] for item in selected}) == len(selected)
    return selected


def summarize(cases: list[dict], out: Path, dataset: str) -> dict:
    traces = []
    for case in cases:
        labels = []
        for chunk in case["chunks"]:
            key = case["id"].replace("/", "__") + f"__{chunk['start']:04d}"
            path = out / "annotations" / f"{key}.json"
            if not path.exists():
                break
            saved = base.read_source(path)
            assert saved["request_sha256"] == chunk["request_sha256"]
            labels += base.validate_labels({"labels": saved["labels"]}, chunk["units"])
        if len(labels) == len(case["units"]):
            safe_id = case["id"].replace("/", "__")
            traces.append({"id": safe_id, "original_id": case["id"],
                           "dataset": dataset, "scenario": case["scenario"],
                           "task": case["scenario"], "group_id": safe_id,
                           "source_path": case["path"], "source_sha256": case["sha256"],
                           "units": [{**unit, **{k: labels[i][k] for k in ("hacking", "needs_review", "reason")}}
                                     for i, unit in enumerate(case["units"])]})
    counts = Counter()
    for trace in traces:
        for unit in trace["units"]:
            counts["all_units"] += 1
            if unit["kind"] == "sentence_or_fragment":
                counts["prose_units"] += 1
                counts["rh_prose_units"] += unit["hacking"]
                counts["review_prose_units"] += unit["needs_review"]
    calls = {}
    for path in (out / "attempts").rglob("*.json"):
        response = base.read_source(path).get("response") or {}
        calls[response.get("id") or str(path)] = (response.get("usage") or {}).get("cost", 0) or 0
    summary = {"status": "complete" if len(traces) == len(cases) else "partial",
               "judge_model": MODEL, "trace_model": TRACE_MODEL, "dataset": dataset,
               "condition": "verbose_prompt" if dataset == "coding" else "original_sales_prompt",
               "expected_traces": len(cases), "labeled_traces": len(traces),
               "expected_prose_units": sum(u["kind"] == "sentence_or_fragment" for c in cases for u in c["units"]),
               **counts, "reported_api_cost_usd_including_retries": round(sum(calls.values()), 6),
               "rubric_sha256": base.sha(base.RUBRIC.encode()),
               "caveat": "Retrospective Haiku judgments, not human ground truth. Coding uses an added verbose prompt; sales uses the original prompt."}
    base.write(out / "summary.json", summary)
    base.write(out / "labels.json", {"summary": summary, "traces": traces})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("coding", "sales"), required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--limit-requests", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    dataset = args.dataset
    out = ROOT / f"runs/emotion-rh-sentence-labels/gemma3-27b-{dataset}-haiku55-v1"
    out.mkdir(parents=True, exist_ok=True)
    base.OUT, base.MODEL = out, MODEL
    selected = freeze_sources(dataset, out)
    cases = base.prepare(selected, expected_model=TRACE_MODEL)
    rubric = out / "rubric.txt"
    if rubric.exists():
        assert rubric.read_text() == base.RUBRIC + "\n"
    else:
        rubric.write_text(base.RUBRIC + "\n")
    manifest = {"model": MODEL, "dataset": dataset, "condition": "verbose_prompt" if dataset == "coding" else "original_sales_prompt",
                "selected": selected, "rubric_sha256": base.sha(base.RUBRIC.encode()),
                "requests": [{"trace_id": c["id"], "start": ch["start"], "request_sha256": ch["request_sha256"]}
                             for c in cases for ch in c["chunks"]]}
    path = out / "manifest.json"
    if path.exists():
        assert base.read_source(path) == manifest, "Frozen manifest changed"
    else:
        base.write(path, manifest)
    jobs = [(case, chunk) for case in cases for chunk in case["chunks"]]
    print(json.dumps({"dataset": dataset, "traces": len(cases), "prose_units": sum(u["kind"] == "sentence_or_fragment" for c in cases for u in c["units"]), "requests": len(jobs)}), flush=True)
    if args.prepare_only:
        return
    if args.limit_requests is not None:
        jobs = jobs[:args.limit_requests]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(base.label_chunk, c, ch): (c["id"], ch["start"]) for c, ch in jobs}
        for n, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                failures.append({"job": futures[future], "error": str(exc)})
                print("FAILED", futures[future], str(exc), flush=True)
            if n % 10 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} requests finished; {len(failures)} failed", flush=True)
    base.write(out / "failures.json", failures)
    summary = summarize(cases, out, dataset)
    print(json.dumps(summary, indent=2), flush=True)
    if failures or (args.limit_requests is None and summary["status"] != "complete"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
