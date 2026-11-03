"""Label the 731 new GLM coding prose units with GPT-5.5 under a $7 run cap.

The 306 older coding prose units were already labeled by GPT-5.5. This run
uses the same frozen rubric, trace context, segmentation and request format.
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
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

import label_glm_hack_traces as base


ROOT = Path(__file__).resolve().parents[3]
SOURCE_SELECTION = ROOT / "runs/emotion-rh-sentence-labels/glm53-coding-deepseek-v41-flash-1017-v1/selection.json"
OUT = ROOT / "runs/emotion-rh-sentence-labels/glm53-coding-new-gpt55-v1"
MODEL = "openai/gpt-5.5"
BUDGET_USD = 7.0
# Standard OpenRouter GPT-5.5 rates. The bound intentionally treats every UTF-8
# request byte as an input token, even though normal text tokenizes much shorter.
INPUT_USD_PER_TOKEN_UPPER = 5 / 1_000_000
OUTPUT_USD_PER_TOKEN_UPPER = 30 / 1_000_000


def selected() -> list[dict]:
    source = base.read_source(SOURCE_SELECTION)
    assert source["selected_prose"] == 1037
    new = [item for item in source["selected"]
           if item["path"].startswith("runs/emotion-rh-traces/glm53-coding-diverse-openrouter-v1/")]
    assert len(new) == 28 and sum(item["prose_units"] for item in new) == 731
    return new


def charged_cost() -> float:
    calls = {}
    for path in (OUT / "attempts").rglob("*.json"):
        response = base.read_source(path).get("response") or {}
        calls[response.get("id") or str(path)] = (response.get("usage") or {}).get("cost", 0) or 0
    return sum(calls.values())


def conservative_call_bound(body: dict) -> float:
    request_bytes = len(json.dumps(body, ensure_ascii=False).encode())
    return 1.1 * (request_bytes * INPUT_USD_PER_TOKEN_UPPER
                  + body["max_tokens"] * OUTPUT_USD_PER_TOKEN_UPPER)


def request_once(body: dict, path: Path, original_request_sha256: str) -> dict:
    body_sha = base.sha(json.dumps(body, ensure_ascii=False, sort_keys=True).encode())
    if path.exists():
        saved = base.read_source(path)
        assert saved["body_sha256"] == body_sha
        return saved["response"]
    spent = charged_cost()
    bound = conservative_call_bound(body)
    if spent + bound > BUDGET_USD:
        raise RuntimeError(f"BUDGET_STOP: spent ${spent:.4f}; next-call upper bound ${bound:.4f}; cap ${BUDGET_USD:.2f}")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        raw = json.load(response)
    # Save every billed response before parsing or validating it.
    base.write(path, {"body_sha256": body_sha, "original_request_sha256": original_request_sha256,
                      "response": raw})
    return raw


def validated(raw: dict, units: list[dict]) -> list[dict]:
    assert raw["model"] == MODEL, f"Unexpected model: {raw.get('model')}"
    choice = raw["choices"][0]
    if choice["finish_reason"] == "content_filter" or choice.get("message", {}).get("refusal"):
        raise RuntimeError("Provider refusal/content filter")
    assert choice["finish_reason"] == "stop", f"Incomplete: {choice['finish_reason']}"
    return base.validate_labels(base.parse_output(choice["message"]["content"]), units)


def label_chunk(case: dict, chunk: dict) -> None:
    key = case["id"].replace("/", "__") + f"__{chunk['start']:04d}"
    dest = OUT / "annotations" / f"{key}.json"
    if dest.exists():
        result = base.read_source(dest)
        assert result["request_sha256"] == chunk["request_sha256"]
        base.validate_labels({"labels": result["labels"]}, chunk["units"])
        return
    attempts = []
    for attempt in range(2):
        path = OUT / "attempts" / key / f"full-{attempt + 1:02d}.json"
        try:
            raw = request_once(chunk["body"], path, chunk["request_sha256"])
            labels = validated(raw, chunk["units"])
            attempts.append(str(path.relative_to(OUT)))
            break
        except Exception as exc:
            if "BUDGET_STOP" in str(exc):
                raise
            attempts.append(f"{path.relative_to(OUT)}: {exc}")
            if "Provider refusal" in str(exc):
                raise
            time.sleep(attempt + 1)
    else:
        # On repeated malformed/truncated completions, label smaller target
        # chunks while retaining the exact same full-trace context and rubric.
        labels = []
        original_payload = json.loads(chunk["body"]["messages"][1]["content"])
        for offset in range(0, len(chunk["units"]), 10):
            units = chunk["units"][offset:offset + 10]
            body = json.loads(json.dumps(chunk["body"]))
            body["messages"][1]["content"] = json.dumps(
                {"context": original_payload["context"], "units_to_label": units}, ensure_ascii=False)
            for attempt in range(2):
                path = OUT / "attempts" / key / f"split-{offset:02d}-{attempt + 1:02d}.json"
                try:
                    raw = request_once(body, path, chunk["request_sha256"])
                    labels.extend(validated(raw, units))
                    attempts.append(str(path.relative_to(OUT)))
                    break
                except Exception as exc:
                    if "BUDGET_STOP" in str(exc):
                        raise
                    attempts.append(f"{path.relative_to(OUT)}: {exc}")
                    if attempt:
                        raise
            else:
                raise RuntimeError(f"Failed subchunk {key} offset {offset}")
        base.validate_labels({"labels": labels}, chunk["units"])
    base.write(dest, {"trace_id": case["id"], "source_sha256": case["sha256"],
                      "request_sha256": chunk["request_sha256"],
                      "rubric_sha256": base.sha(base.RUBRIC.encode()),
                      "model_requested": MODEL, "model_returned": MODEL,
                      "created_at": datetime.now(timezone.utc).isoformat(),
                      "attempt_paths": attempts, "labels": labels})


def summarize(cases: list[dict]) -> dict:
    traces = []
    for case in cases:
        labels = []
        for chunk in case["chunks"]:
            key = case["id"].replace("/", "__") + f"__{chunk['start']:04d}"
            path = OUT / "annotations" / f"{key}.json"
            if not path.exists():
                break
            annotation = base.read_source(path)
            assert annotation["request_sha256"] == chunk["request_sha256"]
            labels.extend(base.validate_labels({"labels": annotation["labels"]}, chunk["units"]))
        if len(labels) == len(case["units"]):
            traces.append({"id": case["id"], "dataset": "coding", "scenario": case["scenario"],
                           "source_path": case["path"], "source_sha256": case["sha256"],
                           "units": [{**unit, **{k: labels[i][k] for k in ("hacking", "needs_review", "reason")}}
                                     for i, unit in enumerate(case["units"])]})
    counts = Counter()
    for trace in traces:
        counts[(trace["scenario"], "traces")] += 1
        for unit in trace["units"]:
            if unit["kind"] == "sentence_or_fragment":
                counts[(trace["scenario"], "prose")] += 1
                counts[(trace["scenario"], "rh")] += unit["hacking"]
                counts[(trace["scenario"], "review")] += unit["needs_review"]
    scenarios = sorted({case["scenario"] for case in cases})
    by_scenario = {s: {k: counts[(s, k)] for k in ("traces", "prose", "rh", "review")}
                   for s in scenarios}
    summary = {"status": "complete" if len(traces) == len(cases) else "partial",
               "judge_model": MODEL, "budget_usd": BUDGET_USD,
               "reported_spend_usd_including_retries": round(charged_cost(), 6),
               "expected_traces": len(cases), "labeled_traces": len(traces),
               "expected_prose_units": 731,
               "labeled_prose_units": sum(row["prose"] for row in by_scenario.values()),
               "rh_prose_units": sum(row["rh"] for row in by_scenario.values()),
               "review_prose_units": sum(row["review"] for row in by_scenario.values()),
               "by_scenario": by_scenario,
               "method": "Same frozen unit segmentation, full-trace context, GPT-5.5 request format and RH rubric as the 306 previously labeled coding prose units.",
               "caveat": "Model labels are not human ground truth; outcomes were not used to select the new traces."}
    base.write(OUT / "summary.json", summary)
    base.write(OUT / "labels.json", {"summary": summary, "traces": traces})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    base.MODEL = MODEL
    new = selected()
    cases = base.prepare(new)
    manifest = {"model": MODEL, "budget_usd": BUDGET_USD,
                "rubric_sha256": base.sha(base.RUBRIC.encode()),
                "source_selection_sha256": base.sha(SOURCE_SELECTION.read_bytes()),
                "selected": new, "expected_traces": 28, "expected_prose_units": 731,
                "requests": [{"trace_id": c["id"], "start": ch["start"],
                              "request_sha256": ch["request_sha256"]}
                             for c in cases for ch in c["chunks"]]}
    path = OUT / "manifest.json"
    if path.exists():
        assert base.read_source(path) == manifest, "Frozen manifest changed"
    else:
        base.write(path, manifest)
    rubric = OUT / "rubric.txt"
    if rubric.exists():
        assert rubric.read_text() == base.RUBRIC + "\n"
    else:
        rubric.write_text(base.RUBRIC + "\n")
    print(f"Selected {len(cases)} new traces, 731 prose units, {len(manifest['requests'])} requests; cap ${BUDGET_USD:.2f}", flush=True)
    if args.prepare_only:
        return
    assert os.environ.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY missing"
    failures = []
    for index, (case, chunk) in enumerate(((c, ch) for c in cases for ch in c["chunks"]), 1):
        try:
            label_chunk(case, chunk)
        except Exception as exc:
            failures.append({"trace_id": case["id"], "chunk_start": chunk["start"], "error": str(exc)})
            print(f"STOP at {index}/{len(manifest['requests'])}: {exc}", flush=True)
            break
        if index % 5 == 0 or index == len(manifest["requests"]):
            print(f"{index}/{len(manifest['requests'])} requests; spent ${charged_cost():.3f}", flush=True)
    base.write(OUT / "failures.json", failures)
    summary = summarize(cases)
    print(json.dumps(summary, indent=2), flush=True)
    if failures or summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
