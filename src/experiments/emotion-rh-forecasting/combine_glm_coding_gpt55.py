"""Combine original and newly judged GLM coding labels into one GPT-5.5 set."""
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


from collections import Counter
import json
from pathlib import Path

import label_glm_hack_traces as base


ROOT = Path(__file__).resolve().parents[3]
LABEL_ROOT = ROOT / "runs/emotion-rh-sentence-labels"
OLD = [LABEL_ROOT / "glm53-hack-traces-gpt55-v1/labels.json",
       LABEL_ROOT / "glm53-nonhack-traces-gpt55-v1/labels.json"]
NEW = LABEL_ROOT / "glm53-coding-new-gpt55-v1/labels.json"
SELECTION = LABEL_ROOT / "glm53-coding-deepseek-v41-flash-1017-v1/selection.json"
OUT = LABEL_ROOT / "glm53-coding-1000-gpt55-v1"


def main() -> None:
    selection = base.read_source(SELECTION)
    fresh = base.read_source(NEW)
    assert fresh["summary"]["status"] == "complete"
    assert fresh["summary"]["judge_model"] == "openai/gpt-5.5"
    by_id = {trace["id"]: trace for trace in fresh["traces"]}
    for path in OLD:
        source = base.read_source(path)
        assert source["summary"]["model"] == "openai/gpt-5.5"
        for trace in source["traces"]:
            if trace["dataset"] == "coding":
                assert trace["id"] not in by_id
                by_id[trace["id"]] = trace
    traces = []
    for item in selection["selected"]:
        trace = by_id[item["id"]]
        assert trace["source_sha256"] == item["sha256"]
        assert trace["source_path"] == item["path"]
        assert sum(unit["kind"] == "sentence_or_fragment" for unit in trace["units"]) == item["prose_units"]
        assert len({unit["unit_id"] for unit in trace["units"]}) == len(trace["units"])
        traces.append(trace)
    assert len(traces) == len(by_id) == 51
    counts = Counter()
    for trace in traces:
        counts["all_units"] += len(trace["units"])
        for unit in trace["units"]:
            if unit["kind"] == "sentence_or_fragment":
                counts["prose_units"] += 1
                counts["rh_prose_units"] += unit["hacking"]
                counts["review_prose_units"] += unit["needs_review"]
    assert counts["prose_units"] == selection["selected_prose"] == 1037
    summary = {"status": "complete", "judge_model": "openai/gpt-5.5",
               "traces": len(traces), **counts,
               "previously_labeled_prose_units": 306,
               "newly_labeled_prose_units": fresh["summary"]["labeled_prose_units"],
               "new_run_budget_usd": fresh["summary"]["budget_usd"],
               "new_run_reported_spend_usd_including_retries": fresh["summary"]["reported_spend_usd_including_retries"],
               "sources": [str(path.relative_to(ROOT)) for path in OLD + [NEW]],
               "selection": str(SELECTION.relative_to(ROOT)),
               "rubric_sha256": base.sha(base.RUBRIC.encode()),
               "caveat": "Full-trace post-hoc model judgments, not human ground truth. The 23 original coding traces were outcome-stratified; the 28 new traces were chosen in fixed scenario/sample order, independent of labels."}
    OUT.mkdir(parents=True, exist_ok=True)
    base.write(OUT / "summary.json", summary)
    base.write(OUT / "labels.json", {"summary": summary, "traces": traces})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
