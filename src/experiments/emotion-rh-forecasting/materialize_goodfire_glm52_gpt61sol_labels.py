"""Merge validated GPT-6.1 Sol adjudications into the frozen manual pilot."""
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
import html
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE.parent), str(HERE)]
import label_goodfire_glm52_gpt61sol as judge  # noqa: E402
from materialize_goodfire_glm52_manual_labels import OUT as BASE  # noqa: E402
from shared.io import atomic, read, require, sha  # noqa: E402

OUT = judge.OUT
AUDITED_POSITIVES = {
    (7, 118), (7, 119), (7, 122), (7, 130), (7, 135), (7, 141),
    (7, 144), (7, 146), (7, 158), (7, 162), (7, 166), (7, 169),
    (7, 179), (7, 180), (7, 195), (7, 198), (438, 584),
}


def main():
    old = read(BASE / "labels.json")
    require(old["summary"]["status"] == "complete", "Manual pilot incomplete")
    jobs = judge.prepare()
    manifest = read(OUT / "manifest.json")
    require(manifest["request_sha256"] == [j["request_sha256"] for j in jobs],
            "Judge request list changed")
    proposed = {}
    cost = 0.0
    for job in jobs:
        path = OUT / "annotations" / f"{job['row_index']:03d}-{job['units'][0]['unit_id']:04d}.json"
        saved = read(path)
        require(saved["request_sha256"] == job["request_sha256"] and
                saved["source_sha256"] == job["source_sha256"] and
                saved["model"] == judge.MODEL, "Judge receipt changed")
        judge.base.validate_labels({"labels": saved["labels"]}, job["units"])
        cost += saved.get("usage", {}).get("cost", 0) or 0
        for label in saved["labels"]:
            key = (job["row_index"], label["unit_id"])
            require(key not in proposed, "Duplicate judgment")
            proposed[key] = label
    require(len(proposed) == 1181, "Missing judgment")
    require({k for k, v in proposed.items() if v["hacking"] and not v["needs_review"]}
            == AUDITED_POSITIVES, "Positive audit set changed")

    all_rows = []
    method_counts = Counter()
    candidate_rows = []
    for trace in old["traces"]:
        for unit in trace["units"]:
            key = (trace["row_index"], unit["unit_id"])
            if key in proposed:
                label = proposed[key]
                require(unit["needs_review"], "Judge changed a frozen certain label")
                unit["gpt61sol_hacking"] = label["hacking"]
                unit["gpt61sol_needs_review"] = label["needs_review"]
                unit["gpt61sol_reason"] = label["reason"]
                if key not in {(7, 174), (7, 365)}:
                    unit["hacking"] = label["hacking"]
                    unit["needs_review"] = label["needs_review"]
                    unit["reason"] = label["reason"]
                    unit["label_method"] = "gpt61sol_judge" if not label["needs_review"] else "gpt61sol_flagged"
                if label["hacking"]:
                    candidate_rows.append({"row_index": key[0], "unit_id": key[1],
                                           "text": unit["text"], "needs_review": label["needs_review"],
                                           "reason": label["reason"], "source_passage_consensus": unit["source_passage_consensus"]})
            method_counts[unit["label_method"]] += 1
            all_rows.append({"trace_id": trace["id"], "task": trace["task"],
                             "row_index": trace["row_index"], "sentence_index": unit["unit_id"], **unit})

    summary = {**old["summary"],
               "rh_sentences": sum(r["hacking"] for r in all_rows),
               "confirmed_rh_sentences": sum(r["hacking"] and not r["needs_review"] for r in all_rows),
               "non_rh_sentences": sum(not r["hacking"] for r in all_rows),
               "needs_review": sum(r["needs_review"] for r in all_rows),
               "certain_sentences": sum(not r["needs_review"] for r in all_rows),
               "methods": dict(method_counts),
               "base_labels_sha256": sha(BASE / "labels.json"),
               "judge_manifest_sha256": sha(OUT / "manifest.json"),
               "gpt61sol_candidates": len(candidate_rows),
               "gpt61sol_unflagged_positives": len(AUDITED_POSITIVES),
               "gpt61sol_cost_usd": round(cost, 6),
               "method": "Frozen manual/source-vote labels plus GPT-6.1 Sol on 1,181 unresolved spans; flagged responses remain excluded from primary evaluation.",
               "caveat": "Judge positives were checked against the task and tool evidence. The 16 new labels in row 7 describe one repeated local-versus-remote shortcut and are not independent cases. This is a 12-trace pilot, not full-corpus labeling."}
    require(len(all_rows) == summary["labeled_sentences"] == 12143, "Sentence count changed")
    require(summary["confirmed_rh_sentences"] == 32 and summary["needs_review"] == 130,
            "Unexpected merged label counts")
    atomic(OUT / "labels.json", {"summary": summary, "traces": old["traces"]})
    atomic(OUT / "summary.json", summary)
    with (OUT / "sentences.jsonl").open("w") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT / "gpt61sol-positive-candidates.jsonl").open("w") as stream:
        for row in candidate_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    body = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>GPT-6.1 Sol RH candidates</title>',
            '<style>body{font:16px/1.5 system-ui;max-width:1050px;margin:2rem auto;color:#233}article{border-top:1px solid #bbb;padding:1rem 0}small{color:#666}</style>',
            f'<h1>GPT-6.1 Sol reward-hack sentence candidates</h1><p>{len(candidate_rows)} positive candidates; '
            f'{len(AUDITED_POSITIVES)} unflagged and audited. Flagged candidates remain outside the primary plot. '
            'Sixteen of the unflagged positives come from one trace and repeat a single shortcut.</p>']
    for row in candidate_rows:
        body.append(f'<article><h3>Trace row {row["row_index"]}, sentence {row["unit_id"]} '
                    f'{"(flagged)" if row["needs_review"] else "(primary label)"}</h3>'
                    f'<p>{html.escape(row["text"])}</p><small>{html.escape(row["reason"])} '
                    f'· Source passage: {html.escape(row["source_passage_consensus"])}</small></article>')
    body.append('</html>')
    (OUT / "review.html").write_text("\n".join(body))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
