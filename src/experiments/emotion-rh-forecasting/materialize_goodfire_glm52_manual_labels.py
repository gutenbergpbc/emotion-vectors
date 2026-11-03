"""Materialize reviewable sentence labels from manual decisions and source passage votes."""
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
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path("/scratch/model-forensics/goodfire/deepswe-glm-5.2-fp8")
OUT = ROOT / "runs/emotion-rh-sentence-labels/goodfire-glm52-deepswe-manual-pilot-v1"
SPANS = ROOT / "runs/emotion-rh-sentence-labels/goodfire-glm52-deepswe-deepseek-pilot-v2/labels.json"
SELECTION = ROOT / "runs/emotion-rh-sentence-labels/goodfire-glm52-deepswe-deepseek-pilot-v2/selection.json"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    index = json.loads((SOURCE / "sentence-index-summary.json").read_text())
    assert sha(SOURCE / "traces.jsonl") == index["source_sha256"]
    assert sha(SOURCE / "judge_passages.jsonl") == index["passage_labels_sha256"]
    selected = json.loads(SELECTION.read_text())
    spans = json.loads(SPANS.read_text())
    assert spans["summary"]["status"] == "complete" and len(spans["traces"]) == 12
    decisions_file = OUT / "adjudications.json"
    decisions = json.loads(decisions_file.read_text())
    overrides = {(d["row_index"], d["unit_id"]): d for d in decisions["decisions"]}
    assert len(overrides) == len(decisions["decisions"])
    passage_labels = {}
    for line in (SOURCE / "judge_passages.jsonl").open():
        p = json.loads(line)
        key = (p["row_index"], p["node_index"], p["channel"])
        assert key not in passage_labels
        passage_labels[key] = p["consensus"]
    raw_by_row = {}
    wanted = {r["row_index"] for r in selected["traces"]}
    with (SOURCE / "traces.jsonl").open("rb") as stream:
        for row_index, raw in enumerate(stream):
            if row_index in wanted:
                raw_by_row[row_index] = (json.loads(raw)["traces"][0], hashlib.sha256(raw).hexdigest())
    traces, counts = [], Counter()
    for old in spans["traces"]:
        row_index = old["row_index"]
        trace, source_hash = raw_by_row[row_index]
        assert old["source_line_sha256"] == source_hash and trace["id"] == old["id"]
        units = []
        for old_unit in old["units"]:
            uid = old_unit["unit_id"]
            field = "reasoning_content" if old_unit["channel"] == "reasoning" else "content"
            source_text = trace["nodes"][old_unit["node_index"]]["message"][field]
            assert source_text[old_unit["char_start"]:old_unit["char_end"]] == old_unit["text"]
            consensus = passage_labels.get((row_index, old_unit["node_index"], old_unit["channel"]), "unjudged")
            assert consensus in ("positive", "negative", "ambiguous", "unjudged")
            override = overrides.get((row_index, uid))
            if override:
                hacking, needs_review = override["hacking"], override["needs_review"]
                reason, method = override["reason"], "manual_adjudication"
                assert consensus in ("positive", "negative")
            elif consensus == "negative":
                hacking, needs_review = False, False
                reason, method = "No source judge flagged this passage in three votes.", "source_unanimous_negative"
            else:
                hacking, needs_review = False, True
                reason, method = "No sentence-level manual decision; excluded from certain-only evaluation.", "unresolved"
            unit = {k: old_unit[k] for k in ("unit_id", "kind", "node_index", "channel",
                                               "char_start", "char_end", "text")}
            unit.update(hacking=hacking, needs_review=needs_review, reason=reason,
                        label_method=method, source_passage_consensus=consensus)
            units.append(unit)
            counts["sentences"] += 1
            counts["rh_sentences"] += hacking
            counts["needs_review"] += needs_review
            counts[("consensus", consensus)] += 1
            counts[("method", method)] += 1
        assert [u["unit_id"] for u in units] == list(range(len(units)))
        traces.append({"id": old["id"], "row_index": row_index, "task": old["task"],
                       "source_line_sha256": source_hash, "units": units})
    assert set(overrides) <= {(t["row_index"], u["unit_id"]) for t in traces for u in t["units"]}
    assert counts["sentences"] == 12143 and len(traces) == 12
    summary = {"status": "complete", "expected_traces": 12, "labeled_traces": 12,
               "labeled_sentences": counts["sentences"], "rh_sentences": counts["rh_sentences"],
               "confirmed_rh_sentences": sum(u["hacking"] and not u["needs_review"]
                                             for t in traces for u in t["units"]),
               "non_rh_sentences": counts["sentences"] - counts["rh_sentences"],
               "needs_review": counts["needs_review"],
               "certain_sentences": counts["sentences"] - counts["needs_review"],
               "methods": {k[1]: v for k, v in counts.items() if isinstance(k, tuple) and k[0] == "method"},
               "source_passage_counts": {k[1]: v for k, v in counts.items() if isinstance(k, tuple) and k[0] == "consensus"},
               "adjudications_sha256": sha(decisions_file), "span_source_sha256": sha(SPANS),
               "selection_sha256": sha(SELECTION), "source_sha256": index["source_sha256"],
               "source_passages_sha256": index["passage_labels_sha256"],
               "method": "Manual decisions for exact disputed sentences; source three-vote negative passages supply the negative pool. Remaining positive-passage, ambiguous, and unjudged spans stay unresolved in the original sequence.",
               "caveat": "This is a targeted manual adjudication, not independent hand review of all 12,143 sentences. Full-trace context informs the manual decisions; unresolved spans are excluded from certain-only evaluation."}
    write(OUT / "summary.json", summary)
    write(OUT / "labels.json", {"summary": summary, "traces": traces})
    with (OUT / "sentences.jsonl").open("w") as stream:
        for t in traces:
            for u in t["units"]:
                stream.write(json.dumps({"trace_id": t["id"], "task": t["task"],
                                         "row_index": t["row_index"], "sentence_index": u["unit_id"], **u},
                                        ensure_ascii=False) + "\n")
    decisions_table = ''.join('<tr><td>' + str(d["row_index"]) + '</td><td>' + str(d["unit_id"]) +
                              '</td><td>' + str(d["hacking"]) + '</td><td>' +
                              html.escape(d["reason"]) + '</td></tr>' for d in decisions["decisions"])
    (OUT / "review.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>GLM 5.2 DeepSWE manual sentence adjudication</title>'
        '<style>body{font:16px/1.5 system-ui;max-width:1100px;margin:2rem auto;color:#172d2b}'
        'td,th{padding:.4rem;border-bottom:1px solid #ddd;text-align:left}</style>'
        '<h1>Manual sentence adjudication</h1>'
        f'<p>{summary["confirmed_rh_sentences"]} confirmed RH sentences, '
        f'{summary["certain_sentences"]} certain sentences, {summary["needs_review"]} unresolved or flagged. '
        'The source three-vote passage negatives supply the negative pool. Manual decisions cover the '
        'source-positive quotes and two DeepSeek-positive disagreements; no claim of individual hand review '
        'for every negative sentence.</p><table><tr><th>Source row</th><th>Sentence</th>'
        '<th>RH</th><th>Decision</th></tr>' + decisions_table + '</table></html>')
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
