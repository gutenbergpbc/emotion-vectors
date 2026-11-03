"""Verify and merge the eight deterministic Goodfire GLM 5.2 sentence-index shards."""

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

import hashlib
import json
from pathlib import Path

SOURCE = Path("/scratch/model-forensics/goodfire/deepswe-glm-5.2-fp8")
RANGES = [(i, i + 70) for i in range(0, 490, 70)] + [(490, 562)]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    shards = []
    for start, stop in RANGES:
        path = SOURCE / f"sentence-index-{start:03d}-{stop:03d}.jsonl"
        summary = json.loads((SOURCE / f"sentence-index-summary-{start:03d}-{stop:03d}.json").read_text())
        assert sha(path) == summary["index_sha256"]
        assert summary["traces"] == stop - start
        assert [r["row_index"] for r in summary["per_trace"]] == list(range(start, stop))
        shards.append((path, summary))
    assert len({s["source_sha256"] for _, s in shards}) == 1
    assert len({s["passage_labels_sha256"] for _, s in shards}) == 1
    dest = SOURCE / "sentence-index.jsonl"
    tmp = dest.with_suffix(".jsonl.tmp")
    with tmp.open("wb") as output:
        for path, _ in shards:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    output.write(block)
    tmp.replace(dest)
    fields = ("traces", "sentences", "reasoning_sentences", "response_sentences",
              "positive_passage_sentences", "negative_passage_sentences", "ambiguous_passage_sentences",
              "unjudged_sentences", "excluded_code_blocks", "fallback_fields")
    summary = {k: sum(s[k] for _, s in shards) for k in fields}
    summary.update({"source": shards[0][1]["source"], "source_revision": shards[0][1]["source_revision"],
                    "splitter": shards[0][1]["splitter"], "source_sha256": shards[0][1]["source_sha256"],
                    "passage_labels_sha256": shards[0][1]["passage_labels_sha256"],
                    "index_sha256": sha(dest),
                    "per_trace": [row for _, s in shards for row in s["per_trace"]]})
    assert summary["traces"] == 562
    assert summary["sentences"] == summary["reasoning_sentences"] + summary["response_sentences"]
    assert summary["sentences"] == sum(summary[k] for k in
               ("positive_passage_sentences", "negative_passage_sentences",
                "ambiguous_passage_sentences", "unjudged_sentences"))
    (SOURCE / "sentence-index-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in fields}, indent=2))


if __name__ == "__main__":
    main()
