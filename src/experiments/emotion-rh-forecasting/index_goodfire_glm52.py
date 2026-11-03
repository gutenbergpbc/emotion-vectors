"""Index exact assistant prose spans in Goodfire DeepSWE GLM 5.2 traces.

Run from model-forensics with: uv run --with pysbd python
src/experiments/emotion-rh-forecasting/index_goodfire_glm52.py
"""

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
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/gpt_oss_coding_early_warning_20261016"))
from label_sentences import segment  # noqa: E402
import pysbd

SOURCE = Path("/scratch/model-forensics/goodfire/deepswe-glm-5.2-fp8")
OUT = SOURCE / "sentence-index.jsonl"


def segment_safe(content):
    """Use the established splitter, with direct pySBD for malformed Markdown."""
    try:
        return segment(content), False
    except AssertionError:
        splitter = pysbd.Segmenter(language="en", clean=False, char_span=True)
        spans = []
        normalized = content.replace("\n", " ").replace("\r", " ")
        for piece in splitter.segment(normalized):
            lo, hi = piece.start, piece.end
            while lo < hi and content[lo].isspace():
                lo += 1
            while hi > lo and content[hi - 1].isspace():
                hi -= 1
            if lo < hi:
                spans.append({"unit_id": len(spans), "kind": "sentence_or_fragment",
                              "char_start": lo, "char_end": hi, "text": content[lo:hi]})
        assert spans and all(content[u["char_start"]:u["char_end"]] == u["text"] for u in spans)
        return spans, True


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=562)
    args = parser.parse_args()
    assert 0 <= args.start < args.stop <= 562
    output_path = SOURCE / f"sentence-index-{args.start:03d}-{args.stop:03d}.jsonl"
    passages = {}
    with (SOURCE / "judge_passages.jsonl").open() as stream:
        for line in stream:
            item = json.loads(line)
            key = (item["row_index"], item["node_index"], item["channel"])
            assert key not in passages
            passages[key] = item["consensus"]
    counts = Counter()
    traces = []
    with (SOURCE / "traces.jsonl").open() as input_stream, output_path.open("w") as output:
        for row_index, line in enumerate(input_stream):
            if row_index < args.start or row_index >= args.stop:
                continue
            row = json.loads(line)
            assert row["ok"] and len(row["traces"]) == 1
            trace = row["traces"][0]
            trace_id = trace["id"]
            task = trace["task"]["data"].get("name")
            local = Counter()
            sentence_index = 0
            for node_index, node in enumerate(trace["nodes"]):
                message = node.get("message") or {}
                if message.get("role") != "assistant":
                    continue
                for field, channel in (("reasoning_content", "reasoning"), ("content", "content")):
                    content = message.get(field)
                    if not isinstance(content, str) or not content.strip():
                        continue
                    consensus = passages.get((row_index, node_index, channel), "unjudged")
                    units, fallback = segment_safe(content)
                    local["fallback_fields"] += fallback
                    for unit in units:
                        local[("units", channel, unit["kind"])] += 1
                        if unit["kind"] != "sentence_or_fragment":
                            continue
                        item = {"row_index": row_index, "trace_id": trace_id, "task": task,
                                "node_index": node_index, "channel": channel,
                                "sentence_index": sentence_index, "char_start": unit["char_start"],
                                "char_end": unit["char_end"], "text": unit["text"],
                                "passage_consensus": consensus}
                        assert content[item["char_start"]:item["char_end"]] == item["text"]
                        output.write(json.dumps(item, ensure_ascii=False) + "\n")
                        local["sentences"] += 1
                        local[("passage_consensus", consensus)] += 1
                        sentence_index += 1
            counts.update(local)
            traces.append({"row_index": row_index, "trace_id": trace_id, "task": task,
                           "sentences": local["sentences"],
                           "positive_passage_sentences": local[("passage_consensus", "positive")],
                           "ambiguous_passage_sentences": local[("passage_consensus", "ambiguous")]})
            if (row_index + 1) % 50 == 0:
                print(f"{row_index + 1} traces, {counts['sentences']} sentences", flush=True)
    assert len(traces) == args.stop - args.start
    summary = {"source": "Goodfire/reward-hack-data/deepswe-glm-5.2-fp8",
               "source_revision": "bd8fb473568b8ef1d12fc2325a5c8af939c15ad9",
               "splitter": "label_sentences.segment (pysbd, sentence_or_fragment only)",
               "traces": len(traces), "sentences": counts["sentences"],
               "reasoning_sentences": counts[("units", "reasoning", "sentence_or_fragment")],
               "response_sentences": counts[("units", "content", "sentence_or_fragment")],
               "positive_passage_sentences": counts[("passage_consensus", "positive")],
               "negative_passage_sentences": counts[("passage_consensus", "negative")],
               "ambiguous_passage_sentences": counts[("passage_consensus", "ambiguous")],
               "unjudged_sentences": counts[("passage_consensus", "unjudged")],
               "excluded_code_blocks": sum(v for k, v in counts.items() if isinstance(k, tuple) and k[0] == "units" and k[2] == "code_block"),
               "fallback_fields": counts["fallback_fields"],
               "source_sha256": digest(SOURCE / "traces.jsonl"),
               "passage_labels_sha256": digest(SOURCE / "judge_passages.jsonl"),
               "index_sha256": digest(output_path), "per_trace": traces}
    (SOURCE / f"sentence-index-summary-{args.start:03d}-{args.stop:03d}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_trace"}, indent=2))


if __name__ == "__main__":
    main()
