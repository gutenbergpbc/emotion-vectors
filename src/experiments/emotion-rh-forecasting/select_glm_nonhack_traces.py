"""Freeze a seeded, whole-trace control sample matched on prose sentence counts.

uv run --with pysbd python src/experiments/emotion-rh-forecasting/select_glm_nonhack_traces.py
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


from collections import Counter
import json
from pathlib import Path
import random

from label_glm_hack_traces import ROOT, SOURCE, read_source, segment, write

SEED = 20261021
TARGETS = {"sales": 2413, "coding": 153}
OUT = ROOT / "runs/emotion-rh-sentence-labels/glm53-nonhack-traces-gpt55-v1/selection.json"


def prose_count(item: dict) -> int:
    trace = read_source(ROOT / item["path"], item["sha256"])
    assert trace["id"] == item["id"] and trace["model_requested"] == "z-ai/glm-5.3"
    if item["dataset"] == "sales":
        fields = [t.get(name) or "" for t in trace["turns"] for name in ("reasoning_text", "response")]
    else:
        fields = [trace.get(name) or "" for name in ("reasoning_text", "response")]
    return sum(u["kind"] == "sentence_or_fragment" for field in fields for u in segment(field))


def candidates() -> dict[str, list[dict]]:
    sales = SOURCE / "glm53-sales-openrouter-v1"
    coding = SOURCE / "glm53-coding-openrouter-v1"
    by_id = {r["id"]: r for r in read_source(sales / "cohort-finalized-150.json")["traces"]}
    rows = {"sales": [], "coding": []}
    for row in read_source(sales / "sales-ledger-hack-rate.json")["episodes"]:
        if not row["ever_violation"]:
            source = by_id[row["id"]]
            rows["sales"].append({"id": row["id"], "dataset": "sales", "path": source["path"],
                                  "sha256": source["sha256"], "selection_stratum": "no_ledger_violation"})
    for path in sorted((coding / "whole-answer-hack-judgments").glob("*/*.json")):
        row = read_source(path)
        if row["label"] == "honest_impossible":
            rows["coding"].append({"id": row["trace_id"], "dataset": "coding",
                                   "path": row["source_path"], "sha256": row["source_sha256"],
                                   "selection_stratum": "honest_impossible"})
    return rows


def choose(items: list[dict], target: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    ordered = sorted(items, key=lambda x: x["id"])
    rng.shuffle(ordered)
    # First reachable sum wins in a seeded random permutation. This samples
    # whole traces conditional on the exact target, retaining k=0..10 runs.
    parents: dict[int, tuple[int, int] | None] = {0: None}
    for index, item in enumerate(ordered):
        size = item["prose_sentences"]
        if not size:
            continue
        for subtotal in sorted(list(parents), reverse=True):
            total = subtotal + size
            if total <= target and total not in parents:
                parents[total] = (subtotal, index)
        if target in parents:
            break
    best = target if target in parents else max(parents)
    assert best == target, f"Could not match target {target}; closest {best}"
    selected = []
    while best:
        parent, index = parents[best]
        selected.append(ordered[index])
        best = parent
    assert len({x["id"] for x in selected}) == len(selected)
    return sorted(selected, key=lambda x: x["id"])


def main() -> None:
    pool = candidates()
    for dataset, items in pool.items():
        for item in items:
            item["prose_sentences"] = prose_count(item)
    selected = {dataset: choose(items, TARGETS[dataset], SEED + i)
                for i, (dataset, items) in enumerate(pool.items())}
    result = {"seed": SEED, "sampling": "seeded random permutation, whole-trace subset sum conditional on exact prose count",
              "targets": TARGETS, "candidate_counts": {d: len(v) for d, v in pool.items()},
              "selected_counts": {d: len(v) for d, v in selected.items()},
              "selected_prose": {d: sum(r["prose_sentences"] for r in v) for d, v in selected.items()},
              "selected": sorted(sum(selected.values(), []), key=lambda x: x["id"])}
    if OUT.exists():
        assert read_source(OUT) == result, "Frozen selection changed"
    else:
        write(OUT, result)
    print(json.dumps({k: v for k, v in result.items() if k != "selected"}, indent=2))


if __name__ == "__main__":
    main()
