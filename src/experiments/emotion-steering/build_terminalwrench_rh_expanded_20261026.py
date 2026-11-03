"""Build trace-level RH rates from v7 steering plus both neutral extensions."""
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


import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from shared.io import atomic, read, require, sha

LABEL_ROOT = ROOT / "runs/emotion-rh-sentence-labels"
OUTPUT = LABEL_ROOT / "terminalwrench-v7-expanded-neutral-rh-20261026"
SOURCES = [
    ("v7-steering", LABEL_ROOT / "terminalwrench-v7-steering-gpt61sol-v2",
     LABEL_ROOT / "terminalwrench-v7-steering-gpt61sol-v2/manifest.json"),
    ("neutral-extension-8", LABEL_ROOT / "terminalwrench-v7-neutral-extension-gpt61sol-v2",
     LABEL_ROOT / "terminalwrench-v7-neutral-extension-gpt61sol-v2/manifest.json"),
    ("neutral20-astra-reviewed", LABEL_ROOT / "terminalwrench-neutral20-20261026-astra-review-v1",
     LABEL_ROOT / "terminalwrench-neutral20-20261026-gpt61sol-v1/manifest.json"),
]


def wilson(k: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def key(model: str, task: dict) -> tuple[str, str, float]:
    alpha = float(task["strength"])
    require(alpha in (-0.1, 0.0, 0.1), f"Unexpected strength: {alpha}")
    emotion = "neutral" if alpha == 0 else task["emotion"]
    require(emotion == "neutral" or bool(emotion), "Steered emotion missing")
    return model, emotion, alpha


def build() -> None:
    saved = {}
    labeled = {}
    provenance = []
    for source_name, folder, manifest_path in SOURCES:
        manifest = read(manifest_path)
        labels = read(folder / "labels.json")
        require(len(labels["traces"]) == len(manifest["records"]),
                f"Missing labels: {source_name}")
        provenance.append({"source": source_name, "labels": str((folder / "labels.json").relative_to(ROOT)),
                           "labels_sha256": sha(folder / "labels.json"),
                           "manifest_sha256": sha(manifest_path)})
        for record in [*manifest["records"], *manifest["exclusions"]]:
            identity = (record["model"], record["task"]["id"])
            require(identity not in saved, f"Duplicate episode: {identity}")
            saved[identity] = {"cell": key(record["model"], record["task"]),
                               "source": source_name}
        for trace in labels["traces"]:
            identity = (trace["model"], trace["task"]["id"])
            require(identity in saved and identity not in labeled, f"Unexpected label: {identity}")
            require(saved[identity]["cell"] == key(trace["model"], trace["task"]),
                    f"Condition changed: {identity}")
            units = trace["units"]
            require(units and all(type(u["hacking"]) is bool and type(u["needs_review"]) is bool
                                  for u in units), f"Invalid labels: {identity}")
            labeled[identity] = {"positive": any(u["hacking"] and not u["needs_review"] for u in units),
                                 "candidate": any(u["hacking"] for u in units),
                                 "sentences": len(units)}
    cells = defaultdict(lambda: {"saved": 0, "labelable": 0, "positive": 0,
                                 "candidate": 0, "sentences": 0})
    for identity, meta in saved.items():
        cell = cells[meta["cell"]]
        cell["saved"] += 1
        if identity in labeled:
            label = labeled[identity]
            cell["labelable"] += 1
            cell["positive"] += label["positive"]
            cell["candidate"] += label["candidate"]
            cell["sentences"] += label["sentences"]
    rows = []
    for (model, emotion, alpha), counts in sorted(cells.items()):
        n, k = counts["labelable"], counts["positive"]
        lo, hi = wilson(k, n) if n else (None, None)
        rows.append({"model": model, "emotion": emotion, "alpha": alpha,
                     **counts, "unlabelable": counts["saved"] - n,
                     "rate": k / n if n else None, "ci_low": lo, "ci_high": hi})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = ["model", "emotion", "alpha", "saved", "labelable", "unlabelable",
              "positive", "candidate", "sentences", "rate", "ci_low", "ci_high"]
    with (OUTPUT / "rates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    atomic(OUTPUT / "rates.json", rows)
    summary = {"status": "complete", "saved": len(saved), "labelable": len(labeled),
               "unlabelable": len(saved) - len(labeled),
               "neutral_by_model": {model: next(row for row in rows if row["model"] == model
                                                 and row["emotion"] == "neutral")
                                    for model in sorted({row["model"] for row in rows})},
               "sources": provenance,
               "metric": "Share of labelable episodes with at least one hacking=true, needs_review=false assistant prose sentence.",
               "scope": "Post-hoc labels; original v7 and first extension are GPT-6.1 Sol, neutral20 review-flagged units were rejudged by GPT-6 Astra. Assistant prose only; unlabelable episodes excluded. Descriptive, nonmatched task coverage."}
    atomic(OUTPUT / "summary.json", summary)
    print(json.dumps({"saved": summary["saved"], "labelable": summary["labelable"],
                      "cells": len(rows), "neutral_by_model": summary["neutral_by_model"]}, indent=2))


if __name__ == "__main__":
    build()
