"""Render Gemma 27B probe accuracy minus token-position baseline from saved metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_kit"))
import figkit as fk  # noqa: E402

ROOT = HERE.parents[3]
EMOTIONS = ["afraid", "anxious", "appreciative", "ashamed", "calm", "desperate",
            "excited", "frustrated", "guilty", "hostile", "sad", "surprised"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("coding-verbose", "sales"))
    args = parser.parse_args()
    run_id = f"gemma-3-27b-{args.dataset}-haiku55-v1"
    root = ROOT / "runs/emotion-rh-forecasting" / run_id
    metrics = json.loads((root / "metrics.json").read_text())
    estimates = metrics["estimates"]
    rows = {(r["model"], r["horizon"]): r for r in estimates if r["cohort"] == "all_available"}
    horizons = list(range(11))
    data = np.array([[100 * (rows[(emotion, h)]["balanced_accuracy"] -
                              rows[("token_position", h)]["balanced_accuracy"])
                      if (emotion, h) in rows and ("token_position", h) in rows else np.nan
                      for h in horizons] for emotion in EMOTIONS])
    finite = np.abs(data[np.isfinite(data)])
    bound = max(10.0, float(np.ceil(finite.max() / 5) * 5)) if len(finite) else 10.0
    fk.use()
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#dddddd")
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound))
    ax.set_xticks(horizons, [str(h) for h in horizons])
    ax.set_yticks(range(12), [e.capitalize() for e in EMOTIONS])
    ax.set_xlabel("Future sentence offset")
    ax.tick_params(length=0)
    colorbar = fig.colorbar(im, ax=ax, fraction=.035, pad=.025)
    colorbar.set_label("Accuracy difference (pp)")
    output = root / "figures" / "probe-minus-position-heatmap.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fk.save(fig, output)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    populations = {p["horizon"]: p for p in metrics["populations"] if p["cohort"] == "all_available"}
    counts = metrics["counts"]
    caption = (f"{counts['traces']} traces and {counts['sentences']} prose sentences; "
               f"{populations[0]['balanced_eval_pairs']} balanced evaluation pairs at h=0 and "
               f"{populations[10]['balanced_eval_pairs']} at h=10. "
               "Each cell is the held-out, balanced-evaluation accuracy of one frozen emotion probe "
               "minus the token-position-only baseline, in percentage points. Positive values favor the "
               "emotion probe. h=0 is current-sentence classification; h>0 predicts exactly h later "
               "eligible prose sentences. Haiku 5.5 labels are retrospective model judgments. "
               "The coding cohort has an added verbose instruction; the sales cohort uses its original prompt. "
               "See metrics.json for every horizon's denominator and sampling interval, and verification.json for the replay audit.\n")
    (root / "figures" / "probe-minus-position-heatmap.caption.md").write_text(caption)


if __name__ == "__main__":
    main()
