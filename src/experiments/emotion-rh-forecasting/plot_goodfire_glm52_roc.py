"""Task-disjoint, probe-specific FPR/TPR curves for the GLM 5.2 DeepSWE pilot.

Run with: uv run --with numpy --with matplotlib --with pysbd --with scikit-learn
          --with transformers python plot_goodfire_glm52_roc.py
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


import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.metrics import auc, roc_curve

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
from emotion_rh_forecasting.analysis import EMOTIONS, fit_threshold, make_pairs, predict_threshold
from plot_goodfire_glm52_forecast import load_rows
from shared.io import atomic, read, require, sha

ROOT = Path(__file__).resolve().parents[3]
LABELS = ROOT / "runs/emotion-rh-sentence-labels/goodfire-glm52-deepswe-gpt61sol-pilot-v1"
OUT = ROOT / "runs/emotion-rh-forecasting/goodfire-glm52-deepswe-gpt61sol-pilot-v1"
PLOT_HORIZONS = (0, 5, 10)
FPR_BUDGETS = (.01, .05, .10)


def roc_for_probe(pairs, probe):
    y = np.asarray([r["hacking"] for r in pairs], dtype=bool)
    folds = np.asarray([r["fold"] for r in pairs], dtype=int)
    x = np.asarray([r["projections"][probe] for r in pairs], dtype=float)
    require(np.isfinite(x).all(), "Missing probe scores")
    percentile = np.empty(len(pairs), dtype=float)
    predicted = np.empty(len(pairs), dtype=bool)
    fold_rules = []
    for fold in sorted(set(folds)):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        require(set(y[train]) == {False, True} and set(y[test]) == {False, True},
                "Each fold needs both classes")
        for key in ("trace_id", "group_id"):
            require(not {pairs[i][key] for i in train} & {pairs[i][key] for i in test},
                    f"{key} crosses folds")
        rule = fit_threshold(x[train], y[train])
        # Train-only empirical CDF makes fold-specific probe scores comparable
        # without using held-out labels or distributions for calibration.
        oriented_train = np.sort(rule["polarity"] * x[train])
        percentile[test] = np.searchsorted(oriented_train, rule["polarity"] * x[test],
                                            side="right") / len(train)
        predicted[test] = predict_threshold(x[test], rule)
        fold_rules.append({"fold": int(fold), "training_n": len(train),
                           "cutoff": rule["cutoff"], "polarity": rule["polarity"]})
    full_fpr, full_tpr, _ = roc_curve(y, percentile, drop_intermediate=False)
    fpr, tpr, _ = roc_curve(y, percentile, drop_intermediate=True)
    rate_fpr = float((predicted & ~y).sum() / (~y).sum())
    rate_tpr = float((predicted & y).sum() / y.sum())
    budgets = {f"tpr_at_fpr_{int(b * 100)}pct": float(full_tpr[full_fpr <= b + 1e-12].max())
               for b in FPR_BUDGETS}
    return {"positive_sentences": int(y.sum()), "negative_sentences": int((~y).sum()),
            "auroc": float(auc(full_fpr, full_tpr)), "operating_fpr": rate_fpr,
            "operating_tpr": rate_tpr, **budgets,
            "fpr": fpr.tolist(), "tpr": tpr.tolist(), "fold_rules": fold_rules}


def plot(curves, counts, out=OUT, subtitle=None, footnote=None):
    colors = {0: "#182a38", 5: "#b04826", 10: "#417f63"}
    styles = {0: "-", 5: "--", 10: ":"}
    fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharex=True, sharey=True)
    for ax, probe in zip(axes.flat, EMOTIONS):
        chance = np.r_[0, np.geomspace(1e-5, 1, 300)]
        ax.plot(chance, chance, color="#b4bec7", linewidth=1, zorder=0)
        for k in PLOT_HORIZONS:
            curve = curves[(probe, k)]
            ax.step(curve["fpr"], curve["tpr"], where="post", color=colors[k],
                    linestyle=styles[k], linewidth=1.7)
        current = curves[(probe, 0)]
        ax.scatter([current["operating_fpr"]], [current["operating_tpr"]],
                   s=50, marker="*", color="#d79c24", edgecolors="#74591d", zorder=5)
        ax.set_title(probe.capitalize())
        ax.text(.97, .05,
                f"k=0 AUROC {current['auroc']:.2f}\n"
                f"Cutoff: TPR {current['operating_tpr']:.0%} · FPR {current['operating_fpr']:.1%}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 2})
        ax.set_xscale("symlog", linthresh=.001)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, .001, .01, .1, 1])
        ax.set_xticklabels(["0", "0.1%", "1%", "10%", "100%"])
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.grid(alpha=.17)
    for ax in axes[:, 0]:
        ax.set_ylabel("True-positive rate")
    for ax in axes[-1]:
        ax.set_xlabel("False-positive rate")
    legend = [Line2D([], [], color=colors[k], linestyle=styles[k], label=f"Probe, k={k}")
              for k in PLOT_HORIZONS]
    legend += [Line2D([], [], color="#d79c24", marker="*", linestyle="None",
                      markersize=11, label="k=0 train-fit cutoff"),
               Line2D([], [], color="#b4bec7", label="Chance")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .03), ncol=5)
    fig.suptitle("GLM 5.2 DeepSWE · per-probe RH sentence detection and forecasting", fontsize=17, y=.99)
    fig.text(.065, .943, subtitle or
             ("Out-of-fold ROC · 12 task-disjoint traces / 3 folds · GPT-6.1 Sol pilot labels · "
              f"k=0: {counts[0]['positive_sentences']} RH and {counts[0]['negative_sentences']:,} non-RH sentences"),
             fontsize=10)
    fig.text(.065, .012, footnote or
             "Full certain-label population; zero FPR retained on near-log axis. "
             "Repeated RH sentences from one trace limit independent evidence.", fontsize=9)
    fig.tight_layout(rect=[0, .09, 1, .915])
    base = out / "figures" / "probe-roc"
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main():
    rows, replay, labels_path = load_rows(LABELS)
    old_metrics = read(OUT / "metrics.json")
    summary = old_metrics["summary"]
    require(summary["labels_sha256"] == sha(labels_path) and
            summary["replay_identity"] == replay["identity"], "Forecast inputs changed")
    folds = summary["folds"]["trace_folds"]
    for row in rows:
        row["fold"] = folds[row["trace_id"]]
    curves = {}
    flat = []
    counts = {}
    for k in range(11):
        pairs = make_pairs(rows, k, certain_only=True)
        require(set(r["fold"] for r in pairs) == set(folds.values()), "A fold disappeared")
        for probe in EMOTIONS:
            result = roc_for_probe(pairs, probe)
            curves[(probe, k)] = result
            counts[k] = {key: result[key] for key in ("positive_sentences", "negative_sentences")}
            flat.append({"probe": probe, "horizon": k, "auroc": result["auroc"],
                         "positive_sentences": result["positive_sentences"],
                         "negative_sentences": result["negative_sentences"],
                         "operating_tpr": result["operating_tpr"],
                         "operating_fpr": result["operating_fpr"],
                         **{key: result[key] for key in result if key.startswith("tpr_at_fpr_")}})
        print(f"k={k}: {counts[k]['positive_sentences']} RH / {counts[k]['negative_sentences']} non-RH", flush=True)
    with (OUT / "probe-roc-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    atomic(OUT / "probe-roc-curves.json", {
        "method": "OOF empirical train-CDF calibration, train-fit polarity and cutoff; certain-label pairs only",
        "label_sha256": sha(labels_path), "replay_identity": replay["identity"],
        "curves": {f"{probe}:k{k}": result for (probe, k), result in curves.items()},
    })
    plot(curves, counts)
    (OUT / "probe-roc-report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>GLM 5.2 probe ROC</title>'
        '<style>body{font:16px/1.5 system-ui;max-width:1600px;margin:2rem auto;color:#172d2b}img{max-width:100%}</style>'
        '<h1>Per-probe false-positive and true-positive rates</h1>'
        '<p>Each panel shows out-of-fold ROC curves for k=0, 5, and 10. The gold star is the k=0 '
        'operating point from a cutoff fitted on the other task folds. Positive means an RH sentence exactly '
        'k positions later. The curves use all certain-label pairs, without balancing negatives. '
        'Probe orientation and score calibration use training folds only.</p>'
        '<p>The near-log x-axis includes zero FPR. AUROC and operating rates are descriptive: 32 RH '
        'sentences at k=0, including 16 repeated statements from one trace, are not 32 independent episodes. '
        'FP8 archive text was replayed through NVFP4 GLM for probe features.</p>'
        '<img src="figures/probe-roc.svg" alt="Twelve per-probe ROC panels">'
        '<p><a href="probe-roc-metrics.csv">All k=0–10 rates and AUROC</a> · '
        '<a href="probe-roc-curves.json">Full ROC coordinates and fold rules</a> · '
        '<a href="report.html">Accuracy-by-horizon plot</a></p></html>')
    atomic(OUT / "probe-roc-verification.json", {"passed": True,
           "labels_sha256": sha(labels_path), "replay_identity": replay["identity"],
           "horizons": list(range(11)), "probes": EMOTIONS,
           "metrics_sha256": sha(OUT / "probe-roc-metrics.csv"),
           "figure_sha256": sha(OUT / "figures/probe-roc.svg")})


if __name__ == "__main__":
    main()
