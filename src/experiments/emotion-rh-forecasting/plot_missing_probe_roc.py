"""Render missing per-emotion ROC curves from frozen forecast inputs.

This is a projection of existing labels, probe scores, and folds. It does not
refit probes or change any forecast run. Curves use every eligible pair in the
run's primary cohort, with train-fold-only polarity and CDF calibration.
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
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE.parent), str(HERE), str(HERE / "figures/_kit")]

import figkit as fk
from emotion_rh_forecasting.analysis import EMOTIONS, assign_folds, make_pairs
from emotion_rh_forecasting.inputs import Sources, load_probes, vector_identity
from plot_glm_hack_forecast import frozen_folds, rows_for_dataset
from plot_goodfire_glm52_roc import roc_for_probe
from shared.io import read, require, sha

BASE = ROOT / "runs/emotion-rh-forecasting"
OUT = BASE / "roc-per-emotion"
HORIZONS = (0, 5, 10)
COLORS = {0: "#1769a3", 5: "#b65a35", 10: "#32836b"}
STYLES = {0: "-", 5: "--", 10: ":"}

# Run, model, dataset, input adapter, dataset subdirectory.
RUNS = [
    ("gemma3-4b-sales-gpt55-v1", "Gemma 3 4B", "Sales", "gemma4-sales", ""),
    ("gemma3-4b-coding-lines-v1", "Gemma 3 4B", "Coding code lines", "gemma4-coding", ""),
    ("gemma4b-terminalwrench-neutral-v7-rh-20261026-v2", "Gemma 3 4B", "TerminalWrench neutral v7 · v2", "saved", ""),
    ("gemma4b-terminalwrench-neutral-v7-rh-20261026-v3", "Gemma 3 4B", "TerminalWrench neutral v7 · v3", "saved", ""),
    ("gemma-3-27b-sales-haiku55-v1", "Gemma 3 27B", "Sales", "saved", ""),
    ("gemma-3-27b-coding-verbose-haiku55-v1", "Gemma 3 27B", "Coding verbose prompt", "saved", ""),
    ("gemma-3-27b-terminalwrench-pilot50-v1", "Gemma 3 27B", "TerminalWrench pilot 50", "saved", ""),
    ("gpt-oss-120b-sales-v1", "GPT-OSS 120B", "Sales", "saved", ""),
    ("gpt-oss-120b-coding-v1", "GPT-OSS 120B", "Coding", "saved", ""),
    ("gpt-oss-120b-terminalwrench-deepseek-balanced50-v1", "GPT-OSS 120B", "TerminalWrench balanced 50", "saved", ""),
    ("gptoss120b-terminalwrench-neutral-v7-rh-20261026-v3", "GPT-OSS 120B", "TerminalWrench neutral v7", "saved", ""),
    ("glm-5.3-balanced-source-best12-v1", "GLM 5.3", "Sales balanced source", "glm-balanced", "sales"),
    ("glm-5.3-balanced-source-best12-v1", "GLM 5.3", "Coding balanced source", "glm-balanced", "coding"),
    ("glm-5.3-coding-1000-gpt55-v1", "GLM 5.3", "Coding (1,037 sentences)", "glm-coding", ""),
    ("glm-5.3-hack-traces-exploratory-v1", "GLM 5.3", "Sales exploratory", "glm-hack", "sales"),
    ("glm-5.3-hack-traces-exploratory-v1", "GLM 5.3", "Coding exploratory", "glm-hack", "coding"),
    ("goodfire-glm52-deepswe-manual-pilot-v1", "GLM 5.2", "DeepSWE manual pilot", "glm52-manual", ""),
]


def verified_replay(replay: Path, labels_path: Path):
    labels = read(labels_path)
    spec = read(replay / "replay-spec.json")
    bundle = load_probes(replay / "probes.json", Sources())
    require(labels["summary"]["status"] == "complete", "Incomplete labels")
    require(spec["labels_sha256"] == sha(labels_path)
            and spec["probe_identity"] == vector_identity(bundle), "Replay identity mismatch")
    require(len(list((replay / "scores").glob("*.receipt.json"))) == len(spec["jobs"]),
            "Incomplete replay")
    return labels, spec, bundle


def load_rows(run: str, mode: str, dataset: str):
    folder = BASE / run
    if mode == "saved":
        rows_path = folder / "sentence-features.json"
        rows = read(rows_path)
        require(rows and all("fold" in r for r in rows), "Frozen rows/folds missing")
        return rows, [str(rows_path)], [sha(rows_path)]
    if mode == "glm52-manual":
        from plot_goodfire_glm52_forecast import load_rows as load_glm52
        rows, _, labels_path = load_glm52()
        summary = read(folder / "metrics.json")["summary"]
        for row in rows:
            row["fold"] = summary["folds"]["trace_folds"][row["trace_id"]]
        return rows, [str(labels_path)], [sha(labels_path)]
    if mode == "gemma4-sales":
        from plot_gemma4b_sales_forecast import LABELS, REPLAY
        labels, spec, bundle = verified_replay(REPLAY, LABELS)
        rows = rows_for_dataset("sales", labels, spec, bundle, REPLAY)
        assign_folds(rows, {"method": "held_out_task"}, 20261022)
        return rows, [str(LABELS)], [sha(LABELS)]
    if mode == "gemma4-coding":
        from plot_gemma4b_coding_line_forecast import LABELS, REPLAY, adapted
        original, spec, bundle = verified_replay(REPLAY, LABELS)
        labels = {**original, "traces": [adapted(t) for t in original["traces"]]}
        rows = rows_for_dataset("coding", labels, spec, bundle, REPLAY)
        assign_folds(rows, {"method": "held_out_task"}, 20261022)
        return rows, [str(LABELS)], [sha(LABELS)]
    if mode == "glm-coding":
        from plot_glm_coding_1000 import LABELS, REPLAY, TASK_FOLDS
        labels, spec, bundle = verified_replay(REPLAY, LABELS)
        rows = rows_for_dataset("coding", labels, spec, bundle, REPLAY)
        for row in rows:
            row["group_id"] = row["task"]
            row["fold"] = TASK_FOLDS[row["task"]]
        assign_folds(rows, {"method": "provided"}, 20261022)
        return rows, [str(LABELS)], [sha(LABELS)]
    from plot_glm_hack_forecast import LABELS, REPLAY
    sources = [(LABELS, REPLAY)]
    if mode == "glm-balanced":
        sources.append((ROOT / "runs/emotion-rh-sentence-labels/glm53-nonhack-traces-gpt55-v1/labels.json",
                        ROOT / "runs/emotion-rh-replay/glm-5.3-nonhack-traces-best12-v1"))
    rows, paths, hashes = [], [], []
    for labels_path, replay in sources:
        labels, spec, bundle = verified_replay(replay, labels_path)
        rows += rows_for_dataset(dataset, labels, spec, bundle, replay)
        paths.append(str(labels_path))
        hashes.append(sha(labels_path))
    frozen_folds(rows, dataset)
    return rows, paths, hashes


def population(run: str, subdir: str, k: int, cohort: str):
    folder = BASE / run / subdir
    metrics = read(folder / "metrics.json")
    if "summary" in metrics:
        return next(p for p in metrics["summary"]["populations"] if p["horizon"] == k)
    return next(p for p in metrics["populations"] if p["cohort"] == cohort and p["horizon"] == k)


def draw_sheet(curves, run_out: Path, emotions: list[str], sheet: int):
    fk.use()
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.97, top=.96, bottom=.12, hspace=.27, wspace=.18)
    chance = np.r_[0, np.geomspace(1e-5, 1, 200)]
    for ax, emotion in zip(axes.flat, emotions):
        ax.plot(chance, chance, color=fk.LIGHT, lw=1)
        for k in HORIZONS:
            c = curves.get(f"{emotion}:k{k}")
            if c:
                ax.step(c["fpr"], c["tpr"], where="post", color=COLORS[k],
                        linestyle=STYLES[k], linewidth=1.8)
                ax.text(.03, .96-k/5*.065, f"k={k} · AUROC {c['auroc']:.2f}",
                        transform=ax.transAxes, ha="left", va="top", color=COLORS[k], fontsize=10)
        ax.set_title(emotion.capitalize(), loc="left", fontsize=12)
        ax.set_xscale("symlog", linthresh=.001)
        ax.set(xlim=(0, 1), ylim=(0, 1.02),
               xticks=[0, .001, .01, .1, 1], xticklabels=["0", "0.1%", "1%", "10%", "100%"],
               yticks=[0, .25, .5, .75, 1], yticklabels=["0%", "25%", "50%", "75%", "100%"])
        ax.grid(alpha=.12)
    for ax in axes[:, 0]:
        ax.set_ylabel("True-positive rate")
    for ax in axes[-1]:
        ax.set_xlabel("False-positive rate")
    stem = run_out / f"roc-emotions-{sheet}"
    fk.save(fig, stem.with_suffix(".png"), words=35, max_panels=4, max_series=4)
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def draw_best(curves, run_out: Path, emotion: str):
    """One selected k=0 emotion; selection is descriptive, not held out."""
    fk.use()
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    fig.subplots_adjust(left=.14, right=.97, top=.95, bottom=.18)
    chance = np.r_[0, np.geomspace(1e-5, 1, 200)]
    ax.plot(chance, chance, color=fk.LIGHT, lw=1)
    for k in HORIZONS:
        c = curves.get(f"{emotion}:k{k}")
        if c:
            ax.step(c["fpr"], c["tpr"], where="post", color=COLORS[k],
                    linestyle=STYLES[k], linewidth=2)
            ax.text(.03, .97-k/5*.085, f"k={k} · AUROC {c['auroc']:.2f}",
                    transform=ax.transAxes, ha="left", va="top", color=COLORS[k], fontsize=11)
    ax.set_xscale("symlog", linthresh=.001)
    ax.set(xlim=(0, 1), ylim=(0, 1.02),
           xticks=[0, .001, .01, .1, 1], xticklabels=["0", "0.1%", "1%", "10%", "100%"],
           yticks=[0, .25, .5, .75, 1], yticklabels=["0%", "25%", "50%", "75%", "100%"],
           xlabel="False-positive rate", ylabel="True-positive rate")
    ax.grid(alpha=.12)
    stem = run_out / "best-k0-roc"
    fk.save(fig, stem.with_suffix(".png"), words=25)
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for run, model, dataset, mode, subdir in RUNS:
        rows, source_paths, source_hashes = load_rows(run, mode, subdir or "")
        cohort = "certain_only" if mode == "glm52-manual" else "all_available"
        run_out = OUT / run / subdir
        run_out.mkdir(parents=True, exist_ok=True)
        curves, flat, unavailable = {}, [], []
        for k in HORIZONS:
            pairs = make_pairs(rows, k, certain_only=(cohort == "certain_only"))
            expected = population(run, subdir, k, cohort)
            require(len(pairs) == expected["eligible_pairs"], "Population changed: " + run)
            for emotion in EMOTIONS:
                try:
                    c = roc_for_probe(pairs, emotion)
                except (ValueError, AssertionError) as exc:
                    unavailable.append({"emotion": emotion, "horizon": k, "reason": str(exc)})
                    continue
                curves[f"{emotion}:k{k}"] = c
                flat.append({"run": run, "model": model, "dataset": dataset, "cohort": cohort,
                             "emotion": emotion, "horizon": k, "auroc": c["auroc"],
                             "positive_units": c["positive_sentences"],
                             "negative_units": c["negative_sentences"],
                             "traces": len({p["trace_id"] for p in pairs}),
                             "groups": len({p["group_id"] for p in pairs}),
                             "tpr_at_fpr_1pct": c["tpr_at_fpr_1pct"],
                             "tpr_at_fpr_5pct": c["tpr_at_fpr_5pct"],
                             "tpr_at_fpr_10pct": c["tpr_at_fpr_10pct"]})
        require(all(f"{e}:k0" in curves for e in EMOTIONS), "Incomplete k=0 ROC: " + run)
        for i in range(3):
            draw_sheet(curves, run_out, EMOTIONS[i*4:(i+1)*4], i+1)
        best = max((r for r in flat if r["horizon"] == 0), key=lambda r: r["auroc"])
        draw_best(curves, run_out, best["emotion"])
        receipt = {"run": run, "model": model, "dataset": dataset, "cohort": cohort,
                   "method": "held-out fold ROC; polarity and empirical CDF fitted on training folds only",
                   "horizons": list(HORIZONS), "source_paths": source_paths,
                   "source_sha256": source_hashes, "curves": curves, "metrics": flat,
                   "unavailable": unavailable}
        (run_out / "roc-data.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        summary += flat
        print(run, subdir, len(flat), "curves", len(unavailable), "unavailable", flush=True)
    with (OUT / "all-roc-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
