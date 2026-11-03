"""Replot saved emotion/RH forecasts on the equivalent -10..0 axis.

No scores, labels, populations, estimates, or uncertainty bands are changed:
each original horizon k is drawn at x=-k. The token-position diagnostic is omitted.
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


import hashlib
import html
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE / "figures/_kit"))
import figkit as fk

RUNS = (
    ("gemma3-4b-sales-gpt55-v1", "Gemma 3 4B", "Sales", "sentences"),
    ("gemma3-4b-coding-lines-v1", "Gemma 3 4B", "Coding", "code lines"),
    ("gemma4b-terminalwrench-neutral-v7-rh-20261026-v3", "Gemma 3 4B", "TerminalWrench (neutral v7)", "sentences"),
    ("gemma-3-27b-sales-haiku55-v1", "Gemma 3 27B", "Sales", "sentences"),
    ("gemma-3-27b-coding-verbose-haiku55-v1", "Gemma 3 27B", "Coding (verbose prompt)", "sentences"),
    ("gemma-3-27b-terminalwrench-pilot50-v1", "Gemma 3 27B", "TerminalWrench (pilot 50)", "sentences"),
    ("gpt-oss-120b-sales-v1", "GPT-OSS 120B", "Sales", "sentences"),
    ("gpt-oss-120b-coding-v1", "GPT-OSS 120B", "Coding", "sentences"),
    ("gpt-oss-120b-terminalwrench-deepseek-balanced50-v1", "GPT-OSS 120B", "TerminalWrench (DeepSeek balanced 50)", "sentences"),
    ("gptoss120b-terminalwrench-neutral-v7-rh-20261026-v3", "GPT-OSS 120B", "TerminalWrench (neutral v7)", "sentences"),
    ("glm-5.3-coding-1000-gpt55-v1", "GLM 5.3", "Coding (1,037 sentences)", "sentences"),
    ("goodfire-glm52-deepswe-manual-pilot-v1", "GLM 5.2", "DeepSWE (manual pilot)", "sentences"),
    ("goodfire-glm52-deepswe-gpt61sol-pilot-v1", "GLM 5.2", "DeepSWE (GPT-6.1 Sol pilot)", "sentences"),
    ("goodfire-glm52-deepswe-gpt61sol-expanded-5k-v1", "GLM 5.2", "DeepSWE (expanded 5k)", "sentences"),
)
EMOTIONS = ("afraid", "anxious", "appreciative", "ashamed", "calm", "desperate",
            "excited", "frustrated", "guilty", "hostile", "sad", "surprised")


def render(run: str, model: str, dataset: str, units: str):
    folder = ROOT / "runs/emotion-rh-forecasting" / run
    metrics_path = folder / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    cohorts = {e["cohort"] for e in metrics["estimates"]}
    cohort = "all_available" if "all_available" in cohorts else "certain_only"
    by_key = {(e["model"], e["horizon"]): e for e in metrics["estimates"]
              if e["cohort"] == cohort}
    assert all((emotion, k) in by_key for emotion in EMOTIONS for k in range(11))
    assert all(by_key[("always_not_hacking", k)]["repeated_sampling_mean_accuracy"] == .5
               for k in range(11))

    fk.use()
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(3, 4, figsize=(17, 10), sharex=True, sharey=True)
    fig.subplots_adjust(left=.055, right=.985, top=.88, bottom=.22, wspace=.16, hspace=.32)
    x = -np.arange(11)
    for ax, emotion in zip(axes.flat, EMOTIONS):
        current = [by_key[(emotion, k)] for k in range(11)]
        ax.plot(x, [100 * e["repeated_sampling_mean_accuracy"] for e in current],
                color="#3169a8", marker="o", markersize=3, linewidth=1.8,
                label="Emotion probe")
        ax.fill_between(x,
                        [100 * e["sampling_interval95"][0] for e in current],
                        [100 * e["sampling_interval95"][1] for e in current],
                        color="#3169a8", alpha=.13)
        ax.axhline(50, color="#a5652e", linestyle="--", linewidth=1,
                   label="Always non-RH (50%)")
        ax.set_title(emotion.capitalize(), fontsize=12)
        ax.set_xlim(-10.25, 0.25)
        ax.set_xticks([-10, -8, -6, -4, -2, 0])
        ax.grid(axis="y", alpha=.12)
    values = [100 * by_key[(e, k)]["repeated_sampling_mean_accuracy"]
              for e in EMOTIONS for k in range(11)]
    axes.flat[0].set_ylim(max(0, 5 * np.floor(min(values) / 5) - 5),
                          min(100, 5 * np.ceil(max(values) / 5) + 5))
    for ax in axes[:, 0]:
        ax.set_ylabel("Balanced accuracy (%)")
    for ax in axes[-1]:
        ax.set_xlabel(f"{units.capitalize()} before target (−k)")
    fig.suptitle(f"{model} · {dataset} · RH forecasting", x=.055, y=.975,
                 ha="left", fontsize=20)
    fig.text(.055, .925, f"{cohort.replace('_', ' ')} cohort · saved estimates at x = −k", fontsize=11)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(.5, .105), ncol=2, frameon=False)
    fig.text(.055, .025,
             "−10 forecasts the label ten units later; 0 classifies the current unit. "
             "Targets can be RH or non-RH, and future RH may be a continuation. "
             "Bands show balanced-subsample variation.", fontsize=9, color="#51565c")
    dest = folder / "figures" / "accuracy-trends-minus10-to-0"
    # The twelve panels match the existing experiment figure the user asked to re-axis.
    fk.save(fig, dest.with_suffix(".png"), max_panels=12, words=65,
            title_words=15, min_pt=8)
    fig.savefig(dest.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    manifest = {
        "source_metrics": str(metrics_path),
        "source_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        "model": model,
        "cohort": cohort,
        "transform": "x = -horizon; plotted probe accuracy, chance baseline, and bands unchanged; token-position baseline omitted",
        "x_values": list(range(-10, 1)),
        "dataset": dataset,
        "unit": units,
    }
    dest.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(dest.with_suffix(".png"))


def write_gallery():
    root = ROOT / "runs/emotion-rh-forecasting"
    models = dict.fromkeys(model for _, model, _, _ in RUNS)
    sections = []
    for model in models:
        cards = []
        for run, item_model, dataset, _ in RUNS:
            if item_model != model:
                continue
            src = f"{run}/figures/accuracy-trends-minus10-to-0.png"
            meta = json.loads((root / run / "figures/accuracy-trends-minus10-to-0.json").read_text())
            title = html.escape(dataset)
            cohort = html.escape(meta["cohort"].replace("_", " "))
            cards.append(f'<article><h3>{title}</h3><p>{cohort} · {html.escape(run)}</p>'
                         f'<a href="{html.escape(src)}"><img loading="lazy" src="{html.escape(src)}" '
                         f'alt="{html.escape(model)} · {title} forecasting"></a></article>')
        sections.append(f'<section><h2>{html.escape(model)}</h2>{"".join(cards)}</section>')
    page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Emotion/RH forecasting · −10 to 0</title>'
            '<style>body{font:16px/1.45 system-ui,sans-serif;color:#1a2530;background:#f4f7fa;'
            'margin:0}header{padding:22px 4vw;background:white;border-bottom:1px solid #d9e2e9}'
            'h1{margin:0;font-size:26px}header p{max-width:850px}main{max-width:1600px;margin:auto;'
            'padding:15px 4vw}nav{display:flex;gap:14px;flex-wrap:wrap}a{color:#146098}'
            'section{margin:28px 0}h2{background:#e8eef4;padding:9px 12px;border-radius:6px}'
            'article{background:white;border:1px solid #dbe3e9;border-radius:8px;padding:14px;'
            'margin:18px 0}h3{margin:0}article p{font-size:13px;color:#526579;margin:2px 0 10px}'
            'img{display:block;max-width:100%;height:auto}</style></head><body>'
            '<header><h1>Emotion/RH forecasting · −10 to 0</h1>'
            '<p>Saved probe accuracy across 12 emotions. The x axis is −horizon, so the target '
            'moves closer from left to right. Token-position diagnostic omitted; blue bands show '
            'balanced-subsample variation. Separate datasets and run versions remain distinct.</p>'
            '<nav>' + ''.join(f'<a href="#{html.escape(model.replace(" ", "-"))}">{html.escape(model)}</a>'
                            for model in models) + '</nav></header><main>'
            + ''.join(section.replace('<section>', f'<section id="{html.escape(model.replace(" ", "-"))}">', 1)
                      for model, section in zip(models, sections))
            + '</main></body></html>')
    target = root / "negative-horizon-gallery.html"
    target.write_text(page)
    print(target)


if __name__ == "__main__":
    for args in RUNS:
        render(*args)
    write_gallery()
