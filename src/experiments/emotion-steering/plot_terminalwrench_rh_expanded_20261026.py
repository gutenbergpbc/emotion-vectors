"""Plot episode-level RH prose rates by v7 steering arm for three models.

Claim: observed arm rates vary; the enlarged neutral cohort is a descriptive reference.
Template: horizontal dot plot with Wilson intervals, one figure per model.
Shows: all labelable steering emotions at alpha -0.10 and +0.10.
Accent: positive-alpha arm; negative-alpha arm is grey.
Text: model, two arm labels, neutral marker, x-axis label, emotion tick labels.
Caption: post-hoc prose labels, denominator, interval, unequal task coverage.
Sketch: neutral vertical line | four emotion rows with paired dots and whiskers.
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
from html import escape
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE / "_kit"))
import figkit as fk

DATA = ROOT / "runs/emotion-rh-sentence-labels/terminalwrench-v7-expanded-neutral-rh-20261026"
OUT = DATA / "plots"
MODELS = {"GPT-OSS 120B": "gpt-oss-120b", "Gemma 4B": "gemma-4b", "GLM 5.3": "glm-5-3"}


def load():
    rows = list(csv.DictReader((DATA / "rates.csv").open()))
    for row in rows:
        for name in ("alpha", "rate", "ci_low", "ci_high"):
            row[name] = float(row[name]) if row[name] else None
        for name in ("saved", "labelable", "unlabelable", "positive", "candidate", "sentences"):
            row[name] = int(row[name])
    return rows


def draw(model, rows, stem):
    neutral = next(row for row in rows if row["model"] == model and row["emotion"] == "neutral")
    arms = [row for row in rows if row["model"] == model and row["emotion"] != "neutral"
            and row["labelable"]]
    emotions = sorted({row["emotion"] for row in arms})
    fig, ax = plt.subplots(figsize=(7, 4.0 if len(emotions) >= 4 else 3.4))
    ax.axvspan(neutral["ci_low"], neutral["ci_high"], color=fk.LIGHT, alpha=.35, zorder=0)
    ax.axvline(neutral["rate"], color=fk.INK, linestyle="--", linewidth=1.3, zorder=1)
    for i, emotion in enumerate(emotions):
        y = len(emotions) - 1 - i
        for alpha, offset, color, marker in ((-.1, .14, fk.GREY, "o"), (.1, -.14, fk.BLUE, "D")):
            matching = [row for row in arms if row["emotion"] == emotion and row["alpha"] == alpha]
            if not matching:
                continue
            row = matching[0]
            ax.hlines(y + offset, row["ci_low"], row["ci_high"], color=color,
                      linewidth=2.1, zorder=2)
            ax.scatter(row["rate"], y + offset, s=42, marker=marker, color=color, zorder=3)
    ax.set_yticks(range(len(emotions)), [emotion.title() for emotion in reversed(emotions)])
    ax.set_xlim(-.015, 1.01)
    ax.set_ylim(-.55, len(emotions) - .1)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="x", color="#E7E7E7", linewidth=.7)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Traces with RH prose")
    ax.set_title(model)
    ax.text(neutral["rate"], -.43, f"Neutral {neutral['positive']}/{neutral['labelable']}",
            ha="left", va="center", fontsize=10, color=fk.INK)
    ax.text(.985, .98, "−0.10", color=fk.MUTED, transform=ax.transAxes,
            ha="right", va="top", fontsize=10)
    ax.text(.985, .89, "+0.10", color=fk.BLUE, transform=ax.transAxes,
            ha="right", va="top", fontsize=10)
    fk.save(fig, stem.with_suffix(".png"), words=25, min_pt=10, max_series=3)
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return neutral, arms


def main():
    fk.use()
    rows = load()
    OUT.mkdir(parents=True, exist_ok=True)
    sections = []
    for model, slug in MODELS.items():
        neutral, arms = draw(model, rows, OUT / slug)
        all_rows = [neutral] + [row for row in rows if row["model"] == model
                                and row["emotion"] != "neutral"]
        table_rows = "".join(
            f"<tr><td>{escape(row['emotion'].title())}</td>"
            f"<td>{row['alpha']:+.2f}</td><td>{row['positive']}/{row['labelable']}</td>"
            f"<td>{row['rate']:.1%}</td><td>{row['unlabelable']}</td></tr>"
            if row["labelable"] else
            f"<tr><td>{escape(row['emotion'].title())}</td><td>{row['alpha']:+.2f}</td>"
            f"<td>—</td><td>—</td><td>{row['unlabelable']}</td></tr>"
            for row in all_rows)
        sections.append(f"<section><h2>{escape(model)}</h2><a href='{slug}.png'>"
                        f"<img src='{slug}.svg' alt='{escape(model)} RH prose rates by steering arm'></a>"
                        f"<p>Neutral: {neutral['positive']}/{neutral['labelable']} labelable traces. "
                        "Dots show steered arm rates; horizontal whiskers and the neutral band "
                        "show 95% Wilson intervals.</p>"
                        f"<p><a href='{slug}.png'>PNG</a> · <a href='{slug}.pdf'>PDF</a> · "
                        f"<a href='{slug}.svg'>SVG</a></p>"
                        "<details><summary>Exact counts and missing prose</summary><table>"
                        "<thead><tr><th>Emotion</th><th>α</th><th>RH / labeled</th>"
                        "<th>Rate</th><th>No prose</th></tr></thead><tbody>"
                        f"{table_rows}</tbody></table></details></section>")
    (OUT / "index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>TerminalWrench RH prose by model</title>"
        "<style>body{font:16px/1.55 system-ui;max-width:1050px;margin:2rem auto;padding:0 1rem;color:#202124}"
        "h1{font-size:2rem}h2{margin-top:0}section{border-top:1px solid #ddd;padding:1.5rem 0}"
        "img{display:block;width:100%;max-width:860px}table{border-collapse:collapse;margin-top:.8rem}"
        "td,th{padding:.35rem 1.4rem .35rem 0;text-align:left;border-bottom:1px solid #eee}"
        "p{max-width:60rem}</style><h1>TerminalWrench reward-hacking prose</h1>"
        "<p>Share of labelable traces with at least one unflagged reward-hacking sentence. "
        "The neutral reference combines the original v7 run, the 8-trace extension, and the new "
        "20-trace extension per model. Steering arms use the saved v7 traces. The original v7 "
        "and first extension were labeled by GPT-6.1 Sol; uncertain sentences in the new "
        "extension were rejudged by GPT-6 Astra. These are descriptive, post-hoc prose labels. "
        "Tasks differ between arms, and tool-only reward hacking may be missed.</p>"
        "<p><a href='../rates.csv'>Rates CSV</a> · <a href='../rates.json'>Rates JSON</a> · "
        "<a href='../summary.json'>Source summary</a></p>" + "".join(sections) + "</html>")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
