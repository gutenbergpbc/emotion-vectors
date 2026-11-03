"""Draw saved pooled metrics as four model plots and one 2x2 comparison."""

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

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = _REPO
sys.path.insert(0, str(ROOT / 'docs/forensics-experiment-writeup/figures/_kit'))
import figkit as fk

SERIES = [('pearson', '#387c74', -.13), ('spearman', '#8a68a1', .13)]
SLUGS = ['glm53', 'gptoss120b', 'gemma4b', 'gemma27b']
fk.use()


def draw(ax, result):
    rows = result['results']
    ax.set_title(result['display_name'], loc='left', fontsize=15)
    ax.set(xlim=(-1, 1), ylim=(len(rows) - .45, -.55))
    ax.set_xticks([-1, -.5, 0, .5, 1])
    ax.set_yticks(range(len(rows)), [r['emotion'] for r in rows])
    ax.tick_params(labelleft=True, labelbottom=True, axis='both', length=0, pad=7)
    ax.grid(False)
    ax.grid(axis='x', color='#eeeeee', linewidth=.8)
    ax.axvline(0, color='#89958b', linewidth=1)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    for i, row in enumerate(rows):
        for name, color, offset in SERIES:
            if row[name] is not None:
                ax.plot(row[name], i + offset, 'o', color=color, markersize=5)
            if row[name + '_ci_low'] is not None:
                ax.hlines(i + offset, row[name + '_ci_low'], row[name + '_ci_high'],
                          color=color, linewidth=2)


def finish(fig, path, center):
    fig.text(fig.subplotpars.left, .965, 'Coding + Sales', va='top', color=fk.MUTED, fontsize=12)
    fig.legend(handles=[Line2D([], [], marker='o', linestyle='none', color=c, label=n.capitalize())
                        for n, c, _ in SERIES], loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(center, .99))
    fig.supxlabel('Correlation with self-report', y=.02, fontsize=13)
    fk.save(fig, path.with_suffix('.png'), max_series=30)
    for ext in ('.pdf', '.svg'):
        fig.savefig(path.with_suffix(ext))
    plt.close(fig)


data = [json.loads((HERE / slug / 'metrics.json').read_text()) for slug in SLUGS]
for slug, result in zip(SLUGS, data):
    fig, ax = plt.subplots(figsize=(7, 5.8), layout='none')
    fig.subplots_adjust(left=.21, right=.97, bottom=.13, top=.84)
    draw(ax, result)
    finish(fig, HERE / slug / 'coding-sales-pooled', .72)
fig, axes = plt.subplots(2, 2, figsize=(10, 9.4), sharex=True, sharey=True, layout='none')
fig.subplots_adjust(left=.145, right=.98, bottom=.09, top=.91, wspace=.62, hspace=.28)
for ax, result in zip(axes.flat, data):
    draw(ax, result)
finish(fig, HERE / 'coding-sales-pooled-2x2', .72)
