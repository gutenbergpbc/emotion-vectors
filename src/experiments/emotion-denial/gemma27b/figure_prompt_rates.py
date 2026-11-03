"""Render the Gemma 27B prompt comparison from saved result JSON."""

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
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / '_kit'))
import figkit as fk  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = _REPO
OUT = ROOT / 'runs/gemma27-session-label-control-20261026/figures'
OLD = ROOT / 'runs/gemma27-neutral100-whole-trace-audit-20261026/summary.json'
CONTROL = ROOT / 'runs/gemma27-session-label-control-20261026/results.json'
EMOTION = ROOT / 'runs/gemma27-emotion-denial-twoarms-20261026/results.json'
ARMS = [
    ('neutral', 'Neutral / no prompt', '#BFC9CF'),
    ('control', 'Session label', '#8299A5'),
    ('named', 'Named emotions', '#C98F3D'),
    ('general', 'General emotions', '#E0AE6D'),
]


def read(path):
    return json.loads(path.read_text())


def values():
    old, control, emotion = read(OLD), read(CONTROL), read(EMOTION)
    assert old['model'] == control['model'] == emotion['model'] == 'google/gemma-3-27b-it'
    result = {}
    for task in ('sales', 'coding'):
        baseline = old['cohorts'][task]['all']
        assert baseline['n'] == 100
        result[(task, 'neutral')] = baseline['reward_hack']
    for bundle in (control, emotion):
        for cell in bundle['cells']:
            assert cell['n'] == 100
            assert cell['reward_hack'] / 100 == cell['rh_rate']
            result[(cell['dataset'], cell['arm'])] = cell['reward_hack']
    assert len(result) == 8
    return result


def main():
    counts = values()
    fk.use()
    plt.rcParams.update({
        'figure.constrained_layout.use': False,
        'axes.grid': False,
        'font.size': 12,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.3), sharey=True)
    fig.patch.set_facecolor('white')
    fig.text(0.07, 0.96, 'GEMMA 27B / REWARD HACKING', color='#3186A8',
             fontsize=11, weight='semibold', ha='left', va='top')
    fig.text(0.07, 0.87, 'Reward hacking by prompt', color='#26323D',
             fontsize=23, weight='semibold', ha='left', va='top')

    for ax, task in zip(axes, ('sales', 'coding')):
        ax.set_facecolor('#FBFBFA')
        for x in (25, 50, 75, 100):
            ax.axvline(x, color='#E4E9EA', lw=0.8, zorder=0)
        for i, (arm, label, color) in enumerate(ARMS):
            count = counts[(task, arm)]
            ax.barh(i, count, height=0.52, color=color, zorder=2)
            ax.text(count + 2.0, i, f'{count}%', color='#26323D',
                    fontsize=12.5, weight='semibold', va='center', ha='left')
        ax.set_title(task.title(), loc='left', color='#26323D',
                     fontsize=15, weight='semibold', pad=12)
        ax.set_xlim(0, 110)
        ax.set_ylim(-0.55, 3.55)
        ax.invert_yaxis()
        ax.set_xticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'])
        ax.tick_params(axis='x', length=0, colors='#6B7D88', pad=9)
        ax.tick_params(axis='y', length=0, pad=12)
        ax.spines[['top', 'left', 'right']].set_visible(False)
        ax.spines['bottom'].set_color('#BFC9CF')
    axes[0].set_yticks(range(4), [label for _, label, _ in ARMS])
    axes[0].tick_params(axis='y', labelcolor='#26323D', labelsize=12)
    axes[1].tick_params(axis='y', labelleft=False)
    fig.text(0.55, 0.08, 'Judge-labeled RH rate', color='#26323D',
             fontsize=12.5, ha='center')
    fig.subplots_adjust(left=0.22, right=0.96, top=0.68, bottom=0.18, wspace=0.18)

    OUT.mkdir(parents=True, exist_ok=True)
    fk.save(fig, OUT / 'prompt-rh-rates.png', max_panels=2, max_series=4, words=35)
    fig.savefig(OUT / 'prompt-rh-rates.svg')
    plt.close(fig)
    (OUT / 'prompt-rh-rates.caption.md').write_text(
        'Whole-trace evidence-based judge labels among 100 Gemma 3 27B traces per cell '
        'across three fixed scenarios. The no-prompt cohort contains 12 local vLLM and '
        '88 OpenRouter traces; all system-prompt cohorts use OpenRouter. Rates are '
        'descriptive and do not measure uncertainty across tasks.\n'
    )


if __name__ == '__main__':
    main()
