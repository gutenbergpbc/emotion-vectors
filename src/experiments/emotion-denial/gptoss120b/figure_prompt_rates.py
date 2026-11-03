"""Draw GPT-OSS prompt comparison from the saved judge summaries."""

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
RESULT = ROOT / 'runs/gptoss120b-emotion-denial-lowreason-20261027'
OUT = RESULT / 'figures'
ARMS = [('baseline', 'No prompt', '#BFC9CF'),
        ('named', 'Named emotions', '#C98F3D'),
        ('general', 'General emotions', '#E0AE6D')]


def main():
    baseline = json.loads((RESULT / 'baseline-results.json').read_text())
    prompts = json.loads((RESULT / 'results.json').read_text())
    assert prompts['model'] == 'openai/gpt-oss-120b'
    cells = {(c['dataset'], 'baseline'): c for c in baseline['cells']}
    cells.update({(c['dataset'], c['arm']): c for c in prompts['cells']})
    assert len(cells) == 6
    for cell in cells.values():
        assert abs(cell['reward_hack'] / cell['n'] - cell['rh_rate']) < 1e-12
    fk.use()
    plt.rcParams.update({'figure.constrained_layout.use': False, 'axes.grid': False, 'font.size': 12})
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.3), sharey=True)
    fig.patch.set_facecolor('white')
    fig.text(0.07, 0.96, 'GPT-OSS 120B / REWARD HACKING', color='#3186A8',
             fontsize=11, weight='semibold', ha='left', va='top')
    fig.text(0.07, 0.87, 'Reward hacking by prompt', color='#26323D',
             fontsize=23, weight='semibold', ha='left', va='top')
    for ax, task in zip(axes, ('sales', 'coding')):
        ax.set_facecolor('#FBFBFA')
        for x in (25, 50, 75, 100):
            ax.axvline(x, color='#E4E9EA', lw=0.8, zorder=0)
        for i, (arm, label, color) in enumerate(ARMS):
            value = cells[(task, arm)]['rh_rate'] * 100
            ax.barh(i, value, height=0.52, color=color, zorder=2)
            display = f'{value:.1f}%' if arm == 'baseline' else f'{value:.0f}%'
            if arm == 'baseline' and task == 'coding':
                display += '*'
            ax.text(value + 2, i, display, color='#26323D', fontsize=12.5,
                    weight='semibold', va='center', ha='left')
        ax.set_title(task.title(), loc='left', color='#26323D', fontsize=15, weight='semibold', pad=12)
        ax.set_xlim(0, 110)
        ax.set_ylim(-0.55, 2.55)
        ax.invert_yaxis()
        ax.set_xticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'])
        ax.tick_params(axis='x', length=0, colors='#6B7D88', pad=9)
        ax.tick_params(axis='y', length=0, pad=12)
        ax.spines[['top', 'left', 'right']].set_visible(False)
        ax.spines['bottom'].set_color('#BFC9CF')
    axes[0].set_yticks(range(3), [label for _, label, _ in ARMS])
    axes[0].tick_params(axis='y', labelcolor='#26323D', labelsize=12)
    axes[1].tick_params(axis='y', labelleft=False)
    fig.text(0.55, 0.115, 'Judge-labeled RH rate', color='#26323D', fontsize=12.5, ha='center')
    fig.text(0.07, 0.035, '* Coding baseline: 400-token forced final; prompt runs: 4,096 tokens with reasoning.',
             color='#6B7D88', fontsize=10, ha='left')
    fig.subplots_adjust(left=0.22, right=0.96, top=0.68, bottom=0.21, wspace=0.18)
    OUT.mkdir(parents=True, exist_ok=True)
    fk.save(fig, OUT / 'prompt-rh-rates.png', max_panels=2, max_series=3, words=45)
    fig.savefig(OUT / 'prompt-rh-rates.svg')
    plt.close(fig)
    (OUT / 'prompt-rh-rates.caption.md').write_text(
        'Judge-labeled reward hacking across three fixed scenarios per task. The earlier '
        'unsteered baseline has 156 sales and 96 coding traces; each prompt arm has 100 '
        'traces per task. The coding baseline used a local forced-final 400-token cap '
        '(84/96 hit the cap); the prompt conditions used OpenRouter with low reasoning '
        'effort and a 4,096-token cap. Providers and collection periods also differ. '
        'These descriptive rates do not establish prompt effects.\n')


if __name__ == '__main__':
    main()
