"""Plot Gemma 3 4B coding code-line forecasts from frozen labels and replay."""
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
from pathlib import Path
import sys

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE.parent), str(HERE)]
from shared.io import atomic, read, require, sha  # noqa: E402
from emotion_rh_forecasting.analysis import EMOTIONS, assign_folds, evaluate, make_pairs  # noqa: E402
from emotion_rh_forecasting.inputs import Sources, load_probes, vector_identity  # noqa: E402
from plot_glm_hack_forecast import rows_for_dataset  # noqa: E402

LABELS = ROOT / 'runs/emotion-rh-code-line-labels/gemma3-4b-coding-v1/labels.json'
REPLAY = ROOT / 'runs/emotion-rh-replay/gemma3-4b-coding-lines-best12-v1'
OUT = ROOT / 'runs/emotion-rh-forecasting/gemma3-4b-coding-lines-v1'


def adapted(trace):
    return {**trace, 'dataset': 'coding',
            'units': [{**unit, 'kind': 'sentence_or_fragment', 'turn': 0,
                       'source_field': 'response', 'hacking': unit['indicative_of_rh'],
                       'needs_review': False}
                      for unit in trace['units']]}


def figure(estimates, destination, subtitle):
    colors = {'emotion': '#377cb5', 'position': '#687d8c', 'baseline': '#d58c4d'}
    fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharex=True, sharey=True)
    by_key = {(r['model'], r['horizon']): r for r in estimates if r['cohort'] == 'all_available'}
    for ax, emotion in zip(axes.flat, EMOTIONS):
        for model, style, color, name in ((emotion, '-', colors['emotion'], 'Emotion probe'),
                                           ('token_position', ':', colors['position'], 'Token position only'),
                                           ('always_not_hacking', '--', colors['baseline'], '50% chance')):
            points = [by_key[(model, h)] for h in range(11)]
            ax.plot(range(11), [p['repeated_sampling_mean_accuracy'] * 100 for p in points],
                    style, color=color, linewidth=1.8, marker='o' if model == emotion else None,
                    markersize=3, label=name)
            if model == emotion:
                ax.fill_between(range(11), [p['sampling_interval95'][0] * 100 for p in points],
                                [p['sampling_interval95'][1] * 100 for p in points], color=color, alpha=.13)
        ax.set_title(emotion.capitalize())
        ax.set_xlim(0, 10)
        ax.set_xticks(range(11))
        ax.set_ylim(20, 85)
        ax.grid(axis='y', alpha=.15)
    for ax in axes[:, 0]:
        ax.set_ylabel('Balanced-evaluation accuracy (%)')
    for ax in axes[-1]:
        ax.set_xlabel('Code-line horizon (+h)')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(.04, .035), ncol=3)
    fig.suptitle('Gemma 3 4B · Coding · emotion probes forecasting violation-evidence lines', fontsize=17, y=.988)
    fig.text(.06, .943, subtitle, fontsize=10)
    fig.text(.06, .018, 'Bands: balanced-subsample variability, not generalization confidence. h=0 is current code line.', fontsize=9)
    fig.tight_layout(rect=[0, .10, 1, .91])
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(destination.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)


def main():
    original = read(LABELS)
    require(original['summary']['status'] == 'complete', 'Code-line labels incomplete')
    labels = {**original, 'traces': [adapted(t) for t in original['traces']]}
    spec = read(REPLAY / 'replay-spec.json')
    bundle = load_probes(REPLAY / 'probes.json', Sources())
    require(spec['labels_sha256'] == sha(LABELS) and spec['probe_identity'] == vector_identity(bundle),
            'Labels and replay vectors differ')
    require(len(list((REPLAY / 'scores').glob('*.receipt.json'))) == len(spec['jobs']), 'Replay incomplete')
    rows = rows_for_dataset('coding', labels, spec, bundle, REPLAY)
    # Each coding scenario has a distinct prompt. Hold out whole tasks for
    # evaluation, keeping all lines of each trace together.
    folds = assign_folds(rows, {'method': 'held_out_task'}, 20261022)
    estimates, populations, failures = [], [], []
    common = {(r['trace_id'], r['unit_id']) for r in make_pairs(rows, 10)}
    for horizon in range(11):
        for cohort in ('all_available', 'common_sources', 'certain_only'):
            pairs = make_pairs(rows, horizon, common if cohort == 'common_sources' else None,
                               cohort == 'certain_only')
            try:
                result, _ = evaluate(pairs, cohort, horizon, 20261022, 2000,
                                     set(folds['trace_folds'].values()))
                estimates.extend(result['estimates'])
                populations.append(result['population'])
            except (ValueError, AssertionError) as exc:
                failures.append({'cohort': cohort, 'horizon': horizon,
                                 'reason': str(exc), 'eligible_pairs': len(pairs)})
                if cohort == 'all_available':
                    raise
    OUT.mkdir(parents=True, exist_ok=True)
    atomic(OUT / 'metrics.json', {'estimates': estimates, 'populations': populations,
                                  'unavailable': failures, 'folds': folds})
    with (OUT / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(estimates[0]))
        writer.writeheader()
        writer.writerows(estimates)
    summary = {'status': 'complete', 'model': bundle['model'], 'probe_identity': spec['probe_identity'],
               'labels_sha256': sha(LABELS), 'replay_identity': spec['identity'],
               'replay_interpretation': spec['interpretation'],
               'traces': len(labels['traces']), 'code_lines': len(rows),
               'evidence_lines': sum(r['hacking'] for r in rows),
               'evaluation': 'Three held-out coding tasks; train-only scalar cutoffs; 2,000 balanced subsamples per horizon',
               'h0': next(p for p in populations if p['cohort'] == 'all_available' and p['horizon'] == 0),
               'h3': next(p for p in populations if p['cohort'] == 'all_available' and p['horizon'] == 3),
               'h10': next(p for p in populations if p['cohort'] == 'all_available' and p['horizon'] == 10),
               'unavailable_cohorts': failures}
    atomic(OUT / 'summary.json', summary)
    subtitle = (f"{summary['traces']} traces · {summary['code_lines']} code lines · "
                f"{summary['evidence_lines']} evidence lines · held-out coding tasks · rule audit")
    figure(estimates, OUT / 'figures/accuracy-trends', subtitle)
    print(summary)


if __name__ == '__main__':
    main()
