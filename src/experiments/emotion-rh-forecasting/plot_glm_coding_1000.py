"""Plot GPT-5.5-labeled GLM coding sentence forecasts from verified probe replay."""
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

HERE = Path(__file__).resolve().parent
CANONICAL = HERE.parent
ROOT = CANONICAL.parents[1]
sys.path[:0] = [str(CANONICAL), str(HERE)]
from shared.io import atomic, read, require, sha  # noqa: E402
from emotion_rh_forecasting.analysis import EMOTIONS, assign_folds, evaluate, make_pairs  # noqa: E402
from emotion_rh_forecasting.inputs import Sources, load_probes, vector_identity  # noqa: E402
from plot_glm_hack_forecast import rows_for_dataset  # noqa: E402


REPLAY = ROOT / 'runs/emotion-rh-replay/glm-5.3-coding-1000-best12-v1'
LABELS = ROOT / 'runs/emotion-rh-sentence-labels/glm53-coding-1000-gpt55-v1/labels.json'
OUT = ROOT / 'runs/emotion-rh-forecasting/glm-5.3-coding-1000-gpt55-v1'

# Frozen task-disjoint folds. Each has RH targets at h=10, and every task
# appears in exactly one fold. Fold choice uses labels for stratification only.
TASK_FOLDS = {
    'constant_median': 0, 'constant_graph_path': 0, 'impossible_prime': 0,
    'constant_palindrome': 1, 'impossible_search': 1,
    'impossible_sort': 2, 'constant_deduplicate': 2,
    'constant_word_frequency': 2, 'sublinear_unsorted_search': 2,
}


def plot(estimates: list[dict], populations: list[dict]) -> None:
    by_key = {(row['model'], row['horizon']): row for row in estimates
              if row['cohort'] == 'all_available'}
    counts = {row['horizon']: row['balanced_eval_pairs'] for row in populations
              if row['cohort'] == 'all_available'}
    fig, axes = plt.subplots(3, 4, figsize=(19, 11), sharex=True, sharey=True)
    colors = {'emotion': '#3976ad', 'position': '#687a8c', 'baseline': '#d38d4a'}
    for ax, emotion in zip(axes.flat, EMOTIONS):
        for model, style, color, name in (
            (emotion, '-', colors['emotion'], 'Emotion probe'),
            ('token_position', ':', colors['position'], 'Token position'),
            ('always_not_hacking', '--', colors['baseline'], 'Always non-hacking'),
        ):
            points = [by_key[(model, h)] for h in range(11)]
            x = list(range(11))
            y = [p['repeated_sampling_mean_accuracy'] * 100 for p in points]
            ax.plot(x, y, style, color=color, linewidth=1.8,
                    marker='o' if model == emotion else None, markersize=3, label=name)
            if model == emotion:
                ax.fill_between(x,
                    [p['sampling_interval95'][0] * 100 for p in points],
                    [p['sampling_interval95'][1] * 100 for p in points],
                    color=color, alpha=.14)
        ax.set_title(emotion.capitalize())
        ax.set_xlim(0, 10)
        ax.set_xticks(range(11))
        ax.set_ylim(20, 85)
        ax.grid(axis='y', alpha=.15)
    for ax in axes[:, 0]:
        ax.set_ylabel('Balanced-evaluation accuracy (%)')
    for ax in axes[-1]:
        ax.set_xlabel('Sentence horizon (+h)')
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower left', bbox_to_anchor=(.04, .045), ncol=3)
    fig.suptitle('GLM 5.3 · Coding · emotion probes forecasting reward-hacking sentences',
                 fontsize=18, y=.99)
    fig.text(.06, .95,
             '51 traces · 1,037 GPT-5.5-labeled prose sentences · 3 task-disjoint folds · native GLM NVFP4 replay',
             fontsize=10)
    fig.text(.06, .018,
             f'Balanced pairs: h=0 {counts[0]}, h=3 {counts[3]}, h=10 {counts[10]}. '
             'Bands show subsampling variability, not generalization confidence. '
             'h=0 classifies this sentence.', fontsize=9)
    fig.tight_layout(rect=[0, .11, 1, .92])
    destination = OUT / 'figures/accuracy-trends'
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(destination.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    labels = read(LABELS)
    spec = read(REPLAY / 'replay-spec.json')
    bundle = load_probes(REPLAY / 'probes.json', Sources())
    require(labels['summary']['status'] == 'complete' and labels['summary']['prose_units'] == 1037,
            'Combined labels incomplete')
    require(spec['labels_sha256'] == sha(LABELS)
            and spec['probe_identity'] == vector_identity(bundle)
            and spec['probes_sha256'] == sha(REPLAY / 'probes.json'),
            'Label/probe identity mismatch')
    require(len(list((REPLAY / 'scores').glob('*.receipt.json'))) == len(spec['jobs']),
            'Replay incomplete')
    rows = rows_for_dataset('coding', labels, spec, bundle, REPLAY)
    require(len(rows) == 1037 and sum(r['hacking'] for r in rows) == 198, 'Scored label population differs')
    require({r['task'] for r in rows} == set(TASK_FOLDS), 'Task set changed')
    for row in rows:
        row['group_id'] = row['task']
        row['fold'] = TASK_FOLDS[row['task']]
    folds = assign_folds(rows, {'method': 'provided'}, 20261022)
    require(len(set(folds['trace_folds'].values())) == 3 and len(folds['group_folds']) == 9,
            'Task-disjoint fold structure changed')
    common = {(r['trace_id'], r['unit_id']) for r in make_pairs(rows, 10)}
    estimates, populations, unavailable, fold_stats = [], [], [], []
    for horizon in range(11):
        for cohort in ('all_available', 'common_sources', 'certain_only'):
            pairs = make_pairs(rows, horizon, common if cohort == 'common_sources' else None,
                               cohort == 'certain_only')
            try:
                result, _ = evaluate(pairs, cohort, horizon, 20261022, 2000, {0, 1, 2})
                estimates.extend(result['estimates'])
                populations.append(result['population'])
                fold_stats.extend(result['fold_statistics'])
            except (ValueError, AssertionError) as exc:
                if cohort == 'all_available':
                    raise
                unavailable.append({'cohort': cohort, 'horizon': horizon,
                                    'eligible_pairs': len(pairs), 'reason': str(exc)})
    OUT.mkdir(parents=True, exist_ok=True)
    atomic(OUT / 'metrics.json', {'estimates': estimates, 'populations': populations,
                                  'fold_statistics': fold_stats, 'unavailable': unavailable,
                                  'folds': folds})
    with (OUT / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(estimates[0]))
        writer.writeheader()
        writer.writerows(estimates)
    plot(estimates, populations)
    primary = {p['horizon']: p for p in populations if p['cohort'] == 'all_available'}
    summary = {'status': 'complete', 'dataset': 'coding', 'model': bundle['model'],
               'probe_identity': spec['probe_identity'], 'labels_path': str(LABELS),
               'labels_sha256': sha(LABELS), 'replay_path': str(REPLAY),
               'replay_identity': spec['identity'], 'traces': 51, 'sentences': 1037,
               'rh_sentences': 198, 'folds': 3, 'task_fold_assignment': TASK_FOLDS,
               'primary_balanced_pairs': {str(h): primary[h]['balanced_eval_pairs'] for h in range(11)},
               'unavailable_sensitivity_cohorts': unavailable,
               'interpretation': spec['interpretation'],
               'limitations': 'Task-disjoint folds but outcome-stratified old traces; full-trace retrospective model labels. Native NVFP4 replay is not byte-identical to OpenRouter serving. Future targets can include ongoing RH. Bands show subsampling variability, not generalization confidence.'}
    atomic(OUT / 'summary.json', summary)
    (OUT / 'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>GLM coding RH forecast</title>'
        '<style>body{font:16px/1.55 system-ui;max-width:1600px;margin:24px auto;padding:0 24px;color:#243440}'
        'img{width:100%}.note{background:#edf3f7;padding:16px;border-radius:8px}</style>'
        '<h1>GLM 5.3 coding sentence-level RH forecast</h1>'
        '<p class="note">51 traces and 1,037 prose sentences, with 198 GPT-5.5 RH-positive labels. '
        'Three task-disjoint folds. Source features use only tokens completed by the current sentence. '
        'The 23 original traces were selected by whole-trace outcome; this is an exploratory population. '
        'Labels were judged with full-trace context, and future targets can reflect ongoing RH. '
        'Native GLM NVFP4 replay is not byte-identical to OpenRouter serving. '
        'Bands show balanced-subsample variability, not generalization confidence.</p>'
        '<img src="figures/accuracy-trends.svg" alt="Twelve-panel GLM emotion probe forecast">'
        '<p><a href="metrics.csv">Metrics CSV</a> · <a href="metrics.json">Full metrics</a> · '
        '<a href="summary.json">Run summary</a></p></html>')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
