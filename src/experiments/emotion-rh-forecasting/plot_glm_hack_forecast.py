"""Fit frozen-GLM emotion probe forecasts from verified replay scores.

uv run --with scikit-learn --with matplotlib python \
  src/experiments/emotion-rh-forecasting/plot_glm_hack_forecast.py
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


from collections import Counter
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CANONICAL = HERE.parent
ROOT = CANONICAL.parents[1]
sys.path[:0] = [str(CANONICAL), str(HERE)]
from shared.io import atomic, digest, read, require, sha  # noqa: E402
from emotion_rh_forecasting.analysis import (EMOTIONS, assign_folds, evaluate, make_pairs)  # noqa: E402
from emotion_rh_forecasting.inputs import Sources, load_probes, vector_identity  # noqa: E402

REPLAY = ROOT / 'runs/emotion-rh-replay/glm-5.3-hack-traces-best12-v1'
LABELS = ROOT / 'runs/emotion-rh-sentence-labels/glm53-hack-traces-gpt55-v1/labels.json'
OUT = ROOT / 'runs/emotion-rh-forecasting/glm-5.3-hack-traces-exploratory-v1'


def rows_for_dataset(dataset, labels, spec, bundle, replay_root=REPLAY):
    by_trace = {t['id']: t for t in labels['traces'] if t['dataset'] == dataset}
    jobs = {}
    for entry in spec['jobs']:
        if entry['trace_id'] not in by_trace:
            continue
        job_path = replay_root / 'inputs' / (entry['id'] + '.json')
        require(sha(job_path) == entry['sha256'], 'Replay job changed')
        job = read(job_path)
        score_path = replay_root / 'scores' / (entry['id'] + '.npz')
        receipt = read(score_path.with_suffix('.receipt.json'))
        require(receipt['replay_identity'] == spec['identity'] and receipt['job_sha256'] == entry['sha256']
                and sha(score_path) == receipt['sha256'], 'Replay score shard changed')
        with np.load(score_path, allow_pickle=False) as npz:
            scores = npz['scores']
        require(scores.shape == (job['tokens'], 12) and np.isfinite(scores).all(), 'Invalid replay scores')
        jobs[entry['id']] = job, scores
    rows = []
    for trace in sorted(by_trace.values(), key=lambda t: t['id']):
        source = read(ROOT / trace['source_path'])
        require(sha(ROOT / trace['source_path']) == trace['source_sha256'], 'Labeled trace source changed')
        reasoning_lengths = ({t['turn']: len(t.get('reasoning_text') or '') for t in source['turns']}
                             if dataset == 'sales' else {0: len(source.get('reasoning_text') or '')})
        job_list = [jobs[e['id']] for e in spec['jobs'] if e['trace_id'] == trace['id']]
        selected = {u['unit_id']: (j, scores, u, offset)
                    for offset, (j, scores) in zip(np.cumsum([0] + [j['output_tokens'] for j, _ in job_list[:-1]]), job_list)
                    for u in j['units']}
        prose = [u for u in trace['units'] if u['kind'] == 'sentence_or_fragment']
        require(len(selected) == len(prose), 'Missing replayed prose spans')
        for index, labeled in enumerate(prose):
            job, scores, unit, offset = selected[labeled['unit_id']]
            require(all(unit[k] == labeled[k] for k in ('unit_id', 'char_start', 'char_end', 'text', 'hacking', 'needs_review')),
                    'Scored sentence differs from annotation')
            token_ix = np.asarray(unit['token_indices'], dtype=int)
            weights = np.asarray(unit['token_weights'], dtype=float)
            require(np.all(weights > 0) and np.all(token_ix < len(scores)), 'Invalid pooling weights')
            values = np.average(scores[token_ix], axis=0, weights=weights)
            position = float(np.average(np.asarray(unit['output_positions']) + offset, weights=weights))
            # The source archive stores reasoning/response in separate fields
            # with independent offsets. Give them one monotonic assistant-turn
            # coordinate for the horizon ordering checks.
            field_offset = reasoning_lengths[labeled['turn']] + 1 if labeled['source_field'] == 'response' else 0
            start, end = labeled['char_start'] + field_offset, labeled['char_end'] + field_offset
            rows.append({'trace_id': trace['id'], 'group_id': trace['id'],
                         'task': trace['id'].split('/')[1], 'sentence_index': index,
                         'unit_id': labeled['unit_id'], 'turn_id': job['id'],
                         'char_start': start, 'char_end': end,
                         'text': labeled['text'], 'hacking': labeled['hacking'],
                         'needs_review': labeled['needs_review'],
                         'projections': dict(zip([p['emotion'] for p in bundle['probes']], map(float, values))),
                         'mean_token_position': position, 'completed_feature_tokens': len(token_ix),
                         'last_feature_token_char_end': end})
    return rows


def frozen_folds(rows, dataset):
    by_trace = {}
    for row in rows:
        by_trace.setdefault(row['trace_id'], []).append(row)
    n = 5 if dataset == 'sales' else 3
    groups = [[] for _ in range(n)]
    # Balance the scarce longest-horizon positive traces across folds. This
    # fixed, label-stratified assignment is recorded and reused for all h.
    positives = sorted((t for t in by_trace if any(r['hacking'] for r in by_trace[t][10:])),
                       key=lambda t: (-sum(r['hacking'] for r in by_trace[t][10:]), t))
    negatives = sorted((t for t in by_trace if t not in positives), key=lambda t: (-len(by_trace[t]), t))
    for trace_id in positives + negatives:
        if trace_id in positives:
            fold = min(range(n), key=lambda i: (sum(sum(r['hacking'] for r in by_trace[t][10:]) for t in groups[i]),
                                                sum(len(by_trace[t]) for t in groups[i]), i))
        else:
            fold = min(range(n), key=lambda i: (sum(len(by_trace[t]) for t in groups[i]), i))
        groups[fold].append(trace_id)
    mapping = {trace_id: fold for fold, group in enumerate(groups) for trace_id in group}
    for row in rows:
        row['fold'] = mapping[row['trace_id']]
    return assign_folds(rows, {'method': 'provided'}, 20261021)


def figure(estimates, dataset, destination, balanced=False):
    colors = {'emotion': '#377cb5', 'position': '#687d8c', 'baseline': '#d58c4d'}
    fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharex=True, sharey=True)
    by_key = {(r['model'], r['horizon']): r for r in estimates if r['cohort'] == 'all_available'}
    for ax, emotion in zip(axes.flat, EMOTIONS):
        for model, style, color, name in ((emotion, '-', colors['emotion'], 'Emotion probe'),
                                           ('token_position', ':', colors['position'], 'Token position'),
                                           ('always_not_hacking', '--', colors['baseline'], 'Always non-hacking')):
            points = [by_key[(model, h)] for h in range(11) if (model, h) in by_key]
            if not points:
                continue
            x = [p['horizon'] for p in points]
            y = [p['repeated_sampling_mean_accuracy']*100 for p in points]
            ax.plot(x, y, style, color=color, linewidth=1.8, marker='o' if model == emotion else None,
                    markersize=3, label=name)
            if model == emotion:
                low = [p['sampling_interval95'][0]*100 for p in points]
                high = [p['sampling_interval95'][1]*100 for p in points]
                ax.fill_between(x, low, high, color=color, alpha=.13)
        ax.set_title(emotion.capitalize())
        ax.set_xlim(0, 10)
        ax.set_xticks(range(11))
        ax.set_ylim(20, 85)
        ax.grid(axis='y', alpha=.15)
    for ax in axes[:, 0]:
        ax.set_ylabel('Balanced-evaluation accuracy (%)')
    for ax in axes[-1]:
        ax.set_xlabel('Sentence horizon (+h)')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(.04, .035), ncol=3)
    fig.suptitle(f'GLM 5.3 · {dataset.title()} · emotion probes forecasting reward-hacking sentences',
                 fontsize=17, y=.988)
    subtitle = ('Outcome-stratified sentence-matched sample' if balanced else 'Exploratory positive-outcome sample')
    fig.text(.06, .943, subtitle + ' · native NVFP4 replay · full-trace GPT-5.5 labels', fontsize=10)
    fig.text(.06, .018, 'Bands: balanced-subsample variability, not generalization confidence. h=0 is current sentence.', fontsize=9)
    fig.tight_layout(rect=[0, .10, 1, .91])
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(destination.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', choices=('hack', 'balanced'), default='hack')
    args = parser.parse_args()
    balanced = args.cohort == 'balanced'
    out = (ROOT / 'runs/emotion-rh-forecasting/glm-5.3-balanced-source-best12-v1'
           if balanced else OUT)
    sources = [(LABELS, REPLAY)]
    if balanced:
        sources.append((ROOT / 'runs/emotion-rh-sentence-labels/glm53-nonhack-traces-gpt55-v1/labels.json',
                        ROOT / 'runs/emotion-rh-replay/glm-5.3-nonhack-traces-best12-v1'))
        combined_path = ROOT / 'runs/emotion-rh-sentence-labels/glm53-balanced-source-gpt55-v1/labels.json'
        combined = read(combined_path)
        require(combined['summary']['status'] == 'complete', 'Combined labels incomplete')
    loaded = []
    for labels_path, replay_root in sources:
        labels = read(labels_path)
        spec = read(replay_root / 'replay-spec.json')
        bundle = load_probes(replay_root / 'probes.json', Sources())
        require(spec['labels_sha256'] == sha(labels_path) and spec['probe_identity'] == vector_identity(bundle)
                and spec['probes_sha256'] == sha(replay_root / 'probes.json'), 'Label/probe identity mismatch')
        require(len(list((replay_root / 'scores').glob('*.receipt.json'))) == len(spec['jobs']), 'Replay is incomplete')
        loaded.append((labels, spec, bundle, replay_root, labels_path))
    require(len({item[1]['probe_identity'] for item in loaded}) == 1, 'Probe bundles differ')
    require(len({t['id'] for labels, *_ in loaded for t in labels['traces']}) == sum(len(labels['traces']) for labels, *_ in loaded), 'Trace repeated across cohorts')
    if balanced:
        require({t['id'] for t in combined['traces']} == {t['id'] for labels, *_ in loaded for t in labels['traces']},
                'Combined label population mismatch')
    bundle = loaded[0][2]
    spec = loaded[0][1]
    summary = {'status': 'complete', 'model': bundle['model'], 'probe_identity': spec['probe_identity'],
               'sources': [{'labels_path': str(path), 'labels_sha256': sha(path),
                            'replay_path': str(root), 'replay_identity': run['identity']}
                           for _, run, _, root, path in loaded],
               'replay_interpretation': spec['interpretation'], 'datasets': {}}
    if balanced:
        summary['combined_labels_sha256'] = sha(combined_path)
        summary['source_cohort_counts'] = combined['summary']['counts']
    for dataset in ('sales', 'coding'):
        rows = [row for labels, run, probes, root, _ in loaded
                for row in rows_for_dataset(dataset, labels, run, probes, root)]
        folds = frozen_folds(rows, dataset)
        out.mkdir(parents=True, exist_ok=True)
        estimates, populations, failures = [], [], []
        for horizon in range(11):
            common = {(r['trace_id'], r['unit_id']) for r in make_pairs(rows, 10)}
            for cohort in ('all_available', 'common_sources', 'certain_only'):
                pairs = make_pairs(rows, horizon, common if cohort == 'common_sources' else None,
                                   cohort == 'certain_only')
                try:
                    result, _ = evaluate(pairs, cohort, horizon, 20261021, 2000, set(folds['trace_folds'].values()))
                    estimates.extend(result['estimates'])
                    populations.append(result['population'])
                except (ValueError, AssertionError) as exc:
                    failures.append({'cohort': cohort, 'horizon': horizon, 'reason': str(exc),
                                     'eligible_pairs': len(pairs)})
                    if cohort == 'all_available':
                        raise
        destination = out / dataset
        destination.mkdir(parents=True, exist_ok=True)
        atomic(destination / 'metrics.json', {'estimates': estimates, 'populations': populations,
                                               'unavailable': failures, 'folds': folds})
        with (destination / 'metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(estimates[0]))
            writer.writeheader()
            writer.writerows(estimates)
        figure(estimates, dataset, destination / 'figures/accuracy-trends', balanced)
        summary['datasets'][dataset] = {'traces': len({r['trace_id'] for r in rows}), 'sentences': len(rows),
                                        'rh_sentences': sum(r['hacking'] for r in rows),
                                        'folds': len(set(folds['trace_folds'].values())),
                                        'unavailable_cohorts': failures,
                                        'k3': next(p for p in populations if p['cohort']=='all_available' and p['horizon']==3),
                                        'k10': next(p for p in populations if p['cohort']=='all_available' and p['horizon']==10)}
    atomic(out / 'summary.json', summary)
    cards = []
    for dataset in ('sales', 'coding'):
        item = summary['datasets'][dataset]
        cards.append(f'<section><h2>{dataset.title()}</h2><p>{item["traces"]} selected traces · '
                     f'{item["sentences"]} labeled prose sentences · {item["rh_sentences"]} positive sentences. '
                     f'At k=3: {item["k3"]["balanced_eval_pairs"]} balanced pairs; '
                     f'at k=10: {item["k10"]["balanced_eval_pairs"]} balanced pairs.</p>'
                     f'<img src="{dataset}/figures/accuracy-trends.svg" alt="{dataset} probe forecast">'
                     f'<p><a href="{dataset}/metrics.csv">Metrics CSV</a> · '
                     f'<a href="{dataset}/metrics.json">Full metrics and folds</a></p></section>')
    note = ('Outcome-stratified sample with equal prose sentence counts from positive- and negative-outcome traces in each dataset. '
            'Selection was by whole trace, preserving sentence sequences; the sample is not prevalence representative. '
            if balanced else 'Exploratory case-only analysis of 40 sales traces with ledger violations and 13 coding traces judged hacks. '
            'This is a selected positive-outcome population, not a representative performance estimate. ')
    (out / 'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>GLM 5.3 emotion/RH forecasts</title>'
        '<style>body{font:16px/1.55 system-ui;max-width:1500px;margin:24px auto;padding:0 24px;color:#243440}'
        'section{border-top:1px solid #d6dee2;margin-top:30px;padding-top:20px}img{width:100%}'
        '.note{background:#edf3f7;padding:16px;border-radius:8px}</style>'
        '<h1>GLM 5.3 emotion probes forecasting reward-hacking sentences</h1>'
        '<p class="note">' + note +
        'GPT-5.5 labels were made with full-trace context; positive horizons can reflect ongoing hacking. '
        'Probes come from the pinned GLM 5.3 NVFP4 checkpoint, while traces were generated by OpenRouter; '
        'the replay is not byte-identical to original serving and has no independent shorter-prefix check. '
        'Bands show balanced-subsample variability, not generalization confidence.</p>'
        + ''.join(cards) + '<p><a href="summary.json">Run summary</a></p></html>')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
