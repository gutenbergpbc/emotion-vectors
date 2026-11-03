"""Update k=0..10 accuracy and probe ROC with the 5,000-sentence cohort."""
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


import argparse
from collections import Counter, defaultdict
import csv
import html
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE.parent), str(HERE)]
from shared.io import atomic, read, require, sha  # noqa: E402
from emotion_rh_forecasting.analysis import EMOTIONS, evaluate  # noqa: E402
from plot_goodfire_glm52_forecast import load_rows as load_pilot_rows, plot as plot_accuracy  # noqa: E402
from plot_goodfire_glm52_roc import roc_for_probe, plot as plot_roc  # noqa: E402

LABELS_PILOT = ROOT / 'runs/emotion-rh-sentence-labels/goodfire-glm52-deepswe-gpt61sol-pilot-v1'
LABELS_NEW = ROOT / 'runs/emotion-rh-sentence-labels/goodfire-glm52-deepswe-gpt61sol-5k-v1'
REPLAY_NEW = ROOT / 'runs/emotion-rh-replay/goodfire-glm52-deepswe-nvfp4-gpt61sol-5k-v1'
OUT = ROOT / 'runs/emotion-rh-forecasting/goodfire-glm52-deepswe-gpt61sol-expanded-5k-v1'
LABELS_SECOND = ROOT / 'runs/emotion-rh-sentence-labels/goodfire-glm52-deepswe-gpt61sol-5k-b-v1'
REPLAY_SECOND = ROOT / 'runs/emotion-rh-replay/goodfire-glm52-deepswe-nvfp4-gpt61sol-5k-b-v1'
OUT_SECOND = ROOT / 'runs/emotion-rh-forecasting/goodfire-glm52-deepswe-gpt61sol-expanded-10k-v1'


def load_new_rows(label_root=LABELS_NEW, replay_root=REPLAY_NEW):
    labels_path = label_root / 'labels.json'
    labels = read(labels_path)
    require(labels['summary']['status'] == 'complete' and len(labels['traces']) in (40, 50),
            '5k labels incomplete')
    spec = read(replay_root / 'replay-spec.json')
    require(spec['selection_sha256'] == sha(label_root / 'selection.json') and
            spec['label_request_manifest_sha256'] == sha(label_root / 'judge-manifest.json') and
            spec['total_sentences'] == 5000, '5k replay inputs changed')
    require(read(replay_root / 'progress.json')['status'] == 'complete', '5k replay incomplete')
    by_trace = {t['id']: t for t in labels['traces']}
    emotions = [p['emotion'] for p in read(replay_root / 'probes.json')['probes']]
    rows = []
    for entry in spec['jobs']:
        job_path = replay_root / 'inputs' / f"{entry['id']}.json"
        require(sha(job_path) == entry['sha256'], 'Replay job changed')
        job = read(job_path)
        score_path = replay_root / 'scores' / f"{entry['id']}.npz"
        receipt = read(score_path.with_suffix('.receipt.json'))
        require(receipt['replay_identity'] == spec['identity'] and
                receipt['job_sha256'] == entry['sha256'] and
                receipt['sha256'] == sha(score_path), 'Replay score changed')
        with np.load(score_path, allow_pickle=False) as archive:
            scores = archive['scores']
        require(scores.shape == (job['tokens'], 12) and np.isfinite(scores).all(),
                'Invalid replay score matrix')
        labels_by_id = {u['unit_id']: u for u in by_trace[entry['trace_id']]['units']}
        for replay in job['units']:
            unit = labels_by_id[replay['unit_id']]
            require(all(unit[k] == replay[k] for k in
                        ('unit_id', 'node_index', 'channel', 'char_start', 'char_end', 'text')),
                    'Sentence differs from replay')
            indices = np.asarray(replay['token_indices'], dtype=int)
            weights = np.asarray(replay['token_weights'], dtype=float)
            require(len(indices) and np.all(indices < len(scores)) and np.all(weights > 0),
                    'Invalid causal pooling')
            values = np.average(scores[indices], axis=0, weights=weights)
            rows.append({'trace_id': entry['trace_id'], 'group_id': entry['task'],
                         'task': entry['task'], 'sentence_index': unit['unit_id'],
                         'unit_id': unit['unit_id'],
                         'turn_id': f"{unit['node_index']}-{unit['channel']}",
                         'char_start': unit['char_start'], 'char_end': unit['char_end'],
                         'text': unit['text'], 'hacking': unit['hacking'],
                         'needs_review': unit['needs_review'],
                         'projections': dict(zip(emotions, map(float, values))),
                         'mean_token_position': float(unit['unit_id']),
                         'completed_feature_tokens': len(indices),
                         'last_feature_token_char_end': unit['char_end']})
    require(len(rows) == 5000, '5k replay coverage changed')
    return rows, spec, labels_path


def make_sparse_pairs(rows, horizon):
    require(type(horizon) is int and 0 <= horizon <= 10, 'Invalid horizon')
    by_trace = defaultdict(dict)
    for row in rows:
        key = row['sentence_index']
        require(key not in by_trace[row['trace_id']], 'Duplicate original sentence position')
        by_trace[row['trace_id']][key] = row
    pairs = []
    for trace_id in sorted(by_trace):
        positions = by_trace[trace_id]
        for index, source in sorted(positions.items()):
            target = positions.get(index + horizon)
            if target is None or source['needs_review'] or target['needs_review']:
                continue
            require(source['last_feature_token_char_end'] <= source['char_end'],
                    'Future token in source features')
            if horizon and source['turn_id'] == target['turn_id']:
                require(source['char_end'] <= target['char_start'], 'Overlapping source/target spans')
            pairs.append(dict(source, horizon=horizon, hacking=target['hacking'],
                              source_hacking=source['hacking'], target_unit_id=target['unit_id'],
                              target_sentence_index=target['sentence_index'],
                              target_turn_id=target['turn_id'],
                              source_needs_review=source['needs_review'],
                              target_needs_review=target['needs_review']))
    return pairs


def assign_task_folds(rows):
    by_trace = defaultdict(list)
    by_group = defaultdict(list)
    for row in rows:
        by_trace[row['trace_id']].append(row)
        by_group[row['group_id']].append(row)
    ten = make_sparse_pairs(rows, 10)
    pos10 = Counter(r['group_id'] for r in ten if r['hacking'])
    pos0 = {g: sum(r['hacking'] and not r['needs_review'] for r in rs)
            for g, rs in by_group.items()}
    folds = [[] for _ in range(3)]
    ordered = sorted(by_group, key=lambda g: (-pos10[g], -pos0[g], len(by_group[g]), g))
    for group in ordered:
        fold = min(range(3), key=lambda k: (
            sum(pos10[g] for g in folds[k]),
            sum(pos0[g] for g in folds[k]),
            sum(len(by_group[g]) for g in folds[k]), k))
        folds[fold].append(group)
    group_mapping = {group: k for k, fold_groups in enumerate(folds) for group in fold_groups}
    mapping = {trace: group_mapping[trace_rows[0]['group_id']]
               for trace, trace_rows in by_trace.items()}
    for row in rows:
        row['fold'] = mapping[row['trace_id']]
    require(len(mapping) == len(by_trace) and len(group_mapping) == len(by_group),
            'Fold assignment changed')
    return {'method': 'task-disjoint-greedy-positive-balance', 'trace_folds': mapping,
            'group_folds': group_mapping}


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-second', action='store_true')
    args = parser.parse_args()
    if args.include_second:
        OUT = OUT_SECOND
    pilot, old_spec, old_labels = load_pilot_rows(LABELS_PILOT)
    new, new_spec, new_labels = load_new_rows()
    for row in pilot:
        row['mean_token_position'] = float(row['sentence_index'])
    rows = pilot + new
    label_paths = [old_labels, new_labels]
    specs = [old_spec, new_spec]
    if args.include_second:
        second, second_spec, second_labels = load_new_rows(LABELS_SECOND, REPLAY_SECOND)
        rows += second
        label_paths.append(second_labels)
        specs.append(second_spec)
    require(len(rows) == (22143 if args.include_second else 17143) and
            len({r['trace_id'] for r in rows}) == (102 if args.include_second else 62),
            'Expanded cohort changed')
    fold_info = assign_task_folds(rows)
    expected_folds = set(fold_info['trace_folds'].values())
    estimates, populations, roc_metrics, curves = [], [], [], {}
    for horizon in range(11):
        pairs = make_sparse_pairs(rows, horizon)
        result, _ = evaluate(pairs, 'certain_only', horizon, 20261023, 2000, expected_folds)
        estimates.extend(result['estimates'])
        populations.append(result['population'])
        for probe in EMOTIONS:
            roc = roc_for_probe(pairs, probe)
            roc_metrics.append({'probe': probe, 'horizon': horizon,
                                **{key: roc[key] for key in roc
                                   if key not in ('fpr', 'tpr', 'fold_rules')}})
            if horizon in (0, 5, 10):
                curves[(probe, horizon)] = roc
        print(f"k={horizon}: {result['population']['balanced_eval_pairs']} balanced pairs, "
              f"{sum(p['hacking'] for p in pairs)} RH targets", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {'status': 'complete', 'traces': len({r['trace_id'] for r in rows}),
               'tasks': len({r['group_id'] for r in rows}), 'sentences': len(rows),
               'rh_sentences': sum(r['hacking'] for r in rows),
               'confirmed_rh_sentences': sum(r['hacking'] and not r['needs_review'] for r in rows),
               'needs_review': sum(r['needs_review'] for r in rows),
               'source': 'Goodfire/reward-hack-data/deepswe-glm-5.2-fp8',
               'labels_sha256': [sha(path) for path in label_paths],
               'replay_identity': [spec['identity'] for spec in specs],
               'probe_model': new_spec['model'], 'folds': fold_info,
               'interpretation': f'FP8 DeepSWE archive text replayed through NVFP4 GLM; 12 full-trace pilot replays plus {90 if args.include_second else 50} local-context trace replays. Original FP8 activations are unavailable.',
               'populations': populations, 'label_method': 'source-vote/manual/GPT-6.1-Sol',
               'caveat': 'The added 5,000-sentence batches were selected from source-positive passages, not randomly. RH sentences cluster within traces and tasks. False-positive rates describe this sampled cohort, not deployment prevalence. The legacy token_position baseline column uses original sentence index for all replay cohorts.'}
    atomic(OUT / 'metrics.json', {'summary': summary, 'estimates': estimates})
    with (OUT / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(estimates[0]))
        writer.writeheader(); writer.writerows(estimates)
    plot_accuracy(estimates, summary, OUT, position_label='Sentence position')
    with (OUT / 'probe-roc-metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(roc_metrics[0]))
        writer.writeheader(); writer.writerows(roc_metrics)
    atomic(OUT / 'probe-roc-curves.json', {
        'method': 'OOF empirical train-CDF calibration, train-fit polarity and cutoff; certain-label exact-position pairs only',
        'labels_sha256': summary['labels_sha256'], 'replay_identity': summary['replay_identity'],
        'curves': {f'{probe}:k{k}': value for (probe, k), value in curves.items()}})
    counts = {k: {'positive_sentences': next(r for r in roc_metrics if r['horizon'] == k)['positive_sentences'],
                  'negative_sentences': next(r for r in roc_metrics if r['horizon'] == k)['negative_sentences']}
              for k in (0, 5, 10)}
    plot_roc(curves, counts, out=OUT,
             subtitle=f"Out-of-fold ROC · {summary['traces']} traces / {summary['tasks']} task-disjoint groups / 3 folds · k=0: "
                      f"{counts[0]['positive_sentences']} RH and {counts[0]['negative_sentences']:,} non-RH sentences",
             footnote='Source-positive-passage enriched sample; zero FPR retained on near-log axis. '
                      'Repeated RH sentences and mixed replay windows limit independent evidence.')
    (OUT / 'report.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Expanded GLM 5.2 RH forecast</title>'
        '<style>body{font:16px/1.5 system-ui;max-width:1600px;margin:2rem auto;color:#172d2b}img{max-width:100%}</style>'
        '<h1>Expanded GLM 5.2 DeepSWE sentence forecast</h1>'
        f'<p>{summary["traces"]} traces across {summary["tasks"]} tasks, {len(rows):,} sampled sentences, {summary["confirmed_rh_sentences"]} certain RH sentences, '
        f'{summary["needs_review"]} flagged. {html.escape(summary["interpretation"])}</p>'
        '<p>Task-disjoint, three-fold out-of-fold evaluation. k=0 detects the current sentence; '
        'k&gt;0 predicts exactly k original sentence positions later. Unselected gaps never shift the horizon. '
        'Accuracy uses balanced test folds; ROC uses every certain-label pair. The added 5,000-sentence '
        'batches were selected from source-positive passages, so the rates are descriptive of this sample. '
        'The position baseline uses original sentence index.</p>'
        '<h2>Accuracy across sentence horizons</h2><img src="figures/accuracy-trends.svg" alt="Accuracy trends">'
        '<h2>False-positive and true-positive rates by probe</h2>'
        '<p>Each panel shows k=0, 5, and 10; the star is the k=0 train-fit operating point.</p>'
        '<img src="figures/probe-roc.svg" alt="Per-probe ROC curves">'
        '<p><a href="metrics.csv">Accuracy metrics</a> · '
        '<a href="probe-roc-metrics.csv">ROC metrics for all k</a> · '
        '<a href="probe-roc-curves.json">ROC coordinates</a></p></html>')
    atomic(OUT / 'verification.json', {'passed': True,
           'labels_sha256': summary['labels_sha256'], 'replay_identity': summary['replay_identity'],
           'horizons': list(range(11)), 'metrics_sha256': sha(OUT / 'metrics.json'),
           'accuracy_figure_sha256': sha(OUT / 'figures/accuracy-trends.svg'),
           'roc_figure_sha256': sha(OUT / 'figures/probe-roc.svg')})
    print(json.dumps({k: summary[k] for k in ('traces', 'sentences', 'confirmed_rh_sentences', 'needs_review')}, indent=2))


if __name__ == '__main__':
    main()
