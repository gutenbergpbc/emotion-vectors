"""Freeze whole, fully rated traces for activation replay; never issue API calls.

Run with Python 3: select_activation_traces.py --model gemma4b --target 2000.
The output is an immutable input contract for model-specific replay workers.
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


import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'runs/emotion-self-report'
EMOTIONS = ['hostile', 'frustrated', 'excited', 'appreciative', 'sad', 'afraid',
            'surprised', 'anxious', 'guilty', 'ashamed', 'desperate', 'calm']
# Dataset, checkpoint run, rating run, rating file (None means response receipts).
SOURCES = {
    'gptoss': [('sales', 'gpt-oss-120b-sales-visible-code-endpoints-20261022-v1',
                'gpt-oss-120b-sales-visible-code-endpoints-20261022-v1', 'ratings.json')],
    'gemma4b': [('sales', 'gemma3-4b-sales-self-report-20261022-v2',
                 'gemma3-4b-sales-self-report-20261022-v2', None),
                ('coding', 'gemma3-4b-coding-visible-code-endpoints-20261022-v3',
                 'gemma3-4b-coding-visible-code-endpoints-20261022-v3', 'ratings.json')],
    'gemma27b': [('sales', 'gemma3-27b-sales-self-report-20261022-v3',
                  'gemma3-27b-sales-self-report-20261022-v3', 'ratings.json'),
                 ('coding', 'gemma3-27b-coding-verbose-self-report-20261022-v4',
                  'gemma3-27b-coding-verbose-self-report-20261022-v4', 'ratings.json')],
    'glm53': [('sales', 'glm53-sales-visible-budget10-20261022-v1',
               'glm53-sales-visible-phase2-budget5-20261022-v1', 'combined-ratings.jsonl'),
              ('coding', 'glm53-coding-visible-budget10-20261022-v1',
               'glm53-coding-visible-completion-budget5-20261022-v2', 'combined-ratings.json')],
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Frozen inputs changed; use a new output directory: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def load_source(dataset, point_run, rating_run, rating_file):
    directory = BASE / point_run
    config_path = directory / 'config.resolved.json'
    if not config_path.exists():
        config_path = directory / 'manifest.json'
    config = read(config_path)
    config = config.get('config', config)
    points_path = directory / 'checkpoints.json'
    points = read(points_path)
    if isinstance(points, dict):
        points = points['checkpoints']
    cases_path = directory / 'cases.json'
    if not cases_path.exists():
        cases_path = directory / 'inputs/traces.json'
    if not cases_path.exists():
        cases_path = Path(config['traces'])
    cases = read(cases_path)
    if isinstance(cases, list):
        cases = {case['id']: case for case in cases}
    for case in cases.values():
        if 'messages' not in case:
            case['messages'] = [{'role': 'user', 'content': case['prompt']},
                                {'role': 'assistant', 'content': case['response']}]
    files = [config_path, points_path, cases_path]
    if rating_file:
        path = BASE / rating_run / rating_file
        files.append(path)
        rows = ([json.loads(line) for line in path.read_text().splitlines() if line]
                if path.suffix == '.jsonl' else read(path))
    else:
        files.extend(sorted((BASE / rating_run / 'responses').glob('*.json')))
        rows = []
        for path in files[3:]:
            record = read(path)
            rows.append({**record['checkpoint'], 'emotion': record['emotion'],
                         'rating': record.get('value'), 'rating_status': record['rating_status'],
                         'provider': record.get('provider'), 'served_model': record.get('served_model')})
    ratings = {}
    for row in rows:
        row = {**row, 'rating': row.get('rating', row.get('value'))}
        key = (row['trace_id'], row['unit_id'], row['emotion'])
        if key in ratings:
            raise ValueError(f'Duplicate rating: {key}')
        ratings[key] = row
    grouped = defaultdict(list)
    seen = set()
    for point in points:
        key = (point['trace_id'], point['unit_id'])
        if key in seen:
            raise ValueError(f'Duplicate checkpoint: {key}')
        seen.add(key)
        body = cases[point['trace_id']]['messages'][point['message_index']]['content']
        if body[point['char_start']:point['char_end']] != point['text']:
            raise ValueError(f'Checkpoint text differs: {key}')
        grouped[point['trace_id']].append(point)
    eligible = []
    for tid, units in grouped.items():
        selected_ratings = []
        for point in units:
            for emotion in EMOTIONS:
                row = ratings.get((tid, point['unit_id'], emotion), {})
                value = row.get('rating')
                if (row.get('rating_status', row.get('status')) != 'numeric'
                        or isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or not 0 <= value <= 10):
                    break
                selected_ratings.append({**point, **row, 'rating_status': 'numeric',
                                         'provider': row.get('provider'), 'dataset': dataset})
        if len(selected_ratings) == len(units) * len(EMOTIONS):
            eligible.append({'dataset': dataset, 'trace_id': tid, 'case': cases[tid],
                             'checkpoints': sorted(units, key=lambda p: (p['message_index'], p['char_end'])),
                             'ratings': selected_ratings, 'source_run': point_run})
    return eligible, {str(path): sha(path) for path in files}, config['model']


def choose(traces, target):
    """Shortest first, reserving one whole trace per available dataset."""
    if target <= 0:
        raise ValueError('Target must be positive')
    ordered = sorted(traces, key=lambda t: (len(t['checkpoints']), t['dataset'], t['trace_id']))
    selected, represented = [], set()
    for trace in ordered:
        if trace['dataset'] not in represented:
            selected.append(trace)
            represented.add(trace['dataset'])
    selected_keys = {(t['dataset'], t['trace_id']) for t in selected}
    count = sum(len(t['checkpoints']) for t in selected)
    for trace in ordered:
        if count >= target:
            break
        if (trace['dataset'], trace['trace_id']) not in selected_keys:
            selected.append(trace)
            count += len(trace['checkpoints'])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=SOURCES, required=True)
    parser.add_argument('--target', type=int, default=2000)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    traces, sources, models = [], {}, set()
    for source in SOURCES[args.model]:
        eligible, hashes, model = load_source(*source)
        traces.extend(eligible)
        sources.update(hashes)
        models.add(model)
    if len(models) != 1:
        raise ValueError('Mixed source models')
    selected = choose(traces, args.target)
    if len({t['trace_id'] for t in selected}) != len(selected):
        raise ValueError('Trace IDs collide between datasets')
    version = 'v2' if args.model == 'glm53' else 'v1'
    out = args.output or ROOT / 'runs/emotion-self-report-activations' / (args.model + '-complete-traces-' + version)
    summary = {'schema_version': 1, 'model': models.pop(), 'emotions': EMOTIONS,
               'target_checkpoints': args.target, 'eligible_traces': len(traces),
               'eligible_checkpoints': sum(len(t['checkpoints']) for t in traces),
               'selected_traces': len(selected),
               'selected_checkpoints': sum(len(t['checkpoints']) for t in selected),
               'dataset_traces': dict(Counter(t['dataset'] for t in selected)),
               'dataset_checkpoints': dict(Counter(p['dataset'] for t in selected for p in
                    [{'dataset': t['dataset']}] * len(t['checkpoints']))),
               'selection_policy': 'Reserve shortest complete trace per available dataset; then shortest remaining traces first until target reached, finishing final trace. Ties use dataset and trace ID. This favors shorter traces; it is not a representative sample.',
               'completeness': 'All frozen self-report checkpoints have numeric ratings for all twelve emotions. Checkpoints may be prose, command or code-block endpoints; not all are sentences.',
               'source_sha256': sources,
               'traces': [{k: t[k] for k in ('dataset', 'trace_id', 'source_run')} |
                          {'checkpoints': len(t['checkpoints'])} for t in selected]}
    for name, value in [('selection.json', summary), ('selected-traces.json', selected),
                        ('cases.json', {t['trace_id']: t['case'] for t in selected}),
                        ('checkpoints.json', [p for t in selected for p in t['checkpoints']]),
                        ('ratings.json', [r for t in selected for r in t['ratings']])]:
        freeze(out / name, value)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('source_sha256', 'traces')}, indent=2))
    print(out)


if __name__ == '__main__':
    main()
