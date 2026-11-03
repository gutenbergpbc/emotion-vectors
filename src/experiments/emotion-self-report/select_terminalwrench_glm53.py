"""Freeze every fully rated GLM 5.3 TerminalWrench trace for probe replay."""
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


from collections import Counter, defaultdict
import math
from pathlib import Path

from select_activation_traces import EMOTIONS, freeze, read, sha

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'runs/emotion-self-report/glm-5.3-terminalwrench-neutral-self-report-v1'
OUT = ROOT / 'runs/emotion-self-report-activations/glm53-terminalwrench-neutral-complete-traces-v1'


def main():
    config = read(SOURCE / 'config.resolved.json')
    points = read(SOURCE / 'checkpoints.json')
    cases = read(SOURCE / 'cases.json')
    ratings = {}
    response_files = sorted((SOURCE / 'responses').glob('*.json'))
    for path in response_files:
        record = read(path)
        point = record['checkpoint']
        key = (point['trace_id'], point['unit_id'], record['emotion'])
        if key in ratings:
            raise ValueError(f'Duplicate response: {key}')
        ratings[key] = (record, path)
    grouped = defaultdict(list)
    for point in points:
        body = cases[point['trace_id']]['messages'][point['message_index']]['content']
        if body[point['char_start']:point['char_end']] != point['text']:
            raise ValueError(f"Checkpoint text mismatch: {point['trace_id']}:{point['unit_id']}")
        grouped[point['trace_id']].append(point)
    selected = []
    used_responses = []
    for trace_id, units in grouped.items():
        rows = []
        paths = []
        for point in units:
            for emotion in EMOTIONS:
                entry = ratings.get((trace_id, point['unit_id'], emotion))
                if entry is None:
                    continue
                record, path = entry
                value = record.get('value')
                if (not record.get('ok') or record.get('rating_status') != 'numeric'
                        or isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or not 0 <= value <= 10
                        or record['checkpoint'] != point):
                    continue
                rows.append({**point, 'emotion': emotion, 'rating': value,
                             'rating_status': 'numeric', 'provider': record.get('provider'),
                             'served_model': record.get('served_model'), 'dataset': 'terminalwrench'})
                paths.append(path)
        if len(rows) == len(units) * len(EMOTIONS):
            selected.append({'dataset': 'terminalwrench', 'trace_id': trace_id,
                             'case': cases[trace_id], 'checkpoints': units, 'ratings': rows,
                             'source_run': SOURCE.name})
            used_responses.extend(paths)
    if len(selected) != 7 or sum(len(t['checkpoints']) for t in selected) != 216:
        raise ValueError('Completed-trace set changed; inspect before replay')
    source_hashes = {str(path): sha(path) for path in
                     [SOURCE / 'config.resolved.json', SOURCE / 'checkpoints.json', SOURCE / 'cases.json', *used_responses]}
    selection = {'schema_version': 1, 'model': config['model'], 'emotions': EMOTIONS,
                 'eligible_traces': len(selected), 'eligible_checkpoints': 216,
                 'selected_traces': len(selected), 'selected_checkpoints': 216,
                 'dataset_traces': {'terminalwrench': len(selected)},
                 'dataset_checkpoints': {'terminalwrench': 216},
                 'selection_policy': 'All fully rated traces from the interrupted GLM 5.3 TerminalWrench run, in source order. Partial traces excluded.',
                 'completeness': 'Each selected checkpoint is a sentence with numeric ratings for all twelve emotions; remaining source traces are incomplete and excluded.',
                 'source_sha256': source_hashes,
                 'traces': [{'dataset': 'terminalwrench', 'trace_id': t['trace_id'],
                             'source_run': SOURCE.name, 'checkpoints': len(t['checkpoints'])} for t in selected]}
    for name, value in [('selection.json', selection), ('selected-traces.json', selected),
                        ('cases.json', {t['trace_id']: t['case'] for t in selected}),
                        ('checkpoints.json', [p for t in selected for p in t['checkpoints']]),
                        ('ratings.json', [r for t in selected for r in t['ratings']])]:
        freeze(OUT / name, value)
    print(f'{len(selected)} traces, 216 sentences, {sum(len(t["ratings"]) for t in selected)} ratings: {OUT}')


if __name__ == '__main__':
    main()
