"""Materialize verified GLM 5.3 activation projections and provenance."""
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
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_ROOT = ROOT / 'runs/emotion-self-report-activations/glm53-complete-traces-v2'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def materialize(root: Path):
    root = Path(root)
    spec = read(root / 'replay-spec.json')
    bundle = read(root / 'probes.json')
    traces = read(root / 'selected-traces.json')
    emotions = [probe['emotion'] for probe in bundle['probes']]
    if len(emotions) != 12 or len(set(emotions)) != 12:
        raise ValueError('Probe bundle must have twelve uniquely ordered emotions')
    if spec['probes_sha256'] != sha(root / 'probes.json'):
        raise ValueError('Probe bundle checksum mismatch')
    jobs = {entry['id']: entry for entry in spec['jobs']}
    if len(jobs) != len(spec['jobs']):
        raise ValueError('Duplicate replay job ID')
    trace_index = {trace['trace_id']: trace for trace in traces}
    rows = []
    seen = set()
    projected_tokens = 0
    for entry in spec['jobs']:
        job_path = root / 'inputs' / f"{entry['id']}.json"
        score_path = root / 'scores' / f"{entry['id']}.npz"
        receipt_path = score_path.with_suffix('.receipt.json')
        job = read(job_path)
        receipt = read(receipt_path)
        score_sha = sha(score_path)
        if sha(job_path) != entry['sha256'] or receipt.get('job_sha256') != entry['sha256']:
            raise ValueError(f"Job/input receipt mismatch: {entry['id']}")
        if receipt.get('replay_identity') != spec['identity'] or receipt.get('sha256') != score_sha:
            raise ValueError(f"Score receipt mismatch: {entry['id']}")
        if job['trace_id'] != entry['trace_id'] or job['tokens'] != entry['tokens']:
            raise ValueError(f"Job identity mismatch: {entry['id']}")
        if job['trace_id'] not in trace_index:
            raise ValueError(f"Unknown selected trace: {job['trace_id']}")
        with np.load(score_path, allow_pickle=False) as archive:
            if set(archive.files) != {'projections'}:
                raise ValueError(f"Unexpected score arrays: {entry['id']}")
            projections = np.asarray(archive['projections'], dtype=np.float32)
        if projections.shape != (job['tokens'], 12) or not np.isfinite(projections).all():
            raise ValueError(f"Invalid projection matrix: {entry['id']}")
        projected_tokens += int(projections.shape[0])
        expected_layers = sorted({int(p['layer']) for p in bundle['probes']})
        if receipt.get('selected_layers') != expected_layers:
            raise ValueError(f"Selected hook layers differ from probe bundle: {entry['id']}")
        readouts = {int(r['unit_id']): r for r in job['readouts']}
        if len(readouts) != len(job['readouts']):
            raise ValueError(f"Duplicate unit in job: {entry['id']}")
        trace = trace_index[job['trace_id']]
        points = {int(p['unit_id']): p for p in trace['checkpoints']
                  if int(p['message_index']) == int(entry['message_index'])}
        if set(readouts) != set(points):
            raise ValueError(f"Checkpoint/readout coverage differs: {entry['id']}")
        for unit_id, readout in readouts.items():
            point = points[unit_id]
            if readout['message_index'] != point['message_index'] or readout['sentence_char_end'] != point['char_end']:
                raise ValueError(f"Checkpoint endpoint mismatch: {job['trace_id']} unit {unit_id}")
            if readout['probe_char_end'] > readout['sentence_char_end']:
                raise ValueError(f"Probe endpoint includes future characters: {job['trace_id']} unit {unit_id}")
            token = int(readout['token_index'])
            if not 0 <= token < len(projections):
                raise ValueError(f"Readout token index out of range: {job['trace_id']} unit {unit_id}")
            for column, probe in enumerate(bundle['probes']):
                key = (job['trace_id'], unit_id, probe['emotion'])
                if key in seen:
                    raise ValueError(f'Duplicate checkpoint/emotion row: {key}')
                seen.add(key)
                value = float(projections[token, column])
                if not math.isfinite(value):
                    raise ValueError(f'Nonfinite projection: {key}')
                rows.append({'trace_id': job['trace_id'], 'unit_id': unit_id,
                    'emotion': probe['emotion'], 'raw_projection': value,
                    'message_index': int(point['message_index']),
                    'sentence_char_end': int(point['char_end']),
                    'probe_char_end': int(readout['probe_char_end']),
                    'dataset': trace['dataset']})
    expected_checkpoints = sum(len(trace['checkpoints']) for trace in traces)
    expected_rows = expected_checkpoints * 12
    if len(rows) != expected_rows or len(seen) != expected_rows:
        raise ValueError(f'Incomplete checkpoint coverage: {len(rows)} rows; expected {expected_rows}')
    for trace in traces:
        checkpoints = {int(p['unit_id']) for p in trace['checkpoints']}
        if any((trace['trace_id'], unit, emotion) not in seen
               for unit in checkpoints for emotion in emotions):
            raise ValueError(f"Incomplete 12-emotion scores for {trace['trace_id']}")
    provenance = {'schema_version': 1, 'model': spec['model'],
        'extraction_model': spec['extraction_model'],
        'provider_model': spec['provider_model'], 'replay_model': spec['replay_model'],
        'provider_quantization': spec['provider_quantization'], 'provider_routes': spec['provider_routes'],
        'provider_route_note': spec['provider_route_note'], 'extraction_quantization': spec['quantization'],
        'extraction_weights_revision': spec['extraction_model']['revision'],
        'extraction_tokenizer_model': spec['extraction_model']['tokenizer_model'],
        'extraction_tokenizer_revision': spec['extraction_model']['tokenizer_revision'],
        'tokenizer_template_sha256': spec['tokenizer_template_sha256'],
        'renderer': spec['renderer'], 'reasoning_effort': spec.get('reasoning_effort', 'max'), 'clear_thinking': False,
        'reasoning_content_handling': 'Absent reasoning streams are rendered as empty reasoning_content; only archived visible conversation content is replayed, and no hidden provider reasoning is reconstructed.',
        'readout': spec['readout'],
        'readout_method': 'post-block residual dot probe vector, raw projection, no intercept',
        'probe_order': emotions,
        'probes': [{'emotion': p['emotion'], 'probe': p['probe'], 'layer': p['layer'],
                    'intercept': p['intercept'], 'centering': p['centering']} for p in bundle['probes']],
        'probe_bundle_sha256': spec['probes_sha256'], 'probe_identity': spec['probe_identity'],
        'selection_identity': spec['source_selection_sha256'],
        'selection_source_sha256': read(root / 'selection.json')['source_sha256'],
        'replay_identity': spec['identity'], 'selected_traces': len(traces),
        'checkpoints': expected_checkpoints, 'score_rows': len(rows),
        'projection_tokens_retained': projected_tokens,
        'score_files': len(spec['jobs']),
        'input_selection': 'Frozen fully rated visible self-report traces; no new self-report requests.'}
    atomic_json(root / 'probe-scores.json', rows)
    atomic_json(root / 'provenance.json', provenance)
    return {'traces': len(traces), 'checkpoints': expected_checkpoints,
            'score_rows': len(rows), 'projection_tokens_retained': projected_tokens}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(materialize(args.root), indent=2))


if __name__ == '__main__':
    main()
