"""Replay frozen GPT-OSS Sales self-report checkpoints with the best 12 probes.

Preparation and Modal replay are separate so a one-trace smoke can validate the
frozen input contract before the complete run. No generation or self-report API
calls are made here.
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
import hashlib
import io
import json
from pathlib import Path
import tarfile

import modal

IS_LOCAL = modal.is_local()
ROOT = Path(__file__).resolve().parents[3] if IS_LOCAL else Path('/root/model-forensics')
HERE = Path(__file__).resolve().parent if IS_LOCAL else ROOT / 'src/experiments/emotion-self-report'
SOURCE = ROOT / 'experiments/gpt_oss_sales_sentence_rh_12probes_20261016'
OUT = ROOT / 'runs/emotion-self-report-activations/gptoss-complete-traces-v1'
RUN_ID = 'gptoss-sales-self-report-activations-v1'
INPUTS = OUT / 'gptoss-replay-inputs.json'
SPEC_PATH = OUT / 'gptoss-sales-replay-spec.json'
REMOTE = Path('/root/model-forensics')
REMOTE_HERE = REMOTE / 'src/experiments/emotion-self-report'
REMOTE_SOURCE = REMOTE / 'experiments/gpt_oss_sales_sentence_rh_12probes_20261016'

app = modal.App('gptoss-sales-self-report-activations')
cache = modal.Volume.from_name('gpt-oss120-probe-model-cache-v1', create_if_missing=True)
data = modal.Volume.from_name('gpt-oss120-probe-runs-v1', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install_from_requirements(str(ROOT / 'src/experiments/emotion-probes/adapters/gptoss120b/requirements-extract.txt'))
    .env({'HF_HOME': '/models', 'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'})
    .add_local_dir(str(ROOT / 'src/environments/steering'), remote_path=str(REMOTE / 'steering'),
                   ignore=['runs/**', 'tests/**', '__pycache__/**'])
    .add_local_dir(str(SOURCE), remote_path=str(REMOTE_SOURCE),
                   ignore=['results/**', 'replay/**', '__pycache__/**'])
    .add_local_dir(str(HERE), remote_path=str(REMOTE_HERE),
                   ignore=['__pycache__/**'])
    .add_local_file(str(INPUTS), remote_path=str(REMOTE_HERE / 'gptoss-replay-inputs.json')))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> dict:
    selected_path = OUT / 'selected-traces.json'
    selected = json.loads(selected_path.read_text())
    cases = []
    for trace in selected:
        case = trace['case']
        for point in trace['checkpoints']:
            mi = int(point['message_index'])
            content = case['messages'][mi]['content']
            end = int(point['char_end'])
            start = int(point['char_start'])
            if content[start:end] != point['text']:
                raise ValueError(f"Checkpoint text mismatch: {trace['trace_id']}:{point['unit_id']}")
            if case['messages'][mi]['role'] != 'assistant' or end <= 0:
                raise ValueError(f"Expected a positive assistant checkpoint: {trace['trace_id']}:{point['unit_id']}")
            cases.append({
                'id': f"{trace['trace_id']}::unit::{point['unit_id']}",
                'trace_id': trace['trace_id'], 'unit_id': point['unit_id'],
                'message_index': mi, 'char_start': start, 'char_end': end,
                'text': point['text'], 'messages': case['messages'][:mi],
                'completion': content[:end],
            })
    payload = {'schema_version': 1, 'cases': cases}
    if INPUTS.exists() and json.loads(INPUTS.read_text()) != payload:
        raise ValueError(f'Prepared replay inputs changed: {INPUTS}')
    INPUTS.parent.mkdir(parents=True, exist_ok=True)
    INPUTS.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n')
    original = json.loads((SOURCE / 'run-spec.json').read_text())
    spec = {
        'run_id': RUN_ID, 'source_spec': original,
        'input_path': str(REMOTE_HERE / 'gptoss-replay-inputs.json'),
        'input_sha256': sha(INPUTS), 'source_spec_sha256': sha(SOURCE / 'run-spec.json'),
        'probes_sha256': sha(SOURCE / 'probes.json'),
        'probe_path': str(REMOTE_SOURCE / 'probes.json'),
        'probe_count': 12, 'trace_count': len(selected), 'checkpoint_count': len(cases),
    }
    if SPEC_PATH.exists() and json.loads(SPEC_PATH.read_text()) != spec:
        raise ValueError(f'Frozen replay spec changed: {SPEC_PATH}')
    SPEC_PATH.write_text(json.dumps(spec, indent=2) + '\n')
    return spec


@app.function(image=image, gpu='H200', cpu=16, memory=131072, timeout=10800,
              volumes={'/models': cache, '/data': data})
def execute(spec, limit=None):
    import sys
    import time
    import numpy as np
    sys.path.insert(0, str(REMOTE))
    from steering.gpt_oss_backend import make_backend
    from steering.backend import load_vector
    from steering.tracking import atomic
    for path, expected in spec['source_spec']['source_hashes'].items():
        assert hashlib.sha256((REMOTE / path).read_bytes()).hexdigest() == expected, path
    assert hashlib.sha256((REMOTE_SOURCE / 'run-spec.json').read_bytes()).hexdigest() == spec['source_spec_sha256']
    assert hashlib.sha256((REMOTE_SOURCE / 'probes.json').read_bytes()).hexdigest() == spec['probes_sha256']
    input_path = Path(spec['input_path'])
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == spec['input_sha256']
    all_cases = json.loads(input_path.read_text())['cases']
    cases = all_cases if limit is None else all_cases[:limit]
    root = Path('/data') / spec['run_id'] / ('smoke' if limit is not None else 'complete')
    root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    backend = make_backend(spec['source_spec'], REMOTE)
    probes = spec['source_spec']['probes']
    vectors = {p['emotion']: load_vector(p, REMOTE) for p in probes}
    emotion_order = [p['emotion'] for p in probes]
    for i, c in enumerate(cases):
        score_id = hashlib.sha256(c['id'].encode()).hexdigest()
        path = root / 'scores' / (score_id + '.npz')
        if path.exists():
            continue
        scored = backend.replay_chat_scores(c['messages'], c['completion'], probes, vectors)
        ix = np.flatnonzero(scored['offsets'][:, 1] > scored['response_char_start'])
        if not len(ix):
            raise ValueError(f"No assistant output tokens for {c['id']}")
        scores = scored['scores'][ix]
        cosines = scored['cosines'][ix]
        offsets = scored['offsets'][ix].copy()
        offsets -= int(scored['response_char_start'])
        # The replay input ends at this checkpoint, so every retained token
        # must end at or before the checkpoint's character endpoint.
        keep = (offsets[:, 1] > 0) & (offsets[:, 1] <= int(c['char_end']))
        scores, cosines = scores[keep], cosines[keep]
        offsets, token_ids = offsets[keep], scored['token_ids'][ix][keep]
        if not len(scores) or offsets[-1, 1] > int(c['char_end']):
            raise ValueError(f"No endpoint token at/before checkpoint: {c['id']}")
        if not np.isfinite(scores).all() or scores.shape[1] != 12:
            raise ValueError(f"Invalid probe scores: {c['id']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix('.tmp').open('wb') as f:
            np.savez_compressed(f, scores=scores, cosines=cosines, offsets=offsets,
                                token_ids=token_ids, emotion_order=np.asarray(emotion_order),
                                checkpoint_id=np.asarray(c['id']),
                                message_index=np.asarray(c['message_index']),
                                sentence_char_end=np.asarray(c['char_end']),
                                probe_char_end=np.asarray(offsets[-1, 1]))
        path.with_suffix('.tmp').replace(path)
        if i % 10 == 0:
            data.commit()
        print(f"Replayed {i + 1}/{len(cases)}: {c['id']}", flush=True)
    atomic(root / 'receipt.json', {'status': 'complete' if limit is None else 'smoke',
           'cases': len(cases), 'elapsed_seconds': time.monotonic() - start,
           'gpu': 'H200', 'replay': 'Native causal decoded-text teacher forcing; complete visible message history before target and assistant prefix truncated exactly at checkpoint; no generation.'})
    data.commit()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as tf:
        for path in sorted(root.rglob('*')):
            if path.is_file(): tf.add(path, arcname=str(path.relative_to(root)))
    return buffer.getvalue()


def download(content: bytes, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(content), mode='r:gz') as archive:
        for entry in archive:
            path = directory / entry.name
            if not entry.isfile() or not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError('Unexpected archive entry')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(entry).read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='Replay every checkpoint in first complete trace')
    parser.add_argument('--prepare-only', action='store_true', help='Freeze replay input and spec for Modal upload')
    args = parser.parse_args()
    spec = prepare()
    if args.prepare_only:
        print(f"Prepared {spec['checkpoint_count']} checkpoints: {INPUTS}")
        return
    smoke_count = len(json.loads(INPUTS.read_text())['cases'])
    if args.smoke:
        selected = json.loads((OUT / 'selected-traces.json').read_text())
        smoke_count = len(selected[0]['checkpoints'])
    with modal.enable_output():
        with app.run():
            content = execute.remote(spec, limit=smoke_count if args.smoke else None)
    download(content, OUT / ('gptoss-smoke' if args.smoke else 'gptoss-complete-replay'))
    print(f"Downloaded {'smoke' if args.smoke else 'complete'} replay ({smoke_count if args.smoke else spec['checkpoint_count']} checkpoints)")


if __name__ == '__main__':
    main()
