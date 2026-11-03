"""Score frozen self-report checkpoints with the pinned Gemma 4B causal replay."""
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
import io
import json
import os
from pathlib import Path
import sys

import modal

HERE = Path(__file__).resolve().parent
PROJECT = (HERE.parents[2] if modal.is_local() else Path('/study'))
ROOT = PROJECT
CANONICAL = ROOT / 'src/experiments'
sys.path.insert(0, str(CANONICAL))
OUT = ROOT / 'runs/emotion-self-report-activations/gemma4b-complete-traces-v1/gemma4b-replay-v1'
MODEL_CACHE = modal.Volume.from_name('canonical-gemma4b-model-cache-v1', create_if_missing=True)
IMAGE = (modal.Image.debian_slim(python_version='3.11')
         .pip_install_from_requirements(str(ROOT / 'src/experiments/emotion-probes/adapters/gemma27b/requirements-extract.txt'))
         .env({'HF_HOME': '/models', 'PYTHONUNBUFFERED': '1'})
         .add_local_dir(CANONICAL / 'shared', '/root/shared')
         .add_local_dir(CANONICAL / 'emotion-rh-forecasting/emotion_rh_forecasting', '/root/emotion_rh_forecasting'))
app = modal.App('emotion-self-report-gemma4b-activations')


@app.function(image=IMAGE, gpu='H100', cpu=12, memory=131072, timeout=21600,
              volumes={'/models': MODEL_CACHE},
              secrets=[modal.Secret.from_dict({'HF_TOKEN': os.environ['HF_TOKEN']})
                       if os.environ.get('HF_TOKEN') else modal.Secret.from_name('huggingface')])
def replay(spec, bundle, jobs, trace_limit=0):
    import numpy as np
    from pathlib import Path
    import sys
    sys.path.insert(0, '/root')
    sys.path.insert(0, '/root/emotion_rh_forecasting')
    from terminalwrench_replay import GemmaReplay
    engine = GemmaReplay(bundle, {**spec, 'chunk_tokens': 1024, 'max_tokens': 131072})
    yield {'type': 'ready', 'runtime': engine.runtime}
    done = 0
    for job in jobs:
        if trace_limit and done >= trace_limit:
            break
        arrays = {'token_ids': np.asarray(job['token_ids'], dtype=np.int64),
                  'offsets': np.asarray(job['offsets'], dtype=np.int64),
                  'kind': np.asarray(job['kind'], dtype=np.int8)}
        path = Path('/tmp') / (job['id'] + '.npz')
        blob, receipt = engine.score(job, arrays, path.parent, smoke=(done == 0))
        yield {'type': 'complete', 'id': job['id'], 'npz': blob, 'receipt': receipt}
        done += 1
    yield {'type': 'finished', 'completed': done}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-limit', type=int, default=0,
                        help='Process all assistant-message replay jobs for the first N frozen traces')
    args = parser.parse_args()
    from shared.io import read, atomic, require, sha
    sys.path.insert(0, str(CANONICAL))
    spec = read(OUT / 'replay-spec.json')
    bundle = read(OUT / 'probes.json')
    require(sha(OUT / 'probes.json') == spec['probes_sha256'], 'Probe bundle checksum mismatch')
    jobs = [read(OUT / 'inputs' / f"{entry['id']}.json") for entry in spec['jobs']]
    trace_ids = []
    for job in jobs:
        if job['trace_id'] not in trace_ids:
            trace_ids.append(job['trace_id'])
    selected_trace_ids = set(trace_ids[:args.trace_limit]) if args.trace_limit else set(trace_ids)
    selected_jobs = [j for j in jobs if j['trace_id'] in selected_trace_ids]
    (OUT / 'scores').mkdir(parents=True, exist_ok=True)
    pending = []
    cached_receipts = []
    for job in selected_jobs:
        entry = next(e for e in spec['jobs'] if e['id'] == job['id'])
        require(sha(OUT / 'inputs' / f"{job['id']}.json") == entry['sha256'], 'Frozen input checksum mismatch')
        path = OUT / 'scores' / (job['id'] + '.npz')
        receipt_path = path.with_suffix('.receipt.json')
        if path.exists() and receipt_path.exists():
            receipt = read(receipt_path)
            require(receipt['replay_identity'] == spec['identity'] and
                    receipt['input_sha256'] == job['input_sha256'] and sha(path) == receipt['sha256'],
                    'Cached projection shard identity/checksum mismatch')
            cached_receipts.append(receipt)
        else:
            pending.append(job)
    runtime = None
    runtime_path = OUT / 'runtime.json'
    if runtime_path.exists():
        runtime = read(runtime_path)
    elif cached_receipts:
        runtime = cached_receipts[0]['runtime']
        atomic(runtime_path, runtime)
    if not pending:
        print(f'All {len(selected_jobs)} requested jobs already have verified projection shards; reusing them.', flush=True)
        return
    print(f'Replaying {len(pending)} jobs; reusing {len(cached_receipts)} verified shards.', flush=True)
    for item in replay.remote_gen(spec, bundle, pending):
        if item['type'] == 'ready':
            runtime = item['runtime']
            atomic(OUT / 'runtime.json', runtime)
            print(f"Model ready on {runtime['gpu']}; {len(selected_jobs)} assistant-message jobs", flush=True)
        elif item['type'] == 'complete':
            path = OUT / 'scores' / (item['id'] + '.npz')
            path.write_bytes(item['npz'])
            require(sha(path) == item['receipt']['sha256'], 'Downloaded projection bytes changed')
            atomic(path.with_suffix('.receipt.json'), item['receipt'])
            print(f"Saved {item['id']} · {item['receipt']['assistant_replay_tokens']} assistant tokens · {item['receipt']['seconds']:.1f}s", flush=True)
        elif item['type'] == 'finished':
            print(f"Modal completed {item['completed']} jobs", flush=True)
    atomic(OUT / 'runtime.json', runtime or {})


@app.local_entrypoint()
def run(trace_limit: int = 0):
    main_args = ['--trace-limit', str(trace_limit)] if trace_limit else []
    sys.argv = [sys.argv[0], *main_args]
    main()
