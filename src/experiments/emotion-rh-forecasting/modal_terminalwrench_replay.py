"""Run frozen Gemma TerminalWrench sentence replay on the existing H200 cache."""

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

import io
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared.io import atomic, read, require, sha

PROJECT = HERE.parents[2] if modal.is_local() else Path('/study')
CANONICAL = PROJECT/'src/experiments'
app = modal.App('canonical-gemma-terminalwrench-rh-forecasting')
cache = modal.Volume.from_name('gemma27-probe-model-cache-v1', create_if_missing=True)
data = modal.Volume.from_name('canonical-emotion-rh-replay-v1', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install_from_requirements(str(PROJECT/'src/experiments/emotion-probes/adapters/gemma27b/requirements-extract.txt'))
         .env({'HF_HOME': '/models', 'PYTHONUNBUFFERED': '1',
               'PYTHONPYCACHEPREFIX': '/tmp/gemma-replay-pycache-clean-v1'})
         .add_local_dir(CANONICAL/'emotion-rh-forecasting/emotion_rh_forecasting', '/root/emotion_rh_forecasting')
         .add_local_dir(CANONICAL/'shared', '/root/shared'))


@app.function(image=image, gpu='H200', cpu=12, memory=131072, timeout=21600,
              volumes={'/models': cache, '/data': data},
              secrets=[modal.Secret.from_dict({'HF_TOKEN': os.environ['HF_TOKEN']})
                       if os.environ.get('HF_TOKEN') else modal.Secret.from_name('huggingface')])
def replay(spec, bundle, limit):
    import numpy as np
    from emotion_rh_forecasting.terminalwrench_replay import GemmaReplay
    directory = Path('/data')/spec['identity']
    output = directory/'scores'
    output.mkdir(exist_ok=True)
    yield {'type': 'progress', 'phase': 'loading_model'}
    engine = GemmaReplay(bundle, spec)
    yield {'type': 'progress', 'phase': 'model_ready', 'runtime': engine.runtime}
    with zipfile.ZipFile(directory/'inputs.zip') as archive:
        selected = spec['jobs'][:limit] if limit else spec['jobs']
        for index, entry in enumerate(selected):
            cid = entry['id']
            path = output/f'{cid}.npz'
            receipt_path = path.with_suffix('.receipt.json')
            if path.exists() and receipt_path.exists():
                receipt = read(receipt_path)
                require(receipt['replay_identity'] == spec['identity'] and sha(path) == receipt['sha256'], 'Invalid replay cache')
                yield {'type': 'complete', 'id': cid, 'npz': path.read_bytes(), 'receipt': receipt, 'cached': True}
                continue
            job = json.loads(archive.read(cid+'.json'))
            raw = archive.read(cid+'.npz')
            import hashlib
            require(hashlib.sha256(raw).hexdigest() == entry['input_sha256'], 'Input token archive changed')
            with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                arrays = {k: z[k].copy() for k in z.files}
            yield {'type': 'progress', 'phase': 'score', 'id': cid, 'index': index, 'tokens': len(arrays['token_ids'])}
            blob, receipt = engine.score(job, arrays, output, smoke=(index == 0))
            data.commit()
            yield {'type': 'complete', 'id': cid, 'npz': blob, 'receipt': receipt, 'cached': False}
    cache.commit()
    yield {'type': 'finished'}


@app.local_entrypoint()
def main(run_dir: str, limit: int = 0):
    root = Path(run_dir).resolve()
    spec, bundle = read(root/'replay-spec.json'), read(root/'probes.json')
    require(sha(root/'probes.json') == spec['probes_sha256'], 'Probe bundle changed')
    remote = '/' + spec['identity']
    existing = {e.path.lstrip('/') for e in data.iterdir('/', recursive=False)}
    if spec['identity'] not in existing:
        with data.batch_upload() as upload:
            upload.put_file(root/'inputs.zip', remote+'/inputs.zip')
    status = {'state': 'running', 'completed': 0, 'total': len(spec['jobs']), 'started_unix': time.time()}
    (root/'scores').mkdir(exist_ok=True)
    for item in replay.remote_gen(spec, bundle, limit):
        if item['type'] == 'complete':
            path = root/'scores'/(item['id']+'.npz')
            path.write_bytes(item.pop('npz'))
            receipt = item.pop('receipt')
            require(sha(path) == receipt['sha256'], 'Downloaded scores changed')
            atomic(path.with_suffix('.receipt.json'), receipt)
            status['completed'] += 1
            print(f'{status["completed"]}/{status["total"]} traces saved · {item["id"]} · {receipt["seconds"]:.1f}s', flush=True)
        else:
            print(json.dumps(item), flush=True)
        status.update(last_event=item, updated_unix=time.time())
        atomic(root/'progress.json', status)
    complete = all((root/'scores'/(j['id']+'.receipt.json')).exists() for j in spec['jobs'])
    status['state'] = 'complete' if complete else 'partial'
    atomic(root/'progress.json', status)
    if complete:
        receipts = [read(root/'scores'/(j['id']+'.receipt.json')) for j in spec['jobs']]
        atomic(root/'replay-receipt.json', {'schema_version': 1, 'status': 'complete', 'identity': spec['identity'],
               'probe_identity': spec['probe_identity'], 'model': bundle['model'], 'interpretation': spec['interpretation'],
               'checks': 'Pinned tokenizer IDs/offsets match; each source boundary ends a forward; shorter-prefix validation on every trace.',
               'causal_checks_passed': True, 'traces': receipts})
