"""Score frozen inputs for the two missing model/dataset cells on Modal."""

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

import json
import os
from pathlib import Path
import sys
import time

import modal

CANON = Path('/study/src/experiments') if Path('/study/src/experiments').exists() else Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CANON), str(CANON / 'emotion-probes')]
from modal_worker import image, models, runs, kernels, secret
from shared.io import atomic, read, require, sha

if modal.is_local():
    image = image.add_local_file(CANON / 'emotion-probes/modal_worker.py', '/study/src/experiments/emotion-probes/modal_worker.py')
    image = image.add_local_file(CANON / 'emotion-rh-forecasting/modal_gemma4b_coding_line_replay.py', '/study/src/experiments/emotion-rh-forecasting/modal_gemma4b_coding_line_replay.py')

FAMILY = os.environ.get('EMOTION_PROBE_MODEL_FAMILY', 'glm53')
GPU = 'H100' if FAMILY == 'gemma4b' else 'B200:4'
app = modal.App('missing-emotion-cell-replay-' + FAMILY)


@app.function(image=image, gpu=GPU, cpu=16, memory=131072, timeout=7200,
              volumes={'/models': models, '/data': runs, '/kernel-cache': kernels}, secrets=[secret], max_containers=1)
def score(remote_name: str):
    import numpy as np
    from vllm import SamplingParams
    from vllm_lens import Hook
    from emotion_probes.extraction import LensExtractor
    sys.path.insert(0, '/study/src/experiments/emotion-rh-forecasting')
    from modal_gemma4b_coding_line_replay import capture_fn
    root = Path('/data') / remote_name
    spec, bundle = read(root / 'replay-spec.json'), read(root / 'probes.json')
    require(sha(root / 'probes.json') == spec['probes_sha256'], 'Probe checksum mismatch')
    is_gemma = 'gemma' in bundle['model']['name'].lower()
    engine = dict(model=bundle['model']['name'], revision=bundle['model']['revision'],
                  tokenizer_model=bundle['model']['tokenizer_model'], tokenizer_revision=bundle['model']['tokenizer_revision'],
                  dtype='bfloat16', quantization=None if is_gemma else 'modelopt_fp4', tensor_parallel_size=1 if is_gemma else 4,
                  max_model_len=max(j['tokens'] for j in spec['jobs']) + 1, max_num_batched_tokens=8192,
                  gpu_memory_utilization=.85, renderer='gemma-native-preserved-streams-v2' if is_gemma else 'glm-native-preserved-streams-v1')
    atomic(root / 'progress.json', dict(status='loading', done=0, total=len(spec['jobs'])))
    runs.commit()
    extractor = LensExtractor(engine)
    for i, entry in enumerate(spec['jobs']):
        job = read(root / 'inputs' / f"{entry['id']}.json")
        require(sha(root / 'inputs' / f"{entry['id']}.json") == entry['sha256'], 'Input checksum mismatch')
        output = root / 'scores' / f"{entry['id']}.npz"
        receipt = output.with_suffix('.receipt.json')
        if receipt.exists():
            prior = read(receipt)
            require(prior['identity'] == spec['identity'] and sha(output) == prior['sha256'], 'Cached shard mismatch')
        else:
            start = time.monotonic()
            capture, layers = capture_fn(bundle['probes'], job['tokens'])
            generated, = extractor.llm.generate([{'prompt_token_ids': job['token_ids']}],
                SamplingParams(temperature=0, max_tokens=1, ignore_eos=True,
                               extra_args={'apply_hooks': [Hook(fn=capture, layer_indices=layers, pre=False)]}), use_tqdm=False)
            require(list(generated.prompt_token_ids) == job['token_ids'], 'Replay token IDs changed')
            result = generated.hook_results
            result = result.get('0', result.get(0, result))
            values = np.asarray(result['scores'], dtype=np.float32)
            require(values.shape == (job['tokens'], 12) and np.isfinite(values).all(), 'Invalid score matrix')
            output.parent.mkdir(exist_ok=True)
            np.savez_compressed(output, scores=values)
            atomic(receipt, dict(id=entry['id'], identity=spec['identity'], sha256=sha(output),
                                job_sha256=entry['sha256'], tokens=job['tokens'], seconds=time.monotonic()-start,
                                model=bundle['model'], projection='Post-block residual dot frozen probe vector; no intercept'))
        atomic(root / 'progress.json', dict(status='scoring', done=i+1, total=len(spec['jobs']), current=entry['id']))
        runs.commit()
    atomic(root / 'progress.json', dict(status='complete', done=len(spec['jobs']), total=len(spec['jobs'])))
    runs.commit()
    models.commit()
    return dict(status='complete', jobs=len(spec['jobs']))


@app.local_entrypoint()
def main(root: str):
    local = Path(root).resolve()
    spec = read(local / 'replay-spec.json')
    remote = local.name + '-' + spec['identity'][:12]
    try:
        require(json.loads(b''.join(runs.read_file(f'/{remote}/replay-spec.json'))) == spec, 'Remote spec differs')
    except FileNotFoundError:
        with runs.batch_upload() as up:
            for name in ['probes.json', 'replay-spec.json']:
                up.put_file(local / name, f'/{remote}/{name}')
            up.put_directory(local / 'inputs', f'/{remote}/inputs')
    connection = local / 'connection.json'
    if connection.exists():
        old = read(connection)
        require(old['identity'] == spec['identity'], 'Existing connection identity differs')
        call = modal.FunctionCall.from_id(old['call_id'])
    else:
        call = score.spawn(remote)
        atomic(connection, dict(call_id=call.object_id, identity=spec['identity'], remote_name=remote, job_url=call.get_dashboard_url()))
    print('Modal job:', call.get_dashboard_url(), flush=True)
    while True:
        try:
            result = call.get(timeout=15)
            break
        except TimeoutError:
            try:
                state = json.loads(b''.join(runs.read_file(f'/{remote}/progress.json')))
                atomic(local / 'progress.json', state)
                print(state, flush=True)
            except FileNotFoundError:
                print('Waiting for GPU allocation', flush=True)
    require(result['status'] == 'complete', 'Replay incomplete')
    for entry in spec['jobs']:
        for ext in ['.npz', '.receipt.json']:
            relative = f"scores/{entry['id']}{ext}"
            target = local / relative
            target.parent.mkdir(exist_ok=True)
            with target.with_suffix(target.suffix + '.download').open('wb') as f:
                for chunk in runs.read_file(f'/{remote}/{relative}'):
                    f.write(chunk)
            target.with_suffix(target.suffix + '.download').replace(target)
        p = local / 'scores' / f"{entry['id']}.npz"
        receipt = read(p.with_suffix('.receipt.json'))
        require(sha(p) == receipt['sha256'] and receipt['job_sha256'] == entry['sha256'], 'Downloaded shard differs')
    atomic(local / 'progress.json', result)
    print('Verified', result['jobs'], 'score shards in', local, flush=True)
