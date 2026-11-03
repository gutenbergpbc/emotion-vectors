"""Capture best12 raw residual projections for frozen GLM 5.3 self-report traces."""
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


import json
import os
from pathlib import Path
import sys

import modal

HERE = Path(__file__).resolve().parent
CANONICAL = Path('/study/src/experiments') if Path('/study/src/experiments').exists() else HERE.parent
ROOT = CANONICAL.parents[1] if len(CANONICAL.parents) > 1 else Path('/study')
sys.path[:0] = [str(CANONICAL), str(CANONICAL / 'emotion-probes')]
from modal_worker import image, models, runs, kernels, secret  # noqa: E402
if not Path('/study/src/experiments').exists():
    image = image.add_local_file(CANONICAL / 'emotion-probes/modal_worker.py',
                                 '/study/src/experiments/emotion-probes/modal_worker.py')

LOCAL = Path(os.environ.get('GLM_REPLAY_ROOT', ROOT / 'runs/emotion-self-report-activations/glm53-complete-traces-v2'))
app = modal.App('canonical-glm53-self-report-activation-replay')


def capture_fn(vectors, n):
    by_layer = {}
    for column, probe in enumerate(vectors):
        by_layer.setdefault(probe['layer'], []).append((column, probe['vector']))
    layers = sorted(by_layer)
    width = len(vectors[0]['vector'])

    def capture(ctx, hidden):
        import numpy as np
        import torch
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
        if get_tensor_model_parallel_rank() != 0:
            return None
        if hidden.ndim != 2 or hidden.shape[1] != width:
            raise ValueError('Residual width changed')
        layer = ctx.layer_idx
        saved = ctx.saved
        # Generation/decode hooks may run after all prompt tokens were saved.
        if 'scores' in saved:
            return None
        offsets = saved.setdefault('offsets', {})
        start = offsets.get(layer, 0)
        stop = min(start + hidden.shape[0], n)
        offsets[layer] = start + hidden.shape[0]
        if stop > start:
            matrix = torch.as_tensor([v for _, v in by_layer[layer]], dtype=torch.float32,
                                     device=hidden.device).T
            projections = (hidden[:stop-start].float() @ matrix).cpu().numpy()
            saved.setdefault('parts', {}).setdefault(layer, []).append((start, projections))
        if layer == layers[-1] and stop >= n:
            result = np.full((n, len(vectors)), np.nan, dtype=np.float32)
            for selected_layer, columns in by_layer.items():
                if offsets.get(selected_layer, 0) < n:
                    raise ValueError('Layer capture incomplete')
                parts = saved['parts'][selected_layer]
                if sum(len(part) for _, part in parts) != n:
                    raise ValueError('Layer tokens missing or duplicated')
                for begin, values in parts:
                    result[begin:begin+len(values), [index for index, _ in columns]] = values
            if not np.isfinite(result).all():
                raise ValueError('Nonfinite projections')
            saved['scores'] = result
            saved.pop('parts')
        return None
    return capture, layers


@app.function(image=image, gpu='B200:4', cpu=16, memory=131072, timeout=86400,
              volumes={'/models': models, '/data': runs, '/kernel-cache': kernels},
              secrets=[secret], max_containers=1)
def worker(remote_name: str):
    import numpy as np
    from vllm import SamplingParams
    from vllm_lens import Hook
    from emotion_probes.extraction import LensExtractor
    from shared.io import atomic, read, require, sha
    root = Path('/data') / remote_name
    spec = read(root / 'replay-spec.json')
    bundle = read(root / 'probes.json')
    require(sha(root / 'probes.json') == spec['probes_sha256'], 'Probe checksum mismatch')
    vectors = bundle['probes']
    require(len(vectors) == 12, 'Expected twelve probes')
    engine_spec = {'model': bundle['model']['name'], 'revision': bundle['model']['revision'],
        'tokenizer_model': bundle['model']['tokenizer_model'],
        'tokenizer_revision': bundle['model']['tokenizer_revision'], 'dtype': 'bfloat16',
        'quantization': 'modelopt_fp4', 'tensor_parallel_size': 4,
        'max_model_len': max(job['tokens'] for job in spec['jobs']) + 1,
        'max_num_batched_tokens': 8192, 'gpu_memory_utilization': .9,
        'renderer': 'glm-native-preserved-streams-v1'}
    atomic(root / 'progress.json', {'status': 'loading', 'total': len(spec['jobs']), 'done': 0,
                                    'identity': spec['identity']})
    runs.commit()
    extractor = LensExtractor(engine_spec)
    done = 0
    pilot_trace = spec['jobs'][0]['trace_id'] if spec['jobs'] else None
    pilot_validated = False
    for index, entry in enumerate(spec['jobs']):
        job_path = root / 'inputs' / (entry['id'] + '.json')
        job = read(job_path)
        require(sha(job_path) == entry['sha256'], 'Job input checksum mismatch')
        output = root / 'scores' / (entry['id'] + '.npz')
        receipt = output.with_suffix('.receipt.json')
        if receipt.exists():
            old = read(receipt)
            require(old['job_sha256'] == entry['sha256'] and old['replay_identity'] == spec['identity']
                    and sha(output) == old['sha256'], 'Cached shard mismatch')
        else:
            hook_fn, layers = capture_fn(vectors, job['tokens'])
            hook = Hook(fn=hook_fn, layer_indices=layers, pre=False)
            generated, = extractor.llm.generate([{'prompt_token_ids': job['token_ids']}],
                SamplingParams(temperature=0, max_tokens=1, ignore_eos=True,
                               extra_args={'apply_hooks': [hook]}), use_tqdm=False)
            require(list(generated.prompt_token_ids) == job['token_ids'], 'Replay tokens changed')
            result = generated.hook_results
            result = result.get('0', result.get(0, result))
            scores = np.asarray(result['scores'], dtype=np.float32)
            require(scores.shape == (job['tokens'], 12) and np.isfinite(scores).all(),
                    'Invalid captured score matrix')
            for readout in job['readouts']:
                idx = readout['token_index']
                require(0 <= idx < job['tokens'] and np.isfinite(scores[idx]).all(),
                        'Invalid causal checkpoint readout')
                require(readout['probe_char_end'] <= readout['sentence_char_end'],
                        'Readout includes a future character')
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output, projections=scores)
            atomic(receipt, {'id': entry['id'], 'job_sha256': entry['sha256'],
                'replay_identity': spec['identity'], 'sha256': sha(output), 'tokens': job['tokens'],
                'selected_layers': layers, 'projection': 'post-block residual dot probe vector'})
        done += 1
        if (not pilot_validated and entry['trace_id'] == pilot_trace
                and (index + 1 == len(spec['jobs']) or spec['jobs'][index + 1]['trace_id'] != pilot_trace)):
            pilot_validated = True
        atomic(root / 'progress.json', {'status': 'scoring', 'total': len(spec['jobs']), 'done': done,
            'current': entry['id'], 'identity': spec['identity'],
            'first_trace_validated': pilot_validated, 'pilot_trace_id': pilot_trace})
        runs.commit()
    atomic(root / 'progress.json', {'status': 'complete', 'total': len(spec['jobs']), 'done': done,
                                    'identity': spec['identity']})
    runs.commit()
    models.commit()
    return {'status': 'complete', 'jobs': done}


def download(remote_name, relative):
    target = LOCAL / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + '.download')
    with tmp.open('wb') as stream:
        for chunk in runs.read_file(f'/{remote_name}/{relative}'):
            stream.write(chunk)
    tmp.replace(target)


@app.local_entrypoint()
def main():
    from shared.io import atomic, read, require, sha
    spec = read(LOCAL / 'replay-spec.json')
    remote_name = LOCAL.name + '-' + spec['identity'][:12]
    try:
        remote_spec = json.loads(b''.join(runs.read_file(f'/{remote_name}/replay-spec.json')))
        require(remote_spec == spec, 'Remote replay specification differs')
    except FileNotFoundError:
        with runs.batch_upload() as upload:
            upload.put_file(LOCAL / 'replay-spec.json', f'/{remote_name}/replay-spec.json')
            upload.put_file(LOCAL / 'probes.json', f'/{remote_name}/probes.json')
            upload.put_directory(LOCAL / 'inputs', f'/{remote_name}/inputs')
    call = worker.spawn(remote_name)
    atomic(LOCAL / 'connection.json', {'remote_name': remote_name, 'call_id': call.object_id,
        'job_url': call.get_dashboard_url(), 'identity': spec['identity']})
    print(f'Modal job: {call.get_dashboard_url()}', flush=True)
    while True:
        try:
            result = call.get(timeout=15)
            break
        except TimeoutError:
            try:
                progress = json.loads(b''.join(runs.read_file(f'/{remote_name}/progress.json')))
                atomic(LOCAL / 'progress.json', progress)
                print(f"{progress['status']}: {progress['done']}/{progress['total']}", flush=True)
            except FileNotFoundError:
                print('Waiting for GPU allocation', flush=True)
    require(result['status'] == 'complete' and result['jobs'] == len(spec['jobs']), 'Replay incomplete')
    for entry in spec['jobs']:
        for ext in ('.npz', '.receipt.json'):
            download(remote_name, f"scores/{entry['id']}{ext}")
        score_path = LOCAL / 'scores' / f"{entry['id']}.npz"
        receipt = read(score_path.with_suffix('.receipt.json'))
        require(receipt['job_sha256'] == entry['sha256'] and sha(score_path) == receipt['sha256'],
                'Downloaded score shard checksum mismatch')
    download(remote_name, 'progress.json')
    print(f"Verified {len(spec['jobs'])} score shards in {LOCAL}", flush=True)
