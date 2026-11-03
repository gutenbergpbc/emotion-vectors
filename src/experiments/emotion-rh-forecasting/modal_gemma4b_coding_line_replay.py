"""Capture twelve frozen Gemma 4B probe projections over coding code lines.

From model-forensics: uv run --with modal python -m modal run \
  src/experiments/emotion-rh-forecasting/modal_gemma4b_coding_line_replay.py
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


import json
import os
from pathlib import Path
import sys
import time

import modal

HERE = Path(__file__).resolve().parent
CANONICAL = Path('/study/src/experiments') if Path('/study/src/experiments').exists() else HERE.parent
ROOT = CANONICAL.parents[1]
sys.path.insert(0, str(CANONICAL / 'emotion-probes'))
sys.path.insert(0, str(CANONICAL))
from shared.io import require  # noqa: E402
from modal_worker import image, models, runs, kernels, secret  # noqa: E402
if not Path('/study/src/experiments').exists():
    image = image.add_local_file(CANONICAL / 'emotion-probes/modal_worker.py',
                                 '/study/src/experiments/emotion-probes/modal_worker.py')

LOCAL = ROOT / 'runs/emotion-rh-replay/gemma3-4b-coding-lines-best12-v1'
app = modal.App('canonical-gemma4b-coding-line-replay')


def capture_fn(vectors, n):
    """Retain per-token raw projections at selected post-block residual layers."""
    by_layer = {}
    for index, p in enumerate(vectors):
        by_layer.setdefault(p['layer'], []).append((index, p['vector']))
    layers = sorted(by_layer)
    width = len(vectors[0]['vector'])

    def capture(ctx, hidden):
        import numpy as np
        import torch
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
        if get_tensor_model_parallel_rank() != 0:
            return None
        require(hidden.ndim == 2 and hidden.shape[1] == width, 'Residual width changed')
        layer = ctx.layer_idx
        saved = ctx.saved
        if 'scores' in saved:
            return None
        offsets = saved.setdefault('offsets', {})
        start = offsets.get(layer, 0)
        stop = min(start + hidden.shape[0], n)
        offsets[layer] = start + hidden.shape[0]
        if stop > start:
            matrix = torch.as_tensor([v for _, v in by_layer[layer]], dtype=torch.float32, device=hidden.device).T
            values = (hidden[:stop-start].float() @ matrix).cpu().numpy()
            saved.setdefault('parts', {}).setdefault(layer, []).append((start, values))
        if layer == layers[-1] and stop >= n:
            output = np.full((n, len(vectors)), np.nan, dtype=np.float32)
            for selected_layer, columns in by_layer.items():
                require(offsets.get(selected_layer, 0) >= n, 'Layer capture incomplete')
                parts = saved['parts'][selected_layer]
                require(sum(len(part) for _, part in parts) == n, 'Layer tokens missing or duplicated')
                for begin, values in parts:
                    output[begin:begin+len(values), [index for index, _ in columns]] = values
            require(np.isfinite(output).all(), 'Nonfinite projections')
            saved['scores'] = output
            saved.pop('parts')
        return None

    return capture, layers


@app.function(image=image, gpu='H100', cpu=8, memory=32768, timeout=86400,
              volumes={'/models': models, '/data': runs, '/kernel-cache': kernels}, secrets=[secret], max_containers=1)
def worker(remote_name: str):
    import numpy as np
    from vllm import SamplingParams
    from vllm_lens import Hook
    from emotion_probes.extraction import LensExtractor
    from shared.io import atomic, digest, read, require, sha

    root = Path('/data') / remote_name
    spec = read(root / 'replay-spec.json')
    bundle = read(root / 'probes.json')
    require(sha(root / 'probes.json') == spec['probes_sha256'], 'Probe vector checksum mismatch')
    vectors = bundle['probes']
    require(len(vectors) == 12, 'Expected twelve probes')
    engine_spec = {'model': bundle['model']['name'], 'revision': bundle['model']['revision'],
                   'tokenizer_model': bundle['model']['tokenizer_model'],
                   'tokenizer_revision': bundle['model']['tokenizer_revision'],
                   'dtype': 'bfloat16', 'quantization': None, 'tensor_parallel_size': 1,
                   'max_model_len': max(j['tokens'] for j in spec['jobs']) + 1,
                   'max_num_batched_tokens': 8192, 'gpu_memory_utilization': .9,
                   'renderer': 'gemma-native-preserved-streams-v2'}
    atomic(root / 'progress.json', {'status': 'loading', 'total': len(spec['jobs']), 'done': 0,
                                     'identity': spec['identity']})
    runs.commit()
    extractor = LensExtractor(engine_spec)
    done = 0
    for entry in spec['jobs']:
        job = read(root / 'inputs' / (entry['id'] + '.json'))
        require(sha(root / 'inputs' / (entry['id'] + '.json')) == entry['sha256'], 'Job changed')
        output = root / 'scores' / (entry['id'] + '.npz')
        receipt = output.with_suffix('.receipt.json')
        if receipt.exists():
            prior = read(receipt)
            require(prior['job_sha256'] == entry['sha256'] and prior['replay_identity'] == spec['identity']
                    and sha(output) == prior['sha256'], 'Cached score shard mismatch')
        else:
            capture, layers = capture_fn(vectors, len(job['token_ids']))
            hook = Hook(fn=capture, layer_indices=layers, pre=False)
            generated, = extractor.llm.generate([{'prompt_token_ids': job['token_ids']}],
                SamplingParams(temperature=0, max_tokens=1, ignore_eos=True,
                               extra_args={'apply_hooks': [hook]}), use_tqdm=False)
            require(list(generated.prompt_token_ids) == job['token_ids'], 'Engine changed replay tokens')
            result = generated.hook_results
            result = result.get('0', result.get(0, result))
            scores = np.asarray(result['scores'], dtype=np.float32)
            require(scores.shape == (job['tokens'], 12) and np.isfinite(scores).all(), 'Invalid score shape')
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output, scores=scores)
            atomic(receipt, {'id': entry['id'], 'job_sha256': entry['sha256'],
                             'replay_identity': spec['identity'], 'sha256': sha(output),
                             'tokens': job['tokens'], 'selected_layers': layers})
        done += 1
        atomic(root / 'progress.json', {'status': 'scoring', 'total': len(spec['jobs']), 'done': done,
                                        'current': entry['id'], 'identity': spec['identity']})
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
    from shared.io import atomic, read, sha
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
        receipt = read(LOCAL / 'scores' / (entry['id'] + '.receipt.json'))
        require(receipt['job_sha256'] == entry['sha256'] and sha(LOCAL / 'scores' / (entry['id'] + '.npz')) == receipt['sha256'],
                'Downloaded score shard mismatch')
    download(remote_name, 'progress.json')
    print(f'Verified {len(spec["jobs"])} replay shards in {LOCAL}', flush=True)
