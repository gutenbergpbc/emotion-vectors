"""Resume the 0.075 Qwen run with larger inference batches."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = 'qwen35-9b-rh-conv1d-075-20261015'
RESULT_VOLUME = modal.Volume.from_name('qwen35-rh-conv1d-075-results-20261015')
MODEL_VOLUME = modal.Volume.from_name('qwen35-9b-hostility-model')
LEGACY = 'sandbox/ayush/emotion_vectors/local_reproduction_20261014/runtime/emotion_concepts'

IMAGE = (
    modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04', add_python='3.12')
    .entrypoint([])
    .uv_pip_install('torch==2.8.0', index_url='https://download.pytorch.org/whl/cu128')
    .uv_pip_install('setuptools', 'wheel', 'packaging', 'ninja')
    .uv_pip_install('causal-conv1d==1.7.0', extra_options='--no-build-isolation')
    .uv_pip_install(
        'transformers==5.17.0', 'flash-linear-attention==0.5.2', 'accelerate',
        'sentencepiece', 'protobuf', 'numpy', 'requests', 'openai', 'pyyaml', 'tqdm',
    )
    .env({'HF_HOME': '/models', 'HF_HUB_CACHE': '/models', 'HF_HUB_OFFLINE': '1'})
    .add_local_dir(str(ROOT / 'steering'), remote_path='/root/model-forensics/steering',
                   ignore=['runs/**', 'tests/**', '__pycache__/**'])
    .add_local_dir(str(ROOT / 'steering/runs' / RUN_ID / 'vectors'),
                   remote_path=f'/root/model-forensics/steering/runs/{RUN_ID}/vectors')
    .add_local_dir(str(ROOT / LEGACY), remote_path=f'/root/model-forensics/{LEGACY}')
)

app = modal.App('qwen35-rh-conv1d-075-resume-b32')


@app.function(image=IMAGE, gpu='H100', cpu=8, memory=32768, timeout=86400,
              volumes={'/data': RESULT_VOLUME, '/models': MODEL_VOLUME})
def execute(spec):
    import sys
    import time
    from itertools import groupby
    sys.path.insert(0, '/root/model-forensics')
    from steering.run import check_spec, tasks
    from steering.backend import load_vector, make_backend
    from steering.tracking import Tracker, atomic, read

    root = Path('/data') / spec['run_id']
    if not (root / 'run_spec.json').exists():
        raise RuntimeError('Cannot resume: the batch-8 run is missing from the results volume')

    checkpoint = {'last': 0.0}
    def persist(force=False):
        current = time.monotonic()
        if force or current - checkpoint['last'] >= 15:
            RESULT_VOLUME.commit()
            checkpoint['last'] = time.monotonic()

    with Tracker(root, on_checkpoint=persist, verify_existing=False) as tracker:
        # Validate the immutable scientific spec and vector compatibility, but
        # leave its batch-8 seed schedule intact. Batch size here changes only
        # how the not-yet-committed tasks are packed for GPU generation.
        check_spec(tracker)
        backend = make_backend(tracker.spec, Path('/root/model-forensics'))
        probes = {p['emotion']: p for p in tracker.spec['probes']}
        vectors = {p['emotion']: load_vector(p, Path('/root/model-forensics'))
                   for p in tracker.spec['probes']}
        for probe in tracker.spec['probes']:
            if len(vectors[probe['emotion']]) != backend.hidden_size:
                raise ValueError(f"Incompatible probe: {probe['emotion']}")

        tracker.stage('generate', 'Resuming remaining responses · batch size 32')
        missing = [task for task in tasks(tracker.spec)
                   if f"generate/{task['id']}" not in tracker.receipts]
        missing.sort(key=lambda t: (t['condition'], t['scenario'], t['sample']))
        for _, grouped in groupby(missing, key=lambda t: (t['condition'], t['scenario'])):
            group = list(grouped)
            for offset in range(0, len(group), 32):
                batch = group[offset:offset + 32]
                started = []
                for task in batch:
                    context = {k: task[k] for k in ('emotion', 'strength', 'scenario', 'sample', 'seed', 'layer')}
                    if tracker.start('generate', task['id'], context):
                        started.append(task)
                if not started:
                    continue
                first = started[0]
                tracker.stage('generate', f"{first['condition']} · {first['scenario']} · batch {first['sample']}–{started[-1]['sample']} (32)")
                started_at = time.perf_counter()
                results = backend.sample_batch(started, probes.get(first['emotion']), vectors.get(first['emotion']))
                elapsed = time.perf_counter() - started_at
                if len(results) != len(started):
                    raise RuntimeError('Backend returned the wrong batch size')
                batch_tokens = sum(result.get('generated_tokens', 0) for result in results)
                for result in results:
                    result['batch_elapsed_seconds'] = elapsed
                    result['batch_size'] = len(results)
                    result['batch_responses_per_second'] = len(results) / elapsed if elapsed else None
                    result['batch_tokens_per_second'] = batch_tokens / elapsed if elapsed else None
                for task, result in zip(started, results):
                    path = root / 'responses' / (task['id'] + '.json')
                    atomic(path, dict(**task, **result, model=tracker.spec['model'],
                                      steering_mode=tracker.spec['steering_mode']))
                    tracker.commit('generate', task['id'], path,
                                   {'scenario': task['scenario'], 'sample': task['sample']})
        tracker.finish_stage('generate')
    RESULT_VOLUME.commit()
    receipts = read(root / 'receipts.json')
    return {'state': read(root / 'progress.json')['state'], 'run_id': spec['run_id'],
            'generation_receipts': sum(r['stage_id'] == 'generate' for r in receipts.values()),
            'expected_generations': len(tasks(spec)), 'batch_size': 32}


@app.local_entrypoint()
def main(config_path='steering/configs/qwen35_9b_rh_conv1d_075.json'):
    from steering.tracking import read
    spec = read(ROOT / config_path)
    if spec['run_id'] != RUN_ID:
        raise ValueError('Unexpected run ID in resume config')
    print(execute.remote(spec))
