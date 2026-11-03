"""Independent Modal run using a CUDA-devel image with causal-conv1d."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = 'qwen35-9b-rh-conv1d-075-20261015'
RESULT_VOLUME = modal.Volume.from_name('qwen35-rh-conv1d-075-results-20261015', create_if_missing=True)
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

app = modal.App('qwen35-rh-conv1d-075')


@app.function(image=IMAGE, gpu='H100', cpu=8, memory=32768, timeout=86400,
              volumes={'/data': RESULT_VOLUME, '/models': MODEL_VOLUME})
def execute(spec, initial_snapshot):
    import sys
    import time
    sys.path.insert(0, '/root/model-forensics')
    from steering.run import check_spec, generate, tasks
    from steering.tracking import Tracker, atomic, read

    root = Path('/data') / spec['run_id']
    root.mkdir(parents=True, exist_ok=True)
    if not (root / 'run_spec.json').exists():
        atomic(root / 'run_spec.json', spec)
        atomic(root / 'progress.json', initial_snapshot)
        atomic(root / 'receipts.json', {})
        (root / 'events.jsonl').touch()

    checkpoint = {'last': 0.0}
    def persist(force=False):
        current = time.monotonic()
        if force or current - checkpoint['last'] >= 15:
            RESULT_VOLUME.commit()
            checkpoint['last'] = time.monotonic()

    with Tracker(root, on_checkpoint=persist, verify_existing=False) as tracker:
        check_spec(tracker)
        generate(tracker)
    RESULT_VOLUME.commit()
    receipts = read(root / 'receipts.json')
    generation = [r for r in receipts.values() if r['stage_id'] == 'generate']
    sample = read(root / generation[-1]['path']) if generation else {}
    return {
        'state': read(root / 'progress.json')['state'],
        'run_id': spec['run_id'],
        'generation_receipts': len(generation),
        'responses_per_second': sample.get('batch_responses_per_second'),
        'tokens_per_second': sample.get('batch_tokens_per_second'),
        'expected_generations': len(tasks(spec)),
    }


@app.local_entrypoint()
def main(config_path='steering/configs/qwen35_9b_rh_conv1d_075.json'):
    from steering.run import initialize
    from steering.tracking import read

    config = ROOT / config_path
    spec = read(config)
    output = ROOT / 'steering/runs' / spec['run_id']
    initialize(config, output)
    snapshot = read(output / 'progress.json')
    print(execute.remote(spec, snapshot))
