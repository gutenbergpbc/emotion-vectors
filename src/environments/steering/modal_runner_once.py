"""Generic Modal generation worker for one steering config.

The local coordinator sets STEERING_CONFIG before calling `modal run`. Results
are written to a run-specific Modal volume, so this worker cannot overwrite the
original pilot or another experiment.
"""
from pathlib import Path
import json
import os

import modal

LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path('/root/model-forensics')
ROOT = REMOTE_ROOT if Path(__file__).resolve().parent == Path('/root') else LOCAL_ROOT
DEFAULT_CONFIG = ROOT / 'steering/configs/qwen35_9b_rh_gamut_six_b32.json'
CONFIG = Path(os.environ.get('STEERING_CONFIG', DEFAULT_CONFIG)).resolve()
if not CONFIG.is_absolute():
    CONFIG = (ROOT / CONFIG).resolve()
REMOTE_CONFIG = Path('/root/model-forensics') / CONFIG.relative_to(LOCAL_ROOT)
SPEC = json.loads(CONFIG.read_text())
RUN_ID = SPEC['run_id']
RESULT_VOLUME_NAME = SPEC.get('modal_result_volume', f"emotion-steering-{RUN_ID}")
RESULT_VOLUME = modal.Volume.from_name(RESULT_VOLUME_NAME, create_if_missing=True)
MODEL_VOLUME = modal.Volume.from_name(SPEC.get('modal_model_volume', 'qwen35-9b-hostility-model'))
VECTOR_RUN_ID = SPEC.get('vector_run_id', 'qwen35-9b-emotions-pilot-20261015')
LEGACY = SPEC['legacy_runtime']

IMAGE = (
    modal.Image.debian_slim(python_version='3.12')
    .uv_pip_install(
        'transformers==5.17.0', 'flash-linear-attention==0.5.2', 'accelerate',
        'sentencepiece', 'protobuf', 'numpy', 'requests', 'openai', 'pyyaml', 'tqdm',
    )
    .env({'HF_HOME': '/models', 'HF_HUB_CACHE': '/models', 'HF_HUB_OFFLINE': '1',
          'STEERING_CONFIG': str(REMOTE_CONFIG)})
    .add_local_dir(str(ROOT / 'steering'), remote_path='/root/model-forensics/steering',
                   ignore=['runs/**', 'tests/**', '__pycache__/**'])
    .add_local_dir(str(ROOT / 'steering/runs' / VECTOR_RUN_ID / 'vectors'),
                   remote_path=f'/root/model-forensics/steering/runs/{VECTOR_RUN_ID}/vectors')
    .add_local_dir(str(ROOT / LEGACY), remote_path=f'/root/model-forensics/{LEGACY}')
)

app = modal.App('emotion-steering-one-run')


@app.function(image=IMAGE, gpu='H100', cpu=8, memory=32768, timeout=86400,
              volumes={'/data': RESULT_VOLUME, '/models': MODEL_VOLUME})
def execute(spec, initial_snapshot):
    import sys
    import time
    from pathlib import Path

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
    elif read(root / 'run_spec.json') != spec:
        raise ValueError('Run spec changed on the results volume; use a new run ID')

    checkpoint = {'last': 0.0}
    def persist(force=False):
        now = time.monotonic()
        if force or now - checkpoint['last'] >= 15:
            RESULT_VOLUME.commit()
            checkpoint['last'] = time.monotonic()

    with Tracker(root, on_checkpoint=persist, verify_existing=False) as tracker:
        check_spec(tracker)
        generate(tracker)
    RESULT_VOLUME.commit()
    receipts = read(root / 'receipts.json')
    generated = sum(r['stage_id'] == 'generate' for r in receipts.values())
    return {'run_id': spec['run_id'], 'state': read(root / 'progress.json')['state'],
            'generation_receipts': generated, 'expected_generations': len(tasks(spec)),
            'batch_size': spec['generation'].get('batch_size')}


@app.local_entrypoint()
def main(config_path: str = str(CONFIG)):
    from steering.run import initialize
    from steering.tracking import read

    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    spec = read(path)
    if spec['run_id'] != RUN_ID:
        raise ValueError('The config selected at module import differs from config_path')
    output = ROOT / 'steering/runs' / RUN_ID
    initialize(path, output)
    print(execute.remote(spec, read(output / 'progress.json')))
