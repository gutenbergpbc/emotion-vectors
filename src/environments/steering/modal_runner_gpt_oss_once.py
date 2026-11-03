"""Dedicated GPT-OSS 120B/H200 Modal worker for one steering run."""
from pathlib import Path
import json
import os
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(os.environ.get('STEERING_CONFIG', ROOT / 'steering/configs/gpt_oss_120b_rh_pm005_pm010_b32.json')).resolve()
SPEC = json.loads(CONFIG.read_text())
RUN_ID = SPEC['run_id']
REMOTE_CONFIG = Path('/root/model-forensics') / CONFIG.relative_to(ROOT)
RESULT_VOLUME = modal.Volume.from_name(SPEC['modal_result_volume'], create_if_missing=True)
MODEL_VOLUME = modal.Volume.from_name(SPEC['modal_model_volume'])

IMAGE = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install_from_requirements(str(ROOT / 'probe-harness/gpt-oss-probes-v1/requirements-extract.txt'))
    # The existing GPT-OSS cache volume has the standard $HF_HOME/hub layout.
    .env({'HF_HOME': '/models', 'HF_HUB_OFFLINE': '1',
          'STEERING_CONFIG': str(REMOTE_CONFIG), 'PYTHONUNBUFFERED': '1'})
    .add_local_dir(str(ROOT / 'steering'), remote_path='/root/model-forensics/steering',
                   ignore=['runs/**', 'tests/**', '__pycache__/**'])
    .add_local_file(str(ROOT / 'probe-harness/gpt-oss-probes-v1/requirements-extract.txt'),
                    remote_path='/root/model-forensics/probe-harness/gpt-oss-probes-v1/requirements-extract.txt')
)
for probe in SPEC['probes']:
    source = ROOT / probe['source']
    IMAGE = IMAGE.add_local_file(
        str(source), remote_path=str(Path('/root/model-forensics') / probe['source']))

app = modal.App('gpt-oss120-rh-emotion-steering-one-run')


@app.function(image=IMAGE, gpu='H200', cpu=16, memory=131072, timeout=86400,
              volumes={'/data': RESULT_VOLUME, '/models': MODEL_VOLUME})
def execute(spec, initial_snapshot):
    import sys
    from pathlib import Path
    sys.path.insert(0, '/root/model-forensics')
    from steering.run import check_spec, tasks
    from steering.tracking import Tracker, atomic, read
    from steering.gpt_oss_generate import generate

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
