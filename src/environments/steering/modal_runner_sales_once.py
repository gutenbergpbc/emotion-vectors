"""Pinned GPT-OSS H200 sales runner, invoked by steering.run_once."""
import json
import os
from pathlib import Path
import time

import modal

ROOT = (Path(__file__).resolve().parents[1] if Path(__file__).parent.name == 'steering'
        else Path('/root/model-forensics'))
CONFIG = Path(os.environ['STEERING_CONFIG']).resolve()
SPEC = json.loads(CONFIG.read_text())
RESULT_VOLUME = modal.Volume.from_name(SPEC['modal_result_volume'], create_if_missing=True)
MODEL_VOLUME = modal.Volume.from_name(SPEC['modal_model_volume'])
IMAGE = (modal.Image.debian_slim(python_version='3.12')
    .pip_install_from_requirements(str(ROOT / 'probe-harness/gpt-oss-probes-v1/requirements-extract.txt'))
    .pip_install('pandas==2.2.3')
    .env({'HF_HOME': '/models', 'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1',
          'STEERING_CONFIG': '/root/model-forensics/' + str(CONFIG.relative_to(ROOT))})
    .add_local_file(str(ROOT / 'probe-harness/gpt-oss-probes-v1/requirements-extract.txt'),
                    remote_path='/root/model-forensics/probe-harness/gpt-oss-probes-v1/requirements-extract.txt')
    .add_local_dir(str(ROOT / 'steering'), remote_path='/root/model-forensics/steering',
                   ignore=['runs/**', 'tests/**', '__pycache__/**'])
    .add_local_dir(str(ROOT / 'sales_env'), remote_path='/root/model-forensics/sales_env',
                   ignore=['__pycache__/**']))
for probe in SPEC['probes']:
    IMAGE = IMAGE.add_local_file(str(ROOT / probe['source']),
                                remote_path='/root/model-forensics/' + probe['source'])

app = modal.App('gpt-oss120-sales-emotion-steering')


@app.function(image=IMAGE, gpu='H200', cpu=16, memory=131072, timeout=86400,
              volumes={'/data': RESULT_VOLUME, '/models': MODEL_VOLUME})
def execute(spec, snapshot, imported_responses, imported_checkpoints):
    import sys
    sys.path.insert(0, '/root/model-forensics')
    from steering.tracking import Tracker, atomic, read, sha
    from steering.sales import generate
    root = Path('/data') / spec['run_id']
    root.mkdir(parents=True, exist_ok=True)
    if not (root / 'run_spec.json').exists():
        atomic(root / 'run_spec.json', spec)
        atomic(root / 'progress.json', snapshot)
        atomic(root / 'receipts.json', {})
        (root / 'events.jsonl').touch()
    elif read(root / 'run_spec.json') != spec:
        raise ValueError('Sales spec differs from frozen remote run')
    checkpoint = {'last': 0.0}

    def persist(force=False):
        if force or time.monotonic() - checkpoint['last'] >= 15:
            RESULT_VOLUME.commit()
            checkpoint['last'] = time.monotonic()

    with Tracker(root, on_checkpoint=persist) as run:
        for item in imported_checkpoints:
            path = root / 'episode_checkpoints' / (item['id'] + '.json')
            if not path.exists() and 'generate/' + item['id'] not in run.receipts:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(item['artifact_text'])
                if sha(path) != item['sha256']:
                    raise ValueError('Imported sales checkpoint hash differs')
                if read(path)['spec_sha256'] != sha(root / 'run_spec.json'):
                    raise ValueError('Imported sales checkpoint refers to another specification')
        for item in imported_responses:
            if run.start('generate', item['id'], {'reused_from': spec['continuation']['source_run_id']}):
                path = root / 'responses' / (item['id'] + '.json')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(item['artifact_text'])
                if sha(path) != item['sha256']:
                    raise ValueError('Imported sales response hash differs')
                run.commit('generate', item['id'], path, {'reused_from': spec['continuation']['source_run_id']})
        generate(run)
    RESULT_VOLUME.commit()
    return dict(run_id=spec['run_id'], state=read(root / 'progress.json')['state'])


@app.local_entrypoint()
def main():
    from steering.run import initialize
    from steering.tracking import read
    root = ROOT / 'steering/runs' / SPEC['run_id']
    initialize(CONFIG, root)
    imported = []
    for item in SPEC.get('continuation', {}).get('responses', []):
        source = root / 'responses' / (item['id'] + '.json')
        imported.append(dict(**item, artifact_text=source.read_text()))
    checkpoints = []
    from steering.tracking import sha
    for item in SPEC.get('continuation', {}).get('checkpoints', []):
        source = root / 'episode_checkpoints' / (item['id'] + '.json')
        checkpoints.append(dict(id=item['id'], sha256=sha(source), artifact_text=source.read_text()))
    print(execute.remote(SPEC, read(root / 'progress.json'), imported, checkpoints))
