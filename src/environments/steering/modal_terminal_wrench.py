"""Serve a prepared TerminalWrench plan on one H200 using the existing cache.

Set TERMINALWRENCH_PLAN to the local plan.json and run ``modal serve`` on this
file. This file does not run Harbor or launch any task containers. Only an
explicit serve/deploy command starts compute.
"""
import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
PLAN = Path(os.environ['TERMINALWRENCH_PLAN']).resolve()
SPEC = json.loads(PLAN.read_text())
VOLUME_NAME = 'terminalwrench-steering-' + SPEC['manifest_sha256'][:16]
MODEL_VOLUME = modal.Volume.from_name('gpt-oss120-probe-model-cache-v1')
RESULT_VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
IMAGE = (modal.Image.debian_slim(python_version='3.12')
    .pip_install_from_requirements(str(ROOT / 'probe-harness/gpt-oss-probes-v1/requirements-extract.txt'))
    .env({'HF_HOME': '/models', 'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'})
    .add_local_dir(str(ROOT / 'steering'), remote_path='/root/model-forensics/steering',
                   ignore=['runs/**', 'tests/**', '__pycache__/**'])
    .add_local_file(str(PLAN), remote_path='/root/terminalwrench-plan.json'))
for source in sorted({p['source'] for p in SPEC['probes']}):
    IMAGE = IMAGE.add_local_file(str(ROOT / source), remote_path='/root/model-forensics/' + source)

app = modal.App('gptoss-terminalwrench-emotion-steering')


@app.function(image=IMAGE, gpu='H200', cpu=16, memory=131072, timeout=86400,
              max_containers=1, scaledown_window=600,
              volumes={'/models': MODEL_VOLUME, '/data': RESULT_VOLUME})
@modal.concurrent(max_inputs=8)
@modal.web_server(8000, startup_timeout=1800)
def serve():
    import subprocess
    subprocess.Popen(['python', '-m', 'steering.terminal_wrench_server',
        '--plan', '/root/terminalwrench-plan.json', '--output', '/data/calls',
        '--host', '0.0.0.0', '--port', '8000', '--modal-volume', VOLUME_NAME],
        cwd='/root/model-forensics')
