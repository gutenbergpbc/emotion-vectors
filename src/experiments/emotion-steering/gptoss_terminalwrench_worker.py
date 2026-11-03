"""One shared, serialized H200 allocation for interactive GPT-OSS episodes."""

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
import modal

HERE = Path(__file__).resolve().parent if modal.is_local() else Path('/study')
PROJECT = HERE.parents[2] if modal.is_local() else Path('/study')
MODEL = json.loads((HERE / 'models.json').read_text())['gpt-oss-120b']
MODEL.pop('answer_prefix', None)
MODEL['max_context'] = 65536
app = modal.App('terminalwrench-gptoss120b-interactive-steering-v1')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install_from_requirements(str(PROJECT / 'src/experiments/emotion-probes/adapters/gptoss120b/requirements-extract.txt'))
         .env({'HF_HOME': '/models', 'HF_HUB_CACHE': MODEL['hub_cache'],
               'PYTHONPATH': '/study', 'PYTHONUNBUFFERED': '1'})
         .add_local_dir(HERE / 'emotion_steering', '/study/emotion_steering', ignore=['__pycache__'])
         .add_local_file(HERE / 'gptoss_terminalwrench_backend.py', '/study/gptoss_terminalwrench_backend.py')
         .add_local_file(HERE / 'models.json', '/study/models.json')) if modal.is_local() else modal.Image.debian_slim(python_version='3.12')
secrets = [modal.Secret.from_dict({'HF_TOKEN': os.environ['HF_TOKEN']})] if os.environ.get('HF_TOKEN') else []


@app.cls(image=image, gpu='H200', cpu=16, memory=131072, timeout=1800,
         scaledown_window=300, max_containers=1,
         volumes={'/models': modal.Volume.from_name(MODEL['cache_volume'], create_if_missing=True)}, secrets=secrets)
class GPTOSSTerminalWrench:
    @modal.enter()
    def load(self):
        from gptoss_terminalwrench_backend import TerminalWrenchGPTOSS
        self.backend = TerminalWrenchGPTOSS({'model': MODEL, 'temperature': 0.7, 'top_p': 0.95})

    @modal.method()
    def complete(self, messages, probe, strength, seed, max_tokens):
        return self.backend.complete(messages, probe, strength, seed, max_tokens)

    @modal.method()
    def info(self):
        return {'model': MODEL, 'max_context': self.backend.max_context,
                'serialized': True, 'channels': 'native analysis/final'}
