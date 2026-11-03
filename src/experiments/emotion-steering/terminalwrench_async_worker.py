"""Bounded continuous serving; no whole-batch completion barrier."""

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

import os
import uuid
import modal
import modal_worker as base

app = modal.App(os.environ.get('TERMINALWRENCH_WORKER_APP', 'terminalwrench-async-v3'))
image = base.image.add_local_file(__file__, '/study/src/experiments/emotion-steering/terminalwrench_async_worker.py')

@app.cls(image=image, gpu=base.MODEL['gpu'], cpu=16, memory=131072,
         timeout=3600, startup_timeout=1800, volumes=base.volumes,
         secrets=[base.secret], max_containers=1, scaledown_window=120)
@modal.concurrent(max_inputs=64)
class AsyncInteractive:
    @modal.enter()
    async def start(self):
        from emotion_steering.async_backend import AsyncLensBackend
        spec = dict(base.SPEC)
        limit = int(os.environ.get('TERMINALWRENCH_INFERENCE_LIMIT', spec['batch_size']))
        if not 1 <= limit <= spec['batch_size']:raise ValueError('Invalid fallback limit')
        spec['batch_size'] = limit
        self.backend = AsyncLensBackend(spec)
        await self.backend.initialize()

    @modal.method()
    async def info(self):
        return dict(model=base.MODEL['name'], identity=base.SPEC['identity'],
                    max_context=base.MODEL['max_context'], batch_size=self.backend.spec['batch_size'],
                    timing_protocol='continuous-lens-v3', live_request_status=True,
                    queued_cancellation=True, execution_cancellation=True)

    @modal.method()
    async def complete(self, messages, probe, strength, seed, max_tokens, request_id=None):
        if not 0 < max_tokens <= 4096:
            raise ValueError('Invalid generation budget')
        return await self.backend.complete_async(dict(messages=messages, probe=probe,
            strength=strength, seed=seed, max_tokens=max_tokens), request_id or uuid.uuid4().hex)

    @modal.method()
    async def request_status(self, request_id):
        return self.backend.status(request_id)

    @modal.method()
    async def cancel_request(self, request_id):
        return await self.backend.cancel(request_id)
