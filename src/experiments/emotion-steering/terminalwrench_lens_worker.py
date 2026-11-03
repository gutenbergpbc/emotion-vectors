"""Persistent, request-isolated Lens inference for interactive Modal trials.

Deploy once per model with EMOTION_STEERING_RUN and TERMINALWRENCH_WORKER_APP.
Concurrent single-turn RPCs are coalesced into batches of at most four.
"""

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

import modal
import modal_worker as base

app = modal.App(os.environ.get("TERMINALWRENCH_WORKER_APP", "terminalwrench-glm53-active-time-v2"))
image = base.image.add_local_file(__file__, "/study/src/experiments/emotion-steering/terminalwrench_lens_worker.py")


from emotion_steering.inference_queue import CompletionQueue


@app.cls(image=image, gpu=base.MODEL["gpu"], cpu=16, memory=131072,
         timeout=3600, startup_timeout=1800, volumes=base.volumes,
         secrets=[base.secret], max_containers=1, scaledown_window=120)
# Reserve control RPC capacity while four episode calls wait for inference.
@modal.concurrent(max_inputs=12)
class LensInteractive:
    @modal.enter()
    def start(self):
        from emotion_steering.backend import LensBackend
        spec = base.SPEC
        if spec["model"]["max_context"] != 65536 or spec["batch_size"] != 4:
            raise ValueError("Interactive plan requires 65536 context and batch size four")
        self.queue = CompletionQueue(LensBackend(spec))

    @modal.method()
    def info(self):
        return {"model": base.MODEL["name"], "revision": base.MODEL["revision"],
                "max_context": base.MODEL["max_context"], "batch_size": 4,
                "identity": base.SPEC["identity"],
                "timing_protocol": "active-time-v2",
                "live_request_status": True, "queued_cancellation": True}

    @modal.method()
    def complete(self, messages, probe, strength, seed, max_tokens, request_id=None):
        if max_tokens != 4096:
            raise ValueError("Interactive plan requires 4096 generated tokens")
        return self.queue.complete(dict(messages=messages, probe=probe,
                                       strength=strength, seed=seed, max_tokens=max_tokens), request_id=request_id)

    @modal.method()
    def request_status(self, request_id):
        return self.queue.status(request_id)

    @modal.method()
    def cancel_request(self, request_id):
        return self.queue.cancel(request_id)
