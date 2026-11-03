"""GPT-OSS native-channel runtime for the pinned interactive sweep."""

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

import threading
import time
from emotion_steering.backend import HFBackend


class TerminalWrenchGPTOSS(HFBackend):
    def __init__(self, spec):
        if spec['model'].get('answer_prefix'):
            raise ValueError('Interactive GPT-OSS must use native analysis/final channels')
        super().__init__(spec)
        self.max_context = min(self.max_context, 65536)
        self._lock = threading.Lock()

    def complete(self, messages, probe, strength, seed, max_tokens):
        if not 0 < max_tokens <= 4096:
            raise ValueError('Generation budget must be at most 4096 including reasoning')
        if strength and probe is None:
            raise ValueError('Nonzero steering requires a probe')
        if probe and len(probe['vector']) != 2880:
            raise ValueError('GPT-OSS requires its own 2880-dimensional vector')
        with self._lock:
            receipt = {'enabled': bool(strength), 'strength': strength,
                       'layer_zero_based': probe['layer_zero_based'] if probe and strength else None,
                       'vector_sha256': probe.get('vector_sha256') if probe and strength else None,
                       'hook_calls': 0, 'tokens_steered': 0,
                       'prefill_tokens': 0, 'decode_tokens': 0}
            handle = None
            if strength:
                # Register before the canonical intervention: record actual forward calls.
                def observe(module, inputs, output):
                    h = output[0] if isinstance(output, tuple) else output
                    n = h.shape[-2]
                    receipt['hook_calls'] += 1
                    receipt['tokens_steered'] += n
                    receipt['prefill_tokens' if receipt['hook_calls'] == 1 else 'decode_tokens'] += n
                handle = self.layers[probe['layer_zero_based']].register_forward_hook(observe)
            started = time.monotonic()
            try:
                result = super().complete(messages, probe, strength, seed, max_tokens)
            finally:
                if handle is not None:
                    handle.remove()
            if strength and not receipt['hook_calls']:
                raise RuntimeError('Steered generation did not traverse selected layer')
            return {**result, 'steering_receipt': receipt,
                    'inference_seconds': time.monotonic() - started,
                    'max_context': self.max_context, 'reasoning_effort': 'low'}
