"""Live queue/execution boundaries for one serialized GPT-OSS worker.

The generator acquires the backend lock before emitting generation_started. A
consumer may safely pause its episode clock before that event, and must resume
it immediately on receipt. Queue/startup require a separate wall-time watchdog.
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

import threading
import time


def timed_call(lock, call, request_id, active_seconds_remaining):
    """Stream boundaries around call(deadline); never emit start while queued."""
    if not isinstance(active_seconds_remaining, (float, int)) or not 0 < active_seconds_remaining <= 900:
        raise ValueError('active_seconds_remaining must be in (0, 900]')
    queued_monotonic = time.monotonic()
    with lock:
        started = time.monotonic()
        deadline = started + active_seconds_remaining
        start_wall = time.time()
        yield {'event': 'generation_started', 'request_id': request_id,
               'wall_time': start_wall, 'monotonic': started,
               'worker_lock_wait_seconds': started - queued_monotonic,
               'active_seconds_remaining': active_seconds_remaining}
        try:
            result = call(deadline)
        except Exception as exc:
            ended = time.monotonic()
            yield {'event': 'generation_failed', 'request_id': request_id,
                   'wall_time': time.time(), 'monotonic': ended,
                   'active_generation_seconds': ended - started,
                   'error': type(exc).__name__ + ': ' + str(exc)}
            raise
        ended = time.monotonic()
        yield {'event': 'generation_finished', 'request_id': request_id,
               'wall_time': time.time(), 'monotonic': ended,
               'active_generation_seconds': ended - started,
               'output': result}


class TimedGPTOSSMixin:
    def initialize_timing(self):
        # The canonical complete() reenters this lock on the same thread; the
        # outer acquisition is what guarantees no lock wait after start.
        self._lock = threading.RLock()

    def complete_timed(self, messages, probe, strength, seed, max_tokens,
                       request_id, active_seconds_remaining):
        def generate(deadline):
            from transformers import StoppingCriteria, StoppingCriteriaList
            original = self.model.generate
            budget_stop = {'stopped': False}

            class DeadlineStop(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    stopped = time.monotonic() >= deadline
                    budget_stop['stopped'] |= stopped
                    return stopped

            def bounded_generate(*args, **kwargs):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Active budget exhausted during prompt preparation')
                criteria = list(kwargs.pop('stopping_criteria', []) or [])
                kwargs['stopping_criteria'] = StoppingCriteriaList([*criteria, DeadlineStop()])
                return original(*args, **kwargs)

            self.model.generate = bounded_generate
            try:
                result = self.complete(messages, probe, strength, seed, max_tokens)
                result['active_budget_stopped'] = budget_stop['stopped']
                result['timing_policy'] = '900s active; startup and inference queue excluded'
                return result
            finally:
                self.model.generate = original

        yield from timed_call(self._lock, generate, request_id, active_seconds_remaining)
