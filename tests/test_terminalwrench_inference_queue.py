"""A blocked fake engine proves timing is observable before RPC completion."""
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import unittest

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src/experiments") / 'emotion-steering'))
from emotion_steering.inference_queue import CompletionQueue


class QueueTimingTest(unittest.TestCase):
    def test_live_execution_is_distinct_from_waiting(self):
        entered, release = threading.Event(), threading.Event()
        class Backend:
            def complete_batch(self, requests):
                entered.set()
                if not release.wait(3): raise TimeoutError('test release missing')
                return requests
        q = CompletionQueue(Backend(), batch_size=1, collect_seconds=0)
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(q.complete, {'seed': 7, 'strength': -.1}, 'first')
            self.assertTrue(entered.wait(2))
            second = pool.submit(q.complete, {'seed': 9, 'strength': .1}, 'second')
            deadline = time.monotonic() + 2
            while q.status('second')['state'] == 'unknown' and time.monotonic() < deadline:
                time.sleep(.005)
            time.sleep(.05)
            a, b = q.status('first'), q.status('second')
            self.assertEqual(a['state'], 'executing')
            self.assertGreater(a['execution_seconds'], .04)
            self.assertEqual(b['state'], 'queued')
            self.assertEqual(b['execution_seconds'], 0)
            self.assertGreater(b['queued_seconds'], .04)
            self.assertFalse(first.done())
            self.assertFalse(second.done())
            release.set()
            out_a, out_b = first.result(2), second.result(2)
        self.assertEqual(out_a['seed'], 7)
        self.assertEqual(out_b['strength'], .1)
        self.assertEqual(out_a['inference_timing']['state'], 'completed')
        self.assertGreater(out_b['inference_timing']['queued_seconds'], .04)
        final = q.status('first')['execution_seconds']
        time.sleep(.01)
        self.assertEqual(q.status('first')['execution_seconds'], final)
        with self.assertRaises(ValueError): q.complete({}, 'first')

    def test_cancel_queued_and_abandon_executing(self):
        from concurrent.futures import CancelledError
        entered, release = threading.Event(), threading.Event()
        calls = []
        class Backend:
            def complete_batch(self, requests):
                calls.extend(requests)
                entered.set()
                if not release.wait(3): raise TimeoutError('test release missing')
                return requests
        q = CompletionQueue(Backend(), batch_size=1, collect_seconds=0)
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(q.complete, {'seed': 1}, 'running')
            self.assertTrue(entered.wait(2))
            second = pool.submit(q.complete, {'seed': 2}, 'waiting')
            deadline = time.monotonic() + 2
            while q.status('waiting')['state'] == 'unknown' and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertTrue(q.cancel('waiting')['cancelled_before_execution'])
            with self.assertRaises(CancelledError): second.result(2)
            cancelled = q.status('waiting')
            self.assertEqual(cancelled['execution_seconds'], 0)
            time.sleep(.01)
            self.assertEqual(q.status('waiting')['queued_seconds'], cancelled['queued_seconds'])
            self.assertFalse(q.cancel('running')['execution_interrupt_supported'])
            self.assertEqual(q.status('running')['state'], 'executing')
            self.assertTrue(q.status('running')['abandoned'])
            release.set()
            output = first.result(2)
        self.assertEqual(output['inference_timing']['state'], 'abandoned')
        self.assertEqual(calls, [{'seed': 1}])
        q.cancel('not-yet-received')
        with self.assertRaises(ValueError): q.complete({'seed': 3}, 'not-yet-received')

    def test_four_request_isolation_and_failure_timing(self):
        class Backend:
            def __init__(self): self.batches = []
            def complete_batch(self, requests):
                self.batches.append(requests)
                if any(x.get('fail') for x in requests): raise ValueError('failed batch')
                return requests
        backend = Backend()
        q = CompletionQueue(backend, collect_seconds=.1)
        requests = [dict(seed=i, strength=(i-2)/10) for i in range(4)]
        with ThreadPoolExecutor(4) as pool:
            outputs = list(pool.map(q.complete, requests))
        self.assertEqual([len(x) for x in backend.batches], [4])
        self.assertEqual([{k:v for k,v in x.items() if k!='inference_timing'} for x in outputs], requests)
        self.assertEqual(len({x['inference_timing']['batch_id'] for x in outputs}), 1)
        with self.assertRaisesRegex(ValueError, 'failed batch'):q.complete({'fail': True}, 'failure')
        self.assertEqual(q.status('failure')['state'], 'failed')
        self.assertEqual(q.complete({'seed': 11})['seed'], 11)

if __name__ == '__main__': unittest.main()
