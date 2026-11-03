"""Concurrency contract: refill free sandbox slots without waiting for peers."""
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from terminalwrench_sweep import run_rolling


class RollingSchedulerTests(unittest.TestCase):
    def test_refills_before_slow_peer_finishes_with_maximum_four(self):
        initial=threading.Barrier(4)
        release_slow=threading.Event()
        lock=threading.Lock()
        active=0
        peak=0
        finished=[]
        refilled_before_slow=False
        def execute(item):
            nonlocal active,peak,refilled_before_slow
            with lock:
                active+=1
                peak=max(peak,active)
            try:
                if item<4:initial.wait(timeout=5)
                if item==0:
                    if not release_slow.wait(timeout=5):raise AssertionError('No rolling refill before peer finished')
                if item==4:
                    with lock:refilled_before_slow=0 not in finished
                    release_slow.set()
                with lock:finished.append(item)
            finally:
                with lock:active-=1
        run_rolling(list(range(12)),execute,4)
        self.assertEqual(peak,4)
        self.assertTrue(refilled_before_slow)
        self.assertEqual(sorted(finished),list(range(12)))

    def test_errors_are_not_silently_discarded(self):
        def execute(item):
            if item==1:raise RuntimeError('preserve this failure')
        with self.assertRaisesRegex(RuntimeError,'preserve this failure'):
            run_rolling(list(range(5)),execute,2)


if __name__=='__main__':unittest.main()
