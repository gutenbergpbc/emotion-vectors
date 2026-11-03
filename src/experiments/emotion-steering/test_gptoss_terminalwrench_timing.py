"""Contract checks for live queue/execution timing without loading model weights."""

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
import unittest
from gptoss_terminalwrench_timing import timed_call, TimedGPTOSSMixin
from types import SimpleNamespace
from unittest.mock import patch


class TimingTests(unittest.TestCase):
    def test_queued_request_does_not_start_clock_until_lock_acquired(self):
        lock=threading.RLock();events=[];active=[]
        entered=threading.Event();release=threading.Event()
        def first(deadline):
            active.append('first');entered.set();release.wait(2);active.remove('first');return {'ok':1}
        def second(deadline):
            self.assertEqual(active,[]);active.append('second');time.sleep(.03);active.remove('second');return {'ok':2}
        t1=threading.Thread(target=lambda:list(timed_call(lock,first,'first',1)))
        t2=threading.Thread(target=lambda:events.extend(timed_call(lock,second,'second',1)))
        t1.start();self.assertTrue(entered.wait(1));t2.start();time.sleep(.06)
        self.assertEqual(events,[])
        release.set();t1.join();t2.join()
        self.assertEqual([e['event'] for e in events],['generation_started','generation_finished'])
        self.assertGreaterEqual(events[0]['worker_lock_wait_seconds'],.04)
        self.assertGreaterEqual(events[1]['active_generation_seconds'],.025)
        self.assertLess(events[1]['active_generation_seconds'],events[0]['worker_lock_wait_seconds'])

    def test_closing_consumer_before_compute_releases_lock(self):
        lock=threading.RLock();called=[]
        g=timed_call(lock,lambda d:called.append(1),'cancel',1)
        self.assertEqual(next(g)['event'],'generation_started');g.close()
        self.assertEqual(called,[])
        got=[]
        t=threading.Thread(target=lambda:got.append(lock.acquire(timeout=.1)))
        # Release from owning thread, not main, for an RLock.
        def acquire():
            ok=lock.acquire(timeout=.1);got.append(ok)
            if ok:lock.release()
        t=threading.Thread(target=acquire);t.start();t.join();self.assertEqual(got,[True])

    def test_failure_emits_elapsed_execution_boundary(self):
        def fail(deadline):
            raise RuntimeError('fixture')
        events=[]
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            for event in timed_call(threading.RLock(),fail,'error',1):events.append(event)
        self.assertEqual([e['event'] for e in events],['generation_started','generation_failed'])
        self.assertGreaterEqual(events[-1]['active_generation_seconds'],0)

    def test_remaining_budget_stops_decode_and_restores_generate(self):
        def generate(*args, stopping_criteria, **kwargs):
            while not any(c(None,None) for c in stopping_criteria):time.sleep(.002)
            return {'response':'partial'}
        class Fake(TimedGPTOSSMixin):
            def complete(self,*args):return self.model.generate()
        fake=Fake();fake.model=SimpleNamespace(generate=generate);fake.initialize_timing()
        tr=SimpleNamespace(StoppingCriteria=object,StoppingCriteriaList=list)
        with patch.dict('sys.modules',{'transformers':tr}):
            events=list(fake.complete_timed([],None,0,1,4096,'bounded',.03))
        self.assertTrue(events[-1]['output']['active_budget_stopped'])
        self.assertIs(fake.model.generate,generate)
        self.assertGreaterEqual(events[-1]['active_generation_seconds'],.03)
        self.assertLess(events[-1]['active_generation_seconds'],.1)

    def test_invalid_budget_rejected_before_compute(self):
        for budget in (0,-1,901,float('nan')):
            with self.assertRaises(ValueError):list(timed_call(threading.RLock(),lambda d:None,'bad',budget))


if __name__=='__main__':unittest.main()
