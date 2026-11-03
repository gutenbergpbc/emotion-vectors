"""GPT proxy live-clock and generator cleanup contracts, with no remote calls."""

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

import asyncio
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from emotion_steering.timed_proxy import TimedProxy

class Clock:
    def __init__(self,exhaust_on_start=False):self.active=0;self.started=False;self.exhaust_on_start=exhaust_on_start
    def snapshot(self,key):return {'remaining_active_seconds':max(0,900-self.active),'active_seconds':self.active,'failure':None}
    def executing(self,request):
        self.started=True
        if self.exhaust_on_start:self.active=901
    def finish(self,request,duration=None):
        if duration is not None:self.active=max(self.active,duration)

class ProxyTests(unittest.TestCase):
    def run_case(self,iterator,clock,root):
        worker=SimpleNamespace(complete_timed=SimpleNamespace(remote_gen=SimpleNamespace(aio=lambda *a,**kw:iterator)))
        proxy=TimedProxy(worker,clock,root,'gpt_oss')
        return asyncio.run(proxy._gpt('episode','request',([],None,0,1,4096)))

    def test_events_saved_and_reconciled_before_result(self):
        async def events():
            yield {'event':'generation_started'}
            yield {'event':'generation_finished','active_generation_seconds':.5,'output':{'response':'action'}}
        with tempfile.TemporaryDirectory() as tmp:
            clock=Clock();out=self.run_case(events(),clock,Path(tmp))
            self.assertTrue(clock.started);self.assertEqual(clock.active,.5);self.assertEqual(out['response'],'action')
            saved=json.loads((Path(tmp)/'timing-events/request.json').read_text());self.assertEqual(len(saved['events']),2)

    def test_never_finishing_generation_still_exhausts_live_clock(self):
        closed=[]
        async def events():
            try:
                yield {'event':'generation_started'}
                await asyncio.sleep(10)
            finally:closed.append(True)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(TimeoutError,'active_budget_timeout'):
                self.run_case(events(),Clock(True),Path(tmp))
        self.assertTrue(closed)

    def test_cleanup_error_does_not_mask_worker_failure(self):
        class Broken:
            async def __anext__(self):return {'event':'generation_failed','active_generation_seconds':.1,'error':'original-worker-error'}
            async def aclose(self):raise RuntimeError('cleanup-error')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'original-worker-error'):
                self.run_case(Broken(),Clock(),Path(tmp))
            self.assertTrue((Path(tmp)/'timing-cleanup-errors/request.json').exists())

    def test_reconciled_over_budget_output_is_not_returned(self):
        async def events():
            yield {'event':'generation_started'}
            yield {'event':'generation_finished','active_generation_seconds':901,'output':{'response':'late'}}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(TimeoutError,'active_budget_timeout'):
                self.run_case(events(),Clock(),Path(tmp))
            saved=json.loads((Path(tmp)/'timing-events/request.json').read_text())
            self.assertEqual(saved['events'][-1]['worker_event']['output']['response'],'late')

if __name__=='__main__':unittest.main()
