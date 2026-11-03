import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path[:0] = [str((Path(__file__).resolve().parents[1] / "src/experiments")), str((Path(__file__).resolve().parents[1] / "src/experiments") / 'emotion-steering')]
from emotion_steering.active_budget import ActiveClock
from emotion_steering.timed_proxy import TimedProxy

class LensProxyTest(unittest.TestCase):
    def setup_proxy(self,status_fn,output_fn,initial=0):
        self.now=[0.0];clock=ActiveClock(now=lambda:self.now[0]);clock.start('episode');self.now[0]=initial;clock.queue('episode','request',live=False)
        self.cancelled=[]
        call=SimpleNamespace(get=output_fn)
        worker=SimpleNamespace(complete=SimpleNamespace(spawn=lambda *a,**k:call),
            request_status=SimpleNamespace(spawn=lambda *a:SimpleNamespace(get=status_fn)),
            cancel_request=SimpleNamespace(spawn=lambda rid:self.cancelled.append(rid)))
        self.tmp=tempfile.TemporaryDirectory()
        def cleanup():
            deadline=time.monotonic()+2
            while self.cancelled and not (Path(self.tmp.name)/'late-inference/request.json').exists() and time.monotonic()<deadline:
                time.sleep(.01)
            self.tmp.cleanup()
        self.addCleanup(cleanup)
        return TimedProxy(worker,clock,self.tmp.name,'glm'),clock

    def test_installed_modal_pending_poll_does_not_cancel_request(self):
        import asyncio
        from modal._functions import _Invocation
        async def pop(**kwargs):return SimpleNamespace(outputs=[],num_unfinished_inputs=1)
        try:asyncio.run(_Invocation.poll_function(SimpleNamespace(pop_function_call_outputs=pop),timeout=0))
        except Exception as exc:pending_error=exc
        self.assertIs(type(pending_error),TimeoutError)
        polls=[0]
        def status(timeout):
            polls[0]+=1
            return {'state':'queued' if polls[0]==1 else 'completed','execution_seconds':0.0}
        def output(timeout):
            if timeout==0:raise pending_error
            return {'response':'ok','inference_timing':{'state':'completed','execution_seconds':0.0}}
        proxy,_=self.setup_proxy(status,output)
        with patch('emotion_steering.timed_proxy.time.monotonic',lambda:self.now[0]):
            result=proxy._lens('episode','request',())
        self.assertEqual(result['response'],'ok')
        self.assertEqual(self.cancelled,[])
        self.assertEqual(polls[0],2)

    def test_completed_status_crossing_budget_is_not_returned(self):
        def status(timeout):
            self.now[0]=901
            return {'state':'completed','execution_seconds':2}
        output={'response':'must not be observed','inference_timing':{'state':'completed','execution_seconds':2}}
        proxy,clock=self.setup_proxy(status,lambda timeout:output,initial=899)
        with patch('emotion_steering.timed_proxy.time.monotonic',lambda:self.now[0]):
            with self.assertRaisesRegex(TimeoutError,'active_budget_timeout'):proxy._lens('episode','request',())
        self.assertEqual(self.cancelled,['request'])
        self.assertGreaterEqual(clock.snapshot('episode')['active_seconds'],901)

    def test_failed_dispatch_is_not_hidden_by_unknown_status(self):
        def output(timeout):raise ValueError('dispatch failed')
        proxy,_=self.setup_proxy(lambda timeout:{'state':'unknown'},output)
        with patch('emotion_steering.timed_proxy.time.monotonic',lambda:self.now[0]):
            with self.assertRaisesRegex(ValueError,'dispatch failed'):proxy._lens('episode','request',())
        self.assertEqual(self.cancelled,['request'])

    def test_output_delivery_crossing_budget_is_rechecked(self):
        def status(timeout):return {'state':'completed','execution_seconds':1}
        def output(timeout):
            self.now[0]=902
            return {'response':'late','inference_timing':{'state':'completed','execution_seconds':1}}
        proxy,_=self.setup_proxy(status,output,initial=899)
        with patch('emotion_steering.timed_proxy.time.monotonic',lambda:self.now[0]):
            with self.assertRaisesRegex(TimeoutError,'active_budget_timeout'):proxy._lens('episode','request',())
        self.assertEqual(self.cancelled,['request'])

if __name__=='__main__':unittest.main()
