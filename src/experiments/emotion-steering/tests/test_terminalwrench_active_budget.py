import asyncio
import contextlib
import datetime
import sys
import tempfile
import types
import unittest
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[2])]
from emotion_steering.active_budget import ActiveClock,ActiveBudgetTimeoutError,ActiveTimeTerminus2,read_clock_file

class ClockTests(unittest.TestCase):
    def setUp(self):self.t=0.;self.clock=ActiveClock(lambda:self.t);self.clock.start('episode')
    def test_queue_excluded_generation_and_terminal_count_live(self):
        self.t=10;self.clock.queue('episode','request')
        self.t=510;self.assertEqual(self.clock.snapshot('episode')['active_seconds'],10)
        self.clock.executing('request')
        self.t=1310;self.assertEqual(self.clock.snapshot('episode')['active_seconds'],810)
        self.clock.finish('request',800)
        self.t=1400;self.assertEqual(self.clock.snapshot('episode')['active_seconds'],900)
        with self.assertRaises(ActiveBudgetTimeoutError):ActiveTimeTerminus2._validate_budget(None,self.clock.snapshot('episode'))
    def test_generation_never_returns_still_exhausts_active_budget(self):
        self.clock.queue('episode','request');self.t=1000;self.clock.executing('request')
        self.t=1901
        with self.assertRaises(ActiveBudgetTimeoutError):ActiveTimeTerminus2._validate_budget(None,self.clock.snapshot('episode'))
    def test_lens_delayed_status_conservatively_counts_transport(self):
        self.clock.queue('episode','request',live=False)
        self.t=100;self.clock.confirm_queued('request',99)
        self.t=130;self.clock.executing('request',20,observed_at=125)
        self.assertEqual(self.clock.snapshot('episode')['active_seconds'],25)
        self.t=200;self.clock.finish('request',85)
        self.assertGreaterEqual(self.clock.snapshot('episode')['active_seconds'],85)
    def test_final_clock_freezes_and_rejects_late_request(self):
        self.t=30;final=self.clock.stop('episode');self.t=999
        self.assertEqual(self.clock.snapshot('episode'),final)
        with self.assertRaisesRegex(ValueError,'already stopped'):self.clock.queue('episode','late')
    def test_shared_snapshot_tracks_active_and_inference_time_without_http_polling(self):
        with tempfile.TemporaryDirectory() as directory:
            t=[0.0];clock=ActiveClock(lambda:t[0],state_dir=directory)
            clock.start('episode');t[0]=20
            self.assertEqual(read_clock_file(Path(directory)/'episode.json',lambda:t[0])['active_seconds'],20)
            clock.queue('episode','request',live=True);t[0]=20.5
            pending=read_clock_file(Path(directory)/'episode.json',lambda:t[0])
            self.assertEqual(pending['active_seconds'],20)
            self.assertNotIn('requests',pending)
            clock.executing('request',execution_seconds=2,observed_at=25);t[0]=30
            self.assertEqual(read_clock_file(Path(directory)/'episode.json',lambda:t[0])['active_seconds'],27)

class HarborClassificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_trial_does_not_rewrite_active_timeout_as_wall_watchdog(self):
        from harbor.trial.trial import Trial
        @contextlib.asynccontextmanager
        async def policy(*args,**kwargs):yield
        async def emit(*args):pass
        async def run(**kwargs):raise ActiveBudgetTimeoutError('active_budget_timeout:900')
        env=types.SimpleNamespace(with_default_user=lambda user:contextlib.nullcontext(),scoped_exec_env=lambda extra:contextlib.nullcontext())
        fake=types.SimpleNamespace(_emit=emit,_now=lambda:datetime.datetime.now(datetime.timezone.utc),_network_plan=lambda cfg:types.SimpleNamespace(agent_env_baseline=None,agent_phase=None),agent_environment=env,_phase_network_policy=policy,_log_context=lambda *args:contextlib.nullcontext(),agent=types.SimpleNamespace(extra_env={},run=run))
        target=types.SimpleNamespace()
        with self.assertRaisesRegex(ActiveBudgetTimeoutError,'active_budget_timeout'):
            await Trial._run_agent_phase(fake,target=target,instruction='test',timeout_sec=21600,user=None)

if __name__=='__main__':unittest.main()
