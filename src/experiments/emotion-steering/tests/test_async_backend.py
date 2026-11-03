import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotion_steering.async_backend import AsyncLensBackend


class FakeEngine:
    def __init__(self):
        self.release = asyncio.Event()
        self.aborted = []
    async def collective_rpc(self, *args, **kwargs):
        pass
    async def generate(self, prompt, params, request_id):
        if request_id == 'slow':
            await self.release.wait()
        yield SimpleNamespace(finished=True, request_id=request_id)
    async def abort(self, request_id):
        self.aborted.append(request_id)


def backend(slots):
    b = AsyncLensBackend.__new__(AsyncLensBackend)
    b.spec = {'batch_size': slots}
    b.llm = FakeEngine()
    b.prepare_requests = lambda requests: ([{}], [{}], [{}])
    b.decode_results = lambda req, prep, results: [{'response': results[0].request_id}]
    return b


def test_short_request_returns_before_long_request():
    async def run():
        b = backend(2)
        await b.initialize()
        slow = asyncio.create_task(b.complete_async({}, 'slow'))
        await asyncio.sleep(0)
        fast = await asyncio.wait_for(b.complete_async({}, 'fast'), 1)
        assert fast['response'] == 'fast' and not slow.done()
        seconds = b.status('fast')['execution_seconds']
        await asyncio.sleep(.01)
        assert b.status('fast')['execution_seconds'] == seconds
        # Replacement work can run while the earlier slow request is still active.
        assert (await b.complete_async({}, 'replacement'))['response'] == 'replacement'
        b.llm.release.set()
        await slow
    asyncio.run(run())


def test_queue_time_and_executing_cancellation():
    async def run():
        b = backend(1)
        await b.initialize()
        slow = asyncio.create_task(b.complete_async({}, 'slow'))
        await asyncio.sleep(0)
        waiting = asyncio.create_task(b.complete_async({}, 'waiting'))
        await asyncio.sleep(.01)
        status = b.status('waiting')
        assert status['state'] == 'queued' and status['execution_seconds'] == 0
        await b.cancel('slow')
        try:
            await slow
        except asyncio.CancelledError:
            pass
        assert b.llm.aborted == ['slow']
        assert (await waiting)['response'] == 'waiting'
        await b.cancel('cancelled-before-arrival')
        try:
            await b.complete_async({}, 'cancelled-before-arrival')
        except ValueError:
            pass
        else:
            raise AssertionError('Cancellation tombstone ignored')
    asyncio.run(run())
