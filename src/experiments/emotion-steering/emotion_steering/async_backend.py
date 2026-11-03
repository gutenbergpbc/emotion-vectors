"""Continuous Lens serving with independent completion and cancellation."""
import asyncio
import time
from .backend import LensBackend


class AsyncLensBackend(LensBackend):
    def build_engine(self, **options):
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        options['disable_log_stats'] = False
        options['worker_extension_cls'] = 'emotion_steering.lens_worker_ext.SelectedLayerExtension'
        return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**options))

    async def initialize(self):
        # Install once before concurrent requests, avoiding first-request races.
        layers = sorted({p['layer_zero_based'] for p in self.spec.get('probes', [])})
        if layers:
            await self.llm.collective_rpc('install_selected_hooks', args=(layers,))
        else:
            await self.llm.collective_rpc('install_hooks')
        self.llm._hooks_installed = True
        self.slots = asyncio.Semaphore(self.spec['batch_size'])
        self.states = {}
        self.running = {}

    def status(self, request_id):
        item = self.states.get(request_id)
        if item is None:
            return dict(request_id=request_id, state='unknown', execution_seconds=0, queued_seconds=0)
        now = time.monotonic()
        start, end = item.get('start'), item.get('end')
        return dict(request_id=request_id, state=item['state'],
                    queued_seconds=(start or end or now)-item['received'],
                    execution_seconds=0 if start is None else (end or now)-start,
                    protocol='continuous-lens-v3')

    async def cancel(self, request_id):
        task = self.running.get(request_id)
        if task is not None:
            task.cancel()
        elif request_id not in self.states:
            self.states[request_id] = dict(state='cancelled', received=time.monotonic(), end=time.monotonic())
        return dict(execution_interrupt_supported=True, state=self.states[request_id]['state'])

    async def complete_async(self, request, request_id):
        if request_id in self.states:
            raise ValueError('Duplicate or cancelled request id')
        state = dict(state='queued', received=time.monotonic())
        self.states[request_id] = state
        self.running[request_id] = asyncio.current_task()
        try:
            prompts, params, prepared = self.prepare_requests([request])
            async with self.slots:
                state.update(state='executing', start=time.monotonic())
                final = None
                async for result in self.llm.generate(prompts[0], params[0], request_id=request_id):
                    if result.finished:
                        state['end'] = time.monotonic()
                        final = result
                if final is None:
                    raise RuntimeError('Engine did not return final output')
                output = self.decode_results([request], prepared, [final])[0]
                state['state'] = 'completed'
                return {**output, 'inference_timing': self.status(request_id)}
        except asyncio.CancelledError:
            await self.llm.abort(request_id)
            state.update(state='cancelled', end=time.monotonic())
            raise
        except BaseException:
            state.update(state='failed', end=time.monotonic())
            raise
        finally:
            self.running.pop(request_id, None)
