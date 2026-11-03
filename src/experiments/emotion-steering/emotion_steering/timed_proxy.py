"""Bridge live worker timing into the local Harbor active-time clock."""
import asyncio
import json
import threading
import time
import uuid
from pathlib import Path
from shared.io import atomic
from .terminalwrench import alias

class TimedProxy:
    def __init__(self,worker,clock,root,family):self.worker=worker;self.clock=clock;self.root=Path(root);self.family=family
    def complete_for_task(self,task,*args):
        key=alias(task);request_id=uuid.uuid4().hex
        self.clock.queue(key,request_id,live=self.family=='gpt_oss')
        try:
            if self.family=='gpt_oss':return asyncio.run(self._gpt(key,request_id,args))
            return self._lens(key,request_id,args)
        except Exception as exc:
            if any(term in str(exc).lower() for term in ('out of memory', 'enginecore', 'enginedead', 'cuda error', 'worker died')):
                atomic(self.root/'worker-restart-required.json', {'request_id':request_id,'error':repr(exc),'at':time.time()})
            if 'active_budget_timeout' not in str(exc):self.clock.fail(key,'inference_failure: '+repr(exc))
            atomic(self.root/'failed-inference'/f'{request_id}.json',{'task':task,'error':repr(exc),'clock':self.clock.snapshot(key,include_requests=True)})
            raise
        finally:self.clock.finish(request_id)
    def _check(self,key):
        state=self.clock.snapshot(key)
        if state.get('stopped'):raise RuntimeError('Episode stopped; cancel outstanding request')
        if state['failure']:raise RuntimeError(state['failure'])
        if state['active_seconds']>=900:raise TimeoutError('active_budget_timeout')
        return state
    async def _gpt(self,key,request_id,args):
        remaining=self.clock.snapshot(key)['remaining_active_seconds']
        iterator=self.worker.complete_timed.remote_gen.aio(*args,request_id=request_id,active_seconds_remaining=remaining)
        pending=None
        events=[]
        primary_error=None
        event_path=self.root/'timing-events'/f'{request_id}.json'
        try:
            while True:
                pending=asyncio.create_task(anext(iterator))
                while not pending.done():
                    self._check(key)
                    await asyncio.wait([pending],timeout=.1)
                event=await pending
                events.append({'received_monotonic':time.monotonic(),'worker_event':event})
                atomic(event_path,{'request_id':request_id,'episode':key,
                                  'record_type':'worker_events_not_agent_trajectory','events':events})
                if event['event']=='generation_started':self.clock.executing(request_id)
                elif event['event']=='generation_finished':
                    duration=event['active_generation_seconds'];self.clock.finish(request_id,duration)
                    self._check(key)
                    output=event['output'];output['inference_timing']={'request_id':request_id,'active_generation_seconds':duration,'protocol':'timed_generator'}
                    return output
                elif event['event']=='generation_failed':
                    self.clock.finish(request_id,event.get('active_generation_seconds'))
                    raise RuntimeError(event.get('error','generation_failed'))
                else:raise ValueError('Unknown GPT timing event: '+repr(event.get('event')))
        except BaseException as exc:
            primary_error=exc
            raise
        finally:
            try:
                if pending is not None and not pending.done():
                    pending.cancel();await asyncio.gather(pending,return_exceptions=True)
                await iterator.aclose()
            except BaseException as cleanup_error:
                try:
                    atomic(self.root/'timing-cleanup-errors'/f'{request_id}.json',
                           {'request_id':request_id,'cleanup_error':repr(cleanup_error),
                            'primary_error':repr(primary_error) if primary_error else None})
                except Exception:
                    pass  # Cleanup diagnostics must not mask the original failure.
                if primary_error is None:raise
    def _lens(self,key,request_id,args):
        from modal.exception import TimeoutError as ModalGetTimeout
        call=None
        def finish_output(output, observed_at):
            timing=output.get('inference_timing')
            if not timing or timing.get('state') != 'completed':
                raise RuntimeError('Missing final Lens timing receipt')
            self.clock.executing(request_id,timing['execution_seconds'],observed_at=observed_at)
            self.clock.finish(request_id,timing['execution_seconds'])
            self._check(key)  # Never expose output that arrived after budget expiry.
            return output
        try:
            call=self.worker.complete.spawn(*args,request_id=request_id)
            while True:
                self._check(key)
                queried_at=time.monotonic()
                control=self.worker.request_status.spawn(request_id)
                try:
                    state=control.get(timeout=5)
                except TimeoutError as exc:
                    # Modal's local poll timeout only means the control RPC has
                    # not returned yet. Keep the inference request and episode alive.
                    # Remote function timeouts/errors must still propagate.
                    if type(exc) is not TimeoutError:
                        raise
                    self._check(key)
                    time.sleep(.1)
                    continue
                if state['state'] in ('executing','completed','failed','abandoned'):
                    self.clock.executing(request_id,state.get('execution_seconds',0),observed_at=queried_at)
                if state['state'] in ('unknown','queued'):self.clock.confirm_queued(request_id,queried_at)
                self._check(key)  # A control query can cross the active deadline.
                if state['state'] in ('completed','failed','abandoned','cancelled'):
                    return finish_output(call.get(timeout=5),queried_at)
                if state['state'] in ('unknown','queued'):
                    try:
                        output=call.get(timeout=0)
                    except (TimeoutError, ModalGetTimeout) as exc:
                        # Modal 1.6.1 poll_function raises builtin TimeoutError
                        # when outputs are not ready. A remote FunctionTimeoutError
                        # subclasses ModalGetTimeout and must still propagate.
                        if type(exc) not in (TimeoutError, ModalGetTimeout):raise
                    else:
                        return finish_output(output,queried_at)
                time.sleep(.1)
        except Exception:
            try:self.worker.cancel_request.spawn(request_id)
            except Exception as cancel_error:
                atomic(self.root/'cancellation-errors'/f'{request_id}.json',{'error':repr(cancel_error)})
            # Running Lens batches cannot interrupt one member. Preserve any late
            # output separately; it is never an agent-observed trajectory response.
            def collect_late():
                try:
                    output=call.get(timeout=3600)
                    atomic(self.root/'late-inference'/f'{request_id}.json',{'request_id':request_id,'agent_observed':False,'output':output})
                except BaseException as exc:atomic(self.root/'late-inference'/f'{request_id}.json',{'request_id':request_id,'agent_observed':False,'error':repr(exc)})
            if call is not None:threading.Thread(target=collect_late,daemon=True).start()
            raise
