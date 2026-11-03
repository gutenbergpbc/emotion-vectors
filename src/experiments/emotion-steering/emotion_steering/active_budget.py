"""Live active-time clock. Only dispatched inference queue intervals are excluded."""
import asyncio
import json
import threading
import time
import urllib.request
import urllib.parse
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from shared.io import atomic

ACTIVE_SECONDS=900
QUEUE_WATCHDOG_SECONDS=5400
WALL_WATCHDOG_SECONDS=21600
TIMING_VERSION='terminalwrench-active-clock-v2.2'

class ActiveClock:
    def __init__(self,now=time.monotonic,state_dir=None):
        self.now=now;self.lock=threading.RLock();self.episodes={};self.requests={};self.request_episodes={}
        self.state_dir=Path(state_dir) if state_dir else None
    def _publish(self,key,force=False):
        if self.state_dir is None:return
        episode=self.episodes[key];now=self.now()
        if not force and now-episode.get('_state_published_at',float('-inf'))<.25:return
        state={k:v for k,v in self.snapshot(key).items() if k!='requests'}
        state['observed_monotonic']=now
        state['inference_pending']=any(q['end'] is None for q in episode['queues'])
        if 'final' in episode:state['stopped']=True
        atomic(self.state_dir/f'{key}.json',state)
        episode['_state_published_at']=now
    def start(self,key):
        with self.lock:
            if key in self.episodes:raise ValueError('Episode clock already started')
            self.episodes[key]={'started':self.now(),'queues':[],'failure':None}
            self._publish(key,force=True)
            return self.snapshot(key)
    def queue(self,key,request_id,live=True):
        with self.lock:
            if key not in self.episodes:raise ValueError('Agent clock must start before inference')
            if 'final' in self.episodes[key]:raise ValueError('Episode already stopped; rejecting late inference')
            q={'request_id':request_id,'start':self.now(),'end':None if live else self.now(),'state':'queued','execution_seconds':0.0}
            self.episodes[key]['queues'].append(q);self.requests[request_id]=q;self.request_episodes[request_id]=key
            self._publish(key,force=True)
    def confirm_queued(self,request_id,confirmed_until):
        with self.lock:
            q=self.requests[request_id]
            if q['state']=='queued' and q['end'] is not None:
                old_end=q['end'];q['end']=max(q['end'],confirmed_until)
                if q['end']>old_end:self._publish(self.request_episodes[request_id],force=False)
    def _episode_for_request(self,request_id):
        return self.request_episodes[request_id]
    def executing(self,request_id,execution_seconds=0.0,observed_at=None):
        with self.lock:
            q=self.requests[request_id]
            if q['state']=='queued':q['end']=max(q['start'],(self.now() if observed_at is None else observed_at)-execution_seconds)
            q['state']='executing';q['execution_seconds']=max(q['execution_seconds'],execution_seconds)
            self._publish(self._episode_for_request(request_id),force=True)
    def finish(self,request_id,execution_seconds=None):
        with self.lock:
            q=self.requests[request_id];now=self.now()
            if q['end'] is None:q['end']=now
            if execution_seconds is not None:
                q['execution_seconds']=max(q['execution_seconds'],execution_seconds)
                # Reconcile delayed signal delivery; never forgive known generation.
                q['end']=min(q['end'],max(q['start'],now-execution_seconds))
            q['state']='finished'
            self._publish(self._episode_for_request(request_id),force=True)
    def fail(self,key,error):
        with self.lock:self.episodes[key]['failure']=error;self._publish(key,force=True)
    def stop(self,key):
        with self.lock:
            if 'final' not in self.episodes[key]:self.episodes[key]['final']={**self.snapshot(key,include_requests=True),'stopped':True}
            self._publish(key,force=True)
            return self.episodes[key]['final']
    def snapshot(self,key,include_requests=False):
        with self.lock:
            episode=self.episodes[key]
            if 'final' in episode:return episode['final']
            now=self.now();wall=max(0.0,now-episode['started'])
            excluded=sum(max(0.0,min(now,q['end'] if q['end'] is not None else now)-max(episode['started'],q['start'])) for q in episode['queues'])
            if excluded>wall+1e-6:raise ValueError('Overlapping or invalid inference queue intervals')
            pending=[q for q in episode['queues'] if q['state']=='queued']
            if any(now-q['start']>QUEUE_WATCHDOG_SECONDS for q in pending):episode['failure']='queue_watchdog_timeout'
            result={'timing_version':TIMING_VERSION,'wall_seconds':wall,'excluded_queue_seconds':excluded,'active_seconds':max(0.0,wall-excluded),'remaining_active_seconds':max(0.0,ACTIVE_SECONDS-(wall-excluded)),'failure':episode['failure']}
            # Harbor polls this endpoint frequently. Returning the entire request
            # history on every poll made response cost grow with episode length.
            if include_requests:result['requests']=[dict(q) for q in episode['queues']]
            return result


def serve_clock(clock,port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):self.respond(True)
        def do_GET(self):self.respond(False)
        def respond(self,start):
            try:
                parsed=urllib.parse.urlsplit(self.path);key=parsed.path.removeprefix('/clock/')
                action=urllib.parse.parse_qs(parsed.query).get('action',[''])[0]
                data=clock.stop(key) if action=='stop' else clock.start(key) if start else clock.snapshot(key);status=200
            except Exception as exc:data={'error':repr(exc)};status=500
            body=json.dumps(data).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    class ClockHTTPServer(ThreadingHTTPServer):
        allow_reuse_address=True
        request_queue_size=128
        daemon_threads=True
    ClockHTTPServer(('127.0.0.1',port),Handler).serve_forever()


def read_clock(url,start=False):
    request=urllib.request.Request(url,method='POST' if start else 'GET',data=b'' if start else None)
    with urllib.request.urlopen(request,timeout=5) as response:return json.load(response)

def read_clock_file(path,now=time.monotonic):
    state=json.loads(Path(path).read_text())
    elapsed=max(0.0,now()-state['observed_monotonic'])
    if not state.get('stopped') and (not state.get('inference_pending') or elapsed>2.0):
        state['active_seconds']+=elapsed
    state['remaining_active_seconds']=max(0.0,ACTIVE_SECONDS-state['active_seconds'])
    return state

# Imported only by the pinned Harbor process, keeping the coordinator lightweight.
try:
    from harbor.agents.terminus_2.terminus_2 import Terminus2
    from harbor.trial.errors import AgentTimeoutError
except ImportError:
    Terminus2=object
    AgentTimeoutError=RuntimeError

class ActiveBudgetTimeoutError(RuntimeError):pass
class BudgetInfrastructureError(RuntimeError):pass

class ActiveTimeTerminus2(Terminus2):
    def __init__(self,*args,budget_url=None,budget_file=None,**kwargs):
        self.budget_url=budget_url;self.budget_file=Path(budget_file) if budget_file else None
        self.budget_logs_dir=Path(kwargs.get('logs_dir',args[0] if args else '.'))
        if not budget_url and not budget_file:raise ValueError('Missing active budget endpoint')
        super().__init__(*args,**kwargs)
    async def _clock(self,start=False):
        if not start and self.budget_file is not None:
            try:return await asyncio.to_thread(read_clock_file,self.budget_file)
            except Exception as file_error:
                if not self.budget_url:raise BudgetInfrastructureError('Active clock snapshot unavailable: '+repr(file_error)) from file_error
                try:return await asyncio.to_thread(read_clock,self.budget_url,False)
                except Exception as network_error:
                    raise BudgetInfrastructureError('Active clock snapshot and endpoint unavailable: '+repr(file_error)+'; '+repr(network_error)) from network_error
        if not self.budget_url:raise BudgetInfrastructureError('Missing active clock start/stop endpoint')
        attempts=1 if start else 3
        last_error=None
        for attempt in range(attempts):
            try:return await asyncio.to_thread(read_clock,self.budget_url,start)
            except (TimeoutError,OSError) as exc:
                last_error=exc
                if attempt+1<attempts:await asyncio.sleep(.2*(attempt+1))
            except Exception as exc:
                raise BudgetInfrastructureError('Active clock unavailable: '+repr(exc)) from exc
        raise BudgetInfrastructureError(f'Active clock unavailable after {attempts} read attempts: '+repr(last_error)) from last_error
    def _validate_budget(self,state):
        if state['failure']:raise BudgetInfrastructureError(state['failure'])
        if state['active_seconds']>=ACTIVE_SECONDS:raise ActiveBudgetTimeoutError('active_budget_timeout: 900 active seconds; inference queue excluded')
    async def run(self,*args,**kwargs):
        await self._clock(True)
        work=asyncio.create_task(super().run(*args,**kwargs))
        try:
            while not work.done():
                self._validate_budget(await self._clock())
                # A half-second poll is enough for the 900-second budget and
                # avoids flooding the local clock service at high concurrency.
                await asyncio.wait([work],timeout=.5)
            result=await work  # Preserve an original model/agent exception.
            self._validate_budget(await self._clock())
            return result
        finally:
            if not work.done():work.cancel()
            await asyncio.gather(work,return_exceptions=True)
            try:
                final=await asyncio.to_thread(read_clock,self.budget_url+'?action=stop',True)
                from shared.io import atomic
                atomic(self.budget_logs_dir/'active-budget.json',final)
            except Exception as exc:
                # Do not overwrite an existing agent/timeout error during cleanup.
                (self.budget_logs_dir/'active-budget-cleanup-error.txt').write_text(repr(exc))
