"""Adopt only completed native trials at a quiescent boundary; never kill agents."""

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

import argparse
import ctypes
import datetime
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent)]
from shared.io import atomic,sha
from emotion_steering.terminalwrench import alias


def argv(pid):return [x.decode() for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]
def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False

def live_children(pid):
    output=subprocess.run(['ps','--ppid',str(pid),'-o','pid=,stat=,args='],text=True,capture_output=True).stdout
    return [line for line in output.splitlines() if not line.split()[1].startswith('Z')]

def active_http_connections(command):
    port=int(command[command.index('--port')+1]) if '--port' in command else 8100
    return [line.split()[1:4] for table in ('/proc/net/tcp','/proc/net/tcp6') for line in Path(table).read_text().splitlines()[1:] if int(line.split()[1].split(':')[1],16)==port and line.split()[3] not in ('0A','06')]

def native_finished(root):
    trials=list((root/'native-trials').glob('steering-*'))
    if not trials:return None
    for trial in trials:
        results=list(trial.glob('jobs/*/*/result.json'))
        if len(results)!=1:return None
        try:json.loads(results[0].read_text())
        except (OSError,json.JSONDecodeError):return None
    return trials


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--stop-only',action='store_true');a=p.parse_args();root=a.root.resolve()
    pid=int((root/'coordinator.pid').read_text());command=argv(pid)
    finishpid=int((root/'finish.pid').read_text());finishcommand=argv(finishpid)
    model_key=command[command.index('--model')+1]
    status_path=HERE.parents[2]/'dashboards/terminalwrench-three-models/status'/f'{model_key.replace("_","-")}.json'
    libc=ctypes.CDLL(None,use_errno=True);notify_fd=libc.inotify_init1(os.O_NONBLOCK)
    if notify_fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
    watched=set();stopped=False;heartbeat_stop=threading.Event()
    def migration_status(message,status='waiting'):
        data=json.loads(status_path.read_text());data.update(status=status,stage='Scheduler migration / model warmup',message=message,running_episodes=0,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat());atomic(status_path,data)
    def heartbeat():
        while not heartbeat_stop.wait(10):migration_status('All native episodes finished; warming the identical model before rolling scheduling.')
    try:
        while alive(pid):
            for directory in [root/'episodes',*root.glob('native-trials/*/jobs/*/*')]:
                if directory.is_dir() and directory not in watched:
                    libc.inotify_add_watch(notify_fd,os.fsencode(directory),128|8);watched.add(directory)
            trials=native_finished(root)
            if trials:
                os.kill(pid,signal.SIGSTOP);stopped=True
                for _ in range(100):
                    if Path(f'/proc/{pid}/stat').read_text().split()[2]=='T':break
                    time.sleep(.001)
                verified=native_finished(root)
                if verified is None or {t.name for t in verified}!={t.name for t in trials}:
                    os.kill(pid,signal.SIGCONT);stopped=False;continue
                # Every native result exists: agent and verifier phases have ended.
                # Harbor CLI cleanup can finish while its coordinator is stopped.
                deadline=time.monotonic()+60
                while live_children(pid) and time.monotonic()<deadline:time.sleep(.05)
                children=live_children(pid)
                if children:
                    os.kill(pid,signal.SIGCONT);stopped=False;continue
                connections=active_http_connections(command)
                plan={alias(t):t for t in json.loads((root/'planned-episodes.json').read_text())}
                recovered=[]
                for trial in trials:
                    dest=root/'episodes'/(trial.name+'.json')
                    if dest.exists():continue
                    result_paths=list(trial.glob('jobs/*/*/result.json'));trajectories=list(trial.glob('jobs/*/*/agent/trajectory.json'))
                    native=json.loads(result_paths[0].read_text());trajectory=json.loads(trajectories[0].read_text()) if len(trajectories)==1 else None
                    failure='incomplete_native_trial' if trajectory is None or native.get('exception_info') else None
                    atomic(dest,{'task':plan[trial.name],'response':'','trajectory_text':json.dumps(trajectory,ensure_ascii=False) if trajectory else '', 'finish_reason':failure or 'eos','termination':failure or 'complete','generation_error':failure,'objective':{'native_verifier':native.get('verifier_result')},'native_artifacts':[{'path':str(x.relative_to(root)),'sha256':sha(x)} for x in result_paths+trajectories],'migration_recovered_after_native_completion':True,'note':'Native reward is not a reward-hacking behavior label'})
                    recovered.append(trial.name)
                if alive(finishpid):os.kill(finishpid,signal.SIGTERM)
                os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT);stopped=False
                while alive(pid):time.sleep(.05)
                if a.stop_only:
                    worker_app=command[command.index('--worker-app')+1]
                    subprocess.run([sys.executable,'-m','modal','app','stop','--yes',worker_app],check=True)
                    atomic(root/'superseded-v1.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preserved_episodes':len(trials),'recovered':recovered,'in_flight_attempts_terminated':0,'model_app_stopped':worker_app})
                    migration_status('v1 attempts retained; model stopped at completed native-trial boundary. Preparing user-approved active-time v2.','superseded')
                    return
                migration_status('All native episodes finished; replacing idle coordinator and canceling post-timeout inference.')
                threading.Thread(target=heartbeat,daemon=True).start()
                receipt={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'old_pid':pid,'preserved_episodes':len(trials),'recovered_completed_native_trials':recovered,'live_children_at_migration':children,'orphan_post_episode_http_connections':connections,'scheduler':'rolling_four','in_flight_attempts_terminated':0,'state':'warming'}
                atomic(root/'scheduler-migration.json',receipt)
                worker_app=command[command.index('--worker-app')+1]
                worker_class=command[command.index('--worker-class')+1] if '--worker-class' in command else 'LensInteractive'
                cancellation={'needed':bool(connections),'method':None}
                if connections:
                    module='gptoss_terminalwrench_worker.py' if model_key=='gpt_oss' else 'terminalwrench_lens_worker.py'
                    env=dict(os.environ,EMOTION_STEERING_RUN=str(root),TERMINALWRENCH_WORKER_APP=worker_app)
                    with (root/'scheduler-worker-reload.log').open('a') as log:
                        subprocess.run([sys.executable,'-m','modal','app','stop','--yes',worker_app],stdout=log,stderr=subprocess.STDOUT,check=True)
                        subprocess.run([sys.executable,'-m','modal','deploy',str(HERE/module)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                    import modal
                    info=modal.Cls.from_name(worker_app,worker_class)().info.remote()
                    cancellation={'needed':True,'method':'Stopped model app after all native episodes ended, redeployed same worker, warmed before starting any new agent timers','worker_info':info}
                with (root/'coordinator.log').open('a') as log:new=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                (root/'coordinator.pid').write_text(str(new.pid))
                with (root/'finish.log').open('a') as log:finish=subprocess.Popen(finishcommand,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                (root/'finish.pid').write_text(str(finish.pid))
                heartbeat_stop.set();receipt.update(state='restarted',new_pid=new.pid,finish_pid=finish.pid,orphan_rpc_cancellation=cancellation);atomic(root/'scheduler-migration.json',receipt)
                print(json.dumps(receipt),flush=True);return
            ready,_,_=select.select([notify_fd],[],[],.05)
            if ready:os.read(notify_fd,65536)
        print('Original coordinator exited before a safe migration boundary',flush=True)
    except BaseException as exc:
        heartbeat_stop.set()
        if stopped and alive(pid):os.kill(pid,signal.SIGCONT)
        migration_status('Migration failed; preserved native artifacts. '+repr(exc),'blocked')
        atomic(root/'scheduler-migration-error.json',{'error':repr(exc),'old_pid':pid,'at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        raise

if __name__=='__main__':main()
