"""Durable continuous-sweep supervisor with bounded OOM fallback and provenance."""

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
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from shared.io import atomic


def terminate_group(proc):
    if proc.poll() is not None:return
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL);proc.wait()


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--suffix',default='reduced-continuous-v3');p.add_argument('--selection',default='coverage');p.add_argument('--inference-slots',type=int);a=p.parse_args()
    root=a.root.resolve();app='terminalwrench-'+a.model+'-'+a.suffix
    spec=json.loads((root/'config.resolved.json').read_text());initial=spec['batch_size'];limits=list(dict.fromkeys([initial,initial//2,4,1]))
    status=HERE.parents[2]/'dashboards/terminalwrench-three-models/status'/(a.model.replace('_','-')+'.json')
    env=dict(os.environ,EMOTION_STEERING_RUN=str(root),TERMINALWRENCH_WORKER_APP=app)
    py=sys.executable;history=[]
    for attempt,limit in enumerate(limits):
        env['TERMINALWRENCH_INFERENCE_LIMIT']=str(limit)
        marker=root/'worker-restart-required.json'
        if marker.exists():marker.rename(root/f'worker-restart-required-{attempt-1}.json')
        receipt=dict(attempt=attempt,inference_limit=limit,sandbox_concurrency=2*limit,at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        history.append(receipt);atomic(root/'serving-attempts.json',history)
        with (root/f'deploy-{attempt}.log').open('a') as log:
            deployed=subprocess.run([py,'-m','modal','deploy',str(HERE/'terminalwrench_async_worker.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
        if deployed.returncode:raise RuntimeError('Deployment failed; inspect deploy log')
        with (root/f'coordinator-{attempt}.log').open('a') as log:
            proc=subprocess.Popen([py,str(HERE/'terminalwrench_sweep.py'),'--model',a.model,'--run-suffix',a.suffix,'--reduced','--continuous','--selection',a.selection,'--inference-slots',str(initial),'--worker-app',app,'--worker-class','AsyncInteractive','--port',str(a.port),'--concurrency',str(2*limit)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (root/'coordinator.pid').write_text(str(proc.pid)+'\n')
        while proc.poll() is None and not marker.exists():time.sleep(5)
        restart=marker.exists()
        if restart:
            subprocess.run([py,'-m','modal','app','stop','--yes',app],stdout=subprocess.DEVNULL)
            try:proc.wait(timeout=120)
            except subprocess.TimeoutExpired:terminate_group(proc)
        subprocess.run([py,'-m','modal','app','stop','--yes',app],stdout=subprocess.DEVNULL)
        receipt.update(returncode=proc.returncode,restart_requested=restart);atomic(root/'serving-attempts.json',history)
        if proc.returncode==0 and not restart:
            with (root/'finish-watcher.log').open('a') as log:
                subprocess.run([py,str(HERE/'terminalwrench_finish.py'),'--run-root',str(root),'--worker-app',app,'--status',str(status)],stdout=log,stderr=subprocess.STDOUT)
            return
        # Startup OOM occurs before TimedProxy can write a restart marker.
        startup=(root/f'coordinator-{attempt}.log').read_text()
        restart=restart or any(x in startup.lower() for x in ['out of memory','enginecore','enginedead','cuda error'])
        if not restart or attempt==len(limits)-1:
            atomic(root/'supervisor-blocked.json',dict(attempt=attempt,reason='Non-retriable failure or fallback limits exhausted',returncode=proc.returncode));return
        # Keep all clean completed episodes; preserve interrupted/infra-failed work separately.
        archive=root/'infrastructure-attempts'/str(attempt);archive.mkdir(parents=True,exist_ok=True)
        retry=set()
        for ep in (root/'episodes').glob('*.json'):
            d=json.loads(ep.read_text())
            error=str(d.get('generation_error') or '')+' '+str(d.get('native_exception') or '')
            if any(term in error.lower() for term in ('inference_failure','budget_infrastructure','enginecore','enginedead','out of memory','cuda error')):
                retry.add(ep.stem);(archive/'episodes').mkdir(exist_ok=True);ep.rename(archive/'episodes'/ep.name)
        for trial in (root/'native-trials').glob('steering-*'):
            if not (root/'episodes'/(trial.name+'.json')).exists():
                retry.add(trial.name);(archive/'native-trials').mkdir(exist_ok=True);trial.rename(archive/'native-trials'/trial.name)
        from emotion_steering.terminalwrench import alias
        for call in (root/'calls').glob('*.json'):
            if alias(json.loads(call.read_text())['task']) in retry:
                (archive/'calls').mkdir(exist_ok=True);call.rename(archive/'calls'/call.name)
        atomic(archive/'retry-receipt.json',dict(reason='Worker OOM/engine failure',episode_aliases=sorted(retry),next_inference_limit=limits[attempt+1]))

if __name__=='__main__':main()
