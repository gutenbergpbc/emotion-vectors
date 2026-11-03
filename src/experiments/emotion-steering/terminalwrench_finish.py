"""Durable finish watcher: stop this model's compute, then judge the saved sweep."""

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
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from shared.io import atomic


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--worker-app',required=True);p.add_argument('--status',type=Path,required=True);a=p.parse_args()
    pid=int((a.run_root/'coordinator.pid').read_text())
    while True:
        try:
            os.kill(pid,0)
            state=Path(f'/proc/{pid}/stat').read_text().split()[2]
            if state=='Z':break
        except (ProcessLookupError,FileNotFoundError):break
        time.sleep(15)
    stopped=subprocess.run([sys.executable,'-m','modal','app','stop','--yes',a.worker_app],capture_output=True,text=True)
    atomic(a.run_root/'compute-stopped.json',{'returncode':stopped.returncode,'stdout':stopped.stdout,'stderr':stopped.stderr,'at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    total=len(list((a.run_root/'episodes').glob('*.json')))
    data=json.loads(a.status.read_text())
    expected=len(json.loads((a.run_root/'planned-episodes.json').read_text()))
    if total!=expected:
        data.update(status='blocked',stage='Sweep interrupted',message=f'{total} episode artifacts retained; model compute stop requested.',updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        atomic(a.status,data);return
    data.update(status='running',stage='Blinded transcript evaluation',running_episodes=0,message='GPU compute stopped; external judge evaluates complete transcripts with the shared TerminalWrench rubric.',updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat());atomic(a.status,data)
    with (a.run_root/'evaluation.log').open('a') as log:
        proc=subprocess.Popen([sys.executable,str(HERE/'terminalwrench_evaluate.py'),str(a.run_root),'--judge','--allow-external-judge'],stdout=log,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            data['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();atomic(a.status,data);time.sleep(15)
    data.update(status='completed' if proc.returncode==0 else 'blocked',stage='Outputs ready' if proc.returncode==0 else 'Evaluation failed',message='Sweep artifacts, blinded labels and comparison viewer saved.' if proc.returncode==0 else 'See evaluation.log; GPU compute stopped.',updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat());atomic(a.status,data)

if __name__=='__main__':main()
