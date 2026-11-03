"""Keep the single inference allocation warm while its coordinator is alive."""

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
import json
from pathlib import Path
import time
import modal

def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--worker-app',required=True);p.add_argument('--worker-class',default='LensInteractive');a=p.parse_args()
    pid=int((a.root/'coordinator.pid').read_text())
    while alive(pid) and not (a.root/'worker-info.json').exists():time.sleep(5)
    if not alive(pid):return
    worker=modal.Cls.from_name(a.worker_app,a.worker_class)()
    (a.root/'keepalive-policy.json').write_text(json.dumps({'purpose':'Control-only keepalive prevents120s idle scale-down during sandbox image setup; no model generation','period_seconds':30,'ends_with_coordinator_pid':pid,'worker_app':a.worker_app},indent=2))
    while alive(pid):
        try:worker.info.spawn().get(timeout=30)
        except Exception as exc:
            print('Worker keepalive failed: '+repr(exc),flush=True)
            if not alive(pid):return
        for _ in range(30):
            if not alive(pid):return
            time.sleep(1)
if __name__=='__main__':main()
