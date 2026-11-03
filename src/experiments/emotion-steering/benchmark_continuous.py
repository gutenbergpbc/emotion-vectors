"""Replay saved first-turn inputs for serving validation; never experiment episodes."""

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
import concurrent.futures
import json
import time
from pathlib import Path
import modal

p=argparse.ArgumentParser();p.add_argument('model');a=p.parse_args()
base=Path(__file__).resolve().parents[3]/'runs/emotion-steering'
old={'gemma':'gemma4b-terminalwrench-interactive-steering-active-time-v2-r2', 'glm':'glm53-terminalwrench-interactive-steering-active-time-v2-r2','gpt_oss':'gptoss120b-terminalwrench-interactive-steering-active-time-v2'}[a.model]
root=base/('terminalwrench-serving-benchmark-'+a.model+'-v3');spec=json.loads((root/'config.resolved.json').read_text())
worker=modal.Cls.from_name('terminalwrench-benchmark-'+a.model+'-v3','AsyncInteractive')()
print('warming',flush=True);print(worker.info.remote(),flush=True)
probes={p['emotion']:p for p in spec['probes']}
rows=[]
for f in sorted((base/old/'calls').glob('*.json')):
 d=json.loads(f.read_text())
 if len(d['messages'])==1:
  rows.append(d)
 if len(rows)==spec['batch_size']:break
if not rows:raise RuntimeError('No replay inputs')
started=time.monotonic()
def run(pair):
 i,d=pair;t=d['task'];o=d['output'];ts=time.monotonic()
 try:
  out=worker.complete.remote(d['messages'],probes.get(t['emotion']),t['strength'],o.get('sampling_seed',t['seed']),4096,request_id='replay-'+str(i))
  row=dict(index=i,task=t,seconds=time.monotonic()-ts,output=out,old_output=o)
 except Exception as exc:row=dict(index=i,error=repr(exc),seconds=time.monotonic()-ts)
 (root/f'replay-{i}.json').write_text(json.dumps(row,indent=2));print({k:v for k,v in row.items() if k not in ['output','old_output','task']},flush=True)
 return row
with concurrent.futures.ThreadPoolExecutor(max_workers=len(rows)) as pool:results=list(pool.map(run,enumerate(rows)))
summary=dict(seconds=time.monotonic()-started,requests=len(rows),errors=sum('error' in x for x in results),tokens=sum(x.get('output',{}).get('generated_tokens',0) for x in results))
(root/'benchmark.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
