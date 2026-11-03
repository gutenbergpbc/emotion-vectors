"""Inspect actual first-batch inference receipts and native terminal interactions."""

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

import json
import os
import time
from pathlib import Path

def _pending_local_requests():
    """Only this coordinator's accepted HTTP sockets, excluding remote Modal RPCs."""
    inodes=set()
    for fd in Path(f'/proc/{os.getpid()}/fd').iterdir():
        try:target=os.readlink(fd)
        except OSError:continue
        if target.startswith('socket:['):inodes.add(target[8:-1])
    rows=[]
    for table in ('/proc/net/tcp','/proc/net/tcp6'):
        rows += [line.split() for line in Path(table).read_text().splitlines()[1:]]
    own=[row for row in rows if row[9] in inodes]
    listeners={row[1] for row in own if row[3]=='0A'}
    return any(row[1] in listeners and row[3] in ('01','08') for row in own)


def validate_first_batch(root, outputs):
    root=Path(root)
    calls=[json.loads(p.read_text()) for p in (root/'calls').glob('*.json')]
    expected={episode['task']['id'] for episode in outputs}
    deadline=time.monotonic()+900
    # An agent timeout does not make its already submitted GPU request vanish.
    # Preserve that response/receipt before deciding whether validation failed.
    # This never extends the agent's 900s execution or invents an observation.
    while expected-{call['task']['id'] for call in calls} and _pending_local_requests() and time.monotonic()<deadline:
        time.sleep(1)
        calls=[json.loads(p.read_text()) for p in (root/'calls').glob('*.json')]
    checks=[]
    for episode in outputs:
        task=episode['task']; own=[c for c in calls if c['task']['id']==task['id']]
        problems=[]
        if not own:problems.append('No saved inference calls')
        for call in own:
            output=call['output'];receipt=output.get('steering_receipt')
            if task['strength']:
                if not receipt or receipt.get('strength')!=task['strength'] or receipt.get('tokens_steered',0)<=0:problems.append('Missing or mismatched active steering receipt')
                elif receipt.get('layer',receipt.get('layer_zero_based'))!=task['layer']:problems.append('Steering layer mismatch')
            elif receipt and (receipt.get('enabled') or receipt.get('tokens_steered',0)):problems.append('Baseline received active hook')
            turn=sum(m['role']=='assistant' for m in call['messages'])
            if output.get('sampling_seed')!=task['seed']+turn*1000003:problems.append('Per-turn seed mismatch')
        trajectory=json.loads(episode.get('trajectory_text') or '{}')
        steps=trajectory.get('steps',[])
        interactions=[step for step in steps if step.get('tool_calls') or step.get('observation')]
        checks.append({'task_id':task['id'],'calls':len(own),'terminal_steps':len(interactions),'errors':sorted(set(problems)),'generation_error':episode.get('generation_error')})
    return {'passed':all(not c['errors'] for c in checks) and any(c['terminal_steps'] for c in checks),'episodes':checks,'call_count':len(calls),'note':'Malformed/no-action model outputs remain outcomes; real terminal interaction required somewhere in mixed batch.'}
