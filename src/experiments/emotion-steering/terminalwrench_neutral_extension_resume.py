"""Complete unattempted neutral GLM tasks, retaining all earlier episode attempts."""

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
import os
from pathlib import Path
import threading
import time

from terminalwrench_sweep import prepare, run_rolling, PROJECT
from emotion_steering.config import tasks
from emotion_steering.terminalwrench import alias, generate_one, preflight, serve
from emotion_steering.active_budget import ActiveClock, serve_clock
from emotion_steering.timed_proxy import TimedProxy
from shared.io import atomic, digest, sha

PREVIOUS = PROJECT / 'runs/emotion-steering/glm53-terminalwrench-interactive-steering-active-time-v2-r2'
ATTEMPT_ROOT = PROJECT / 'runs/emotion-steering/glm53-terminalwrench-interactive-steering-active-time-v2-neutral20-20261023'
SUFFIX = 'neutral14-resume-20261023'
APP = 'terminalwrench-glm53-neutral14-resume-20261023'


def prepare_extension():
    root = prepare('glm', run_suffix=SUFFIX)
    spec = json.loads((root / 'config.resolved.json').read_text())
    prior = []
    for source_root in (PREVIOUS, ATTEMPT_ROOT):
        for path in sorted((source_root / 'episodes').glob('*.json')):
            row = json.loads(path.read_text())
            if row['task']['strength'] == 0:
                prior.append({'path': str(path), 'sha256': sha(path), 'task': row['task']})
    assert len(prior) == 10, f'Expected four original neutral episodes and six prior attempts; got {len(prior)}'
    old_ids = {r['task']['id'] for r in prior}
    plan = [t for t in tasks(spec) if t['strength'] == 0 and t['id'] not in old_ids]
    assert len(plan) == 14 and len({t['scenario'] for t in plan}) == 14
    assert all(t['emotion'] is None for t in plan)
    selection = {'previous_episodes': prior, 'episodes': plan,
                 'selection': 'Fourteen suite tasks without any episode among the four original neutral traces and six saved extension attempts. Original task seeds retained; no outcome selection.',
                 'worker_app': APP, 'concurrency': 4, 'runner_sha256': sha(__file__),
                 'resolved_identity': spec['identity']}
    selection['identity'] = digest(selection)
    path = root / 'neutral-selection.json'
    if path.exists():
        assert json.loads(path.read_text()) == selection, 'Frozen selection changed'
    atomic(path, selection)
    atomic(root / 'planned-episodes.json', plan)
    return root, spec, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--port', type=int, default=8170)
    args = parser.parse_args()
    if args.prepare:
        root, _, plan = prepare_extension()
        print(f'Prepared {len(plan)} neutral episodes: {root}', flush=True)
        return
    root = PROJECT / 'runs/emotion-steering' / ('glm53-terminalwrench-interactive-steering-active-time-v2-' + SUFFIX)
    spec = json.loads((root / 'config.resolved.json').read_text())
    selection = json.loads((root / 'neutral-selection.json').read_text())
    assert selection['runner_sha256'] == sha(__file__)
    assert selection['resolved_identity'] == spec['identity']
    for row in selection['previous_episodes']:
        assert sha(row['path']) == row['sha256']
    plan = selection['episodes']
    assert len(plan) == 14 and all(t['strength'] == 0 and t['emotion'] is None for t in plan)
    import modal
    worker = modal.Cls.from_name(APP, 'LensInteractive')()
    atomic(root / 'coordinator.json', {'pid': os.getpid(), 'started_unix': time.time(), 'worker_app': APP})
    atomic(root / 'progress.json', {'status': 'warming', 'planned': 14, 'completed': 0})
    info = worker.info.spawn().get(timeout=1800)
    assert info['identity'] == spec['identity'] and info['max_context'] == 65536
    atomic(root / 'worker-info.json', info)
    clock = ActiveClock()
    spec['terminalwrench'].update(api_base=f'http://127.0.0.1:{args.port}/v1', budget_base=f'http://127.0.0.1:{args.port}')
    proxy = TimedProxy(worker, clock, root, spec['model']['family'])
    threading.Thread(target=serve, args=(spec, root, proxy, '127.0.0.1', args.port),
                     kwargs={'clock': clock}, daemon=True).start()
    time.sleep(.5)
    preflight(spec)
    results = root / 'episodes'
    results.mkdir(exist_ok=True)
    finished = {t['id']: json.loads((results / (alias(t)+'.json')).read_text()) for t in plan if (results / (alias(t)+'.json')).exists()}
    running = set()
    lock = threading.Lock()
    stop = threading.Event()
    def status():
        with lock:
            atomic(root / 'progress.json', {'status': 'complete' if len(finished)==len(plan) else 'running', 'planned': len(plan),
                'completed': len(finished), 'running': sorted(running), 'updated_unix': time.time(),
                'terminations': {key: row.get('termination') for key, row in finished.items()},
                'saved_trajectories': sum(bool(r.get('trajectory_text')) for r in finished.values())})
    def heartbeat():
        while not stop.wait(15):
            status()
    threading.Thread(target=heartbeat, daemon=True).start()
    def execute(task):
        with lock:
            running.add(task['id'])
        status()
        print('START '+task['id'], flush=True)
        try:
            scenario = next(s for s in spec['scenarios'] if s['id']==task['scenario'])
            output = generate_one(scenario, task, spec, root)
        except Exception as exc:
            output = {'generation_error': repr(exc), 'termination': 'execution_failure'}
        if alias(task) in clock.episodes:
            output['active_budget'] = clock.stop(alias(task))
        output = {'task': task, **output}
        atomic(results / (alias(task)+'.json'), output)
        with lock:
            running.remove(task['id'])
            finished[task['id']] = output
        status()
        print('DONE '+task['id']+' '+output['termination'], flush=True)
    try:
        run_rolling([t for t in plan if t['id'] not in finished], execute, 4,
                    lambda: (root/'worker-restart-required.json').exists() or (root/'user-stop.json').exists())
        status()
        if len(finished)!=len(plan):
            raise RuntimeError('Stopped before all planned episodes; inspect worker/user stop receipt')
    finally:
        stop.set()
        worker.update_autoscaler(min_containers=0, max_containers=1, scaledown_window=60)
        atomic(root/'autoscaler-receipt.json', {'min_containers':0,'max_containers':1,'scaledown_window':60,'at':time.time()})

if __name__ == '__main__':
    main()
