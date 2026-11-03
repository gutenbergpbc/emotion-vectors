"""Generate neutral GLM traces until twenty valid trajectories or twenty attempts."""

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
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import threading
import time

from terminalwrench_sweep import prepare, PROJECT
from emotion_steering.config import tasks
from emotion_steering.terminalwrench import alias, generate_one, preflight, serve
from emotion_steering.active_budget import ActiveClock
from emotion_steering.timed_proxy import TimedProxy
from shared.io import atomic, digest, read, sha

PREVIOUS = PROJECT / 'runs/emotion-steering/glm53-terminalwrench-interactive-steering-active-time-v2-r2'
ATTEMPTS = [
    PROJECT / 'runs/emotion-steering/glm53-terminalwrench-interactive-steering-active-time-v2-neutral20-20261023',
    PROJECT / 'runs/emotion-steering/glm53-terminalwrench-interactive-steering-active-time-v2-neutral14-resume-20261023',
]
SUFFIX = 'neutral-to20-clockfix-20261023'
APP = 'terminalwrench-glm53-neutral-to20-clockfix-20261023'
TARGET = 20
MAX_ATTEMPTS = 20
CONCURRENCY = 4


def valid_trace(row):
    if row.get('termination') not in ('complete', 'active_budget_timeout'):
        return False
    try:
        trajectory = json.loads(row.get('trajectory_text') or 'null')
    except (TypeError, ValueError):
        return False
    return isinstance(trajectory, dict) and any(s.get('source') == 'agent' for s in trajectory.get('steps', []))


def prepare_run():
    root = prepare('glm', run_suffix=SUFFIX)
    spec = read(root / 'config.resolved.json')
    prior, original_ids = [], set()
    for path in sorted((PREVIOUS / 'episodes').glob('*.json')):
        row = read(path)
        if row['task'].get('strength') == 0:
            prior.append({'path': str(path), 'sha256': sha(path), 'task': row['task'],
                          'valid_trace': valid_trace(row)})
            original_ids.add(row['task']['id'])
    for source_root in ATTEMPTS:
        for path in sorted((source_root / 'episodes').glob('*.json')):
            row = read(path)
            if row['task'].get('strength') == 0:
                prior.append({'path': str(path), 'sha256': sha(path), 'task': row['task'],
                              'valid_trace': valid_trace(row)})
    successes = sum(r['valid_trace'] for r in prior)
    assert successes == 6, f'Prior valid-trace count changed: {successes}'
    # Retain the four original neutral task groups as controls; fill the remaining twenty.
    plan = [t for t in tasks(spec) if t['strength'] == 0 and t['id'] not in original_ids]
    assert len(plan) == 20 and len({t['scenario'] for t in plan}) == 20
    selection = {'prior_episodes': prior, 'prior_valid_traces': successes, 'target_valid_traces': TARGET,
                 'max_new_attempts': MAX_ATTEMPTS, 'initial_attempts': plan,
                 'selection': 'One unsteered attempt for each of the twenty suite tasks beyond the four original neutral controls; failed attempts may be retried by replacing unstarted tasks.',
                 'worker_app': APP, 'concurrency': CONCURRENCY, 'runner_sha256': sha(__file__),
                 'resolved_identity': spec['identity']}
    selection['identity'] = digest(selection)
    atomic(root / 'neutral-selection.json', selection)
    atomic(root / 'planned-episodes.json', plan)
    return root, spec, plan, successes


def retry_task(task, previous_attempts):
    sample = previous_attempts + 1
    return {**task, 'id': f"baseline/{task['scenario']}/{sample:04d}", 'condition': 'baseline',
            'emotion': None, 'strength': 0.0, 'sample': sample, 'seed': task['seed'] + sample}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--port', type=int, default=8170)
    args = parser.parse_args()
    root, spec, plan, prior_successes = prepare_run()
    if args.prepare:
        print(f'Prepared {len(plan)} initial neutral tasks; {prior_successes}/{TARGET} valid traces already saved.', flush=True)
        return

    import modal
    worker = modal.Cls.from_name(APP, 'LensInteractive')()
    atomic(root / 'coordinator.json', {'pid': os.getpid(), 'started_unix': time.time(), 'worker_app': APP})
    atomic(root / 'progress.json', {'status': 'warming', 'target': TARGET, 'prior_valid_traces': prior_successes,
                                    'planned_attempts': MAX_ATTEMPTS, 'attempts_started': 0, 'new_valid_traces': 0})
    info = worker.info.spawn().get(timeout=1800)
    if info.get('identity') != spec['identity'] or info.get('max_context') != 65536:
        raise RuntimeError('Warmed GLM worker identity/context mismatch')
    atomic(root / 'worker-info.json', info)

    clock = ActiveClock()
    spec['terminalwrench'].update(api_base=f'http://127.0.0.1:{args.port}/v1',
                                  budget_base=f'http://127.0.0.1:{args.port}')
    proxy = TimedProxy(worker, clock, root, spec['model']['family'])
    threading.Thread(target=serve, args=(spec, root, proxy, '127.0.0.1', args.port),
                     kwargs={'clock': clock}, daemon=True).start()
    time.sleep(.5)
    preflight(spec)
    results = root / 'episodes'
    results.mkdir(exist_ok=True)
    running, finished = {}, {}
    attempts_by_scenario = {}
    status_lock = threading.Lock()
    stop = threading.Event()
    attempts_started = 0
    new_successes = 0
    pending = deque(plan)

    def status(state='running', detail=None):
        with status_lock:
            atomic(root / 'progress.json', {'status': state, 'target_valid_traces': TARGET,
                'prior_valid_traces': prior_successes, 'new_valid_traces': new_successes,
                'total_valid_traces': prior_successes + new_successes, 'max_new_attempts': MAX_ATTEMPTS,
                'attempts_started': attempts_started, 'saved_attempts': len(finished),
                'running': sorted(t['id'] for t in running.values()), 'pending_initial_tasks': len(pending),
                'terminations': {key: row.get('termination') for key, row in finished.items()},
                'detail': detail, 'updated_unix': time.time()})

    def execute(task):
        print('START ' + task['id'], flush=True)
        try:
            scenario = next(s for s in spec['scenarios'] if s['id'] == task['scenario'])
            output = generate_one(scenario, task, spec, root)
        except Exception as exc:
            output = {'generation_error': repr(exc), 'termination': 'execution_failure'}
        if alias(task) in clock.episodes:
            output['active_budget'] = clock.stop(alias(task))
        row = {'task': task, **output}
        atomic(results / (alias(task) + '.json'), row)
        print('DONE ' + task['id'] + ' ' + str(row.get('termination')), flush=True)
        return task, row

    def launch(pool):
        nonlocal attempts_started
        while pending and len(running) < CONCURRENCY and attempts_started < MAX_ATTEMPTS \
                and prior_successes + new_successes < TARGET:
            task = pending.popleft()
            attempts_started += 1
            running[pool.submit(execute, task)] = task
        status()

    try:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            launch(pool)
            while running:
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    task = running.pop(future)
                    saved_task, row = future.result()
                    finished[saved_task['id']] = row
                    if valid_trace(row):
                        new_successes += 1
                    else:
                        # Reserve the next available attempt for a retry of this task.
                        if pending and attempts_started < MAX_ATTEMPTS:
                            pending.pop()
                            seen = attempts_by_scenario.get(task['scenario'], 0)
                            attempts_by_scenario[task['scenario']] = seen + 1
                            pending.appendleft(retry_task(task, seen))
                    status()
                launch(pool)
        total = prior_successes + new_successes
        state = 'target_reached' if total >= TARGET else 'attempt_limit_reached'
        status(state)
        atomic(root / 'final-summary.json', {'status': state, 'target_valid_traces': TARGET,
            'prior_valid_traces': prior_successes, 'new_valid_traces': new_successes,
            'total_valid_traces': total, 'max_new_attempts': MAX_ATTEMPTS,
            'attempts_started': attempts_started, 'attempts': finished,
            'selection_identity': read(root / 'neutral-selection.json')['identity']})
    finally:
        stop.set()
        worker.update_autoscaler(min_containers=0, max_containers=1, scaledown_window=60)
        atomic(root / 'autoscaler-receipt.json', {'min_containers': 0, 'max_containers': 1,
                                                   'scaledown_window': 60, 'at': time.time()})


if __name__ == '__main__':
    main()
