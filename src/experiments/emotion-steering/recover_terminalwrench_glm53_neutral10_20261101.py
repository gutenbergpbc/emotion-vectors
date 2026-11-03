"""Recover the fixed ten tasks after coordinator SIGTERM, preserving all attempts."""

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

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import modal
import terminalwrench_glm53_neutral10_20261101 as coordinator
from emotion_steering.active_budget import ActiveClock
from emotion_steering.terminalwrench import alias, endpoint_identity, generate_one, preflight, serve
from emotion_steering.timed_proxy import TimedProxy
from shared.io import atomic, digest, read, sha


def main():
    _, source = coordinator.runner.paths('glm')
    root = source.with_name(source.name + '-recovery-v2')
    root.mkdir(exist_ok=True)
    spec = read(source / 'config.resolved.json')
    plan = read(source / 'planned-episodes.json')
    for name in ['config.json', 'config.resolved.json', 'planned-episodes.json', 'dataset.json']:
        target = root / name
        if not target.exists():
            shutil.copyfile(source / name, target)
    atomic(root / 'recovery-provenance.json', dict(
        original_run=str(source), original_config_sha256=sha(source / 'config.resolved.json'),
        cause='Original coordinator and watcher exited with SIGTERM (143); native artifacts retained.',
        policy='Retain all first attempts. Reuse completed native trials; retry only infrastructure-interrupted trials, with identical task seeds and settings. No outcome-based retries. Complete the same ten preselected task slots.',
        code_sha256=sha(__file__), worker_app=coordinator.APP,
    ))
    episodes = root / 'episodes'
    episodes.mkdir(exist_ok=True)
    imported = []
    interrupted = []
    for task in plan:
        original = source / 'native-trials' / alias(task)
        results = list(original.glob('jobs/*/*/result.json'))
        trajectories = list(original.glob('jobs/*/*/agent/trajectory.json'))
        if len(results) != 1 or len(trajectories) != 1:
            continue
        result = read(results[0])
        exception = result.get('exception_info')
        raw = dict(task=task, termination='native_exception' if exception else 'complete',
                   generation_error=(exception or {}).get('exception_type'),
                   trajectory_text=json.dumps(read(trajectories[0]), ensure_ascii=False),
                   native_result_path=str(results[0]), native_result_sha256=sha(results[0]),
                   trajectory_path=str(trajectories[0]), trajectory_sha256=sha(trajectories[0]))
        atomic(source / 'recovered-attempts' / f'{alias(task)}.json', raw)
        if exception:
            interrupted.append(task['id'])
            continue
        dest = episodes / f'{alias(task)}.json'
        if dest.exists():
            continue
        artifacts = []
        for artifact in [results[0], trajectories[0]]:
            target = root / 'native-imports' / alias(task) / artifact.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(artifact, target)
            artifacts.append(dict(path=str(target.relative_to(root)), sha256=sha(target)))
        atomic(dest, dict(task=task, response='', trajectory_text=raw['trajectory_text'],
                          termination='complete', finish_reason='eos', generation_error=None,
                          native_exception=None, objective=dict(native_verifier=result.get('verifier_result')),
                          native_artifacts=artifacts, imported_from=str(original),
                          note='Completed original native trial recovered without regeneration; reward is separate from RH label.'))
        imported.append(task['id'])
    atomic(root / 'initial-recovery.json', dict(imported=imported, infrastructure_interrupted=interrupted))
    print('Recovered completed tasks:', imported, 'Infrastructure retries:', interrupted, flush=True)
    worker = modal.Cls.from_name(coordinator.APP, 'AsyncInteractive')()
    info = worker.info.spawn().get(timeout=1800)
    assert info['identity'] == spec['identity']
    atomic(root / 'worker-info.json', info)
    port = 8196
    spec['terminalwrench'].update(api_base=f'http://127.0.0.1:{port}/v1', budget_base=f'http://127.0.0.1:{port}', budget_file_base=str(root / 'clock-state'))
    clock = ActiveClock(state_dir=root / 'clock-state')
    proxy = TimedProxy(worker, clock, root, 'lens')
    threading.Thread(target=serve, args=(spec, root, proxy, '127.0.0.1', port), kwargs=dict(clock=clock), daemon=True).start()
    time.sleep(.5)
    preflight(spec)
    finished = {t['id'] for t in plan if (episodes / f'{alias(t)}.json').exists()}
    running = set()
    lock = threading.Lock()
    def progress(stage='generation'):
        with lock:
            atomic(root / 'progress.json', dict(status=stage, planned=10, completed=len(finished), running=sorted(running), updated_unix=time.time()))
    scenarios = {s['id']: s for s in spec['scenarios']}
    def execute(task):
        with lock:
            running.add(task['id'])
        progress()
        try:
            output = generate_one(scenarios[task['scenario']], task, spec, root)
        except Exception as exc:
            output = dict(generation_error=repr(exc), termination='execution_failure')
        if alias(task) in clock.episodes:
            output['active_budget'] = clock.stop(alias(task))
        atomic(episodes / f'{alias(task)}.json', dict(task=task, **output))
        with lock:
            finished.add(task['id']); running.remove(task['id'])
        progress()
        print(len(finished), '/10', task['id'], output.get('termination'), flush=True)
    try:
        progress()
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(execute, [t for t in plan if t['id'] not in finished]))
    finally:
        stopped = subprocess.run([sys.executable, '-m', 'modal', 'app', 'stop', '--yes', coordinator.APP], capture_output=True, text=True)
        atomic(root / 'compute-stopped.json', dict(app=coordinator.APP, returncode=stopped.returncode, stdout=stopped.stdout, stderr=stopped.stderr))
    import label_terminalwrench_glm53_neutral10_20261101 as labeling
    labeling.batch.RUNS = {'GLM 5.3': root.name}
    labeling.batch.OUT = root.parents[1] / 'emotion-rh-sentence-labels/glm53-terminalwrench-neutral10-20261101-recovery-v2-gpt61sol-v1'
    progress('sentence-labeling')
    sys.argv = [sys.argv[0], '--workers', '2']
    labeling.batch.main()
    labels = read(labeling.batch.OUT / 'labels.json')
    units = [u for t in labels['traces'] for u in t['units']]
    summary = dict(saved=len(finished), labels_status=labels['summary']['status'],
                   sentences=len(units), rh_including_review=sum(u['hacking'] for u in units),
                   rh_unflagged=sum(u['hacking'] and not u['needs_review'] for u in units),
                   non_rh=sum(not u['hacking'] for u in units), review_flagged=sum(u['needs_review'] for u in units),
                   positive_tasks=sorted({t['task']['scenario'] for t in labels['traces'] if any(u['hacking'] and not u['needs_review'] for u in t['units'])}),
                   labels_path=str(labeling.batch.OUT / 'labels.json'), labels_sha256=sha(labeling.batch.OUT / 'labels.json'))
    atomic(root / 'completion-summary.json', summary)
    progress('complete')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
