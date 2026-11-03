"""Finalize the neutral extension and stop only its dedicated model app."""
from __future__ import annotations

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
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared.io import atomic, digest, read, sha

APP = 'terminalwrench-glm53-neutral20-20261023'


def process_identity(pid):
    """Return state/start ticks, including zombies and excluding PID reuse."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[0], fields[19]
    except FileNotFoundError:
        return None


def audit(root):
    selection = read(root / 'neutral-selection.json')
    spec = read(root / 'config.resolved.json')
    errors = []
    if selection['worker_app'] != APP:
        raise ValueError('Unexpected worker app; refusing unrelated compute cleanup')
    for name, value in [('selection', selection), ('resolved config', spec)]:
        if digest({k: v for k, v in value.items() if k != 'identity'}) != value['identity']:
            errors.append(f'{name} identity mismatch')
    if selection['resolved_identity'] != spec['identity']:
        errors.append('Selection/resolved identity mismatch')
    if selection['runner_sha256'] != sha(HERE / 'terminalwrench_neutral_extension.py'):
        errors.append('Bound runner source changed')
    project = HERE.parents[2]
    for relative, expected in spec['source_hashes'].items():
        frozen = root / 'runtime-source' / relative
        if not frozen.is_file() or sha(frozen) != expected:
            errors.append(f'Frozen runtime checksum mismatch: {relative}')
        # Running modules were loaded from the current checkout.
        live = project / relative
        if not live.is_file() or sha(live) != expected:
            errors.append(f'Live runtime source changed: {relative}')
    for row in selection['previous_episodes']:
        if not Path(row['path']).is_file() or sha(row['path']) != row['sha256']:
            errors.append(f'Previous episode checksum mismatch: {row["path"]}')
    for scenario in spec['scenarios']:
        folder = Path(scenario['task_path'])
        actual = {str(p.relative_to(folder)): sha(p) for p in folder.rglob('*') if p.is_file()}
        if actual != scenario['task_hashes']:
            errors.append(f'Task source changed: {scenario["id"]}')
    expected = selection['episodes']
    if len(expected) != 20 or len({t['id'] for t in expected}) != 20:
        errors.append('Expected exactly twenty unique tasks')
    rows, expected_paths = [], set()
    for task in expected:
        path = root / 'episodes' / ('steering-' + digest(task['id'])[:24] + '.json')
        expected_paths.add(path)
        row = {'task_id': task['id'], 'path': str(path), 'valid_trajectory': False, 'errors': []}
        if not path.exists():
            row['errors'].append('Missing episode artifact')
            rows.append(row)
            continue
        row['sha256'] = sha(path)
        try:
            data = read(path)
            if data['task'] != task or task['strength'] != 0 or task['emotion'] is not None:
                row['errors'].append('Episode task differs from frozen neutral selection')
            row.update(termination=data.get('termination'), generation_error=data.get('generation_error'))
            trajectory = json.loads(data.get('trajectory_text') or 'null')
            steps = trajectory.get('steps', []) if isinstance(trajectory, dict) else []
            row['assistant_steps'] = sum(s.get('source') == 'agent' for s in steps)
            row['valid_trajectory'] = bool(steps and row['assistant_steps'])
            if not row['valid_trajectory']:
                row['errors'].append('No parseable trajectory with assistant steps')
            artifacts = data.get('native_artifacts', [])
            if not artifacts:
                row['errors'].append('Missing native artifact receipts')
            for artifact in artifacts:
                native = (root / artifact['path']).resolve()
                if not native.is_relative_to(root.resolve()) or not native.is_file() or sha(native) != artifact['sha256']:
                    row['errors'].append('Native artifact checksum/path mismatch: ' + artifact['path'])
        except (ValueError, KeyError, TypeError, OSError) as exc:
            row['errors'].append(f'{type(exc).__name__}: {exc}')
        if row['errors']:
            row['valid_trajectory'] = False
        rows.append(row)
    extras = sorted(str(p) for p in (root / 'episodes').glob('*.json') if p not in expected_paths)
    if extras:
        errors.append('Unexpected episode files: ' + ', '.join(extras))
    valid = sum(r['valid_trajectory'] for r in rows)
    failures = [r for r in rows if r['errors'] or r.get('generation_error')]
    return {'status': 'complete' if valid == 20 and not errors and not failures else 'complete_with_failures' if valid == 20 and not errors else 'incomplete',
            'planned': 20, 'saved_episode_artifacts': sum('sha256' in r for r in rows),
            'valid_trajectories': valid, 'episodes_with_errors_or_nonstandard_termination': len(failures),
            'terminations': dict(Counter(r.get('termination', 'missing') for r in rows)),
            'provenance_errors': errors, 'episodes': rows, 'updated_unix': time.time(),
            'scope': 'Structural/source verification; native reward and episode completion do not establish reward hacking.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    selection = read(root / 'neutral-selection.json')
    if selection['worker_app'] != APP or root.name != 'glm53-terminalwrench-interactive-steering-active-time-v2-neutral20-20261023':
        raise ValueError('This supervisor only handles the dedicated neutral20 run')
    pid = int((root / 'coordinator.pid').read_text())
    initial = process_identity(pid)
    atomic(root / 'supervisor.json', {'pid': os.getpid(), 'coordinator_pid': pid,
           'coordinator_process_identity': initial, 'started_unix': time.time(), 'source_sha256': sha(__file__)})
    while initial:
        current = process_identity(pid)
        if current is None or current[0] == 'Z' or current[1] != initial[1]:
            break
        time.sleep(10)
    try:
        summary = audit(root)
    except Exception as exc:
        summary = {'status': 'audit_failed', 'error': repr(exc), 'updated_unix': time.time()}
    # Model startup/preflight failures also reach this branch after process exit.
    try:
        stopped = subprocess.run([sys.executable, '-m', 'modal', 'app', 'stop', '--yes', APP],
                                 capture_output=True, text=True, timeout=120)
        cleanup = {'app': APP, 'returncode': stopped.returncode, 'stdout': stopped.stdout,
                   'stderr': stopped.stderr, 'at': time.time()}
    except Exception as exc:
        cleanup = {'app': APP, 'error': repr(exc), 'at': time.time()}
    atomic(root / 'compute-stopped.json', cleanup)
    summary['compute_stop_succeeded'] = cleanup.get('returncode') == 0
    if not summary['compute_stop_succeeded']:
        summary['cleanup_requires_attention'] = True
    atomic(root / 'final-summary.json', summary)
    atomic(root / 'progress.json', {k: v for k, v in summary.items() if k != 'episodes'})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
