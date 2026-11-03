"""Generate ten fresh, unsteered GLM 5.3 TerminalWrench episodes; resume safely."""

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
from collections import defaultdict
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import terminalwrench_neutral20_20261026 as runner
from shared.io import atomic, digest, read, sha
from emotion_steering.config import tasks

runner.RUN_SUFFIX = 'neutral10-sample3-20261101'
APP = 'terminalwrench-glm53-neutral10-20261101'


def prepare():
    source, root = runner.paths('glm')
    if root.exists():
        raise ValueError(f'Run already exists: {root}')
    spec = read(source / 'config.resolved.json')
    by_stratum = defaultdict(list)
    for scenario in spec['scenarios']:
        by_stratum[scenario['stratum']].append(scenario['id'])
    assert len(by_stratum) == 6
    # Ten distinct tasks: one from each stratum, plus a second from four strata.
    # Freeze selection before generation; no outcome-based selection or stopping.
    order = sorted(by_stratum, key=lambda s: digest(['glm-neutral10-20261101', s]))
    selected = []
    for i, stratum in enumerate(order):
        ranked = sorted(by_stratum[stratum], key=lambda s: digest(['glm-neutral10-20261101', stratum, s]))
        selected.extend(ranked[:2 if i < 4 else 1])
    spec.update(run_id=root.name, samples=4)
    spec.pop('identity', None)
    # Pin the code actually used, retaining previous hashes as provenance.
    prior_hashes = spec['source_hashes']
    spec['source_hashes'] = {name: sha(runner.PROJECT / name) for name in prior_hashes}
    spec['identity'] = digest(spec)
    plan = [t for t in tasks(spec) if t['condition'] == 'baseline' and t['sample'] == 3 and t['scenario'] in selected]
    assert len(plan) == len({t['scenario'] for t in plan}) == 10
    assert all(t['emotion'] is None and t['strength'] == 0 for t in plan)
    cfg = read(source / 'config.json')
    cfg.update(run_id=root.name, samples=4)
    root.mkdir(parents=True)
    for name, value in [('config.json', cfg), ('config.resolved.json', spec),
                        ('dataset.json', read(source / 'dataset.json')), ('planned-episodes.json', plan)]:
        atomic(root / name, value)
    atomic(root / 'selection.json', dict(
        source_run=str(source), model='glm', sample=3, scenario_ids=selected,
        runner_sha256=sha(runner.__file__), coordinator_sha256=sha(__file__),
        source_config_sha256=sha(source / 'config.resolved.json'),
        source_dataset_sha256=sha(source / 'dataset.json'), previous_source_hashes=prior_hashes,
        method='Ten distinct tasks across six strata, selected deterministically before outcomes; fresh sample 3; fixed ten-attempt budget; unsteered; all outcomes retained.',
        worker_app=APP,
    ))
    atomic(root / 'progress.json', dict(status='prepared', planned=10, completed=0))
    print(root)
    for task in plan:
        print(task['id'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--port', type=int, default=8196)
    parser.add_argument('--concurrency', type=int, default=4)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        _, root = runner.paths('glm')
        assert read(root / 'selection.json')['coordinator_sha256'] == sha(__file__)
        runner.run('glm', args.port, args.concurrency, app_name=APP)
