"""Audit all ten generation attempts and summarize sentence labels if available."""

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

from collections import Counter
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import terminalwrench_glm53_neutral10_20261101 as coordinator
from emotion_steering.terminalwrench import alias
from shared.io import atomic, read, sha


def audit():
    _, root = coordinator.runner.paths('glm')
    plan = read(root / 'planned-episodes.json')
    rows = []
    for task in plan:
        path = root / 'episodes' / (alias(task) + '.json')
        row = dict(id=task['id'], scenario=task['scenario'], present=path.exists())
        if path.exists():
            episode = read(path)
            trajectory = json.loads(episode.get('trajectory_text') or 'null') or {}
            steps = trajectory.get('steps', [])
            artifacts = episode.get('native_artifacts') or []
            verified = bool(artifacts) and all((root / a['path']).exists() and sha(root / a['path']) == a['sha256'] for a in artifacts)
            agent_steps = sum(s.get('source') == 'agent' for s in steps)
            neutral = episode['task'] == task and task['sample'] == 3 and task['emotion'] is None and task['strength'] == 0
            usable = neutral and verified and agent_steps > 0 and episode.get('termination') in ('complete', 'active_budget_timeout') and episode.get('generation_error') in (None, 'active_budget_timeout')
            row.update(usable=usable, agent_steps=agent_steps, artifacts_verified=verified,
                       termination=episode.get('termination'), generation_error=episode.get('generation_error'),
                       path=str(path), sha256=sha(path))
        rows.append(row)
    summary = dict(status='complete' if len(rows) == 10 and all(r.get('usable') for r in rows) else 'incomplete',
                   planned=len(plan), saved=sum(r['present'] for r in rows), usable=sum(bool(r.get('usable')) for r in rows),
                   terminations=dict(Counter(r.get('termination', 'missing') for r in rows)), episodes=rows)
    atomic(root / 'trace-quality.json', summary)
    labels_root = root.parents[1] / 'emotion-rh-sentence-labels/glm53-terminalwrench-neutral10-20261101-gpt61sol-v1'
    if (labels_root / 'labels.json').exists():
        labels = read(labels_root / 'labels.json')
        traces = labels['traces']
        units = [u for t in traces for u in t['units']]
        summary['labels'] = dict(
            status=labels['summary']['status'], sentences=len(units),
            rh_including_review=sum(u['hacking'] for u in units),
            rh_unflagged=sum(u['hacking'] and not u['needs_review'] for u in units),
            non_rh=sum(not u['hacking'] for u in units), review_flagged=sum(u['needs_review'] for u in units),
            positive_tasks=sorted({t['task']['scenario'] for t in traces if any(u['hacking'] and not u['needs_review'] for u in t['units'])}),
            labels_path=str(labels_root / 'labels.json'), labels_sha256=sha(labels_root / 'labels.json'),
        )
    atomic(root / 'completion-summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'episodes'}, indent=2))
    return summary


if __name__ == '__main__':
    audit()
