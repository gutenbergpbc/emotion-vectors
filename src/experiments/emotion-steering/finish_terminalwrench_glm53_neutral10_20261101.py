"""Finish this fixed ten-trace batch: audit, sentence-label, and save a report."""

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
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import terminalwrench_glm53_neutral10_20261101 as coordinator
from audit_terminalwrench_glm53_neutral10_20261101 import audit
from shared.io import atomic, read


def main():
    _, root = coordinator.runner.paths('glm')
    deadline = time.monotonic() + 3 * 3600
    previous = None
    while True:
        saved = len(list((root / 'episodes').glob('*.json')))
        if saved != previous:
            print(f'{saved}/10 traces saved', flush=True)
            previous = saved
        atomic(root / 'pipeline-progress.json', dict(stage='generation', saved=saved, planned=10, updated_unix=time.time()))
        if saved == 10 and (root / 'compute-stopped.json').exists():
            break
        if (root / 'compute-stopped.json').exists() and saved < 10:
            raise RuntimeError('Generation stopped before ten traces were saved; inspect generation logs.')
        if time.monotonic() > deadline:
            raise RuntimeError('Three-hour completion watchdog reached.')
        time.sleep(10)
    quality = audit()
    if quality['usable'] != 10:
        print(f"Only {quality['usable']}/10 traces passed quality checks; all artifacts retained.", flush=True)
    atomic(root / 'pipeline-progress.json', dict(stage='sentence-labeling', saved=10, planned=10, updated_unix=time.time()))
    with (root / 'sentence-labeling.log').open('a') as log:
        result = subprocess.run([sys.executable, '-u', str(HERE / 'label_terminalwrench_glm53_neutral10_20261101.py'), '--workers', '2'], stdout=log, stderr=subprocess.STDOUT)
    summary = audit()
    labels = summary.get('labels', {})
    passed = result.returncode == 0 and labels.get('status') == 'complete' and summary['usable'] == 10
    atomic(root / 'pipeline-progress.json', dict(stage='complete' if passed else 'needs-attention', saved=10, planned=10,
                                                label_returncode=result.returncode, summary=labels, updated_unix=time.time()))
    report = [
        '# GLM 5.3: ten additional neutral TerminalWrench traces', '',
        'Ten distinct tasks were selected across six task categories before generation. All use fresh replicate sample 3, zero steering, and the original pinned NVFP4 checkpoint and active-time policy. All attempts and partial active-time timeouts are retained.', '',
        f"Saved: {summary['saved']}/10. Usable: {summary['usable']}/10. Terminations: {json.dumps(summary['terminations'])}.", '',
        f"Sentence labels: {labels.get('sentences', 'unavailable')}. Unflagged RH: {labels.get('rh_unflagged', 'unavailable')}. RH including review flags: {labels.get('rh_including_review', 'unavailable')}. Non-RH: {labels.get('non_rh', 'unavailable')}. Review-flagged: {labels.get('review_flagged', 'unavailable')}.", '',
        'Positive tasks: ' + (', '.join(labels.get('positive_tasks', [])) or 'none'), '',
        'Labels use the existing GPT-6.1 Sol full-trajectory rubric on assistant prose. Native task reward is separate from the RH label; tool-only hacking may be missed. This task-selected batch does not estimate population RH prevalence.', '',
        '[Detailed audit](completion-summary.json) · [Generation plan](selection.json) · [Labeling log](sentence-labeling.log)',
    ]
    if labels.get('labels_path'):
        report.extend(['', 'Sentence labels: ' + labels['labels_path']])
    (root / 'RESULTS.md').write_text('\n'.join(report) + '\n')
    print(json.dumps(dict(passed=passed, **labels), indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
