"""Resume the final labeling pass serially, respecting transient retry intervals."""

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
import label_terminalwrench_glm53_neutral10_20261101 as labeling

batch = labeling.batch
RUN = 'glm53-terminalwrench-interactive-steering-active-time-v2-rh-top4-continuous-v7-neutral10-sample3-20261101-recovery-v2'
PROJECT = HERE.parents[2]
root = PROJECT / 'runs/emotion-steering' / RUN
batch.RUNS = {'GLM 5.3': RUN}
batch.OUT = PROJECT / 'runs/emotion-rh-sentence-labels/glm53-terminalwrench-neutral10-20261101-recovery-v2-gpt61sol-v1'
retry_at = time.time()
for folder in (batch.OUT / 'attempts').iterdir():
    if (batch.OUT / 'annotations' / (folder.name + '.json')).exists():
        continue
    paths = sorted(folder.glob('*.json'))
    if not paths:
        continue
    latest = paths[-1]
    attempt = json.loads(latest.read_text())
    if attempt.get('error') == 'HTTP 402' and 'in_flight_budget_exhausted' in attempt.get('detail', ''):
        retry_at = max(retry_at, latest.stat().st_mtime + 120)
print(f'Waiting {max(0, retry_at-time.time()):.0f}s for the provider retry interval.', flush=True)
while time.time() < retry_at:
    time.sleep(min(10, retry_at-time.time()))
sys.argv = [sys.argv[0], '--workers', '1']
batch.main()
result = subprocess.run([sys.executable, str(HERE / 'summarize_terminalwrench_glm53_neutral10_20261101.py')])
if result.returncode:
    raise SystemExit(result.returncode)
summary = json.loads((root / 'verified-summary.json').read_text())
batch.atomic(root / 'progress.json', dict(status='complete', planned=10, completed=10,
                                         verified_summary='verified-summary.json', updated_unix=time.time()))
print('All ten traces generated, labeled, and verified.', flush=True)
