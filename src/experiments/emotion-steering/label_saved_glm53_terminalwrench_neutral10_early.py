"""Populate the final labeling cache for saved traces while generation continues."""

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
from pathlib import Path
import sys
import threading

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import label_terminalwrench_glm53_neutral10_20261101 as labeling
from shared.io import atomic, read, sha

batch = labeling.batch
PROJECT = HERE.parents[2]
RUN = 'glm53-terminalwrench-interactive-steering-active-time-v2-rh-top4-continuous-v7-neutral10-sample3-20261101-recovery-v2'
ROOT = PROJECT / 'runs/emotion-steering' / RUN
FINAL = PROJECT / 'runs/emotion-rh-sentence-labels/glm53-terminalwrench-neutral10-20261101-recovery-v2-gpt61sol-v1'
batch.RUNS = {'GLM 5.3': RUN}
batch.OUT = FINAL.with_name(FINAL.name + '-early-snapshot')
if (batch.OUT / 'manifest.json').exists():
    frozen = read(batch.OUT / 'manifest.json')
    allowed = {r['id'] for r in frozen['records'] + frozen['exclusions']}
    source_rows = batch.source_rows
    batch.source_rows = lambda: [r for r in source_rows() if f'{r[0]}/{r[2].stem}' in allowed]

manifest, requests = batch.prepare()
config = dict(output=str(FINAL), model=batch.MODEL, allowed_returned_models=[batch.MODEL], max_attempts=4)
stop = threading.Event()
with ThreadPoolExecutor(max_workers=1) as pool:
    results = list(pool.map(lambda req: batch.judge.annotate(req, config, stop), requests))
units = [label for result in results for label in result['labels']]
summary = dict(status='partial_batch', planned_traces=10, labeled_saved_traces=len(manifest['records']),
               sentences=len(units), rh_including_review=sum(u['hacking'] for u in units),
               rh_unflagged=sum(u['hacking'] and not u['needs_review'] for u in units),
               non_rh=sum(not u['hacking'] for u in units), review_flagged=sum(u['needs_review'] for u in units),
               manifest_path=str(batch.OUT / 'manifest.json'), manifest_sha256=sha(batch.OUT / 'manifest.json'),
               note='Partial result only. Exact request-ID/hash-bound annotations are cached for the final ten-trace labeling pass; no duplicate judge calls are needed.')
atomic(ROOT / 'early-labeling-summary.json', summary)
atomic(FINAL / 'early-labeling-source-receipt.json', summary)
print(json.dumps(summary, indent=2), flush=True)
