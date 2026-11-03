"""Resume a frozen run from extraction without scanning every token-cache file."""

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

from pathlib import Path
import time

from shared.io import atomic, read, require, sha
from shared.progress import Progress, STAGES, lock, now
from emotion_probes.config import source_hashes


def extract_with_policy(root, spec, progress, smoke_only, extract):
    """Apply an explicitly recorded validation override without changing readout."""
    from emotion_probes import extraction
    path = root / 'execution-policy.json'
    if not path.exists():
        return extract(root, spec, progress, smoke_only)
    policy = read(path)
    require(policy.get('identity') == spec['identity'], 'Execution policy belongs to another run')
    require(policy.get('capture_smoke') == 'skip_tolerance_checks', 'Unknown capture validation policy')
    require(not smoke_only, 'A smoke-only run cannot skip its smoke checks')
    evidence = {**policy, 'policy_sha256': sha(path), 'entrypoint_sha256': sha(Path(__file__)),
                'status': 'skipped', 'passed': None,
                'checks_skipped': ['repeated-forward tolerance', 'cross-forward raw-pooling tolerance'],
                'checks_retained': ['input token identity', 'target token counts at every layer',
                                    'activation shape', 'finite activation values', 'shard checksums']}
    def record_skip(root, spec, extractor, progress, cases):
        atomic(root / 'smoke.json', evidence)
        atomic(root / 'probes/execution-policy.json', evidence)
        progress.stage('extract', message='Extracting activations; tolerance checks skipped by request')
    original = extraction.smoke
    extraction.smoke = record_skip
    try:
        return extract(root, spec, progress, smoke_only)
    finally:
        extraction.smoke = original


def execute(root, through='verify', smoke_only=False):
    from emotion_probes.extraction import extract
    from emotion_probes.fitting import fit
    from emotion_probes.evaluation import calibrate, evaluate
    from emotion_probes.reporting import report, verify
    root = Path(root)
    spec = read(root / 'config.resolved.json')
    through = 'extract' if smoke_only else through
    require(through in STAGES[STAGES.index('extract'):], 'Cached-token resume starts at extraction')
    require(spec['source_hashes'] == source_hashes(), 'Scientific code changed since preparation')
    with lock(root), Progress(root, spec['run_id']) as progress:
        # These small manifests attest the completed tokenization. Individual
        # token records are checked on demand by the existing extraction code.
        summary = read(root / 'tokenization.json')
        total = sum(spec['counts'][p] for p in ('training', 'eval'))
        require(summary['identity'] == spec['identity'] and summary['cases'] == total,
                'Missing or incompatible completed token cache')
        for partition in ('training', 'eval'):
            manifest = read(root / 'tokens' / partition / 'manifest.json')
            require(manifest['identity'] == spec['identity']
                    and len(manifest['records']) == spec['counts'][partition],
                    f'Incomplete token manifest: {partition}')
        for name, checksum in spec['input_hashes'].items():
            require(sha(root / 'inputs' / name) == checksum, 'Prepared input changed')
        atomic(root / 'logs' / f'cached-token-resume-{time.time_ns()}.json', {
            'identity': spec['identity'], 'started_at': now(), 'cases': total,
            'entrypoint_sha256': sha(Path(__file__)),
            'mode': 'reuse completed token cache; validate individual records during extraction',
        })
        progress.stage('prepare', 4, 4, 'input files', 'Frozen input files verified')
        progress.complete_stage('prepare')
        progress.stage('tokenize', total, total, 'cached cases', 'Reusing completed token cache; standalone scan skipped')
        progress.complete_stage('tokenize')
        tasks = {'extract': extract, 'fit': fit, 'calibrate': calibrate,
                 'evaluate': evaluate, 'report': report, 'verify': verify}
        for name in STAGES[STAGES.index('extract'):STAGES.index(through)+1]:
            progress.stage(name, message=f'Starting {name}')
            if name == 'extract':
                if not extract_with_policy(root, spec, progress, smoke_only, extract):
                    next(s for s in progress.data['stages'] if s['id'] == name)['state'] = 'pending'
                    progress.finish(False)
                    return
            else:
                tasks[name](root, spec, progress)
            progress.complete_stage(name)
        progress.finish(through == 'verify')
