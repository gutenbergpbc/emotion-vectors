"""Canonical baseline collection, historical import, analysis and verification."""
import json
from pathlib import Path
import shutil

import matplotlib
import numpy as np
import scipy
from scipy.stats import rankdata

from shared.io import atomic, digest, read, require, sha
from shared.progress import Progress, lock
from .analysis import analyze, join
from .collection import collect, import_legacy, request_body
from .config import CANONICAL, PROJECT, QUESTIONS, load_config, run_root
from .inputs import load_inputs
from .reporting import report, write_csv

STAGES = ['prepare', 'collect', 'analyze', 'report', 'verify']


def resolve(cfg):
    cases, points, scores, files, legacy = load_inputs(cfg)
    sources = list(Path(__file__).parent.glob('*.py')) + [CANONICAL / name for name in
               ('emotion-self-report/run.py', 'shared/io.py', 'shared/progress.py', 'shared/server.py', 'emotion-probes/emotion_probes/data.py')]
    spec = {**cfg, 'question': QUESTIONS[cfg['prompt']], 'scale': [0, 10],
            'inputs': {key: {'path': str(path), 'sha256': sha(path)} for key, path in sorted(files.items())},
            'counts': {'traces': len({p['trace_id'] for p in points}), 'checkpoints': len(points),
                       'requests': len(points) * len(cfg['emotions']), 'matched_probe_scores': len(scores)},
            'checkpoint_hash': digest(points),
            'source_hashes': {str(p.relative_to(CANONICAL)): sha(p) for p in sorted(sources)},
            'dependencies': {'numpy': np.__version__, 'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__},
            'replay': 'Independent visible-text prefix at each assistant sentence/checkpoint; no future messages, outcome labels, probe scores or earlier ratings in requests.',
            'collection_settings': legacy['settings'] if legacy else {
                'model': cfg['model'], **{k: cfg['api'][k] for k in ('temperature', 'max_tokens', 'reasoning', 'provider')}},
            'legacy_manifest': legacy}
    if not cfg['checkpoints'] and cfg['mode'] == 'collect':
        from importlib.metadata import version
        spec['dependencies']['pysbd'] = version('pysbd')
    spec['identity'] = digest(spec)
    return spec, (cases, points, scores)


def freeze_inputs(root, spec):
    for name, source in spec['inputs'].items():
        path = Path(source['path'])
        destination = root / 'inputs' / (name + path.suffix)
        destination.parent.mkdir(exist_ok=True)
        if destination.exists():
            require(sha(destination) == source['sha256'], f'Frozen input changed: {name}')
        else:
            shutil.copyfile(path, destination)
            require(sha(destination) == source['sha256'], f'Input changed during copy: {name}')


def verify(root, spec, points, scores):
    """Check cardinality, exact joins, and coefficients using independent formulas."""
    ratings = read(root / 'ratings.json')
    expected = {(p['trace_id'], p['unit_id'], e) for p in points for e in spec['emotions']}
    require(len(ratings) == len(expected) and {(r['trace_id'], r['unit_id'], r['emotion']) for r in ratings} == expected,
            'Missing/duplicate/unexpected rating rows')
    for row in ratings:
        if row['rating_status'] == 'numeric':
            require(type(row['rating']) is int and 0 <= row['rating'] <= 10, 'Invalid 0–10 rating')
        else:
            require(row['rating_status'] in ('explicit_null', 'non_numeric_response') and row['rating'] is None,
                    'Missing rating was coerced or request failed')
    paired = read(root / 'paired-scores.json')
    require(paired == join(ratings, scores), 'Saved probe-score join changed')
    result = read(root / 'metrics.json')
    require(result['identity'] == spec['identity'], 'Result identity changed')
    for record in result['results']:
        rows = [r for r in paired if r['emotion'] == record['emotion'] and
                r['label'] not in spec['exclude_labels'] and r['rating'] is not None and spec['correlation_score'] in r]
        require(len(rows) == record['pairs'] and len({r['trace_id'] for r in rows}) == record['traces'],
                'Correlation denominator mismatch')
        x, y = [np.array([r[k] for r in rows], dtype=float) for k in (spec['correlation_score'], 'rating')]
        for name, a, b in [('pearson', x, y), ('spearman', rankdata(x), rankdata(y))]:
            expected_value = None
            if len(a) >= 2 and np.ptp(a) > 0 and np.ptp(b) > 0:
                expected_value = float(np.corrcoef(a, b)[0, 1])
            require(record[name] is None if expected_value is None else record[name] is not None and
                    np.isclose(record[name], expected_value, rtol=1e-12, atol=1e-12), 'Correlation does not reproduce')
            low, high = record[name + '_ci_low'], record[name + '_ci_high']
            require(low is None and high is None or low is not None and high is not None and -1 <= low <= high <= 1,
                    'Invalid correlation interval')
            require(record[name + '_valid_bootstraps'] + record[name + '_undefined_bootstraps'] +
                    record[name + '_skipped_bootstraps'] == spec['bootstrap_replicates'],
                    'Bootstrap draw count mismatch')
    for name, source in spec['inputs'].items():
        require(sha(root / 'inputs' / (name + Path(source['path']).suffix)) == source['sha256'], 'Frozen input hash mismatch')
    for name in ('report.html', 'metrics.csv', 'figures/score-correlations.png', 'figures/score-correlations.svg',
                 'figures/score-correlations.pdf', 'figures/rating-distribution.png'):
        require((root / name).stat().st_size > 0, f'Missing artifact: {name}')
    return dict(passed=True, identity=spec['identity'], requests=len(ratings),
                checks=['frozen input hashes', 'complete checkpoint/emotion grid', '0–10/missing rating semantics',
                        'exact checkpoint join', 'no future probe token', 'independent Pearson/Spearman recomputation',
                        'bootstrap interval/count checks', 'figure/report artifacts'])


def execute(root, spec, inputs, resume=False, through='verify', send=None):
    root = Path(root)
    cases, points, scores = inputs
    require(through in STAGES, 'Unknown self-report stage')
    with lock(root):
        if (root / 'config.resolved.json').exists():
            require(resume, 'Run exists; use --resume or a new run_id')
            require(read(root / 'config.resolved.json') == spec, 'Configuration, inputs, dependencies or code changed; choose a new run_id')
        else:
            atomic(root / 'config.resolved.json', spec)
        if (root / 'verification.json').exists():
            receipt = read(root / 'verification.json')
            require(receipt['passed'], 'Prior verification failed')
            for name, checksum in receipt['output_hashes'].items():
                require(sha(root / name) == checksum, f'Completed output changed: {name}')
            verify(root, spec, points, scores)
            return read(root / 'metrics.json')
        for name in ('dashboard.html', 'dashboard.js', 'dashboard.css'):
            shutil.copyfile(PROJECT / 'dashboards' / name, root / ('index.html' if name == 'dashboard.html' else name))
        progress = Progress(root, f"{spec['model']} · {spec['environment']} · self-report baseline")
        if [s['id'] for s in progress.data['stages']] != STAGES:
            progress.data['stages'] = [dict(id=s, label=s.title(), state='pending') for s in STAGES]
        with progress:
            progress.stage('prepare', message='Freezing cases and sentence checkpoints')
            freeze_inputs(root, spec)
            atomic(root / 'checkpoints.json', points)
            atomic(root / 'cases.json', cases)
            example = request_body(spec, cases[points[0]['trace_id']], points[0], spec['emotions'][0])
            if spec['mode'] == 'import':
                example = spec['legacy_manifest']['settings'] | {'messages': example['messages']}
            atomic(root / 'example-request.json', example)
            progress.complete_stage('prepare')
            if through == 'prepare':
                progress.finish(False)
                return
            progress.stage('collect', 0, spec['counts']['requests'], 'requests')
            if spec['mode'] == 'import':
                ratings = import_legacy(spec, cases, points, root / 'inputs/legacy_responses.jsonl', spec['legacy_manifest'], progress)
            else:
                ratings = collect(root, spec, cases, points, progress, send)
            ratings.sort(key=lambda r: (r['trace_id'], r['sentence_index'], spec['emotions'].index(r['emotion'])))
            atomic(root / 'ratings.json', ratings)
            write_csv(root / 'ratings.csv', ratings)
            progress.complete_stage('collect')
            if through == 'collect':
                progress.finish(False)
                return
            progress.stage('analyze', message='Pairing exact checkpoints and resampling whole traces')
            paired = join(ratings, scores)
            atomic(root / 'paired-scores.json', paired)
            result = analyze(paired, spec, progress)
            atomic(root / 'metrics.json', result)
            progress.complete_stage('analyze')
            if through == 'analyze':
                progress.finish(False)
                return result
            progress.stage('report', message='Writing Pearson/Spearman figure and rating distributions')
            report(root, result, paired)
            progress.complete_stage('report')
            if through == 'report':
                progress.finish(False)
                return result
            progress.stage('verify', message='Independently checking exported coefficients and artifacts')
            receipt = verify(root, spec, points, scores)
            paths = [p for p in root.rglob('*') if p.is_file() and
                     p.name not in ('progress.json', 'events.jsonl', '.run.lock', 'verification.json') and
                     'logs' not in p.relative_to(root).parts]
            receipt['output_hashes'] = {str(p.relative_to(root)): sha(p) for p in sorted(paths)}
            atomic(root / 'verification.json', receipt)
            progress.complete_stage('verify')
            progress.finish(True)
            return result


def main(config, dry_run=False, resume=False, serve_port=None, through='verify'):
    cfg = load_config(config)
    root = run_root(cfg)
    if serve_port:
        from shared.server import serve
        serve(root, serve_port)
        return
    spec, inputs = resolve(cfg)
    if dry_run:
        print(json.dumps({'run_dir': str(root), 'model': spec['model'], 'environment': spec['environment'],
                          'mode': spec['mode'], 'counts': spec['counts'], 'scale': spec['scale'],
                          'prompt': spec['prompt'], 'api_calls': spec['counts']['requests'] if spec['mode'] == 'collect' else 0,
                          'identity': spec['identity']}, indent=2))
        return
    execute(root, spec, inputs, resume=resume, through=through)
    print(f'Run: {root}')
