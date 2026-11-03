"""JSON-configured sentence RH experiment with resumable, verified artifacts."""
import csv
import json
from pathlib import Path
import re
import shutil

import matplotlib
import numpy as np
import sklearn
from sklearn.metrics import roc_curve

from shared.io import atomic, digest, read, require, save_npz, sha
from shared.progress import Progress, lock
from .analysis import (EMOTIONS, BASELINES, assign_folds, balanced_groups, draw_balanced,
                       evaluate, make_pairs, predict_threshold, score)
from .features import extract, verify_cached_features
from .inputs import Sources, load_dataset, load_probes
from .reporting import FORMATS, figure_names, report

CANONICAL = Path(__file__).resolve().parents[2]
PROJECT = CANONICAL.parents[1]
STAGES = ['prepare', 'score', 'evaluate', 'report', 'verify']


def load_config(path):
    path = Path(path).resolve()
    supplied = read(path)
    defaults = dict(schema_version=1, experiment='emotion-rh-forecasting', horizons=list(range(11)),
                    seed=20261016, repeats=2000, cohorts=['all_available', 'common_sources', 'certain_only'],
                    scoring={'backend': 'cached'}, evaluation={'method': 'held_out_task'})
    require(isinstance(supplied, dict) and not (set(supplied)-set(defaults)-{'run_id', 'title', 'probe_bundle', 'dataset', 'reference'}),
            'Unknown forecasting config fields')
    cfg = {**defaults, **supplied}
    require(type(cfg['schema_version']) is int and cfg['schema_version'] == 1 and cfg['experiment'] == 'emotion-rh-forecasting', 'Unsupported forecasting config')
    require(isinstance(cfg.get('run_id'), str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', cfg['run_id']), 'Unsafe/missing run_id')
    cfg.setdefault('title', cfg['run_id'])
    require(isinstance(cfg['title'], str), 'Title must be a string')
    require(isinstance(cfg.get('probe_bundle'), str) and cfg['probe_bundle'], 'Supply a twelve-probe bundle path')
    cfg['probe_bundle'] = str((path.parent/cfg['probe_bundle']).resolve())
    data = cfg.get('dataset')
    require(isinstance(data, dict) and set(data) == {'format', 'path'}, 'Dataset needs format and path')
    require(data['format'] in ('legacy-coding', 'legacy-sales', 'sentence-traces-v1', 'token-replay-v1'), 'Unknown dataset format')
    require(isinstance(data['path'], str) and data['path'], 'Missing dataset path')
    data['path'] = str((path.parent/data['path']).resolve())
    hs = cfg['horizons']
    require(isinstance(hs, list) and hs and all(type(h) is int and 0 <= h <= 10 for h in hs)
            and hs == sorted(set(hs)), 'Horizons must be unique ascending integers in 0..10')
    require(type(cfg['seed']) is int and 0 <= cfg['seed'] < 2**32-1, 'Invalid random seed')
    require(type(cfg['repeats']) is int and cfg['repeats'] >= 2, 'At least two balanced draws required')
    require(isinstance(cfg['cohorts'], list) and 'all_available' in cfg['cohorts']
            and len(set(cfg['cohorts'])) == len(cfg['cohorts'])
            and set(cfg['cohorts']) <= {'all_available', 'common_sources', 'certain_only'}, 'Invalid cohorts')
    ev = cfg['evaluation']
    require(isinstance(ev, dict) and not (set(ev)-{'method', 'folds'})
            and ev.get('method') in ('held_out_task', 'provided', 'grouped'), 'Invalid evaluation method')
    if ev['method'] == 'grouped':
        ev.setdefault('folds', 5)
        require(type(ev['folds']) is int and ev['folds'] >= 2, 'Need at least two grouped folds')
    else:
        require(set(ev) == {'method'}, 'folds applies only to grouped evaluation')
    scoring = cfg['scoring']
    require(isinstance(scoring, dict) and scoring.get('backend') in ('cached', 'local-hf'), 'Scoring backend must be cached or local-hf')
    if scoring['backend'] == 'cached':
        require(set(scoring) == {'backend'} and data['format'] in ('legacy-coding', 'legacy-sales', 'token-replay-v1'),
                'Cached adapter requires a verified token replay dataset')
    else:
        require(data['format'] == 'sentence-traces-v1', 'Local replay expects sentence-traces-v1 inputs')
        hf = dict(backend='local-hf', model_class='AutoModelForCausalLM', dtype='bfloat16', device='cuda',
                  attention='sdpa', layers_path='model.layers', max_tokens=131072, template_kwargs={}, answer_prefix='')
        require(not (set(scoring)-set(hf)), 'Unknown local-hf fields')
        cfg['scoring'] = hf = {**hf, **scoring}
        require(hf['model_class'] in ('AutoModelForCausalLM', 'Gemma3ForConditionalGeneration', 'AutoModelForImageTextToText'), 'Unsupported model class')
        require(hf['dtype'] in ('float32', 'float16', 'bfloat16') and isinstance(hf['device'], str), 'Invalid dtype/device')
        require(hf['attention'] in ('eager', 'sdpa', 'flash_attention_2', 'flex_attention'), 'Invalid attention implementation')
        require(isinstance(hf['template_kwargs'], dict) and isinstance(hf['answer_prefix'], str), 'Invalid replay template settings')
        require(type(hf['max_tokens']) is int and hf['max_tokens'] > 0 and isinstance(hf['layers_path'], str) and hf['layers_path'], 'Invalid context/layers path')
    if 'reference' in cfg:
        ref = cfg['reference']
        require(isinstance(ref, dict) and set(ref) == {'format', 'path'} and ref['format'] in ('legacy-coding', 'legacy-sales'), 'Invalid reference config')
        require(isinstance(ref['path'], str) and ref['path'], 'Reference needs a result-bundle.json path')
        ref['path'] = str((path.parent/ref['path']).resolve())
    return cfg


def resolve(cfg):
    sources = Sources()
    bundle = load_probes(cfg['probe_bundle'], sources)
    traces, jobs = load_dataset(cfg['dataset'], bundle, sources)
    if 'reference' in cfg:
        sources.path(cfg['reference']['path'])  # Used only after analysis, in verification.
    code = list(Path(__file__).parent.glob('*.py')) + [CANONICAL/p for p in ('shared/io.py', 'shared/progress.py', 'shared/server.py')]
    deps = {'numpy': np.__version__, 'scikit-learn': sklearn.__version__, 'matplotlib': matplotlib.__version__}
    if cfg['scoring']['backend'] == 'local-hf':
        import importlib.metadata
        for name in ('torch', 'transformers', 'tokenizers'):
            try:
                deps[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                deps[name] = 'not-installed'
    spec = {**cfg, 'model': bundle['model'], 'probe_digest': digest(bundle), 'input_hashes': sources.files,
            'source_hashes': {str(p.relative_to(CANONICAL)): sha(p) for p in sorted(code)}, 'dependencies': deps,
            'counts': {'traces': len(traces), 'turns': len(jobs), 'sentences': sum(len(t['units']) for t in traces)},
            'protocol': {'feature': 'Character-weighted raw projection on tokens completed by the source sentence end; assistant output only.',
                         'target': 'RH boolean at exactly i+h eligible prose sentences within the same trace; h=0 is contemporaneous.',
                         'training': 'Frozen emotion vectors. Refit RH threshold and polarity on other folds at each horizon/cohort.',
                         'evaluation': 'Balance each test fold separately; same sampled rows for every probe and baseline.',
                         'interval': 'Central 95% of repeated majority-class subsamples, conditional on fixed data and cutoffs; not generalization confidence.'}}
    if cfg['dataset']['format'] == 'token-replay-v1':
        replay = read(Path(cfg['dataset']['path'])/'replay-spec.json')
        spec['protocol']['replay'] = replay['interpretation']
        spec['protocol']['token_position'] = 'Cumulative assistant replay tokens (message and command text); prompts and tool responses excluded.'
    spec['identity'] = digest(spec)
    return spec, bundle, traces, jobs


def run_root(cfg):
    return PROJECT / 'runs/emotion-rh-forecasting' / cfg['run_id']


def freeze(root, spec, bundle, traces, jobs):
    folder = root / 'inputs/source-files'
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for original, checksum in spec['input_hashes'].items():
        destination = folder / (checksum+Path(original).suffix)
        if not destination.exists():
            shutil.copyfile(original, destination)
        require(sha(destination) == checksum, 'Frozen source file checksum mismatch')
        paths[original] = str(destination.relative_to(root))
    frozen_jobs = {k: {**j, **({'cache': str(root/paths[j['cache']])} if 'cache' in j else {})} for k, j in jobs.items()}
    atomic(root / 'inputs/probes.json', bundle)
    atomic(root / 'inputs/traces.json', traces)
    atomic(root / 'inputs/jobs.json', frozen_jobs)
    atomic(root / 'inputs/source-files.json', paths)
    return frozen_jobs


def checkpoint_dir(root, cohort, h):
    return root / 'results' / cohort / f'h{h:02d}'


def pairs_for(rows, spec):
    common = {(r['trace_id'], r['unit_id']) for r in make_pairs(rows, max(spec['horizons']))}
    for h in spec['horizons']:
        for cohort in spec['cohorts']:
            yield cohort, h, make_pairs(rows, h, common if cohort == 'common_sources' else None, cohort == 'certain_only')


def reference_check(root, spec, result):
    if 'reference' not in spec:
        return {'requested': False}
    paths = read(root / 'inputs/source-files.json')
    ref = read(root / paths[spec['reference']['path']])
    old_key = 'horizon' if spec['reference']['format'] == 'legacy-coding' else 'k'
    expected = {(r['cohort'], r[old_key], r['model']): r for r in ref['estimates']}
    actual = {(r['cohort'], r['horizon'], r['model']): r for r in result['estimates']}
    relevant = {key: value for key, value in expected.items() if key[0] in spec['cohorts'] and key[1] in spec['horizons']}
    require(relevant, 'Reference has no overlapping cohort/horizon results')
    fields = ['accuracy', 'balanced_accuracy', 'rh_recall', 'non_rh_specificity', 'repeated_sampling_mean_accuracy', 'sampling_interval95',
              'n', 'rh_n', 'non_rh_n', 'tp', 'tn', 'fp', 'fn']
    error = 0.
    for key, old in relevant.items():
        require(key in actual, f'Missing reference cell: {key}')
        for field in fields:
            a, b = np.asarray(actual[key][field]), np.asarray(old[field])
            delta = float(np.max(np.abs(a-b)))
            require(delta <= 1e-12, f'Reference mismatch {key}/{field}: maximum error {delta}')
            error = max(error, delta)
    for old in ref['populations']:
        key = old['cohort'], old[old_key]
        if key[0] not in spec['cohorts'] or key[1] not in spec['horizons']:
            continue
        new = next(p for p in result['populations'] if (p['cohort'], p['horizon']) == key)
        for field in ('eligible_pairs', 'balanced_eval_pairs', 'balanced_rh', 'balanced_non_rh'):
            require(new[field] == old[field], f'Reference population differs: {key}/{field}')
    return {'requested': True, 'passed': True, 'compared_cells': len(relevant), 'maximum_absolute_error': error,
            'horizons': sorted({k[1] for k in relevant}), 'reference_sha256': spec['input_hashes'][spec['reference']['path']]}


def verify(root, spec, rows, result):
    """Audit pair identity, train-only rule optima, predictions and draw summaries."""
    paths = read(root / 'inputs/source-files.json')
    for original, checksum in spec['input_hashes'].items():
        require(sha(root / paths[original]) == checksum, 'Frozen input was modified')
    require(result['identity'] == spec['identity'] and result['emotions'] == EMOTIONS, 'Result identity differs')
    features_checked = verify_cached_features(root, spec, rows)
    checked = 0
    for cohort, h, pairs in pairs_for(rows, spec):
        directory = checkpoint_dir(root, cohort, h)
        saved = read(directory / 'metrics.json')
        require(len(pairs) == len(saved['pair_index']), 'Pair index count differs')
        with np.load(directory / 'oof-predictions.npz', allow_pickle=False) as z:
            models = z['models'].astype(str).tolist()
            require(models == EMOTIONS+BASELINES, 'Prediction model order differs')
            y, folds = z['labels'], z['folds']
            require(y.dtype == bool and np.array_equal(y, [r['hacking'] for r in pairs])
                    and np.array_equal(folds, [r['fold'] for r in pairs]), 'Prediction labels/folds differ')
            for i, row in enumerate(saved['pair_index']):
                require(all(row[k] == pairs[i][k] for k in row if k != 'selected'), 'Pair source/target identity differs')
            for rule in saved['rules']:
                model, fold = rule['model'], rule['fold']
                train, test = folds != fold, folds == fold
                for key in ('group_id', 'trace_id'):
                    require(not {r[key] for i, r in enumerate(pairs) if train[i]} & {r[key] for i, r in enumerate(pairs) if test[i]}, 'Cross-fold leakage')
                x = np.array([r['mean_token_position'] if model == 'token_position' else r['projections'][model] for r in pairs])
                fitted = predict_threshold(x[train], rule)
                ba = .5*(fitted[y[train]].mean()+(~fitted[~y[train]]).mean())
                fpr, tpr, _ = roc_curve(y[train], x[train], drop_intermediate=False)
                optimal = max(float((.5*(tpr+1-fpr)).max()), float((.5*(1-tpr+fpr)).max()))
                require(abs(ba-optimal) <= 1e-12 and abs(ba-rule['training_balanced_accuracy']) <= 1e-12, 'Training rule is not BA-optimal')
                require(rule['training_n'] == int(train.sum()), 'Training count differs')
                require(np.array_equal(z['predictions'][models.index(model), test], predict_threshold(x[test], rule)), 'Saved predictions differ from frozen rule')
                checked += 1
            require(not z['predictions'][models.index('always_not_hacking')].any(), 'Constant baseline differs')
            require(np.array_equal(z['predictions'][models.index('oracle_label_persistence')], [r['source_hacking'] for r in pairs]), 'Oracle diagnostic differs')
            groups = balanced_groups(y, folds)
            chosen = draw_balanced(groups, np.random.default_rng(spec['seed']))
            require(np.array_equal(z['selected_indices'], chosen), 'Seeded balanced sample differs')
            require([r['selected'] for r in saved['pair_index']] == [i in set(chosen.tolist()) for i in range(len(pairs))], 'Pair selection flags differ')
            samples = z['sampling_accuracies']
            require(samples.shape == (spec['repeats'], len(models)), 'Incomplete sampling draws')
            rng = np.random.default_rng(spec['seed']+1)
            for repeat in range(spec['repeats']):
                ix = draw_balanced(groups, rng)
                require(np.array_equal(samples[repeat], (z['predictions'][:, ix] == y[ix]).mean(axis=1)), 'Sampling draw does not reproduce')
            for j, estimate in enumerate(saved['estimates']):
                require(estimate['model'] == models[j], 'Estimate model order differs')
                require(all(estimate[k] == v for k, v in score(y[chosen], z['predictions'][j, chosen]).items()), 'Confusion metrics differ')
                require(estimate['repeated_sampling_mean_accuracy'] == float(samples[:, j].mean()) and
                        estimate['sampling_interval95'] == np.percentile(samples[:, j], [2.5, 97.5]).tolist(), 'Sampling summary differs')
                require(estimate in result['estimates'], 'Aggregate metrics differ')
            require(saved['population'] in result['populations'] and all(s in result['fold_statistics'] for s in saved['fold_statistics']), 'Aggregate counts differ')
    require(read(root / 'chart-data.json')['estimates'] == result['estimates'], 'Plot data differs')
    with (root / 'metrics.csv').open(newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == len(result['estimates']) and all(float(c['repeated_sampling_mean_accuracy']) == r['repeated_sampling_mean_accuracy']
            for c, r in zip(csv_rows, result['estimates'])), 'CSV differs')
    return dict(passed=True, identity=spec['identity'], rules_checked=checked, source_sentences_checked=features_checked,
                reference=reference_check(root, spec, result), checks=[
                    'Frozen input hashes and probe identity', 'Original sentence order; no cross-trace pairs',
                    ('Cached features re-pooled from frozen token scores' if spec['scoring']['backend'] == 'cached'
                     else 'Local prefix token identity and sentence-span evidence'),
                    'Group/trace isolation', 'Independent ROC rule optimum',
                    'Saved-rule predictions', 'Seeded balanced samples shared across probes',
                    'Every repeated draw and metric summary', 'CSV/JSON plot data parity'])


def execute(root, spec, bundle, traces, jobs, resume=False):
    root = Path(root)
    with lock(root):
        config_path = root / 'config.resolved.json'
        if config_path.exists():
            require(resume, 'Run exists; use --resume or a new run_id')
            require(read(config_path) == spec, 'Inputs, config, dependencies or code changed; use a new run_id')
        else:
            require(all(p.name == '.run.lock' for p in root.iterdir()), 'Output directory is not empty')
            atomic(config_path, spec)
        if (root / 'verification.json').exists():
            receipt = read(root / 'verification.json')
            require(receipt['passed'] and receipt['identity'] == spec['identity'], 'Invalid prior receipt')
            require(all(sha(root/name) == checksum for name, checksum in receipt['output_hashes'].items()), 'Completed output was modified')
            return read(root / 'metrics.json')
        with Progress(root, spec['title']) as progress:
            progress.data['stages'] = [{'id': s, 'label': s.title(), 'state': 'pending'} for s in STAGES]
            progress.stage('prepare', message='Freezing probe identities, labeled traces and replay evidence')
            for name in ('dashboard.html', 'dashboard.css', 'dashboard.js'):
                shutil.copyfile(PROJECT / 'dashboards' / name, root / ('index.html' if name == 'dashboard.html' else name))
            jobs = freeze(root, spec, bundle, traces, jobs)
            progress.complete_stage('prepare')
            progress.stage('score', 0, spec['counts']['sentences'], 'sentences', 'Pooling completed source-sentence tokens')
            rows, drops = extract(root, spec, bundle, traces, jobs, progress)
            folds = assign_folds(rows, spec['evaluation'], spec['seed'])
            atomic(root / 'sentence-features.json', rows)
            atomic(root / 'dropped-features.json', drops)
            atomic(root / 'folds.json', folds)
            progress.complete_stage('score')
            result = dict(identity=spec['identity'], experiment=spec['experiment'], model=spec['model'], emotions=EMOTIONS,
                          horizons=spec['horizons'], protocol=spec['protocol'], interval=spec['protocol']['interval'],
                          counts={**spec['counts'], 'unscorable_sentences': len(drops), 'folds': len(set(folds['trace_folds'].values()))},
                          estimates=[], populations=[], fold_statistics=[])
            total = len(spec['horizons'])*len(spec['cohorts'])
            expected_folds = sorted(set(folds['trace_folds'].values()))
            for done, (cohort, h, pairs) in enumerate(pairs_for(rows, spec), 1):
                progress.stage('evaluate', done-1, total, 'cohort-horizons', f'{cohort} · +{h} sentences')
                directory = checkpoint_dir(root, cohort, h)
                binding = digest({'identity': spec['identity'], 'pairs': pairs, 'cohort': cohort, 'horizon': h})
                if (directory / 'receipt.json').exists():
                    cached = read(directory / 'receipt.json')
                    require(cached['binding'] == binding and all(sha(directory/n) == value for n, value in cached['files'].items()), 'Corrupt evaluation checkpoint')
                    values = read(directory / 'metrics.json')
                else:
                    values, arrays = evaluate(pairs, cohort, h, spec['seed'], spec['repeats'], expected_folds)
                    directory.mkdir(parents=True, exist_ok=True)
                    atomic(directory / 'metrics.json', values)
                    save_npz(directory / 'oof-predictions.npz', **arrays)
                    atomic(directory / 'receipt.json', {'binding': binding, 'files': {n: sha(directory/n) for n in ('metrics.json', 'oof-predictions.npz')}})
                result['estimates'].extend(values['estimates'])
                result['fold_statistics'].extend(values['fold_statistics'])
                result['populations'].append(values['population'])
                progress.stage('evaluate', done, total, 'cohort-horizons', f'{cohort} · +{h} sentences')
                print(f'{cohort} +{h}: {len(pairs)} eligible / {values["population"]["balanced_eval_pairs"]} balanced pairs', flush=True)
            progress.complete_stage('evaluate')
            atomic(root / 'metrics.json', result)
            progress.stage('report', message='Rendering twelve-panel trends and forecasting heatmaps')
            report(root, spec, result)
            progress.complete_stage('report')
            progress.stage('verify', message='Auditing predictions and comparing archived reference results')
            receipt = verify(root, spec, rows, result)
            for name in figure_names(spec):
                for ext in FORMATS:
                    require((root / 'figures' / f'{name}.{ext}').stat().st_size > 0, 'Missing figure')
            receipt['output_hashes'] = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()
                                       and p.name not in ('progress.json', 'events.jsonl', '.run.lock', 'verification.json') and 'logs' not in p.relative_to(root).parts}
            atomic(root / 'verification.json', receipt)
            progress.complete_stage('verify')
            progress.finish(True)
            return result


def main(config, dry_run=False, resume=False, serve_port=None):
    cfg = load_config(config)
    root = run_root(cfg)
    if serve_port:
        from shared.server import serve
        serve(root, serve_port)
        return
    spec, bundle, traces, jobs = resolve(cfg)
    if dry_run:
        print(json.dumps({'run_dir': str(root), 'model': bundle['model']['name'], 'counts': spec['counts'],
                          'backend': cfg['scoring']['backend'], 'horizons': cfg['horizons'], 'cohorts': cfg['cohorts'],
                          'evaluation': cfg['evaluation'], 'identity': spec['identity']}, indent=2))
        return
    execute(root, spec, bundle, traces, jobs, resume)
    print(json.dumps({'run_dir': str(root), 'verified': True, 'report': str(root/'report.html')}, indent=2))
