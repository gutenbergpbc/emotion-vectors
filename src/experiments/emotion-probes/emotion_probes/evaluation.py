import csv
import numpy as np

from shared.io import atomic, read, save_npz, sha
from .data import EMOTIONS, RECIPES
from .extraction import cases_for, feature_manifest, layer_features
from .statistics import bootstrap, bootstrap_setup, choose_threshold, measure, verify_threshold


def calibrate(root, spec, progress):
    models = read(root / 'selected_models.json')['models']
    cases = cases_for(root, spec)['eval']
    manifest = feature_manifest(root, spec, 'eval', cases)
    rows = read(root / 'inputs/split_manifest.json')['rows']
    labels = np.array([r['label'] for r in rows])
    cal = np.array([r['split'] == 'calibration' for r in rows])
    scores = {}
    for layer in sorted({m['layer'] for m in models.values()}):
        x = layer_features(root, 'eval', layer, manifest)
        for key, model in models.items():
            if model['layer'] == layer:
                scores[key] = x @ np.asarray(model['vector'])
        progress.stage('calibrate', len(scores), 36, 'probes', f'Scoring selected layer {layer}')
    calibrated, metrics = {}, {}
    for key, model in models.items():
        cut = choose_threshold(labels[cal], scores[key][cal], model['emotion'])
        verify_threshold(labels[cal], scores[key][cal], model['emotion'], cut)
        calibrated[key] = {**model, 'threshold': cut}
        metrics[key] = measure(labels[cal], scores[key][cal], cut, model['emotion'])
    # Freeze all cutoffs and probe selections before computing held-out metrics.
    atomic(root / 'calibrated_models.json', {'identity': spec['identity'], 'models': calibrated})
    atomic(root / 'calibration_results.json', metrics)
    selected = {}
    for emotion in EMOTIONS:
        keys = [f'{emotion}__{r}' for r in RECIPES]
        best = max(keys, key=lambda k: (metrics[k]['balanced_accuracy'], -RECIPES.index(calibrated[k]['recipe'])))
        selected[emotion] = best
    atomic(root / 'probes/selected-12.json', {'method': 'highest calibration mixed BA; ties in manifest contrast order', 'probes': selected})
    for partition in ('calibration', 'heldout'):
        mask = np.array([r['split'] == partition for r in rows])
        save_npz(root / f'probes/scores/{partition}.npz', ids=np.array([r['id'] for r in rows])[mask],
                 labels=labels[mask], **{k: s[mask] for k, s in scores.items()})
        atomic(root / f'probes/scores/{partition}.rows.json', [r for r in rows if r['split'] == partition])
    progress.stage('calibrate', 36, 36, 'probes', 'Cutoffs and selected twelve probes frozen')


def evaluate(root, spec, progress):
    models = read(root / 'calibrated_models.json')['models']
    rows = read(root / 'probes/scores/heldout.rows.json')
    labels = np.array([r['label'] for r in rows])
    calibration = read(root / 'calibration_results.json')
    setup = bootstrap_setup(rows, spec['bootstrap_draws'], spec['bootstrap_seed'])
    results = {}
    with np.load(root / 'probes/scores/heldout.npz', allow_pickle=False) as scores:
        for key, model in models.items():
            s = scores[key]
            results[key] = {'emotion': model['emotion'], 'recipe': model['recipe'], 'layer': model['layer'],
                            'threshold': model['threshold'], 'calibration': calibration[key],
                            'heldout': measure(labels, s, model['threshold'], model['emotion']),
                            'heldout_bootstrap': bootstrap(s, model['threshold'], model['emotion'], setup)}
            atomic(root / 'metrics.json', {'identity': spec['identity'], 'model': spec['model'],
                   'synthetic': spec['backend'] == 'synthetic', 'quantization': spec['quantization'],
                   'state': 'complete' if len(results) == 36 else 'partial', 'results': results})
            progress.stage('evaluate', len(results), 36, 'probes', key)
    with (root / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['emotion', 'contrast', 'layer', 'threshold', 'heldout_mixed_ba', 'ci_low', 'ci_high'])
        for value in results.values():
            writer.writerow([value['emotion'], value['recipe'], value['layer'], value['threshold'],
                             value['heldout']['balanced_accuracy'], *value['heldout_bootstrap']['balanced_accuracy_95_interval']])
    bundle = root / 'probes'
    atomic(bundle / 'manifest.json', {
        'schema_version': 1, 'identity': spec['identity'], 'model': spec['model'], 'revision': spec['revision'],
        'tokenizer_model': spec['tokenizer_model'], 'tokenizer_revision': spec['tokenizer_revision'],
        'training_family': spec.get('training_family', 'diversity'),
        'training_readout': spec.get('training_readout', spec['readout']),
        'quantization': spec['quantization'], 'centering': spec['centering'], 'readout': spec['readout'],
        'synthetic': spec['backend'] == 'synthetic', 'emotions': EMOTIONS, 'contrasts': RECIPES,
        'layers': spec['layers'], 'width': spec['width'], 'input_hashes': spec['input_hashes'],
        'vector_variants': {
            'primary': {'centering': spec['centering'], 'vectors': 'vectors.npz',
                        'all_layers': 'all-layer-vectors.npz', 'calibrated': True,
                        'layer_selection': 'training-only grouped cross-validation'},
            'uncentered': {'centering': 'none', 'vectors': 'uncentered-vectors.npz',
                           'all_layers': 'all-layer-uncentered-vectors.npz', 'calibrated': False,
                           'layer_selection': 'same layers as primary; no separate selection or calibration'},
        },
        'models': {k: {a: b for a, b in m.items() if a != 'vector'} for k, m in models.items()},
        'files': {str(p.relative_to(bundle)): sha(p) for p in sorted(bundle.rglob('*')) if p.is_file() and p.name != 'manifest.json'},
    })
