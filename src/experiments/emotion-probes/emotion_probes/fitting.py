"""Training-only source-grouped CV; condition-weighted mean differences."""
import numpy as np
from sklearn.metrics import roc_auc_score

from shared.io import atomic, read, require, save_npz
from .data import EMOTIONS, RECIPES, training_conditions
from .extraction import cases_for, feature_manifest, layer_features


def normalize(value):
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    require(np.isfinite(norm).all() and (norm > 1e-10).all(), 'Degenerate/nonfinite direction')
    return value / norm


def directions(means, centering='within_contrast'):
    positive = np.stack([means[(e, e)] for e in EMOTIONS])
    neutral = np.stack([means[(e, 'neutral')] for e in EMOTIONS])
    other = (positive.sum(axis=0) - positive) / 11
    raw = {'neutral': positive-neutral, 'other11_neutral': positive-(11*other+neutral)/12,
           'other11': positive-other}
    raw = {r: normalize(v) for r, v in raw.items()}
    return {r: normalize(v-v.mean(axis=0)) for r, v in raw.items()} if centering == 'within_contrast' else raw


def negative_conditions(emotion, recipe):
    return ([(e, e) for e in EMOTIONS if e != emotion] if recipe != 'neutral' else []) + (
        [(emotion, 'neutral')] if recipe != 'other11' else [])


def winning_layer(values):
    return int(np.flatnonzero(values >= np.max(values)-1e-6)[0])


def fit(root, spec, progress):
    cases = cases_for(root, spec)['training']
    manifest = feature_manifest(root, spec, 'training', cases)
    conditions = {key: np.asarray(indices) for key, indices in training_conditions(cases).items()}
    folds = np.array([c['fold'] for c in cases])
    keys = [f'{e}__{r}' for e in EMOTIONS for r in RECIPES]
    cv = {key: np.empty((5, spec['layers'])) for key in keys}
    vectors = {key: np.empty((spec['layers'], spec['width']), dtype=np.float32) for key in keys}
    uncentered = {key: np.empty_like(value) for key, value in vectors.items()}
    total = spec['layers'] * 5
    for layer in range(spec['layers']):
        checkpoint = root / 'checkpoints' / 'fit' / f'{layer:03d}.json'
        vector_path = checkpoint.with_suffix('.npz')
        if checkpoint.exists() and vector_path.exists():
            from shared.io import sha
            saved = read(checkpoint)
            require(saved['identity'] == spec['identity'] and saved['vectors_sha256'] == sha(vector_path), 'Stale/corrupt fit checkpoint')
            with np.load(vector_path, allow_pickle=False) as cache:
                for key in keys:
                    cv[key][:, layer] = saved['aurocs'][key]
                    vectors[key][layer] = cache[key]
                    uncentered[key][layer] = cache[f'uncentered__{key}']
            progress.stage('fit', (layer+1)*5, total, 'layer-folds', f'Cached layer {layer}')
            continue
        x = layer_features(root, 'training', layer, manifest)
        for fold in range(5):
            means = {key: x[ix[folds[ix] != fold]].mean(axis=0, dtype=np.float64) for key, ix in conditions.items()}
            fitted = directions(means, spec['centering'])
            for ei, emotion in enumerate(EMOTIONS):
                for recipe in RECIPES:
                    neg = negative_conditions(emotion, recipe)
                    cond = [(emotion, emotion)] + neg
                    indices = [conditions[k][folds[conditions[k]] == fold] for k in cond]
                    require(all(len(ix) for ix in indices), 'Missing validation condition')
                    ix = np.concatenate(indices)
                    labels = np.concatenate([np.full(len(ix), j == 0) for j, ix in enumerate(indices)])
                    weights = np.concatenate([np.full(len(ix), (1 if j == 0 else 1/len(neg))/len(ix)) for j, ix in enumerate(indices)])
                    cv[f'{emotion}__{recipe}'][fold, layer] = roc_auc_score(labels, x[ix] @ fitted[recipe][ei], sample_weight=weights)
        means = {key: x[ix].mean(axis=0, dtype=np.float64) for key, ix in conditions.items()}
        full = directions(means, spec['centering'])
        raw = directions(means, 'none')
        layer_vectors = {f'{e}__{r}': full[r][i].astype(np.float32) for i, e in enumerate(EMOTIONS) for r in RECIPES}
        raw_vectors = {f'{e}__{r}': raw[r][i].astype(np.float32) for i, e in enumerate(EMOTIONS) for r in RECIPES}
        for key in keys:
            vectors[key][layer] = layer_vectors[key]
            uncentered[key][layer] = raw_vectors[key]
        save_npz(vector_path, **layer_vectors, **{f'uncentered__{key}': value for key, value in raw_vectors.items()})
        from shared.io import sha
        atomic(checkpoint, {'identity': spec['identity'], 'vectors_sha256': sha(vector_path),
                            'aurocs': {key: cv[key][:, layer].tolist() for key in keys}})
        progress.stage('fit', (layer+1)*5, total, 'layer-folds', f'Committed layer {layer}')
    selected, summary = {}, {}
    for key in keys:
        means = cv[key].mean(axis=0)
        layer = winning_layer(means)
        emotion, recipe = key.split('__')
        selected[key] = {'emotion': emotion, 'recipe': recipe, 'layer': layer,
                         'vector': vectors[key][layer].tolist(), 'intercept': 0.0,
                         'training_cv_auroc': float(means[layer]), 'centering': spec['centering']}
        summary[key] = {'fold_aurocs': cv[key].tolist(), 'mean_auroc': means.tolist(), 'selected_layer': layer}
    atomic(root / 'cv_results.json', summary)
    atomic(root / 'selected_models.json', {'identity': spec['identity'], 'models': selected})
    save_npz(root / 'probes/all-layer-vectors.npz', **vectors)
    save_npz(root / 'probes/vectors.npz', **{key: vectors[key][model['layer']] for key, model in selected.items()})
    # Preserve the pre-centering directions at every layer and at the primary
    # probes' selected layers. Primary thresholds do not apply to this variant.
    save_npz(root / 'probes/all-layer-uncentered-vectors.npz', **uncentered)
    save_npz(root / 'probes/uncentered-vectors.npz', **{key: uncentered[key][model['layer']] for key, model in selected.items()})
