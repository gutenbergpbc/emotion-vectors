import json
from pathlib import Path

import numpy as np
import pytest

from shared.io import atomic, read
from emotion_probes.config import load_config, resolve
from emotion_probes.data import EMOTIONS, LABELS, RECIPES, validate_dataset
from emotion_probes.extraction import feature_manifest, cases_for, layer_features
from emotion_probes.fitting import directions
from emotion_probes.pipeline import execute, prepare
from emotion_probes.statistics import choose_threshold, measure, verify_threshold


def fixture_dataset(root):
    root.mkdir()
    def case(cid, emotion, label, group, fold=None):
        text = f'{cid}: a distinct example for {label}.'
        row = dict(id=cid, emotion=emotion, label=label, group_id=group, pair_id=group,
                   partition='training' if fold is not None else 'eval', render_mode='assistant_only',
                   messages=[{'role': 'user', 'content': 'Explain.'}, {'role': 'assistant', 'content': text}],
                   target=dict(message_index=1, stream='content', char_start=0, char_end=len(text), text=text))
        if fold is not None:
            row['fold'] = fold
        return row
    train = [case(f'train-{e}-{fold}-{label}', e, label, f'train-{e}-{fold}', fold)
             for e in EMOTIONS for fold in range(5) for label in (e, 'neutral')]
    evaluation, rows, aliases = [], [], {}
    for split in ('calibration', 'heldout'):
        for group in range(3):
            for label in LABELS:
                cid = f'{split}-{group}-{label}'
                group_id = f'{split}-{group}'
                evaluation.append(case(cid, label if label != 'neutral' else EMOTIONS[0], label, group_id))
                rows.append(dict(id=cid, label=label, group_id=group_id, split=split))
                aliases[cid] = [cid]
    for name, data in [('training_cases.json', train), ('eval_cases.json', evaluation),
                       ('split_manifest.json', dict(rows=rows, aliases=aliases)), ('protocol.json', {})]:
        atomic(root / name, data)


def fixture_config(tmp_path):
    fixture_dataset(tmp_path / 'data')
    path = tmp_path / 'config.json'
    atomic(path, {'run_id': 'test-synthetic', 'model': 'synthetic/demo', 'revision': 'test',
                  'backend': 'synthetic', 'executor': 'local', 'layers': 3, 'width': 16,
                  'dataset': 'data', 'dataset_contract': 'custom', 'bootstrap_draws': 30})
    return resolve(load_config(path))


def test_condition_weighting_and_cutoff_ties():
    labels = np.array(LABELS + ['neutral']*100)
    scores = np.array([1.] + [0.]*(len(labels)-1))
    cutoff = choose_threshold(labels, scores, EMOTIONS[0])
    verify_threshold(labels, scores, EMOTIONS[0], cutoff)
    assert measure(labels, scores, cutoff, EMOTIONS[0])['balanced_accuracy'] == 1
    # A neutral-heavy population must not change equally weighted label BA.
    assert measure(labels, np.zeros(len(labels)), 1, EMOTIONS[0])['balanced_accuracy'] == .5


def test_centering_and_condition_translation():
    rng = np.random.default_rng(92)
    means = {(e, label): rng.normal(size=16) for e in EMOTIONS for label in (e, 'neutral')}
    raw, centered = directions(means, 'none'), directions(means)
    shift = rng.normal(size=16)
    shifted = directions({key: value+shift for key, value in means.items()})
    for recipe in RECIPES:
        assert np.allclose(centered[recipe], shifted[recipe])
        assert np.allclose(np.linalg.norm(centered[recipe], axis=1), 1)
        assert not np.allclose(raw[recipe], centered[recipe])


def test_source_leakage_and_unknown_config_rejected(tmp_path):
    fixture_dataset(tmp_path / 'data')
    split = read(tmp_path / 'data/split_manifest.json')
    split['rows'][-1]['group_id'] = split['rows'][0]['group_id']
    atomic(tmp_path / 'data/split_manifest.json', split)
    with pytest.raises(ValueError, match='crosses evaluation'):
        validate_dataset(tmp_path / 'data')
    atomic(tmp_path / 'bad.json', {'typo': 'wrong'})
    with pytest.raises(ValueError, match='Unknown config'):
        load_config(tmp_path / 'bad.json')


def test_complete_pipeline_resume_and_corruption(tmp_path):
    spec = fixture_config(tmp_path)
    root = tmp_path / 'run'
    prepare(root, spec)
    execute(root)
    assert read(root / 'verification.json')['passed']
    assert read(root / 'progress.json')['state'] == 'complete'
    assert len(read(root / 'probes/manifest.json')['models']) == 36
    assert not read(root / 'probes/manifest.json')['vector_variants']['uncentered']['calibrated']
    # Independently reconstruct the uncentered export from training activations.
    cases = cases_for(root, spec)['training']
    manifest = feature_manifest(root, spec, 'training', cases)
    models = read(root / 'selected_models.json')['models']
    with np.load(root / 'probes/all-layer-uncentered-vectors.npz') as raw_layers, \
         np.load(root / 'probes/uncentered-vectors.npz') as raw_selected:
        before_resume = {key: value.copy() for key, value in raw_layers.items()}
        for layer in range(spec['layers']):
            x = layer_features(root, 'training', layer, manifest)
            means = {(e, label): x[[i for i, c in enumerate(cases) if (c['emotion'], c['label']) == (e, label)]].mean(axis=0, dtype=np.float64)
                     for e in EMOTIONS for label in (e, 'neutral')}
            expected = directions(means, 'none')
            for ei, emotion in enumerate(EMOTIONS):
                for recipe in RECIPES:
                    key = f'{emotion}__{recipe}'
                    np.testing.assert_allclose(raw_layers[key][layer], expected[recipe][ei], atol=1e-7)
                    np.testing.assert_array_equal(raw_selected[key], raw_layers[key][models[key]['layer']])
    selected = read(root / 'probes/selected-12.json')['probes']
    calibration = read(root / 'calibration_results.json')
    for emotion in EMOTIONS:
        assert calibration[selected[emotion]]['balanced_accuracy'] == max(calibration[f'{emotion}__{r}']['balanced_accuracy'] for r in RECIPES)
    cache = root / 'activations/training/00000.npy'
    timestamp = cache.stat().st_mtime_ns
    prepare(root, spec, resume=True)
    execute(root, through='fit')
    assert cache.stat().st_mtime_ns == timestamp
    with np.load(root / 'probes/all-layer-uncentered-vectors.npz') as resumed:
        for key, values in before_resume.items():
            np.testing.assert_array_equal(resumed[key], values)
    assert read(root / 'progress.json')['state'] == 'waiting'
    with cache.open('r+b') as stream:
        stream.seek(-1, 2)
        stream.write(b'!')
    with pytest.raises(ValueError, match='checksum'):
        feature_manifest(root, spec, 'training', cases_for(root, spec)['training'])
    with pytest.raises(ValueError, match='changed'):
        prepare(root, {**spec, 'centering': 'none'}, resume=True)
