import hashlib
from pathlib import Path

import numpy as np
import pytest

from emotion_probes.config import load_config, resolve
from emotion_probes.data import EMOTIONS, LABELS, RECIPES, training_conditions, validate_dataset
from emotion_probes.extraction import feature_manifest, layer_features
from emotion_probes.fitting import directions
from emotion_probes.pipeline import execute, prepare
from emotion_probes.rendering import tokenize_case
from prepare_stories import accepted
from shared.io import atomic, read
from test_pipeline import fixture_dataset


def story_fixture(root):
    fixture_dataset(root)
    rows = []
    for label in LABELS:
        for topic in range(1, 21):
            text = f'{label}, topic {topic}: distinct generated training narrative.'
            rows.append({'id': f'{label}-{topic}', 'label': label, 'render_mode': 'raw_story',
                         'topic_index': topic, 'group_id': f'story-topic-{topic:02d}',
                         'fold': (topic-1)//4, 'text': text,
                         'text_sha256': hashlib.sha256(text.encode()).hexdigest()})
    atomic(root / 'training_cases.json', rows)
    return rows


def test_shared_neutral_rows_and_topic_folds(tmp_path):
    rows = story_fixture(tmp_path / 'data')
    validate_dataset(tmp_path / 'data', training_family='story')
    conditions = training_conditions(rows)
    neutral = [i for i, row in enumerate(rows) if row['label'] == 'neutral']
    assert len(neutral) == 20
    assert all(conditions[(emotion, 'neutral')] == neutral for emotion in EMOTIONS)
    assert len(set(i for indices in conditions.values() for i in indices)) == 260
    rows[-1]['fold'] = 0
    atomic(tmp_path / 'data/training_cases.json', rows)
    with pytest.raises(ValueError, match='crosses folds'):
        validate_dataset(tmp_path / 'data', training_family='story')


def test_raw_story_pools_special_tokens_and_has_no_chat_prompt():
    class Tokenizer:
        def __call__(self, text, *, add_special_tokens, truncation):
            assert add_special_tokens and not truncation
            return {'input_ids': [2] + list(text.encode())}
        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError('A story must not use chat formatting')
    case = {'id': 'story', 'render_mode': 'raw_story', 'text': 'hello'}
    record = tokenize_case(case, Tokenizer(), 100)
    assert record['input_ids'] == [2, 104, 101, 108, 108, 111]
    assert record['target_token_span'] == [0, 6]
    assert record['target_token_count'] == 6
    with pytest.raises(ValueError, match='no silent truncation'):
        tokenize_case(case, Tokenizer(), 5)


def test_story_pipeline_uses_single_shared_neutral_mean(tmp_path):
    rows = story_fixture(tmp_path / 'data')
    atomic(tmp_path / 'config.json', {'run_id': 'test-story-synthetic', 'model': 'synthetic/demo',
        'revision': 'test', 'backend': 'synthetic', 'layers': 3, 'width': 16, 'dataset': 'data',
        'dataset_contract': 'custom', 'training_family': 'story', 'bootstrap_draws': 30})
    spec = resolve(load_config(tmp_path / 'config.json'))
    root = tmp_path / 'run'
    prepare(root, spec)
    execute(root)
    assert read(root / 'verification.json')['passed']
    manifest = read(root / 'probes/manifest.json')
    assert manifest['training_family'] == 'story'
    assert 'raw-text-all-token-mean' in manifest['training_readout']
    assert 'story probes' in (root / 'report.html').read_text()
    features = feature_manifest(root, spec, 'training', rows)
    x = layer_features(root, 'training', 0, features)
    labels = np.array([row['label'] for row in rows])
    means = {(emotion, label): x[labels == label].mean(axis=0, dtype=np.float64)
             for emotion in EMOTIONS for label in (emotion, 'neutral')}
    expected = directions(means)
    with np.load(root / 'probes/all-layer-vectors.npz') as vectors:
        for ei, emotion in enumerate(EMOTIONS):
            for recipe in RECIPES:
                np.testing.assert_allclose(vectors[f'{emotion}__{recipe}'][0], expected[recipe][ei], atol=1e-7)


def test_semantic_acceptance_does_not_silently_lower_requirements():
    assert accepted({'fit': 3, 'topic_fit': 3, 'explicit_wording': False}, 'emotion')
    assert not accepted({'fit': 2, 'topic_fit': 5, 'explicit_wording': False}, 'emotion')
    assert not accepted({'fit': 5, 'topic_fit': 5, 'explicit_wording': True}, 'emotion')
    assert not accepted({'neutrality': 4, 'role_format_ok': True}, 'neutral')
    with pytest.raises(ValueError, match='audit rating'):
        accepted({'fit': '5', 'topic_fit': 5, 'explicit_wording': False}, 'emotion')
