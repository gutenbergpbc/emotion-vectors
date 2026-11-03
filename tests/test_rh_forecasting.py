"""Scientific regression and input/resume checks for sentence RH forecasting."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from emotion_rh_forecasting.analysis import (EMOTIONS, assign_folds, balanced_groups, draw_balanced,
                                            evaluate, fit_threshold, make_pairs, predict_threshold)
from emotion_rh_forecasting.features import pool_completed
from emotion_rh_forecasting.inputs import Sources, load_dataset, load_probes, vector_identity
from emotion_rh_forecasting.pipeline import execute, load_config, resolve
from shared.io import atomic, read, save_npz, sha


def sentences():
    rows = []
    for task in range(3):
        for i in range(14):
            rows.append(dict(trace_id=f't{task}', task=f'task{task}', group_id=f'g{task}', fold=task,
                             unit_id=i, sentence_index=i, turn_id=f'turn{task}', char_start=3*i, char_end=3*i+2,
                             hacking=bool(i % 2), needs_review=False,
                             projections={e: float(i % 2)+j*.01 for j, e in enumerate(EMOTIONS)},
                             mean_token_position=float(i), last_feature_token_char_end=3*i+2))
    return rows


def test_strict_source_token_boundary_and_character_weights():
    offsets = np.array([[0, 2], [2, 6], [6, 9], [9, 12]])
    scores = np.repeat(np.array([2., 4., 1000., 99999.])[:, None], 12, axis=1)
    pooled = pool_completed(scores, offsets, 1, 8)
    # The token [6,9] overlaps the source but is unfinished at its boundary.
    assert pooled['projections']['afraid'] == (2+4*4)/5
    assert pooled['mean_token_position'] == 4/5
    assert pooled['last_feature_token_char_end'] == 6
    assert pool_completed(scores, offsets, 6, 8) is None
    scores[2:] = -1e10
    assert pool_completed(scores, offsets, 1, 8) == pooled


def test_pairs_do_not_compress_missing_or_uncertain_sentences():
    rows = sentences()
    rows[1]['projections'] = None
    rows[2]['needs_review'] = True
    pairs = make_pairs(rows, 2, certain_only=True)
    assert not any(p['trace_id'] == 't0' and p['unit_id'] in (0, 1, 2) for p in pairs)
    primary = make_pairs(rows, 2)
    assert primary[0]['unit_id'] == 0 and primary[0]['target_unit_id'] == 2
    assert primary[1]['unit_id'] == 2 and primary[1]['target_unit_id'] == 4
    zero = make_pairs(rows, 0)
    assert len(zero) == len(rows)-1
    assert all(r['unit_id'] == r['target_unit_id'] and r['source_hacking'] == r['hacking'] for r in zero)
    common = {(r['trace_id'], r['unit_id']) for r in make_pairs(rows, 10)}
    assert all({(r['trace_id'], r['unit_id']) for r in make_pairs(rows, h, common)} == common for h in range(11))


def test_fold_group_leakage_rejected():
    rows = sentences()
    rows[14]['group_id'] = rows[0]['group_id']
    with pytest.raises(ValueError, match='crosses folds'):
        assign_folds(rows, {'method': 'held_out_task'}, 1)


def test_threshold_optimum_matches_exhaustive_candidates():
    rng = np.random.default_rng(42)
    for _ in range(20):
        x, y = rng.integers(-4, 5, 30), rng.integers(0, 2, 30).astype(bool)
        rule = fit_threshold(x, y)
        values = np.unique(x)
        best = 0
        for cutoff in np.r_[values[0]-1, (values[:-1]+values[1:])/2, values[-1]+1]:
            for polarity in (1, -1):
                pred = x >= cutoff if polarity == 1 else x < cutoff
                best = max(best, .5*(pred[y].mean()+(~pred[~y]).mean()))
        pred = predict_threshold(x, rule)
        assert .5*(pred[y].mean()+(~pred[~y]).mean()) == pytest.approx(best)
    assert fit_threshold(np.ones(4), [0, 0, 1, 1])['polarity'] == 1


def test_no_heldout_labels_used_to_fit_cutoff():
    rows = make_pairs(sentences(), 1)
    first, _ = evaluate(rows, 'all_available', 1, 7, 10, [0, 1, 2])
    changed = [dict(r, hacking=not r['hacking']) if r['fold'] == 0 else r for r in rows]
    second, _ = evaluate(changed, 'all_available', 1, 7, 10, [0, 1, 2])
    assert [r for r in first['rules'] if r['fold'] == 0] == [r for r in second['rules'] if r['fold'] == 0]


def test_balancing_preserves_each_fold_and_missing_classes_fail():
    y = np.array([True, False, False, False, True, True, False], dtype=bool)
    folds = np.array([0, 0, 0, 0, 1, 1, 1])
    groups = balanced_groups(y, folds)
    for seed in range(20):
        selected = draw_balanced(groups, np.random.default_rng(seed))
        for fold in (0, 1):
            ix = selected[folds[selected] == fold]
            assert int(y[ix].sum())*2 == len(ix)
    with pytest.raises(ValueError, match='one target class'):
        balanced_groups(np.ones(7, bool), folds)


@pytest.fixture
def source(tmp_path):
    root = tmp_path/'legacy'
    root.mkdir()
    models = {f'{e}__neutral': dict(emotion=e, layer=0, vector=[1., 0., 0.]) for e in EMOTIONS}
    atomic(root/'probes.json', {'models': models})
    cases, labels = [], []
    for t in range(3):
        text = 'aa. '*14
        c = dict(id=f't{t}', scenario=f'task{t}', response=text, prompt=f'prompt {t}', source_response_sha256=f'source-{t}')
        cases.append(c)
        units = [dict(unit_id=i, kind='sentence_or_fragment', text='aa.', char_start=i*4, char_end=i*4+3,
                      hacking=bool(i % 2), needs_review=False) for i in range(14)]
        labels.append(dict(id=c['id'], source_response_sha256=c['source_response_sha256'],
                           response_text_sha256=hashlib.sha256(text.encode()).hexdigest(), units=units))
    atomic(root/'cases.json', cases)
    atomic(root/'sentence-labels.json', labels)
    spec = dict(model={'name': 'test/checkpoint', 'revision': 'frozen-test-revision'},
                cases_sha256=sha(root/'cases.json'), labels_sha256=sha(root/'sentence-labels.json'),
                probes=[dict(emotion=e, probe=f'{e}__neutral', layer_zero_based=0, source_sha256=sha(root/'probes.json')) for e in EMOTIONS])
    atomic(root/'run-spec.json', spec)
    spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    atomic(root/'replay/replay-receipt.json', {'status': 'complete', 'spec_sha256': spec_hash})
    atomic(root/'replay/smoke.json', {'status': 'passed'})
    for t in range(3):
        scores = np.array([[float(i % 2)+j*.01 for j in range(12)] for i in range(14)])
        path = root/'replay/scores'/f't{t}.npz'
        save_npz(path, scores=scores, offsets=np.array([[i*4, i*4+3] for i in range(14)]), response_char_start=np.array(0))
        atomic(path.with_suffix('.receipt.json'), {'spec_sha256': spec_hash, 'sha256': sha(path)})
    config = tmp_path/'config.json'
    atomic(config, dict(experiment='emotion-rh-forecasting', run_id='test', probe_bundle=str(root),
                        dataset={'format': 'legacy-coding', 'path': str(root)}, horizons=[0, 1, 10], repeats=8))
    return config


def test_probe_cache_binding_and_source_text(source):
    cfg = load_config(source)
    sources = Sources()
    bundle = load_probes(cfg['probe_bundle'], sources)
    original = vector_identity(bundle)
    bundle['probes'][0]['vector'][0] = 2
    assert original != vector_identity(bundle)
    with pytest.raises(ValueError, match='does not match'):
        load_dataset(cfg['dataset'], bundle, sources)


def test_portable_multiturn_input_and_grouped_folds(tmp_path):
    config = {'format': 'sentence-traces-v1', 'path': str(tmp_path/'traces.json')}
    trace = dict(id='episode-1', task='terminal-task', group_id='terminal-task', turns=[
        dict(id='first', messages=[{'role': 'user', 'content': 'Do a task'}], completion='First.',
             units=[dict(unit_id='s1', char_start=0, char_end=6, text='First.', hacking=False)]),
        dict(id='second', messages=[{'role': 'user', 'content': 'Do a task'}, {'role': 'assistant', 'content': 'First.'},
                                   {'role': 'user', 'content': 'Feedback'}], completion='Next.',
             units=[dict(unit_id='s2', char_start=0, char_end=5, text='Next.', hacking=True)])])
    atomic(config['path'], {'schema_version': 1, 'traces': [trace]})
    traces, jobs = load_dataset(config, {}, Sources())
    assert len(jobs) == 2 and len(traces[0]['units']) == 2
    trace['turns'][1]['units'][0]['text'] = 'Wrong'
    atomic(config['path'], {'schema_version': 1, 'traces': [trace]})
    with pytest.raises(ValueError, match='differs'):
        load_dataset(config, {}, Sources())


def test_end_to_end_and_resume_integrity(source, tmp_path):
    cfg = load_config(source)
    spec, bundle, traces, jobs = resolve(cfg)
    out = tmp_path/'out'
    result = execute(out, spec, bundle, traces, jobs)
    assert len(result['estimates']) == 3*3*15
    assert read(out/'verification.json')['passed']
    assert read(out/'progress.json')['state'] == 'complete'
    assert all((out/'figures'/f'{name}.png').exists() for name in ('accuracy-trends', 'accuracy-trends-future', 'accuracy-heatmap'))
    assert execute(out, spec, bundle, traces, jobs, resume=True) == result
    with pytest.raises(ValueError, match='Run exists'):
        execute(out, spec, bundle, traces, jobs)
    (out/'metrics.csv').write_text('corrupt')
    with pytest.raises(ValueError, match='modified'):
        execute(out, spec, bundle, traces, jobs, resume=True)
