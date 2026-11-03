from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import pearsonr, spearmanr

from shared.io import atomic, digest, read, sha
from emotion_self_report.analysis import analyze, correlations, join
from emotion_self_report.collection import call, parse_rating, request_body
from emotion_self_report.config import QUESTIONS
from emotion_self_report.inputs import load_inputs
from emotion_self_report.pipeline import execute, load_config, resolve
from emotion_self_report.reporting import write_csv


def fixture(tmp_path, scores=True):
    cases, points, score_rows = [], [], []
    for i in range(5):
        response = 'First sentence. Second sentence. Third sentence.'
        tid = f'trace-{i}'
        cases.append(dict(id=tid, prompt=f'Task {i}', response=response,
                          label='ambiguous' if i == 4 else 'honest', scenario=f'task-{i % 2}'))
        for j, (lo, hi) in enumerate([(0, 15), (16, 32), (33, len(response))]):
            points.append(dict(trace_id=tid, unit_id=j, char_start=lo, char_end=hi))
            for e in ('sad', 'calm'):
                score_rows.append(dict(trace_id=tid, unit_id=j, emotion=e,
                                       raw_projection=i + j * 1.7, projection_score_1_to_10=1 + i + j,
                                       sentence_char_end=hi, probe_char_end=hi, label=cases[-1]['label']))
    atomic(tmp_path / 'traces.json', cases)
    atomic(tmp_path / 'points.json', points)
    cfg = dict(experiment='emotion-self-report', run_id='fixture', model='synthetic/demo', environment='coding',
               traces='traces.json', checkpoints='points.json', emotions=['sad', 'calm'], synthetic=True,
               bootstrap_replicates=100, api={'request_interval_seconds': 0, 'workers': 2})
    if scores:
        write_csv(tmp_path / 'scores.csv', score_rows)
        atomic(tmp_path / 'provenance.json', dict(model='synthetic/demo', cases_sha256=sha(tmp_path / 'traces.json')))
        cfg['probe_scores'] = dict(path='scores.csv', provenance='provenance.json')
    atomic(tmp_path / 'config.json', cfg)
    return tmp_path / 'config.json'


def fake_send(body):
    prefix = body['messages'][-2]['content']
    # Exercise legitimate zeros and refusal/null semantics in a full run.
    if 'calm' in body['messages'][-1]['content']:
        content = 'null' if len(prefix) < 20 else 'I cannot rate that.'
    else:
        content = str((len(prefix) + int(body['messages'][0]['content'][-1])) % 11)
    return dict(model=body['model'], provider='Test', usage={'total_tokens': 15},
                choices=[dict(finish_reason='stop', message={'content': content})])


@pytest.mark.parametrize('text,value,status', [('0', 0, 'numeric'), ('10', 10, 'numeric'),
                        ('null', None, 'explicit_null'), ('true', None, 'non_numeric_response'),
                        ('5.0', None, 'non_numeric_response'), ('11', None, 'non_numeric_response'),
                        ('-1', None, 'non_numeric_response'), ('"5"', None, 'non_numeric_response'),
                        ('I cannot feel.', None, 'non_numeric_response')])
def test_rating_parser(text, value, status):
    assert parse_rating(text) == (value, status)


def test_prefix_has_context_but_no_future_or_labels(tmp_path):
    path = fixture(tmp_path)
    cfg = load_config(path)
    cases, points, _, _, _ = load_inputs(cfg)
    first = points[0]
    body = request_body(cfg, cases[first['trace_id']], first, 'sad')
    assert body['messages'][1]['content'] == 'First sentence.'
    assert 'Second sentence.' not in json.dumps(body)
    assert 'ambiguous' not in json.dumps(body)
    original = deepcopy(cases[first['trace_id']])
    request_body(cfg, original, first, 'calm')
    assert original == cases[first['trace_id']]
    assert body['messages'][-1]['content'] == QUESTIONS['task_stance'].format(emotion='sad')


def test_multiturn_prefix_keeps_tools_and_omits_future(tmp_path):
    messages = [dict(role='user', content='Task'), dict(role='assistant', content='Inspect.',
                tool_calls=[dict(id='call-1', type='function', function=dict(name='shell', arguments='{}'))]),
                dict(role='tool', content='Tool result', tool_call_id='call-1'),
                dict(role='assistant', content='I can do it. Future sentence.', tool_calls=[{'id': 'future'}]),
                dict(role='user', content='Future instruction')]
    atomic(tmp_path / 'traces.json', [dict(id='a', messages=messages)])
    atomic(tmp_path / 'points.json', [dict(trace_id='a', unit_id=0, message_index=3, char_start=0, char_end=12)])
    atomic(tmp_path / 'config.json', dict(run_id='multi', model='test', environment='terminalwrench',
                                         traces='traces.json', checkpoints='points.json'))
    cfg = load_config(tmp_path / 'config.json')
    cases, points, *_ = load_inputs(cfg)
    body = request_body(cfg, cases['a'], points[0], 'sad')
    assert body['messages'][:3] == messages[:3]
    assert body['messages'][3] == dict(role='assistant', content='I can do it.')
    assert 'Future' not in json.dumps(body)


def test_future_scores_duplicates_and_model_mismatch_fail(tmp_path):
    path = fixture(tmp_path)
    cfg = load_config(path)
    rows = list(csv.DictReader((tmp_path / 'scores.csv').open()))
    original = deepcopy(rows)
    rows[0]['probe_char_end'] = '9999'
    write_csv(tmp_path / 'scores.csv', rows)
    with pytest.raises(ValueError, match='future'):
        load_inputs(cfg)
    write_csv(tmp_path / 'scores.csv', original + [original[0]])
    with pytest.raises(ValueError, match='Duplicate probe'):
        load_inputs(cfg)
    write_csv(tmp_path / 'scores.csv', original)
    atomic(tmp_path / 'provenance.json', dict(model='wrong', cases_sha256=sha(tmp_path / 'traces.json')))
    with pytest.raises(ValueError, match='model mismatch'):
        load_inputs(cfg)


def test_correlations_ties_constants_and_cluster_bootstrap():
    a, b = [1, 1, 3, 4, 4], [5, 3, 2, 2, 0]
    p, s = correlations(a, b)
    assert p == pytest.approx(pearsonr(a, b).statistic)
    assert s == pytest.approx(spearmanr(a, b).statistic)
    assert correlations([1, 1], [2, 3]) == (None, None)
    rows = [dict(trace_id=str(i), unit_id=j, emotion='sad', rating=(i+j) % 5,
                 rating_status='numeric', raw_projection=i-j, provider='Test', label='honest')
            for i in range(4) for j in range(i+2)]
    spec = dict(identity='test', model='demo', environment='coding', synthetic=True, mode='collect',
                prompt='task_stance', correlation_score='raw_projection', bootstrap_replicates=100,
                seed=42, exclude_labels=['ambiguous'], emotions=['sad'])
    result = analyze(rows, spec)['results'][0]
    # Independent whole-cluster implementation, unequal trace lengths.
    rng = np.random.default_rng(42)
    bootstrap = [[], []]
    for _ in range(100):
        sampled = [r for i in rng.integers(4, size=4) for r in rows if r['trace_id'] == str(i)]
        x, y = [np.array([r[k] for r in sampled]) for k in ('raw_projection', 'rating')]
        if np.ptp(x) and np.ptp(y):
            bootstrap[0].append(pearsonr(x, y).statistic)
            bootstrap[1].append(spearmanr(x, y).statistic)
    for i, name in enumerate(('pearson', 'spearman')):
        assert [result[name+'_ci_low'], result[name+'_ci_high']] == pytest.approx(np.quantile(bootstrap[i], [.025, .975]))
    assert result['traces'] == 4 and result['pairs'] == len(rows)


def test_collection_resume_missing_ratings_and_verification(tmp_path):
    spec, inputs = resolve(load_config(fixture(tmp_path)))
    root = tmp_path / 'run'
    execute(root, spec, inputs, through='prepare')
    assert not (root / 'responses').exists()
    result = execute(root, spec, inputs, resume=True, send=fake_send)
    assert read(root / 'verification.json')['passed']
    assert read(root / 'progress.json')['state'] == 'complete'
    assert result['rating_status_counts'] == dict(numeric=15, explicit_null=5, non_numeric_response=10)
    assert result['results'][0]['pairs'] == 12
    assert result['results'][1]['pairs'] == 0
    def forbidden(_):
        raise AssertionError('Resume must not call the API')
    assert execute(root, spec, inputs, resume=True, send=forbidden) == result
    (root / 'ratings.csv').write_text('tampered')
    with pytest.raises(ValueError, match='Completed output changed'):
        execute(root, spec, inputs, resume=True, send=forbidden)


def test_call_retries_transport_but_not_nonnumeric_and_checks_cache(tmp_path):
    spec, (cases, points, _) = resolve(load_config(fixture(tmp_path)))
    attempts = []
    def send(body):
        attempts.append(body)
        if len(attempts) == 1:
            raise TimeoutError('temporary failure')
        return fake_send(body)
    row = call(tmp_path / 'run', spec, cases[points[0]['trace_id']], points[0], 'calm', send, sleep=lambda _: None)
    assert row['ok'] and row['rating_status'] == 'explicit_null' and len(row['attempts']) == 2
    call(tmp_path / 'run', spec, cases[points[0]['trace_id']], points[0], 'calm', send)
    assert len(attempts) == 2
    changed = deepcopy(points[0]); changed['char_end'] = 32
    with pytest.raises(ValueError, match='binding changed'):
        call(tmp_path / 'run', spec, cases[points[0]['trace_id']], changed, 'calm', send)


def test_no_probe_scores_is_valid_collection(tmp_path):
    spec, inputs = resolve(load_config(fixture(tmp_path, scores=False)))
    result = execute(tmp_path / 'run', spec, inputs, send=fake_send)
    assert all(r['status'] == 'no_pairs' for r in result['results'])
    assert result['results'][0]['missing_probe_score'] == 12


def test_legacy_import_reparses_and_preserves_zero(tmp_path):
    cfg_path = fixture(tmp_path)
    cfg = load_config(cfg_path)
    cases, points, *_ = load_inputs(cfg)
    legacy = tmp_path / 'legacy'; legacy.mkdir()
    plan = dict(source_cases=str(tmp_path / 'traces.json'), source_cases_sha256=sha(tmp_path / 'traces.json'),
                checkpoints=read(tmp_path / 'points.json'))
    atomic(legacy / 'plan.json', plan)
    settings = dict(model=cfg['model'], temperature=0, max_tokens=2048)
    manifest = dict(settings=settings, question=QUESTIONS['task_stance'],
                    plan_sha256=sha(legacy / 'plan.json'), source_cases_sha256=sha(tmp_path / 'traces.json'))
    atomic(legacy / 'manifest.json', manifest)
    rows = []
    for point in points:
        for emotion in cfg['emotions']:
            body = settings | {'messages': request_body(cfg, cases[point['trace_id']], point, emotion)['messages']}
            response = dict(model=cfg['model'], provider='Test', usage={}, choices=[dict(finish_reason='stop', message={'content': '0'})])
            rows.append(dict(checkpoint=point, emotion=emotion, request=body, ok=True, value=0,
                             rating_status='numeric', attempts=[dict(ok=True, request=body, response=response)]))
    (legacy / 'responses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    raw = read(cfg_path); raw.pop('traces'); raw.pop('checkpoints')
    raw.update(mode='import', legacy_run='legacy')
    atomic(cfg_path, raw)
    spec, inputs = resolve(load_config(cfg_path))
    result = execute(tmp_path / 'run', spec, inputs)
    assert result['rating_status_counts'] == {'numeric': 30}
    assert all(r['rating'] == 0 for r in read(tmp_path / 'run/ratings.json'))
    assert all(r['pearson'] is None for r in result['results'])


def test_auto_segmentation_preserves_code_and_drops_heading_targets(tmp_path):
    from emotion_self_report.segmentation import segment
    text = '# Header\n\nI can\ntry this. Next sentence.\n\n```python\nx = 1\n```\n\nFinal fragment'
    units = segment(text)
    assert any(u['kind'] == 'code_block' for u in units)
    prose = [u for u in units if u['kind'] == 'sentence_or_fragment']
    assert [u['text'] for u in prose] == ['I can\ntry this.', 'Next sentence.', 'Final fragment']
    assert all(text[u['char_start']:u['char_end']] == u['text'] for u in units)


def test_config_and_changed_input_rejected(tmp_path):
    path = fixture(tmp_path)
    spec, inputs = resolve(load_config(path))
    execute(tmp_path / 'run', spec, inputs, through='prepare')
    data = read(tmp_path / 'points.json'); data[0]['char_end'] -= 1
    atomic(tmp_path / 'points.json', data)
    with pytest.raises(ValueError, match='boundary mismatch'):
        resolve(load_config(path))
    cfg = read(path); cfg['rating_min'] = 1; atomic(path, cfg)
    with pytest.raises(ValueError, match='Unknown self-report'):
        load_config(path)
