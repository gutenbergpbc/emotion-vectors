"""Complete-trace selection and causal score-contract regression checks."""
import json
from pathlib import Path
import sys

import pytest

DIRECTORY = (Path(__file__).resolve().parents[1] / "src/experiments") / 'emotion-self-report'
sys.path.insert(0, str(DIRECTORY))
import select_activation_traces as selection
from finalize_activations import validate


def test_selection_keeps_whole_traces_and_both_datasets():
    traces = [{'dataset': d, 'trace_id': tid, 'checkpoints': list(range(n))}
              for d, tid, n in [('sales', 's', 6), ('coding', 'a', 2), ('coding', 'b', 3), ('coding', 'c', 4)]]
    chosen = selection.choose(traces, 10)
    assert {t['trace_id'] for t in chosen} == {'s', 'a', 'b'}
    assert sum(len(t['checkpoints']) for t in chosen) == 11
    assert selection.choose(traces, 100) == selection.choose(list(reversed(traces)), 100)


def test_value_schema_and_incomplete_trace_exclusion(tmp_path, monkeypatch):
    monkeypatch.setattr(selection, 'BASE', tmp_path)
    directory = tmp_path / 'source'
    directory.mkdir()
    cases = {tid: {'id': tid, 'messages': [{'role': 'assistant', 'content': 'a.b.'}]} for tid in ['full', 'partial']}
    points = [{'trace_id': tid, 'unit_id': i, 'message_index': 0, 'char_start': i * 2,
               'char_end': (i + 1) * 2, 'text': text} for tid in cases for i, text in enumerate(['a.', 'b.'])]
    rows = [{'trace_id': p['trace_id'], 'unit_id': p['unit_id'], 'emotion': e,
             'value': 0, 'rating_status': 'numeric'} for p in points for e in selection.EMOTIONS]
    rows[-1]['value'] = None
    for filename, value in [('config.resolved.json', {'model': 'test'}), ('cases.json', cases),
                            ('checkpoints.json', points), ('ratings.json', rows)]:
        (directory / filename).write_text(json.dumps(value))
    traces, _, _ = selection.load_source('sales', 'source', 'source', 'ratings.json')
    assert [t['trace_id'] for t in traces] == ['full']
    assert len(traces[0]['ratings']) == 24
    assert all(r['rating'] == 0 for r in traces[0]['ratings'])


def score_fixture():
    points = [{'trace_id': 't', 'unit_id': 1, 'message_index': 2, 'char_end': 10}]
    ratings = [{'trace_id': 't', 'unit_id': 1, 'emotion': e, 'rating': 0, 'rating_status': 'numeric'}
               for e in selection.EMOTIONS]
    scores = [{'trace_id': 't', 'unit_id': 1, 'emotion': e, 'message_index': 2,
               'sentence_char_end': 10, 'probe_char_end': 9, 'raw_projection': .5} for e in selection.EMOTIONS]
    return points, ratings, scores


@pytest.mark.parametrize('failure', ['missing', 'duplicate', 'future', 'message', 'nan'])
def test_reject_invalid_projection_contract(failure):
    points, ratings, scores = score_fixture()
    assert len(validate(points, ratings, scores)) == 12
    if failure == 'missing':
        scores.pop()
    elif failure == 'duplicate':
        scores.append(scores[0])
    elif failure == 'future':
        scores[0]['probe_char_end'] = 11
    elif failure == 'message':
        scores[0]['message_index'] = 4
    else:
        scores[0]['raw_projection'] = float('nan')
    with pytest.raises(ValueError):
        validate(points, ratings, scores)


def test_finalizer_exports_complete_trajectories_and_reports(tmp_path, monkeypatch):
    import finalize_activations
    points, ratings, scores = [], [], []
    for i in range(4):
        tid, uid = f't{i // 2}', i % 2
        point = {'trace_id': tid, 'unit_id': uid, 'sentence_index': uid + 1,
                 'message_index': 1, 'char_start': uid * 2, 'char_end': (uid + 1) * 2,
                 'text': 'x.', 'label': 'unlabeled'}
        points.append(point)
        for e in selection.EMOTIONS:
            ratings.append({**point, 'emotion': e, 'rating': i, 'rating_status': 'numeric'})
            scores.append({'trace_id': tid, 'unit_id': uid, 'emotion': e, 'raw_projection': i * .2,
                           'message_index': 1, 'sentence_char_end': point['char_end'],
                           'probe_char_end': point['char_end']})
    manifest = {'model': 'synthetic-test-fixture', 'selected_checkpoints': 4, 'selected_traces': 2,
                'dataset_traces': {'sales': 2}, 'dataset_checkpoints': {'sales': 4},
                'selection_policy': 'Synthetic test only', 'completeness': 'Synthetic test only',
                'traces': [{'trace_id': f't{i}', 'dataset': 'sales', 'checkpoints': 2} for i in range(2)]}
    for name, value in [('selection.json', manifest), ('checkpoints.json', points),
                        ('ratings.json', ratings), ('probe-scores.json', scores),
                        ('provenance.json', {'model': 'synthetic-test-fixture'})]:
        (tmp_path / name).write_text(json.dumps(value))
    monkeypatch.setattr(sys, 'argv', ['finalize_activations.py', str(tmp_path), '--bootstrap-replicates', '10'])
    finalize_activations.main()
    assert json.loads((tmp_path / 'verification.json').read_text())['projection_rows'] == 48
    assert len(json.loads((tmp_path / 'trajectories.json').read_text())) == 2
    assert (tmp_path / 'sales/figures/score-correlations.png').exists()
    assert 'Checkpoint text' in (tmp_path / 'trajectories.html').read_text()
