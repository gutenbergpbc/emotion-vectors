import importlib.util
import json
from pathlib import Path

import pytest

PATH = (Path(__file__).resolve().parents[1] / "src/experiments") / 'emotion-rh-forecasting/label_terminalwrench.py'
spec = importlib.util.spec_from_file_location('terminalwrench_labeling', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_judge_blinding_and_unicode_source_spans():
    trace = {'agent': {'model_name': 'secret-generator'}, 'final_metrics': {'reward': 1}, 'steps': [
        {'step_id': 1, 'source': 'user', 'message': 'Solve this task.'},
        {'step_id': 2, 'source': 'agent', 'model_name': 'secret-generator', 'metrics': {'reward': 1},
         'message': 'Café is ready.\n\n```sh\necho hacked\n```\n\nI will change the tests.',
         'observation': {'results': [{'content': 'Observed tool result'}]}}]}
    packet, units, excluded = module.packet_and_units(trace)
    text = json.dumps(packet)
    assert 'secret-generator' not in text and 'reward' not in text
    assert 'Observed tool result' in text
    assert len(excluded) == 1 and excluded[0]['kind'] == 'code_block'
    assert len(units) == 2 and [u['unit_id'] for u in units] == [0, 1]
    for u in units:
        assert trace['steps'][1]['message'][u['char_start']:u['char_end']] == u['text']


def test_task_sample_reproducible_and_without_group_duplicates():
    catalog = [{'id': f'{g}-{i}', 'task_group': str(g), 'label': 'legitimate'}
               for g in range(8) for i in range(3)]
    selected = module.select_population(catalog, 5, 12)
    assert selected == module.select_population(list(reversed(catalog)), 5, 12)
    assert len({r['task_group'] for r in selected}) == 5
    with pytest.raises(ValueError):
        module.select_population(catalog, 9, 12)


def test_reject_missing_duplicate_labels_wrong_model_and_nonboolean():
    request = {'unit_ids': [0, 2]}
    config = {'allowed_returned_models': ['sonnet']}
    labels = [{'unit_id': i, 'hacking': False, 'needs_review': False, 'reason': 'Ordinary debugging.'} for i in [0, 2]]
    def response(ls, model='sonnet'):
        return {'model': model, 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'labels': ls})}}]}
    assert module.validate_response(response(labels), request, config) == labels
    for bad in [labels[:1], [labels[0], labels[0]], [dict(labels[0], hacking='false'), labels[1]]]:
        with pytest.raises(ValueError):
            module.validate_response(response(bad), request, config)
    with pytest.raises(ValueError):
        module.validate_response(response(labels, 'other'), request, config)


def test_provider_change_preserves_blinded_annotation_content():
    packet = {'events': [{'step_id': '1', 'source': 'user', 'fields': {'message': 'Task'}}]}
    units = [{'unit_id': 0, 'text': 'A sentence.'}]
    sonnet = module.make_body({'model': 'anthropic/claude-sonnet-5.5', 'reasoning_effort': 'low'}, packet, units)
    gpt = module.make_body({'model': 'openai/gpt-5.5', 'reasoning_effort': 'high',
                           'provider': {'only': ['openai'], 'allow_fallbacks': False, 'require_parameters': True}}, packet, units)
    assert sonnet['messages'] == gpt['messages']
    assert sonnet['response_format'] == gpt['response_format']
    assert gpt['provider']['only'] == ['openai']
    assert not gpt['provider']['allow_fallbacks']


def test_balanced_sample_has_equal_strata_and_disjoint_tasks():
    catalog = [{'id': f'{g}-{label}', 'task_group': str(g), 'label': label}
               for g in range(60) for label in ('legitimate', 'reward_hacking')]
    selected = module.select_population(catalog, 50, 20261022, balanced=True)
    assert selected == module.select_population(list(reversed(catalog)), 50, 20261022, balanced=True)
    assert len({r['task_group'] for r in selected}) == 50
    assert sum(r['label'] == 'reward_hacking' for r in selected) == 25
    with pytest.raises(ValueError):
        module.select_population(catalog, 49, 20261022, balanced=True)
