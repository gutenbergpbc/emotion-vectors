"""Gemma replay keeps the marked occurrence and rejects incompatible configs."""
from pathlib import Path

import pytest

from emotion_probes.config import load_config
from emotion_probes.gemma_rendering import render_case
from emotion_probes.rendering import tokenize_case
from shared.io import atomic, read


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert messages[0]['role'] == 'user'
        assert all(m['role'] in ('user', 'assistant') for m in messages)
        assert all(a['role'] != b['role'] for a, b in zip(messages, messages[1:]))
        return '<bos>' + ''.join(f"<{m['role']}>{m['content'].strip()}</turn>" for m in messages)

    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(c) for c in text], 'offset_mapping': [(i, i+1) for i in range(len(text))]}


@pytest.mark.parametrize('stream', ['reasoning', 'content'])
def test_gemma_repeated_target_and_role_mapping(stream):
    case = {'id': 'repeat', 'render_mode': 'sales_with_reasoning', 'messages': [
        {'role': 'system', 'content': 'System rule'},
        {'role': 'user', 'content': 'same sentence'},
        {'role': 'assistant', 'content': 'Tool requested'},
        {'role': 'tool', 'content': 'Tool evidence'},
        {'role': 'user', 'content': 'Continue'},
        {'role': 'assistant', 'content': 'same sentence', 'reasoning': 'same sentence. same sentence'},
    ], 'target': {'message_index': 5, 'stream': stream, 'char_start': 15 if stream == 'reasoning' else 0,
                  'char_end': 28 if stream == 'reasoning' else 13, 'text': 'same sentence'}}
    tokenizer = CharacterTokenizer()
    rendered, start, end = render_case(case, tokenizer)
    assert rendered[start:end] == 'same sentence'
    assert 'System rule\n\nsame sentence' in rendered
    assert '[Tool response]\nTool evidence\n\nContinue' in rendered
    assert '<think>\nsame sentence. same sentence\n</think>' in rendered
    if stream == 'reasoning':
        assert rendered[start-15:start] == 'same sentence. '
    record = tokenize_case(case, tokenizer, 10000, renderer='gemma-native-preserved-streams-v2')
    lo, hi = record['target_token_span']
    assert ''.join(map(chr, record['input_ids'][lo:hi])) == 'same sentence'
    assert len(record['input_ids']) == end
    with pytest.raises(ValueError, match='no silent truncation'):
        tokenize_case(case, tokenizer, 10, renderer='gemma-native-preserved-streams-v2')


def test_gemma_assistant_first_history():
    case = {'id': 'assistant-first', 'render_mode': 'assistant_only', 'messages': [
        {'role': 'assistant', 'content': 'Prior context'}, {'role': 'assistant', 'content': 'Target'}],
        'target': {'message_index': 1, 'stream': 'content', 'char_start': 0, 'char_end': 6, 'text': 'Target'}}
    rendered, start, end = render_case(case, CharacterTokenizer())
    assert rendered.startswith('<bos><user></turn><assistant>Prior context\n\nTarget')
    assert rendered[start:end] == 'Target'


@pytest.mark.parametrize('changes', [
    {'model': 'unknown/model'}, {'quantization': 'fp8'}, {'renderer': 'glm-native-preserved-streams-v1'},
    {'width': 4096}, {'tokenizer_model': 'google/gemma-3-27b-it'}, {'gpu': 'H100:2'},
])
def test_gemma_config_rejects_wrong_model_precision_and_template(tmp_path, changes):
    source = (Path(__file__).resolve().parents[1] / "src/experiments") / '../../configs/gemma-3-4b-diversity.json'
    assert load_config(source)['quantization'] is None
    atomic(tmp_path / 'config.json', {**read(source), **changes})
    with pytest.raises(ValueError):
        load_config(tmp_path / 'config.json')
