"""Native GLM chat replay; pool only the marked assistant content/reasoning span."""
import hashlib

from shared.io import digest, require

OPEN, CLOSE = '\ue000PROBE_TARGET_START\ue001', '\ue000PROBE_TARGET_END\ue001'


def render_case(case, tokenizer, reasoning_effort='max'):
    target = case['target']
    require(target['message_index'] == len(case['messages']) - 1, 'Target must be the final supplied message')

    def build(mark):
        messages = []
        for i, message in enumerate(case['messages']):
            role = message['role']
            require(role in ('system', 'user', 'assistant', 'tool'), f'Unsupported role: {role}')
            content, reasoning = message.get('content', ''), message.get('reasoning', '')
            require(OPEN not in content + reasoning and CLOSE not in content + reasoning, 'Reserved marker in source')
            if case['render_mode'] == 'sales_with_reasoning':
                content, reasoning = content.strip(), reasoning.strip()
            if i == target['message_index']:
                require(role == 'assistant', 'Target must be assistant-only')
                body = reasoning if target['stream'] == 'reasoning' else content
                a, b = target['char_start'], target['char_end']
                require(body[a:b] == target['text'], f"Target offsets changed: {case['id']}")
                if mark:
                    body = body[:a] + OPEN + body[a:b] + CLOSE + body[b:]
                if target['stream'] == 'reasoning':
                    reasoning = body
                else:
                    content = body
            item = {'role': role, 'content': content}
            if role == 'assistant':
                # Explicit empty values prevent accidental Jinja variable inheritance.
                item['reasoning_content'] = reasoning
            else:
                require(not reasoning, 'Reasoning on a non-assistant message')
            # Frozen tool messages have no tool-call metadata; preserve native
            # observation rendering without inventing names or tool calls.
            messages.append(item)
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                             clear_thinking=False, reasoning_effort=reasoning_effort)

    marked, rendered = build(True), build(False)
    require(marked.count(OPEN) == marked.count(CLOSE) == 1, 'Template lost or duplicated target markers')
    start, end = marked.index(OPEN), marked.index(CLOSE) - len(OPEN)
    require(marked.replace(OPEN, '').replace(CLOSE, '') == rendered, 'Marker round-trip changed native rendering')
    require(rendered[start:end] == target['text'], 'Rendered target mismatch')
    return rendered, start, end


def tokenize_case(case, tokenizer, max_tokens, reasoning_effort='max', renderer='glm-native-preserved-streams-v1'):
    if case['render_mode'] == 'raw_story':
        text = case['text']
        ids = tokenizer(text, add_special_tokens=True, truncation=False)['input_ids']
        require(ids and len(ids) <= min(max_tokens, 8192), f"{case['id']}: story token limit exceeded; no silent truncation")
        return {
            'id': case['id'], 'input_ids': ids, 'case_sha256': digest(case),
            'rendered_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'token_ids_sha256': digest(ids), 'target_token_ids_sha256': digest(ids),
            'target_char_span': [0, len(text)], 'target_token_span': [0, len(ids)],
            'prefix_token_count': len(ids), 'target_token_count': len(ids),
            'pooling': 'all raw-text tokens including tokenizer special tokens',
        }
    if renderer == 'gemma-native-preserved-streams-v2':
        from .gemma_rendering import render_case as render_gemma
        rendered, start, end = render_gemma(case, tokenizer)
    else:
        require(renderer == 'glm-native-preserved-streams-v1', 'Unknown replay adapter')
        rendered, start, end = render_case(case, tokenizer, reasoning_effort)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    positions = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if b > a and b > start and a < end]
    require(positions and positions == list(range(positions[0], positions[-1] + 1)), 'Noncontiguous/empty target tokens')
    ids = encoded['input_ids'][:positions[-1] + 1]
    require(len(ids) <= max_tokens, f"{case['id']}: {len(ids)} tokens exceeds max_model_len={max_tokens}; no silent truncation")
    return {
        'id': case['id'], 'input_ids': ids, 'case_sha256': digest(case),
        'rendered_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
        'token_ids_sha256': digest(ids), 'target_token_ids_sha256': digest(ids[positions[0]:]),
        'target_char_span': [start, end], 'target_token_span': [positions[0], positions[-1] + 1],
        'prefix_token_count': len(ids), 'target_token_count': len(positions),
    }
