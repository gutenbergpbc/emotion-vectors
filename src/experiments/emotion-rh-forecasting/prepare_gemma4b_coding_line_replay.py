"""Freeze Gemma-native replay jobs for audited coding code lines.

uv run --with 'transformers==5.17.0' --with 'jinja2==3.1.6' --with scikit-learn python \
  src/experiments/emotion-rh-forecasting/prepare_gemma4b_coding_line_replay.py
"""
from __future__ import annotations

# Resolve repository imports when this file is executed directly.
from pathlib import Path as _Path
import sys as _sys
_REPO = next((p for p in _Path(__file__).resolve().parents
              if (p / "src/experiments/shared").is_dir()), None)
if _REPO is not None:
    _sys.path[:0] = [str(_Path(__file__).resolve().parent), str(_REPO), str(_REPO / "src/environments"),
                    str(_REPO / "src/experiments")] + [
        str(_REPO / "src/experiments" / p) for p in
        ("emotion-probes", "emotion-rh-forecasting", "emotion-steering", "emotion-self-report")]


from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
CANONICAL = HERE.parent
ROOT = CANONICAL.parents[1]
sys.path[:0] = [str(CANONICAL), str(CANONICAL / 'emotion-probes'), str(HERE)]
from shared.io import atomic, digest, read, require, sha  # noqa: E402
from emotion_probes.gemma_rendering import render_case  # noqa: E402
from emotion_rh_forecasting.inputs import Sources, load_probes, vector_identity  # noqa: E402

LABELS = ROOT / 'runs/emotion-rh-code-line-labels/gemma3-4b-coding-v1/labels.json'
PROBES = ROOT / 'data/probes/gemma4b'
OUT = ROOT / 'runs/emotion-rh-replay/gemma3-4b-coding-lines-best12-v1'


def adapted(trace):
    """Map audited code-line fields onto the shared causal replay contract."""
    return {**trace, 'dataset': 'coding',
            'units': [{**unit, 'kind': 'sentence_or_fragment', 'turn': 0,
                       'source_field': 'response', 'hacking': unit['indicative_of_rh'],
                       'needs_review': False}
                      for unit in trace['units']]}


def fields_for(trace, turn):
    source = read(ROOT / trace['source_path'])
    require(sha(ROOT / trace['source_path']) == trace['source_sha256'], 'Trace source checksum changed')
    if trace['dataset'] == 'coding':
        return source['prompt'], [], source['reasoning_text'] or '', source['response'] or ''
    prior = []
    for previous in source['turns']:
        if previous['turn'] == turn:
            break
        prior.append({'role': 'assistant', 'content': previous['response'] or '',
                      'reasoning': previous['reasoning_text'] or ''})
        prior.append({'role': 'user', 'content': previous['feedback'] or ''})
    return None, [*source['initial_messages'], *prior], next(t['reasoning_text'] or '' for t in source['turns'] if t['turn'] == turn), next(t['response'] or '' for t in source['turns'] if t['turn'] == turn)


def make_job(trace, turn, tokenizer):
    prompt, messages, reasoning, response = fields_for(trace, turn)
    if prompt is not None:
        messages = [{'role': 'user', 'content': prompt}]
    # Gemma's chat template trims assistant content at role boundaries. Freeze
    # that normalization explicitly so span markers cannot change the trim.
    messages.append({'role': 'assistant', 'content': response.strip(), 'reasoning': reasoning.strip()})
    starts = {}
    rendered_lengths = {}
    leading = {}
    rendered = None
    for field, stream in (('reasoning_text', 'reasoning'), ('response', 'content')):
        body = reasoning if stream == 'reasoning' else response
        if not body:
            continue
        rendered_body = body.strip()
        require(rendered_body, 'Assistant stream is whitespace only')
        leading[field] = len(body) - len(body.lstrip())
        case = {'id': trace['id'] + '/' + str(turn) + '/' + field, 'messages': messages,
                'render_mode': 'native', 'target': {'message_index': len(messages)-1,
                'stream': stream, 'char_start': 0, 'char_end': len(rendered_body), 'text': rendered_body}}
        text, start, end = render_case(case, tokenizer)
        require(rendered is None or rendered == text, 'Stream renders differ')
        rendered = text
        starts[field] = start
        rendered_lengths[field] = len(rendered_body)
        require(text[start:end] == rendered_body, 'Rendered stream changed')
    require(rendered is not None, 'Empty assistant turn')
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded['input_ids']
    offsets = encoded['offset_mapping']
    require(len(ids) == len(offsets) and len(ids) <= 131072, 'Tokenizer/context mismatch')
    output_token_ids = sorted({i for field, start in starts.items()
                               for i, (a, b) in enumerate(offsets)
                               if b > a and b > start and a < start + rendered_lengths[field]})
    output_positions = {index: position for position, index in enumerate(output_token_ids)}
    units = []
    for unit in trace['units']:
        if unit['turn'] != turn or unit['kind'] != 'sentence_or_fragment':
            continue
        field_body = reasoning if unit['source_field'] == 'reasoning_text' else response
        require(leading[unit['source_field']] <= unit['char_start'] < unit['char_end'] <=
                leading[unit['source_field']] + rendered_lengths[unit['source_field']],
                'Labeled unit falls outside retained assistant stream')
        start = starts[unit['source_field']] + unit['char_start'] - leading[unit['source_field']]
        end = starts[unit['source_field']] + unit['char_end'] - leading[unit['source_field']]
        require(rendered[start:end] == unit['text'], 'Sentence/source offset differs')
        selected = [(i, min(b, end)-max(a, start)) for i, (a, b) in enumerate(offsets)
                    if b > a and a < end and b > start and b <= end]
        fallback = False
        if not selected:
            # A one-character fragment may share its token with later text.
            # Keep the readout causal by using the last already-completed
            # assistant-output token, never the overlapping future token.
            previous = [i for i in output_token_ids if offsets[i][1] <= end]
            require(previous, f'No causal token for {trace["id"]}/{unit["unit_id"]}')
            selected = [(previous[-1], 1)]
            fallback = True
        require(all(i in output_positions for i, _ in selected), 'Scoring prompt token')
        units.append({'unit_id': unit['unit_id'], 'source_field': unit['source_field'], 'turn': turn,
                      'char_start': unit['char_start'], 'char_end': unit['char_end'], 'text': unit['text'],
                      'hacking': unit['hacking'], 'needs_review': unit['needs_review'],
                      **({'prior_token_fallback': True} if fallback else {}),
                      'token_indices': [i for i, _ in selected], 'token_weights': [w for _, w in selected],
                      'output_positions': [output_positions[i] for i, _ in selected]})
    jobid = trace['id'].replace('/', '__') + f'__turn{turn}'
    return {'id': jobid, 'trace_id': trace['id'], 'dataset': trace['dataset'], 'turn': turn,
            'source_sha256': trace['source_sha256'], 'rendered_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
            'token_ids': ids, 'token_ids_sha256': digest(ids), 'tokens': len(ids),
            'output_tokens': len(output_token_ids), 'units': units}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    source = read(LABELS)
    expected_traces = source['summary'].get('expected_traces', source['summary'].get('traces'))
    require(source['summary']['status'] == 'complete' and len(source['traces']) == expected_traces, 'Incomplete labels')
    bundle = load_probes(PROBES, Sources())
    require(bundle['model']['name'] == 'google/gemma-3-4b-it', 'Wrong probe checkpoint')
    tokenizer = AutoTokenizer.from_pretrained(bundle['model']['tokenizer_model'],
        revision=bundle['model']['tokenizer_revision'], trust_remote_code=False)
    require(tokenizer.is_fast, 'Offset tokenizer required')
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for original in source['traces']:
        trace = adapted(original)
        for turn in sorted({u['turn'] for u in trace['units']}):
            job = make_job(trace, turn, tokenizer)
            path = OUT / 'inputs' / f'{job["id"]}.json'
            if path.exists():
                require(read(path) == job, 'Replay job changed')
            else:
                atomic(path, job)
            jobs.append({'id': job['id'], 'sha256': sha(path), 'trace_id': trace['id'], 'turn': turn,
                         'tokens': job['tokens'], 'output_tokens': job['output_tokens'], 'units': len(job['units'])})
    expected_lines = source['summary']['code_lines']
    require(sum(j['units'] for j in jobs) == expected_lines, 'Code-line count changed')
    vectors = {p['emotion']: p['vector'] for p in bundle['probes']}
    vectors_path = OUT / 'probes.json'
    if vectors_path.exists():
        require(read(vectors_path) == bundle, 'Selected probes changed')
    else:
        atomic(vectors_path, bundle)
    spec = {'schema_version': 1, 'model': bundle['model'], 'probe_identity': vector_identity(bundle),
            'probes_sha256': sha(vectors_path), 'labels_sha256': sha(LABELS),
            'source_probe_manifest_sha256': sha(PROBES / 'manifest.json'),
            'tokenizer_template_sha256': digest(tokenizer.get_chat_template()),
            'interpretation': 'Gemma-native chat-template replay of archived OpenRouter coding responses. Checkpoint is pinned local BF16, not a verified byte-identical OpenRouter serving checkpoint. Code-line labels are deterministic evidence annotations, not intent judgments.',
            'jobs': jobs, 'counts': {'coding': len(source['traces'])},
            'total_tokens': sum(j['tokens'] for j in jobs), 'total_code_lines': sum(j['units'] for j in jobs)}
    spec['identity'] = digest(spec)
    spec_path = OUT / 'replay-spec.json'
    if spec_path.exists():
        require(read(spec_path) == spec, 'Replay specification changed')
    else:
        atomic(spec_path, spec)
    print(json.dumps({'jobs': len(jobs), 'total_tokens': spec['total_tokens'],
                      'max_tokens': max(j['tokens'] for j in jobs), 'code_lines': spec['total_code_lines'],
                      'output': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
