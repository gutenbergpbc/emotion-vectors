"""Freeze Gemma 3 4B causal prefixes for the selected self-report traces."""
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
sys.path[:0] = [str(CANONICAL), str(CANONICAL / 'emotion-probes')]
from shared.io import atomic, digest, read, require, sha  # noqa: E402
from emotion_probes.gemma_rendering import render_case  # noqa: E402

SELECTION = ROOT / 'runs/emotion-self-report-activations/gemma4b-complete-traces-v1'
PROBES = ROOT / 'data/probes/gemma4b/probes.json'
OUT = SELECTION / 'gemma4b-replay-v1'


def make_job(trace, tokenizer):
    case = trace['case']
    checkpoints = trace['checkpoints']
    by_message = {}
    for cp in checkpoints:
        by_message.setdefault(cp['message_index'], []).append(cp)
    jobs = []
    for message_index, units in sorted(by_message.items()):
        messages = [dict(m) for m in case['messages'][:message_index + 1]]
        for previous in messages[:-1]:
            if previous['role'] == 'assistant':
                previous['content'] = previous.get('content', '').strip()
        final = messages[-1]
        require(final['role'] == 'assistant', 'Selected target must be assistant text')
        raw_message = final.get('content', '')
        leading_trim = len(raw_message) - len(raw_message.lstrip())
        normalized = raw_message.strip()
        translated = []
        for cp in units:
            start, end = cp['char_start'] - leading_trim, cp['char_end'] - leading_trim
            require(0 <= start < end <= len(normalized), 'Checkpoint falls outside Gemma whitespace normalization')
            require(normalized[start:end] == cp['text'], 'Whitespace normalization changed checkpoint text')
            translated.append(({**cp, 'char_start': start, 'char_end': end,
                                'source_char_start': cp['char_start'], 'source_char_end': cp['char_end']}))
        max_char = max(cp['char_end'] for cp in translated)
        visible = normalized[:max_char]
        final['content'] = visible
        for cp in translated:
            require(visible[cp['char_start']:cp['char_end']] == cp['text'], 'Frozen checkpoint text changed')
        render_case_input = {'messages': messages, 'render_mode': 'native', 'target': {
            'message_index': len(messages) - 1, 'stream': 'content', 'char_start': 0,
            'char_end': len(visible), 'text': visible}}
        rendered, target_start, target_end = render_case(render_case_input, tokenizer)
        require(rendered[target_start:target_end] == visible, 'Gemma template changed selected assistant text')
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded['input_ids']
        offsets = [list(pair) for pair in encoded['offset_mapping']]
        require(len(ids) == len(offsets) and len(ids) < 131072, 'Tokenizer/context mismatch')
        translated_units = []
        for cp in translated:
            end = target_start + cp['char_end']
            start = target_start + cp['char_start']
            translated_units.append({**cp, 'rendered_char_start': start,
                                     'rendered_char_end': end})
        job_id = hashlib.sha256(f"{trace['trace_id']}:{message_index}".encode()).hexdigest()[:20]
        job = {'id': job_id, 'trace_id': trace['trace_id'], 'dataset': trace['dataset'],
               'message_index': message_index, 'rendered': rendered,
               'rendered_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
               'target_start': target_start, 'target_end': target_end,
               'target_leading_trim': leading_trim,
               'units': translated_units, 'source_boundaries': [
                   {'char_end': u['rendered_char_end']} for u in translated_units],
               'token_ids': ids, 'offsets': offsets,
               'kind': [1 if b > target_start and a < target_end else 0 for a, b in offsets],
               'tokens': len(ids)}
        job['source_sha256'] = sha(SELECTION / 'selected-traces.json')
        job['input_sha256'] = digest({k: v for k, v in job.items() if k != 'input_sha256'})
        jobs.append(job)
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-limit', type=int, default=0)
    args = parser.parse_args()
    selection = read(SELECTION / 'selected-traces.json')
    probes = read(PROBES)
    source_spec = {'probe_identity': digest(probes)}
    require(len(probes['probes']) == 12, 'Expected the existing best-12 probe bundle')
    tokenizer = AutoTokenizer.from_pretrained(probes['model']['name'],
        revision=probes['model']['revision'], use_fast=True, trust_remote_code=False)
    require(tokenizer.is_fast, 'Offset tokenizer required')
    selected = selection[:args.trace_limit] if args.trace_limit else selection
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [job for trace in selected for job in make_job(trace, tokenizer)]
    for job in jobs:
        path = OUT / 'inputs' / f"{job['id']}.json"
        if path.exists():
            prior = read(path)
            if prior != job:
                require({k: v for k, v in prior.items() if k not in ('source_sha256', 'input_sha256')} ==
                        {k: v for k, v in job.items() if k not in ('source_sha256', 'input_sha256')},
                        'Frozen Gemma replay job changed')
                atomic(path, job)
        else:
            atomic(path, job)
    atomic(OUT / 'probes.json', probes)
    spec = {
        'schema_version': 1, 'model': probes['model'], 'probe_identity': source_spec['probe_identity'],
        'selection_sha256': sha(SELECTION / 'selected-traces.json'),
        'probes_sha256': sha(OUT / 'probes.json'), 'tokenizer_chat_template_sha256': digest(tokenizer.get_chat_template()),
        'renderer': 'gemma-native-preserved-streams-v2', 'readout': 'float32 dot product of each selected best12 probe with post-block residual; causal prefix; last token wholly completed by sentence endpoint',
        'jobs': [{'id': j['id'], 'trace_id': j['trace_id'], 'message_index': j['message_index'],
                  'tokens': j['tokens'], 'units': len(j['units']), 'sha256': sha(OUT / 'inputs' / f"{j['id']}.json")} for j in jobs],
    }
    spec['identity'] = digest(spec)
    atomic(OUT / 'replay-spec.json', spec)
    print(json.dumps({'jobs': len(jobs), 'units': sum(len(j['units']) for j in jobs),
                      'tokens': sum(j['tokens'] for j in jobs), 'output': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
