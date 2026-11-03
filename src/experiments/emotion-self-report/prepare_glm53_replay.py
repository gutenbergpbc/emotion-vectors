"""Prepare native-template GLM 5.3 replay prompts for frozen self-report traces."""
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


import hashlib
import json
import argparse
from collections import defaultdict
from pathlib import Path
import sys

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
CANONICAL = HERE.parent
ROOT = CANONICAL.parents[1]
sys.path[:0] = [str(CANONICAL), str(CANONICAL / 'emotion-probes')]
from shared.io import atomic, digest, read, require, sha  # noqa: E402
from emotion_probes.rendering import render_case  # noqa: E402

FROZEN = ROOT / 'runs/emotion-self-report-activations/glm53-complete-traces-v2'
PROBE_BUNDLE = ROOT / 'runs/emotion-rh-replay/glm-5.3-coding-1000-best12-v1/probes.json'
OUT = FROZEN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=FROZEN)
    args = parser.parse_args()
    out = args.root.resolve()
    terminalwrench = 'terminalwrench' in out.name
    reasoning_effort = 'low' if terminalwrench else 'max'
    bundle = read(PROBE_BUNDLE)
    require(len(bundle['probes']) == 12, 'Expected the existing best12 probe bundle')
    tokenizer = AutoTokenizer.from_pretrained(bundle['model']['tokenizer_model'],
        revision=bundle['model']['tokenizer_revision'], trust_remote_code=False)
    require(tokenizer.is_fast, 'Fast tokenizer offsets are required')
    traces = read(out / 'selected-traces.json')
    jobs = []
    for trace in traces:
        case = trace['case']
        by_message = defaultdict(list)
        for point in trace['checkpoints']:
            by_message[point['message_index']].append(point)
        for msg_i, checkpoints in sorted(by_message.items()):
            assistant = case['messages'][msg_i]
            require(assistant['role'] == 'assistant', 'Checkpoint message is not assistant')
            body = assistant['content']
            require(all(0 < p['char_end'] <= len(body) for p in checkpoints),
                    'Checkpoint is outside assistant message')
            max_end = max(p['char_end'] for p in checkpoints)
            # Prefix ends at this checkpoint turn and endpoint. Later turns and
            # appended self-report questions never enter the model context.
            messages = [*case['messages'][:msg_i], {**assistant, 'content': body[:max_end]}]
            target = {'message_index': msg_i, 'stream': 'content', 'char_start': 0,
                      'char_end': max_end, 'text': body[:max_end]}
            render_input = {'id': trace['trace_id'], 'messages': messages,
                            'render_mode': 'native', 'target': target}
            rendered, body_start, body_end = render_case(render_input, tokenizer, reasoning_effort)
            encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
            token_ids, offsets = encoded['input_ids'], encoded['offset_mapping']
            require(body_end == body_start + max_end and rendered[body_start:body_end] == body[:max_end],
                    'Native GLM rendering changed assistant text')
            readouts = []
            for point in checkpoints:
                char_end = body_start + point['char_end']
                eligible = [i for i, (a, b) in enumerate(offsets)
                            if b > body_start and b > a and b <= char_end]
                require(eligible, f"No causal token for {trace['trace_id']} unit {point['unit_id']}")
                idx = eligible[-1]
                require(offsets[idx][1] <= char_end, 'Readout token overlaps future text')
                readouts.append({'unit_id': point['unit_id'], 'message_index': msg_i,
                    'sentence_char_end': point['char_end'], 'probe_char_end': offsets[idx][1] - body_start,
                    'token_index': idx})
            job_id = hashlib.sha256(f"{trace['trace_id']}:{msg_i}".encode()).hexdigest()[:20]
            job = {'id': job_id, 'trace_id': trace['trace_id'], 'dataset': trace['dataset'],
                   'token_ids': token_ids, 'tokens': len(token_ids), 'token_ids_sha256': digest(token_ids),
                   'rendered_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
                   'source_run': trace['source_run'], 'readouts': readouts}
            path = out / 'inputs' / f'{job_id}.json'
            if path.exists():
                require(read(path) == job, f'Frozen GLM job changed: {job_id}')
            else:
                atomic(path, job)
            jobs.append({'id': job_id, 'sha256': sha(path), 'trace_id': trace['trace_id'],
                         'message_index': msg_i, 'tokens': len(token_ids), 'checkpoints': len(readouts)})
    atomic(out / 'probes.json', bundle)
    spec = {'schema_version': 1, 'identity': '',
            'model': read(out / 'selection.json')['model'],
            'extraction_model': bundle['model'],
            'provider_model': 'z-ai/glm-5.3', 'replay_model': 'Inferact/GLM-5.3-NVFP4',
            'quantization': 'modelopt NVFP4', 'provider_quantization': 'FP8 routes recorded in source manifests (OpenRouter-served; exact remote weights not independently verified)',
            'provider_routes': ({'terminalwrench': ['novita/fp8']} if terminalwrench else {'coding': ['sail-research/fp8'], 'sales': ['novita/fp8']}),
            'reasoning_effort': reasoning_effort,
            'provider_route_note': 'Routes are copied from the source self-report collection manifests/configs; the exact remote serving weights are not verified byte-identical to the NVFP4 replay checkpoint.',
            'probe_identity': hashlib.sha256(json.dumps(bundle['probes'], sort_keys=True).encode()).hexdigest(),
            'probes_sha256': sha(out / 'probes.json'),
            'tokenizer_template_sha256': digest(tokenizer.get_chat_template()),
            'renderer': 'GLM native preserved streams; final assistant visible prefix only',
            'readout': 'Post-block residual dot probe vector; last tokenizer token whose character end is <= checkpoint sentence_char_end; no intercept; no token after checkpoint endpoint.',
            'source_selection_sha256': sha(out / 'selection.json'), 'jobs': jobs,
            'total_checkpoints': sum(j['checkpoints'] for j in jobs),
            'total_prompt_tokens': sum(j['tokens'] for j in jobs)}
    spec['identity'] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    atomic(out / 'replay-spec.json', spec)
    print(json.dumps({'traces': len(jobs), 'checkpoints': spec['total_checkpoints'],
                      'prompt_tokens': spec['total_prompt_tokens'], 'max_tokens': max(j['tokens'] for j in jobs),
                      'out': str(out)}, indent=2))


if __name__ == '__main__':
    main()
