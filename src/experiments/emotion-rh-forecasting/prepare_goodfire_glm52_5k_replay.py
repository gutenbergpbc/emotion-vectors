"""Freeze local-context NVFP4 activation replay for the 5,000 judged spans.

Each selected source passage is replayed with system/user and up to 30 prior
nodes, trimming older nodes only if needed to fit the 131072-token context.
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


import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE.parent), str(HERE)]
from shared.io import atomic, digest, read, require, sha  # noqa: E402
from emotion_rh_forecasting.inputs import Sources, load_probes, vector_identity  # noqa: E402
from prepare_goodfire_glm52_replay import job_for, PROBES  # noqa: E402
from select_goodfire_glm52_gpt61sol_5k import OUT as LABELS, SOURCE  # noqa: E402

OUT = ROOT / 'runs/emotion-rh-replay/goodfire-glm52-deepswe-nvfp4-gpt61sol-5k-v1'


def main():
    global LABELS, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=Path, default=LABELS)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    LABELS, OUT = args.labels, args.out
    selection = read(LABELS / 'selection.json')
    meta = {t['row_index']: t for t in selection['traces']}
    units_by_trace = defaultdict(lambda: defaultdict(list))
    with (LABELS / 'selected-sentences.jsonl').open() as stream:
        for line in stream:
            u = json.loads(line)
            u['unit_id'] = u['sentence_index']
            units_by_trace[u['row_index']][(u['node_index'], u['channel'])].append(u)
    require(sum(len(x) for p in units_by_trace.values() for x in p.values()) == 5000, 'Selection changed')
    bundle = load_probes(PROBES, Sources())
    require(bundle['model']['name'] == 'nvidia/GLM-5.2-NVFP4', 'Wrong probe model')
    tokenizer = AutoTokenizer.from_pretrained(bundle['model']['tokenizer_model'],
                                              revision=bundle['model']['tokenizer_revision'])
    require(tokenizer.is_fast, 'Fast offset tokenizer required')
    OUT.mkdir(parents=True, exist_ok=True)
    probes_path = OUT / 'probes.json'
    if probes_path.exists():
        require(read(probes_path) == bundle, 'Probe bundle changed')
    else:
        atomic(probes_path, bundle)
    jobs = []
    with (SOURCE / 'traces.jsonl').open('rb') as stream:
        for row_index, raw in enumerate(stream):
            if row_index not in meta:
                continue
            entry = meta[row_index]
            require(hashlib.sha256(raw).hexdigest() == entry['source_line_sha256'], 'Trace changed')
            trace = json.loads(raw)['traces'][0]
            require(trace['id'] == entry['trace_id'], 'Trace identity changed')
            case = {'row_index': row_index, 'task': entry['task'], 'sha256': entry['source_line_sha256']}
            for (node_index, channel), units in sorted(units_by_trace[row_index].items()):
                units.sort(key=lambda u: u['unit_id'])
                job_id = f"{trace['id']}__n{node_index:04d}_{channel}"
                job = None
                previous_used = None
                for previous in (30, 20, 10, 4, 0):
                    lo = max(2, node_index - previous)
                    prior = list(range(lo, node_index + 1))
                    if prior:
                        # Do not begin a local window with an orphaned tool result.
                        first_assistant = next((i for i in prior
                                                if (trace['nodes'][i].get('message') or {}).get('role') == 'assistant'), None)
                        if first_assistant is not None:
                            prior = [i for i in prior if i >= first_assistant]
                    nodes = sorted(set([0, 1, *prior]))
                    try:
                        job = job_for(trace, case, tokenizer, units, nodes, job_id)
                        previous_used = previous
                        break
                    except ValueError as exc:
                        if 'Trace exceeds GLM context' not in str(exc):
                            raise
                require(job is not None, f'Positive passage exceeds GLM context: {job_id}')
                job['context_nodes'] = nodes
                job['prior_node_budget'] = previous_used
                path = OUT / 'inputs' / f"{job_id}.json"
                if path.exists():
                    require(read(path) == job, 'Replay job changed')
                else:
                    atomic(path, job)
                jobs.append({'id': job_id, 'trace_id': trace['id'], 'row_index': row_index,
                             'task': entry['task'], 'sha256': sha(path),
                             'tokens': job['tokens'], 'output_tokens': job['output_tokens'],
                             'units': len(job['units']), 'prior_node_budget': previous_used})
                print(f"Prepared {row_index}/{node_index}/{channel}: {job['tokens']} tokens, "
                      f"{len(units)} sentences, {previous_used} prior nodes", flush=True)
    require(sum(j['units'] for j in jobs) == 5000 and len({j['id'] for j in jobs}) == len(jobs),
            'Replay coverage changed')
    spec = {'schema_version': 1, 'model': bundle['model'],
            'probe_identity': vector_identity(bundle), 'probes_sha256': sha(probes_path),
            'source_probe_manifest_sha256': sha(PROBES / 'manifest.json'),
            'selection_sha256': sha(LABELS / 'selection.json'),
            'label_request_manifest_sha256': sha(LABELS / 'judge-manifest.json'),
            'tokenizer_template_sha256': digest(tokenizer.get_chat_template()),
            'interpretation': 'NVFP4 GLM replay of archived FP8 DeepSWE message/tool text in local causal windows. Each window contains the system/user context and up to 30 preceding nodes; older history is omitted. These are not original FP8 activations and differ from full-trace pilot replay.',
            'jobs': jobs, 'total_tokens': sum(j['tokens'] for j in jobs),
            'total_sentences': sum(j['units'] for j in jobs)}
    spec['identity'] = digest(spec)
    path = OUT / 'replay-spec.json'
    if path.exists():
        require(read(path) == spec, 'Replay specification changed')
    else:
        atomic(path, spec)
    print(json.dumps({'jobs': len(jobs), 'tokens': spec['total_tokens'],
                      'max_tokens': max(j['tokens'] for j in jobs),
                      'sentences': spec['total_sentences']}, indent=2))


if __name__ == '__main__':
    main()
