"""Join causal Gemma 4B token projections to the frozen self-report checkpoints."""
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


import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CANONICAL = HERE.parent
ROOT = CANONICAL.parents[1]
import sys
sys.path.insert(0, str(CANONICAL))
from shared.io import atomic, read, require, sha  # noqa: E402

SELECTION = ROOT / 'runs/emotion-self-report-activations/gemma4b-complete-traces-v1'
REPLAY = SELECTION / 'gemma4b-replay-v1'
OUT = SELECTION


def checkpoint_kind(message, char_start, text, dataset):
    if dataset == 'coding':
        before = message[:char_start]
        if before.count('```') % 2:
            return 'code'
    first = text.lstrip().splitlines()[0] if text.strip() else ''
    if first.startswith('```'):
        return 'code_block'
    if first.startswith(('$ ', '>>>', 'PS>', 'In [', 'Out[')):
        return 'command'
    if dataset == 'coding' and (first.startswith(('def ', 'class ', 'import ', 'from '))
                                or first.startswith(('{', '[', '    '))):
        return 'code'
    return 'prose'


def main():
    selected = read(SELECTION / 'selected-traces.json')
    spec = read(REPLAY / 'replay-spec.json')
    probes_bundle = read(REPLAY / 'probes.json')
    jobs = {entry['id']: read(REPLAY / 'inputs' / f"{entry['id']}.json") for entry in spec['jobs']}
    scores = []
    per_probe = probes_bundle['probes']
    require(len(per_probe) == 12, 'Expected twelve selected probes')
    for entry in spec['jobs']:
        job = jobs[entry['id']]
        archive_path = REPLAY / 'scores' / (entry['id'] + '.npz')
        receipt = read(archive_path.with_suffix('.receipt.json'))
        require(receipt['replay_identity'] == spec['identity'], 'Projection shard belongs to another replay')
        require(sha(archive_path) == receipt['sha256'], 'Projection shard checksum mismatch')
        with np.load(archive_path, allow_pickle=False) as archive:
            projections = archive['scores']
            offsets = archive['offsets']
            native_indices = archive['native_token_index']
            require(projections.shape == (len(offsets), 12), 'Unexpected token projection shape')
            require(np.isfinite(projections).all(), 'Nonfinite per-token projections')
        trace = next(t for t in selected if t['trace_id'] == job['trace_id'])
        message = trace['case']['messages'][job['message_index']]['content']
        for unit in job['units']:
            endpoint = unit['rendered_char_end']
            valid = np.flatnonzero((offsets[:, 1] <= endpoint) &
                                   (offsets[:, 1] > job['target_start']) &
                                   (offsets[:, 1] > offsets[:, 0]))
            require(len(valid) > 0, f"No completed assistant token at {job['trace_id']} unit {unit['unit_id']}")
            token_row = max(valid, key=lambda i: (int(offsets[i, 1]), int(native_indices[i])))
            probe_char_end = (int(offsets[token_row, 1]) - int(job['target_start'])
                              + int(job['target_leading_trim']))
            sentence_char_end = unit['source_char_end']
            sentence_char_start = unit['source_char_start']
            require(0 < probe_char_end <= sentence_char_end, 'Probe token extends past checkpoint endpoint')
            require(unit['text'] == message[sentence_char_start:sentence_char_end], 'Checkpoint text changed')
            for probe_index, probe in enumerate(per_probe):
                scores.append({
                    'trace_id': job['trace_id'], 'unit_id': unit['unit_id'],
                    'emotion': probe['emotion'], 'raw_projection': float(projections[token_row, probe_index]),
                    'message_index': job['message_index'], 'sentence_char_end': sentence_char_end,
                    'probe_char_end': probe_char_end,
                    'probe_token_index': int(native_indices[token_row]),
                    'checkpoint_kind': checkpoint_kind(message, sentence_char_start, unit['text'], job['dataset']),
                    'readout': 'last_causally_completed_assistant_token',
                    'probe_id': probe['probe'], 'probe_source': probe.get('source', 'gemma3-4b-sales-best12-v1'),
                    'probe_layer': probe['layer'], 'probe_recipe': probe.get('recipe', 'best12_existing_probe'),
                })
    expected = sum(len(t['checkpoints']) for t in selected) * 12
    require(len(scores) == expected, f'Score coverage is {len(scores)}, expected {expected}')
    require(all(np.isfinite(row['raw_projection']) for row in scores), 'A probe score is nonfinite')
    if (OUT / 'probe-scores.json').exists():
        prior = read(OUT / 'probe-scores.json')
        require(prior == scores, 'Existing score output differs; refusing replacement')
    else:
        atomic(OUT / 'probe-scores.json', scores)
    provenance = {
        'schema_version': 1,
        'model': probes_bundle['model']['name'],
        'extraction_model': {'name': probes_bundle['model']['name'], 'revision': probes_bundle['model']['revision']},
        'model_weights': {'repository': probes_bundle['model']['name'], 'revision': probes_bundle['model']['revision'],
                          'format': 'safetensors', 'dtype': 'bfloat16', 'quantization': None},
        'probe_bundle_sha256': sha(REPLAY / 'probes.json'),
        'probe_identity': spec.get('probe_identity'),
        'probes': [{'id': p['probe'], 'emotion': p['emotion'], 'layer': p['layer'],
                    'source': p.get('source', 'gemma3-4b-sales-best12-v1'), 'recipe': p.get('recipe', 'best12_existing_probe'), 'centering': p['centering']}
                   for p in per_probe],
        'readout': 'Float32 dot product of the post-block residual with each selected probe. For every checkpoint, use the last token wholly completed by that checkpoint endpoint; no subsequent token is included. Per-token projections and tokenizer offsets are retained in gemma4b-replay-v1/scores/*.npz.',
        'tokenizer': {'model': probes_bundle['model']['name'], 'revision': probes_bundle['model']['revision'],
                      'chat_template_sha256': spec['tokenizer_chat_template_sha256'],
                      'renderer': spec['renderer']},
        'conversation_normalization': 'Gemma native chat-template replay strips leading/trailing whitespace from prior assistant messages and strips the current assistant message before truncating to each message’s last selected checkpoint. Original source character offsets are restored in probe-scores.json; checkpoint text is validated byte-for-byte.',
        'selection_sha256': sha(SELECTION / 'selected-traces.json'),
        'cases_sha256': sha(SELECTION / 'cases.json'),
        'traces_sha256': sha(SELECTION / 'selected-traces.json'),
        'replay_identity': spec['identity'],
        'runtime': read(REPLAY / 'runtime.json') if (REPLAY / 'runtime.json').exists() else {},
        'coverage': {'traces': len(selected), 'checkpoints': expected // 12,
                     'emotions_per_checkpoint': 12, 'score_rows': len(scores)},
    }
    if (OUT / 'provenance.json').exists():
        require(read(OUT / 'provenance.json') == provenance, 'Existing provenance differs; refusing replacement')
    else:
        atomic(OUT / 'provenance.json', provenance)
    print(json.dumps({'score_rows': len(scores), 'checkpoints': expected // 12,
                      'provenance': str(OUT / 'provenance.json'),
                      'probe_scores': str(OUT / 'probe-scores.json')}, indent=2))


if __name__ == '__main__':
    main()
