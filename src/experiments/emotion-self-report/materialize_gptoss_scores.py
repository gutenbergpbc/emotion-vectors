"""Materialize endpoint scores and provenance after GPT-OSS replay download."""
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
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'runs/emotion-self-report-activations/gptoss-complete-traces-v1'
SOURCE = ROOT / 'experiments/gpt_oss_sales_sentence_rh_12probes_20261016'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_kind(text: str) -> str:
    stripped = text.lstrip()
    if stripped.startswith('```COMMAND') or stripped == 'FINALIZE_AND_END_DAY':
        return 'command'
    if stripped == '```' or stripped.startswith('```\n'):
        return 'formatting'
    if stripped.startswith('```'):
        return 'code_block'
    return 'prose'


def main():
    points = json.loads((OUT / 'checkpoints.json').read_text())
    selection = json.loads((OUT / 'selection.json').read_text())
    selected = json.loads((OUT / 'selected-traces.json').read_text())
    by_key = {(p['trace_id'], p['unit_id']): p for p in points}
    source_spec = json.loads((SOURCE / 'run-spec.json').read_text())
    ratings = json.loads((OUT / 'ratings.json').read_text())
    from collections import Counter
    rating_providers = Counter((r.get('provider'), r.get('served_model')) for r in ratings)
    probe_bundle_path = SOURCE / 'probes.json'
    probe_bundle = json.loads(probe_bundle_path.read_text())
    probes = source_spec['probes']
    probe_by_emotion = {p['emotion']: p for p in probes}
    score_root = OUT / 'gptoss-complete-replay/scores'
    scores_out = []
    token_projection_count = 0
    for point in points:
        checkpoint_id = f"{point['trace_id']}::unit::{point['unit_id']}"
        score_id = hashlib.sha256(checkpoint_id.encode()).hexdigest()
        path = score_root / f'{score_id}.npz'
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['checkpoint_id'].item()) != checkpoint_id:
                raise ValueError(f'Checkpoint identity mismatch in {path}')
            orders = saved['emotion_order'].tolist()
            values = saved['scores']
            offsets = saved['offsets']
            if values.ndim != 2 or values.shape[1] != 12 or len(values) != len(offsets) or not len(values):
                raise ValueError(f'Unexpected token score shape in {path}')
            token_projection_count += len(values) * values.shape[1]
            probe_char_end = int(offsets[-1, 1])
            if not 0 < probe_char_end <= int(point['char_end']):
                raise ValueError(f'Projection is outside checkpoint endpoint: {checkpoint_id}')
            if orders != list(probe_by_emotion):
                raise ValueError(f'Probe order differs from pinned run spec: {checkpoint_id}')
            for j, emotion in enumerate(orders):
                raw = float(values[-1, j])
                if not np.isfinite(raw):
                    raise ValueError(f'Nonfinite endpoint projection: {checkpoint_id}/{emotion}')
                scores_out.append({
                    'trace_id': point['trace_id'], 'unit_id': point['unit_id'],
                    'emotion': emotion, 'raw_projection': raw,
                    'message_index': point['message_index'],
                    'sentence_char_end': point['char_end'],
                    'probe_char_end': probe_char_end,
                    'checkpoint_kind': checkpoint_kind(point['text']),
                })

    probes_out = []
    for p in probes:
        key = p['probe']
        vector = probe_bundle['models'][key]['vector']
        probes_out.append({
            'emotion': p['emotion'], 'probe': key,
            'layer_zero_based': p['layer_zero_based'],
            'balanced_accuracy': p['balanced_accuracy'],
            'training_mix': p['training_mix'], 'recipe': p['recipe'],
            'vector_norm_recorded': p['vector_norm'],
            'vector_sha256_json': hashlib.sha256(
                json.dumps(vector, separators=(',', ':')).encode()).hexdigest(),
            'probe_bundle': str(probe_bundle_path.relative_to(ROOT)),
            'probe_bundle_sha256': sha(probe_bundle_path),
            'catalog_source': p['catalog_source'],
            'catalog_probe': p['catalog_probe'],
            'catalog_sha256': p['catalog_sha256'],
            'selection_metric': p['selection_metric'],
        })

    provenance = {
        'schema_version': 1,
        'model': selection['model'],
        'extraction_model': source_spec['model'],
        'tokenizer': {
            'name': source_spec['model']['name'],
            'revision': source_spec['model']['revision'],
            'template': 'AutoTokenizer loaded at the pinned model revision; backend calls tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, reasoning_effort="low") for the native conversation before appending the exact assistant prefix.',
            'template_kwargs': source_spec['model'].get('template_kwargs', {}),
        },
        'probes': probes_out,
        'readout': {
            'method': 'raw linear projection: block output hidden vector dot saved probe vector',
            'activation_site': 'GPT-OSS transformer block output before final normalization',
            'causal_endpoint': 'Last completed assistant output token whose character offset ends at or before the frozen checkpoint char_end; the replay input itself is truncated exactly to that checkpoint endpoint.',
            'aggregation': 'The exported checkpoint projection is the primary last completed token readout. All available per-token projections and token offsets are retained in gptoss-complete-replay/scores/*.npz.',
            'future_context': 'No later assistant text, subsequent tool result, or self-report question is included in the replayed message sequence.',
        },
        'runtime': {
            'gpu': 'NVIDIA H200 via Modal',
            'device': source_spec['model']['device'],
            'dtype': source_spec['model']['dtype'],
            'quantization': source_spec['model']['runtime']['quantization'],
            'attention': source_spec['model']['runtime']['attention'],
            'kernel_repository': source_spec['model']['runtime']['kernel_repository'],
            'kernel_revision': source_spec['model']['runtime']['kernel_revision'],
            'packages': source_spec['model']['runtime']['packages'],
            'modal_run_id': 'gptoss-sales-self-report-activations-v1',
        },
        'source_identity': {
            'selection_source_sha256': selection['source_sha256'],
            'selection_json_sha256': sha(OUT / 'selection.json'),
            'selected_traces_sha256': sha(OUT / 'selected-traces.json'),
            'checkpoints_sha256': sha(OUT / 'checkpoints.json'),
            'ratings_sha256': sha(OUT / 'ratings.json'),
            'replay_inputs_sha256': sha(OUT / 'gptoss-replay-inputs.json'),
            'run_spec_sha256': sha(SOURCE / 'run-spec.json'),
            'probe_bundle_sha256': sha(probe_bundle_path),
        },
        'coverage': {
            'traces': selection['selected_traces'],
            'checkpoints': selection['selected_checkpoints'],
            'endpoint_projection_rows': len(scores_out),
            'token_projection_values_retained': token_projection_count,
            'checkpoint_kinds': {kind: sum(1 for p in points if checkpoint_kind(p['text']) == kind)
                                 for kind in ('command', 'code_block', 'formatting', 'prose')},
            'existing_rating_providers': [
                {'provider': provider, 'served_model': model, 'ratings': count}
                for (provider, model), count in sorted(rating_providers.items())
            ],
            'new_self_report_requests_made_during_extraction': False,
        },
    }
    (OUT / 'probe-scores.json').write_text(json.dumps(scores_out, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(provenance['coverage'], indent=2))


if __name__ == '__main__':
    main()
