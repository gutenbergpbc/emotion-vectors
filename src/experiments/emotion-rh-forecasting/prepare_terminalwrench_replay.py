"""Freeze selected emotion probes and exact ATIF token/span evidence for GPU replay."""

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
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared.io import atomic, digest, read, require, sha
from emotion_rh_forecasting.inputs import validate_bundle, vector_identity
from emotion_rh_forecasting.terminalwrench_replay import sentence_cuts


def prepare(config_path):
    config_path = Path(config_path).resolve()
    cfg = read(config_path)
    for key in ('labels', 'selection', 'archive', 'output'):
        cfg[key] = str((config_path.parent / cfg[key]).resolve())
    root, label_root, selection_root = Path(cfg['output']), Path(cfg['labels']), Path(cfg['selection'])
    root.mkdir(parents=True, exist_ok=True)
    require(read(label_root/'independent-validation.json')['status'] == 'passed', 'Annotation audit missing')
    labels = read(label_root/'labels.json')['traces']
    require(read(label_root/'summary.json')['status'] == 'complete', 'Incomplete annotations')
    folds = read(label_root/'folds.json')
    require(folds['labels_sha256'] == sha(label_root/'labels.json'), 'Fold/label binding differs')
    selection, sources = read(selection_root/'selected-probes.json'), read(selection_root/'config.resolved.json')['inputs']
    probes, source_hashes = [], {}
    for selected in selection['probes']:
        source = sources[selected['source_index']]
        directory = Path(source['metadata']['path'])
        for name in ('calibrated_models.json', 'probe_vectors.npz'):
            require(sha(directory/name) == source['files'][name], 'Probe source changed')
            source_hashes[str(directory/name)] = source['files'][name]
        metadata = read(directory/'calibrated_models.json')['models'][selected['probe']]
        require(metadata['layer'] == selected['layer'] and metadata['emotion'] == selected['emotion'], 'Selection metadata differs')
        with np.load(directory/'probe_vectors.npz', allow_pickle=False) as vectors:
            vector = vectors[selected['probe']].astype(float).tolist()
        require(np.allclose(vector, metadata['vector'], rtol=1e-7, atol=1e-9), 'Vector differs from selected model')
        probes.append({**selected, 'vector': vector, 'intercept': 0})
    model_meta = sources[0]['metadata']
    require(all(s['metadata']['revision'] == model_meta['revision'] for s in sources), 'Probe checkpoints differ')
    bundle = validate_bundle({'schema_version': 1, 'model': {'name': model_meta['model'], 'revision': model_meta['revision']},
                              'selection': selection['selection'], 'probes': probes})
    atomic(root/'probes.json', bundle)
    is_gpt = bundle['model']['name'] == 'openai/gpt-oss-120b'
    traces, jobs = [], []
    archive = Path(cfg['archive'])
    for trace in labels:
        cid = trace['id']
        raw = read(label_root/'inputs/traces'/f'{cid}.json')
        require(sha(label_root/'inputs/traces'/f'{cid}.json') == trace['source_sha256'], 'Original annotation source changed')
        meta_path, array_path = archive/f'{cid}.json', archive/f'{cid}.npz'
        meta = read(meta_path)
        require(meta['source_sha256'] == trace['source_sha256'], 'Archive is a different trajectory')
        rendered = meta['rendered']
        require(hashlib.sha256(rendered.encode()).hexdigest() == meta['rendered_sha256'], 'Archived rendering checksum differs')
        steps = {str(s['step_id']): s for s in raw['steps']}
        if is_gpt:
            segments = {str(s['step']): s for s in meta['segments']
                        if s.get('role') == 'assistant' and s.get('channel') == 'final'}
        else:
            segments = {str(s['step']): s for s in meta['segments'] if s['kind'] == 'assistant'}
        units = []
        for unit in trace['units']:
            original = steps[unit['step_id']]['message']
            segment = segments[unit['step_id']]
            require((original if is_gpt else original.strip()) == segment['text'], 'Archived assistant segment differs from annotation text')
            trim = 0 if is_gpt else len(original)-len(original.lstrip())
            if is_gpt:
                segment = {**segment, 'char_start': segment['start']}
            lo, hi = segment['char_start']+unit['char_start']-trim, segment['char_start']+unit['char_end']-trim
            require(rendered[lo:hi] == unit['text'], 'Rendered sentence mapping failed')
            units.append({**unit, 'source_char_start': unit['char_start'], 'source_char_end': unit['char_end'],
                          'char_start': lo, 'char_end': hi, 'turn_id': cid})
        input_path = root/'inputs'/f'{cid}.npz'
        input_path.parent.mkdir(exist_ok=True)
        if is_gpt:
            arrays = dict(token_ids=np.asarray(meta['input_ids'], dtype=np.int64),
                          offsets=np.asarray([[t['start'],t['end']] for t in meta['tokens']],dtype=np.int64),
                          kind=np.asarray([1 if t['role']=='assistant' else 0 for t in meta['tokens']],dtype=np.int8))
        else:
            with np.load(array_path, allow_pickle=False) as saved:
                arrays = {k: saved[k].copy() for k in ('token_ids', 'offsets', 'kind')}
        sentence_cuts(arrays['offsets'], units)
        np.savez_compressed(input_path, **arrays)
        job = {'id': cid, 'source_sha256': trace['source_sha256'], 'input_sha256': sha(input_path),
               'rendered': rendered, 'rendered_sha256': meta['rendered_sha256'],
               'source_boundaries': [{'unit_id': u['unit_id'], 'char_start': u['char_start'], 'char_end': u['char_end']} for u in units],
               'archive_metadata_sha256': sha(meta_path), 'archive_arrays_sha256': None if is_gpt else sha(array_path),
               'native_context_tokens': len(arrays['token_ids'])}
        atomic(root/'inputs'/f'{cid}.json', job)
        jobs.append({k: v for k, v in job.items() if k not in ('rendered', 'source_boundaries')})
        traces.append({'id': cid, 'task': trace['task'], 'group_id': trace['group_id'], 'fold': folds['trace_folds'][cid],
                       'turn_ids': [cid], 'units': units})
    atomic(root/'dataset-traces.json', traces)
    spec = {'schema_version': 1, 'config': cfg, 'model': bundle['model'], 'probe_identity': vector_identity(bundle),
            'probes_sha256': sha(root/'probes.json'), 'jobs': jobs, 'labels_sha256': sha(label_root/'labels.json'),
            'folds_sha256': sha(label_root/'folds.json'), 'dataset_traces_sha256': sha(root/'dataset-traces.json'),
            'selection_sha256': sha(selection_root/'selected-probes.json'), 'probe_source_hashes': source_hashes,
            'renderer': 'atif-gemma-replay-v1 (archived rendering verified against pinned tokenizer)',
            'interpretation': 'Gemma native-template replay of archived ATIF message/tool text; GPT-5.5 full-trajectory sentence labels. Reconstructed replay, not original generation activations.',
            'chunk_tokens': 512, 'max_tokens': 131072,
            'source_hashes': {p.name: sha(p) for p in (Path(__file__), HERE/'modal_terminalwrench_replay.py', HERE/'emotion_rh_forecasting/terminalwrench_replay.py')}}
    if is_gpt:
        spec.update(renderer='gpt-oss-harmony-preserved-streams-v1',
                    interpretation='GPT-OSS native Harmony replay of archived ATIF message/tool text; DeepSeek V4.1 Flash 1017 full-trajectory sentence labels. Reconstructed activations, not original serving activations.',
                    validation_protocol='gptoss-causal-schedule-v1', attention='eager',
                    kernel_revision='b26c936cb02fb91920cbccfc7454284b47d86b0d')
        spec['source_hashes'].update({p.name:sha(p) for p in (HERE/'modal_gptoss_terminalwrench_replay.py', HERE/'emotion_rh_forecasting/gptoss_terminalwrench_replay.py')})
    spec['identity'] = digest(spec)
    if (root/'replay-spec.json').exists():
        require(read(root/'replay-spec.json')['identity'] == spec['identity'], 'Replay identity changed; use a new output directory')
    atomic(root/'replay-spec.json', spec)
    with zipfile.ZipFile(root/'inputs.zip', 'w', compression=zipfile.ZIP_DEFLATED) as zipfile_out:
        for path in sorted((root/'inputs').iterdir()):
            zipfile_out.write(path, path.name)
    print(json.dumps({'output': str(root), 'traces': len(traces), 'sentences': sum(len(t['units']) for t in traces),
                      'native_context_tokens': sum(j['native_context_tokens'] for j in jobs), 'identity': spec['identity']}, indent=2))
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    prepare(parser.parse_args().config)
