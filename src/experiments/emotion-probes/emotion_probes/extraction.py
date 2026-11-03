from __future__ import annotations

import importlib.metadata
from pathlib import Path
import time

import numpy as np

from shared.io import atomic, digest, read, require, save_array, sha
from .data import EMOTIONS, validate_dataset
from .rendering import tokenize_case


def cases_for(root, spec):
    training, evaluation, _, _ = validate_dataset(root / 'inputs', spec['dataset_contract'] == 'unified-twelve-v1', spec.get('training_family', 'diversity'))
    return {'training': training, 'eval': evaluation}


def tokenize(root, spec, progress):
    cases = cases_for(root, spec)
    if spec['backend'] == 'synthetic':
        progress.stage('tokenize', 0, 0, 'cases', 'SYNTHETIC DEMO: no tokenizer or model invoked')
        return
    from transformers import AutoConfig, AutoTokenizer
    model_config = AutoConfig.from_pretrained(spec['model'], revision=spec['revision'], trust_remote_code=False)
    text_config = getattr(model_config, 'text_config', model_config)
    require((text_config.num_hidden_layers, text_config.hidden_size) == (spec['layers'], spec['width']), 'Model architecture differs from config')
    if spec['renderer'] == 'gemma-native-preserved-streams-v2':
        require(model_config.model_type == 'gemma3' and text_config.model_type == 'gemma3_text', 'Unexpected Gemma architecture')
        require(spec['quantization'] is None and not getattr(model_config, 'quantization_config', None), 'Expected unquantized Gemma checkpoint')
    else:
        require(model_config.model_type == 'glm_moe_dsa', 'Unexpected GLM architecture')
        quant = model_config.quantization_config
        if spec['quantization'] == 'modelopt_fp4':
            require(quant['quant_method'] == 'modelopt' and quant['quant_algo'] == 'NVFP4', 'Expected native NVFP4 checkpoint')
        else:
            require(quant['quant_method'] == spec['quantization'], 'Checkpoint precision mismatch')
    require(spec['max_model_len'] <= text_config.max_position_embeddings, 'Requested context exceeds checkpoint limit')
    tokenizer = AutoTokenizer.from_pretrained(spec['tokenizer_model'], revision=spec['tokenizer_revision'], trust_remote_code=False)
    require(tokenizer.is_fast, 'Offset-mapping tokenizer required')
    template = tokenizer.get_chat_template()
    atomic(root / 'tokenizer.json', {'model': spec['tokenizer_model'], 'revision': spec['tokenizer_revision'], 'chat_template_sha256': digest(template),
                                   'model_type': model_config.model_type,
                                   'versions': versions(['transformers', 'tokenizers', 'huggingface-hub'])})
    (root / 'chat_template.jinja').write_text(template)
    total = sum(map(len, cases.values()))
    completed = 0
    lengths = []
    for partition, rows in cases.items():
        folder = root / 'tokens' / partition
        folder.mkdir(parents=True, exist_ok=True)
        records = []
        for i, case in enumerate(rows):
            path = folder / f'{i:05d}.json'
            if path.exists():
                record = read(path)
                require(record['identity'] == spec['identity'] and record['case_sha256'] == digest(case), 'Stale token checkpoint')
                require(record['token_ids_sha256'] == digest(record['input_ids']), 'Corrupt token checkpoint')
            else:
                record = {**tokenize_case(case, tokenizer, spec['max_model_len'], spec['reasoning_effort'], spec['renderer']), 'identity': spec['identity']}
                atomic(path, record)
            records.append({'file': path.name, 'sha256': sha(path), 'id': case['id'],
                            'prefix_token_count': record['prefix_token_count']})
            lengths.append(record['prefix_token_count'])
            completed += 1
            if completed % 20 == 0 or completed == total:
                progress.stage('tokenize', completed, total, 'cases', f"{partition}: {case['id']}")
        atomic(folder / 'manifest.json', {'identity': spec['identity'], 'records': records})
    atomic(root / 'tokenization.json', {'cases': total, 'max_prefix_tokens': max(lengths),
                                       'total_prefix_tokens': sum(lengths), 'identity': spec['identity']})


def versions(names):
    return {name: importlib.metadata.version(name) for name in names}


def make_capture_fn(lo, hi, layers, width):
    """Lens supplies the summed full residual, replicated after TP all-reduce.

    Offsets are per request and layer. Prefix caching/preemption are disabled;
    each target token must be counted exactly once even with chunked prefill.
    Only TP rank zero saves values. Never return a tensor (which would steer).
    """
    def capture(ctx, hidden):
        import torch
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
        if get_tensor_model_parallel_rank() != 0:
            return None
        if hidden.ndim != 2 or hidden.shape[1] != width:
            raise ValueError(f'Expected full residual width {width}, got {tuple(hidden.shape)}')
        saved = ctx.saved
        if 'means' in saved:
            return None
        saved.setdefault('offsets', {})
        saved.setdefault('counts', {})
        saved.setdefault('sums', {})
        layer = str(ctx.layer_idx)
        offset = saved['offsets'].get(layer, 0)
        end = offset + hidden.shape[0]
        a, b = max(lo, offset) - offset, min(hi, end) - offset
        if b > a:
            value = hidden[a:b].float().sum(dim=0)
            saved['sums'][layer] = saved['sums'].get(layer, 0) + value
            saved['counts'][layer] = saved['counts'].get(layer, 0) + b - a
        saved['offsets'][layer] = end
        if ctx.layer_idx == layers[-1] and end >= hi:
            if not all(saved['counts'].get(str(i)) == hi - lo for i in layers):
                raise ValueError(f'Missing/duplicated target-token capture: {saved["counts"]}')
            saved['means'] = (torch.stack([saved['sums'][str(i)] for i in layers]) / (hi - lo)).cpu().numpy()
            saved.pop('sums')
        return None
    return capture


class LensExtractor:
    def __init__(self, spec):
        from vllm import LLM
        self.spec = spec
        options = {'limit_mm_per_prompt': {'image': 0}} if spec['renderer'] == 'gemma-native-preserved-streams-v2' else {}
        self.llm = LLM(model=spec['model'], revision=spec['revision'], tokenizer=spec['tokenizer_model'], tokenizer_revision=spec['tokenizer_revision'],
                       dtype=spec['dtype'], quantization=spec['quantization'],
                       tensor_parallel_size=spec['tensor_parallel_size'], pipeline_parallel_size=1,
                       max_model_len=spec['max_model_len'] + 1,
                       max_num_batched_tokens=spec['max_num_batched_tokens'], max_num_seqs=1,
                       gpu_memory_utilization=spec['gpu_memory_utilization'],
                       enforce_eager=True, enable_prefix_caching=False, enable_chunked_prefill=True,
                       skip_tokenizer_init=True, trust_remote_code=False, seed=0, **options)

    def capture(self, record):
        from vllm import SamplingParams
        from vllm_lens import Hook
        layers = list(range(self.spec['layers']))
        lo, hi = record['target_token_span']
        hook = Hook(fn=make_capture_fn(lo, hi, layers, self.spec['width']), layer_indices=layers, pre=False)
        output, = self.llm.generate([{'prompt_token_ids': record['input_ids']}],
                                   SamplingParams(temperature=0, max_tokens=1, ignore_eos=True,
                                                  extra_args={'apply_hooks': [hook]}), use_tqdm=False)
        require(list(output.prompt_token_ids) == record['input_ids'], 'Engine changed input tokens')
        results = output.hook_results
        results = results.get('0', results.get(0, results))
        require('means' in results, 'Lens did not return complete post-block captures; inspect worker errors')
        result = np.asarray(results['means'], dtype=np.float32)
        require(result.shape == (self.spec['layers'], self.spec['width']) and np.isfinite(result).all(), 'Invalid residual capture')
        return result


def check_shard(folder, entry, spec, rows):
    require(entry['identity'] == spec['identity'], 'Stale activation checkpoint')
    require(entry['ids'] == [c['id'] for c in rows] and entry['case_hashes'] == [digest(c) for c in rows], 'Activation row identity mismatch')
    path = folder / entry['file']
    require(path.name == entry['file'] and sha(path) == entry['sha256'], 'Activation shard checksum/path mismatch')
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    require(array.shape == (len(rows), spec['layers'], spec['width']), 'Activation shape mismatch')
    require(array.dtype == np.float32 and np.isfinite(array).all(), 'Invalid activation values')
    return array


def feature_manifest(root, spec, partition, rows):
    folder = root / 'activations' / partition
    manifest = read(folder / 'manifest.json')
    require(manifest['identity'] == spec['identity'], 'Feature manifest identity mismatch')
    offset = 0
    for entry in manifest['files']:
        n = len(entry['ids'])
        check_shard(folder, entry, spec, rows[offset:offset+n])
        offset += n
    require(offset == len(rows) == manifest['rows'], 'Incomplete features')
    return manifest


def layer_features(root, partition, layer, manifest):
    return np.concatenate([np.load(root / 'activations' / partition / e['file'], mmap_mode='r', allow_pickle=False)[:, layer]
                           for e in manifest['files']])


def synthetic_feature(case, spec):
    rng = np.random.default_rng(int(digest(case['id'])[:16], 16))
    value = rng.normal(size=(spec['layers'], spec['width'])).astype(np.float32)
    if case['label'] in EMOTIONS:
        value[:, EMOTIONS.index(case['label'])] += np.linspace(1, 3, spec['layers'])
    return value


def extract(root, spec, progress, smoke_only=False):
    cases = cases_for(root, spec)
    total = sum(map(len, cases.values()))
    progress.stage('extract', 0, total, 'cases', 'Checking committed activation shards')
    pending, manifests = [], {}
    for partition, rows in cases.items():
        folder = root / 'activations' / partition
        folder.mkdir(parents=True, exist_ok=True)
        entries = []
        for start in range(0, len(rows), spec['shard_size']):
            batch = rows[start:start+spec['shard_size']]
            receipt = folder / f'{start:05d}.json'
            if receipt.exists():
                entry = read(receipt)
                check_shard(folder, entry, spec, batch)
                entries.append(entry)
            else:
                pending.append((partition, start, batch))
        manifests[partition] = entries
    completed = sum(len(e['ids']) for entries in manifests.values() for e in entries)
    progress.stage('extract', completed, total, 'cases', f'{completed} verified cached cases')
    extractor = None
    if (pending or smoke_only) and spec['backend'] != 'synthetic':
        progress.stage('extract', completed, total, 'cases', f"Loading pinned {spec['model']} {spec['quantization'] or spec['dtype']} checkpoint")
        start = time.monotonic()
        extractor = LensExtractor(spec)
        atomic(root / 'runtime.json', {'versions': versions(['torch', 'vllm', 'vllm-lens', 'transformers']),
                                      'identity': spec['identity'], 'load_seconds': time.monotonic() - start,
                                      'tensor_parallel_size': spec['tensor_parallel_size'],
                                      'quantization': spec['quantization']})
        # Repeat a training pair, longest prefix and reasoning example, then
        # independently compare Lens pooling with its raw residual capture API.
        smoke(root, spec, extractor, progress, cases)
    if smoke_only:
        return False
    for partition, start, batch in pending:
        folder = root / 'activations' / partition
        values, evidence = [], []
        for i, case in enumerate(batch, start):
            if extractor is None:
                require(spec['backend'] == 'synthetic', 'No model extractor available')
                value = synthetic_feature(case, spec)
                evidence.append({'id': case['id'], 'synthetic': True})
            else:
                token_path = root / 'tokens' / partition / f'{i:05d}.json'
                record = read(token_path)
                require(record['identity'] == spec['identity'] and record['case_sha256'] == digest(case), 'Stale token record')
                require(record['token_ids_sha256'] == digest(record['input_ids']), 'Corrupt token IDs')
                value = extractor.capture(record)
                evidence.append({k: v for k, v in record.items() if k != 'input_ids'})
            values.append(value)
        path = folder / f'{start:05d}.npy'
        save_array(path, np.stack(values))
        entry = {'file': path.name, 'sha256': sha(path), 'identity': spec['identity'],
                 'ids': [c['id'] for c in batch], 'case_hashes': [digest(c) for c in batch], 'evidence': evidence}
        atomic(path.with_suffix('.json'), entry)
        manifests[partition].append(entry)
        completed += len(batch)
        progress.stage('extract', completed, total, 'cases', f"{partition}: committed through {batch[-1]['id']}")
    for partition, rows in cases.items():
        atomic(root / 'activations' / partition / 'manifest.json', {
            'identity': spec['identity'], 'rows': len(rows),
            'files': sorted(manifests[partition], key=lambda e: e['file']),
        })
        feature_manifest(root, spec, partition, rows)
    return True


def smoke(root, spec, extractor, progress, cases):
    from vllm import SamplingParams
    training = cases['training']
    if spec.get('training_family') == 'story':
        selected = [('training', 0), next(('training', i) for i, c in enumerate(training)
                    if c['label'] == 'neutral' and c['group_id'] == training[0]['group_id'])]
    else:
        pair_id = training[0]['pair_id']
        selected = [('training', i) for i, c in enumerate(training) if c['pair_id'] == pair_id][:2]
    all_records = [(part, i, r) for part in cases for i, r in enumerate(read(root / 'tokens' / part / 'manifest.json')['records'])]
    part, i, _ = max(all_records, key=lambda x: x[2]['prefix_token_count'])
    selected.append((part, i))
    selected.append(next((part, i) for part, rows in cases.items() for i, c in enumerate(rows) if c.get('target', {}).get('stream') == 'reasoning'))
    results = []
    for part, i in dict.fromkeys(selected):
        record = read(root / 'tokens' / part / f'{i:05d}.json')
        start = time.monotonic()
        a = extractor.capture(record)
        first_seconds = time.monotonic() - start
        start = time.monotonic()
        b = extractor.capture(record)
        repeat_seconds = time.monotonic() - start
        relative = np.linalg.norm(a-b, axis=1) / np.maximum(np.linalg.norm(a, axis=1), 1e-12)
        repeatable = relative.max() <= 1e-5
        require(repeatable or spec['repeatability_policy'] == 'record_only',
                f"Repeated captures differ for {record['id']}")
        results.append({'id': record['id'], 'prefix_tokens': len(record['input_ids']),
                        'first_seconds': first_seconds, 'repeat_seconds': repeat_seconds,
                        'repeat_prefix_tokens_per_second': len(record['input_ids']) / repeat_seconds,
                        'repeat_max_relative_l2': float(relative.max()),
                        'repeatability_within_limit': bool(repeatable)})
    # A short real prefix validates target-only pooling against independently
    # returned raw full-residual activations, including first and last blocks.
    part, i, _ = min(all_records, key=lambda x: x[2]['prefix_token_count'])
    record = read(root / 'tokens' / part / f'{i:05d}.json')
    layers = sorted({0, spec['layers']//2, spec['layers']-1})
    output, = extractor.llm.generate([{'prompt_token_ids': record['input_ids']}], SamplingParams(
        temperature=0, max_tokens=1, ignore_eos=True, extra_args={'output_residual_stream': layers}), use_tqdm=False)
    raw = output.activations['residual_stream']
    raw = raw.float().cpu().numpy() if hasattr(raw, 'cpu') else np.asarray(raw)
    require(raw.ndim == 3 and raw.shape[0] == len(layers) and raw.shape[2] == spec['width'], 'Raw residual smoke shape mismatch')
    lo, hi = record['target_token_span']
    require(raw.shape[1] >= hi, 'Raw residual smoke missed prompt tokens')
    expected = raw[:, lo:hi].mean(axis=1)
    actual = extractor.capture(record)[layers]
    relative = np.linalg.norm(actual-expected, axis=1) / np.maximum(np.linalg.norm(expected, axis=1), 1e-12)
    require(relative.max() <= 1e-5, 'Target pooling differs from raw residual mean')
    atomic(root / 'smoke.json', {'passed': True, 'identity': spec['identity'],
                               'repeatability_policy': spec['repeatability_policy'],
                               'repeatability_limit': 1e-5,
                               'repeatability_limit_exceeded': any(not r['repeatability_within_limit'] for r in results),
                               'cases': results,
                               'raw_pooling_max_relative_l2': float(relative.max()), 'layers': layers})
