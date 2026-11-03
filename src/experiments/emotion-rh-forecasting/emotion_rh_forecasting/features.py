"""Strict source-sentence pooling and optional local Hugging Face replay."""
from contextlib import ExitStack
import inspect

import numpy as np

from shared.io import atomic, digest, read, require, sha
from .analysis import EMOTIONS


def pool_completed(scores, offsets, start, end, position_offset=0):
    """Never let a token crossing the source sentence boundary become a feature."""
    scores, offsets = np.asarray(scores, float), np.asarray(offsets)
    require(offsets.shape == (len(scores), 2) and scores.shape == (len(offsets), 12), 'Invalid token score shape')
    require(np.isfinite(scores).all() and np.issubdtype(offsets.dtype, np.integer), 'Invalid token scores/offsets')
    require(np.all(offsets[:, 1] >= offsets[:, 0]), 'Reversed token offsets')
    complete = offsets[:, 1] <= end
    weights = np.maximum(0, np.minimum(offsets[complete, 1], end)-np.maximum(offsets[complete, 0], start)).astype(float)
    if weights.sum() == 0:
        return None
    feature = np.average(scores[complete], axis=0, weights=weights)
    positions = np.arange(len(scores))[complete] + position_offset
    return dict(projections=dict(zip(EMOTIONS, feature.tolist())),
                mean_token_position=float(np.average(positions, weights=weights)),
                completed_feature_tokens=int((weights > 0).sum()),
                last_feature_token_char_end=int(offsets[complete][weights > 0, 1].max()))


class LocalHF:
    """Batch-one prefix forwards, configurable block path; no generation.

    New GPU/runtime combinations need their own validation. Existing GPT-OSS
    reproduction uses its verified archived causal replay instead.
    """
    def __init__(self, bundle, config):
        import torch
        import transformers
        self.torch, self.bundle, self.config = torch, bundle, config
        cfg = bundle['model']
        cls = getattr(transformers, config['model_class'])
        self.model = cls.from_pretrained(cfg['name'], revision=cfg['revision'],
            torch_dtype=getattr(torch, config['dtype']), device_map=config['device'],
            attn_implementation=config['attention'], trust_remote_code=False).eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            cfg.get('tokenizer_model', cfg['name']), revision=cfg.get('tokenizer_revision', cfg['revision']), trust_remote_code=False)
        require(self.tokenizer.is_fast, 'Offset-mapping tokenizer required')
        self.layers = self.model
        for part in config['layers_path'].split('.'):
            self.layers = getattr(self.layers, part)
        require(max(p['layer'] for p in bundle['probes']) < len(self.layers), 'Probe layer exceeds checkpoint layers')

    def prepare(self, job):
        prefix = self.tokenizer.apply_chat_template(job['messages'], tokenize=False, add_generation_prompt=True,
                                                    **self.config['template_kwargs']) + self.config['answer_prefix']
        text = prefix + job['completion']
        tokens = self.tokenizer(text, return_tensors='pt', return_offsets_mapping=True, add_special_tokens=False, truncation=False)
        offsets = tokens.pop('offset_mapping')[0].numpy()
        output_ix = np.flatnonzero(offsets[:, 1] > len(prefix))
        return tokens, offsets, output_ix, len(prefix)

    def feature(self, prepared, unit, position_offset):
        tokens, offsets, output_ix, start = prepared
        end = start + unit['char_end']
        used = np.flatnonzero((offsets[:, 1] <= end) & (offsets[:, 1] > start+unit['char_start']) & (offsets[:, 1] > offsets[:, 0]))
        if not len(used):
            return None
        count = int(used[-1])+1
        require(np.all(offsets[:count, 1] <= end), 'Tokenizer prefix includes text beyond the source boundary')
        model_config = getattr(self.model.config, 'text_config', self.model.config)
        limit = min(self.config['max_tokens'], getattr(model_config, 'max_position_embeddings', self.config['max_tokens']))
        require(count <= limit, f'Sentence prefix has {count} tokens, exceeding {limit}; truncation is forbidden')
        captured = {}
        with ExitStack() as stack:
            for p in self.bundle['probes']:
                def hook(module, args, output, p=p):
                    h = output[0] if isinstance(output, tuple) else output
                    require(h.shape[0] == 1 and h.shape[-1] == self.bundle['width'], 'Probe/checkpoint residual width mismatch')
                    vector = self.torch.as_tensor(p['vector'], dtype=self.torch.float32, device=h.device)
                    captured[p['emotion']] = (h[0].float() @ vector).detach().cpu().numpy()
                handle = self.layers[p['layer']].register_forward_hook(hook)
                stack.callback(handle.remove)
            args = {k: v[:, :count].to(self.model.device) for k, v in tokens.items()}
            if 'logits_to_keep' in inspect.signature(self.model.forward).parameters:
                args['logits_to_keep'] = 1
            with self.torch.inference_mode():
                self.model(**args, use_cache=False, return_dict=True)
        require(set(captured) == set(EMOTIONS), 'Missing probe hooks')
        ix = output_ix[output_ix < count]
        scores = np.stack([captured[e][ix] for e in EMOTIONS], axis=-1)
        feature = pool_completed(scores, offsets[ix]-start, unit['char_start'], unit['char_end'], position_offset)
        if feature is not None:
            feature.update(prefix_token_count=count,
                           prefix_token_ids_sha256=digest(tokens['input_ids'][0, :count].tolist()),
                           chat_template_sha256=digest(self.tokenizer.get_chat_template()))
        return feature


def extract(root, spec, bundle, traces, jobs, progress):
    rows, drops, cache = [], [], {}
    local = None
    total = sum(len(t['units']) for t in traces)
    for trace in traces:
        cumulative, positions = 0, {}
        for turn_id in trace['turn_ids']:
            job = jobs[turn_id]
            if turn_id not in cache:
                if spec['scoring']['backend'] == 'cached':
                    require(sha(job['cache']) == job['sha256'], 'Replay changed since preparation')
                    with np.load(job['cache'], allow_pickle=False) as z:
                        start = int(z['response_char_start'])
                        ix = np.flatnonzero(z['offsets'][:, 1] > start)
                        order = [job['emotions'].index(e) for e in EMOTIONS]
                        cache[turn_id] = (z['scores'][ix][:, order].astype(float), z['offsets'][ix]-start)
                else:
                    if local is None:
                        local = LocalHF(bundle, spec['scoring'])
                    cache[turn_id] = local.prepare(job)
            positions[turn_id] = cumulative
            cumulative += len(cache[turn_id][0]) if local is None else len(cache[turn_id][2])
        for index, unit in enumerate(trace['units']):
            row = dict(trace_id=trace['id'], group_id=trace['group_id'], task=trace['task'],
                       sentence_index=index, **{k: unit[k] for k in ('unit_id', 'turn_id', 'char_start', 'char_end', 'text', 'hacking', 'needs_review')})
            if 'fold' in trace:
                row['fold'] = trace['fold']
            key = digest([trace['id'], unit['unit_id']])
            checkpoint = root / 'features' / (key + '.json')
            binding = digest({'identity': spec['identity'], 'source': row})
            if checkpoint.exists():
                saved = read(checkpoint)
                require(saved['binding'] == binding and saved['checksum'] == digest(saved['feature']), 'Stale/corrupt feature checkpoint')
                feature = saved['feature']
            else:
                values = cache[unit['turn_id']]
                feature = (pool_completed(*values, unit['char_start'], unit['char_end'], positions[unit['turn_id']])
                           if local is None else local.feature(values, unit, positions[unit['turn_id']]))
                atomic(checkpoint, dict(binding=binding, feature=feature, checksum=digest(feature)))
            if feature is None:
                row['projections'] = None
                drops.append(dict(trace_id=trace['id'], unit_id=unit['unit_id'], reason='No completed token overlaps source sentence'))
            else:
                row.update(feature)
            rows.append(row)
            if len(rows) % 25 == 0 or len(rows) == total:
                progress.stage('score', len(rows), total, 'sentences', f'{trace["id"]} · sentence {index+1}')
    return rows, drops


def verify_cached_features(root, spec, rows):
    """Re-pool archived tokens independently of feature checkpoints and labels."""
    require(digest(read(root/'inputs/probes.json')) == spec['probe_digest'], 'Frozen probe bundle changed')
    traces, jobs = read(root/'inputs/traces.json'), read(root/'inputs/jobs.json')
    require(len(rows) == sum(len(t['units']) for t in traces), 'Missing original sentence rows')
    by_key = {(r['trace_id'], r['unit_id']): r for r in rows}
    require(len(by_key) == len(rows), 'Duplicate sentence feature rows')
    arrays = {}
    for trace in traces:
        cumulative, positions = 0, {}
        for key in trace['turn_ids']:
            job = jobs[key]
            if spec['scoring']['backend'] == 'cached':
                if key not in arrays:
                    require(sha(job['cache']) == job['sha256'], 'Frozen replay shard changed')
                    with np.load(job['cache'], allow_pickle=False) as z:
                        start = int(z['response_char_start'])
                        ix = np.flatnonzero(z['offsets'][:, 1] > start)
                        arrays[key] = (z['scores'][ix][:, [job['emotions'].index(e) for e in EMOTIONS]].astype(float), z['offsets'][ix]-start)
                positions[key] = cumulative
                cumulative += len(arrays[key][0])
        for i, u in enumerate(trace['units']):
            row = by_key[trace['id'], u['unit_id']]
            require(all(row[k] == u[k] for k in ('turn_id', 'char_start', 'char_end', 'text', 'hacking', 'needs_review'))
                    and row['sentence_index'] == i and row['task'] == trace['task'] and row['group_id'] == trace['group_id'],
                    'Feature row differs from original trace/annotation')
            if spec['scoring']['backend'] == 'cached':
                feature = pool_completed(*arrays[u['turn_id']], u['char_start'], u['char_end'], positions[u['turn_id']])
                require((feature is None and row['projections'] is None) or
                        (feature is not None and all(row[k] == v for k, v in feature.items())),
                        'Source features differ from frozen token replay')
            if row['projections'] is not None:
                require(row['last_feature_token_char_end'] <= u['char_end'], 'Future token feature')
                if spec['scoring']['backend'] == 'local-hf':
                    require(row['prefix_token_count'] > 0 and len(row['prefix_token_ids_sha256']) == 64,
                            'Missing local causal prefix evidence')
    return len(rows)
