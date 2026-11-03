"""Gemma replay of frozen ATIF rendering, ending forwards at sentence boundaries."""
from collections import defaultdict
import gc
import hashlib
import importlib.metadata
import io
from pathlib import Path
import time

import numpy as np

from shared.io import atomic, digest, read, require, sha


def sentence_cuts(offsets, units):
    offsets = np.asarray(offsets)
    require(np.all(np.diff(offsets[:, 1]) >= 0), 'Nonmonotone tokenizer offsets')
    cuts = []
    for unit in units:
        cut = int(np.searchsorted(offsets[:, 1], unit['char_end'], side='right'))
        require(cut > 0 and np.all(offsets[:cut, 1] <= unit['char_end']), 'Future token at sentence boundary')
        cuts.append(cut)
    return sorted(set(cuts))


def chunk_ranges(count, cuts, chunk_size):
    """Every source boundary is a forward-call boundary; later IDs are not fed."""
    position = 0
    for boundary in sorted(set(cuts + [count])):
        if boundary > count:
            continue
        while position < boundary:
            end = min(position + chunk_size, boundary)
            yield position, end
            position = end


class GemmaReplay:
    def __init__(self, bundle, spec):
        import torch
        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
        self.torch, self.bundle, self.spec = torch, bundle, spec
        cfg = bundle['model']
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['name'], revision=cfg['revision'], use_fast=True)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            cfg['name'], revision=cfg['revision'], torch_dtype=torch.bfloat16,
            device_map={'': 'cuda'}, attn_implementation='sdpa').eval()
        self.groups, self.captured, self.handles = defaultdict(list), {}, []
        for j, probe in enumerate(bundle['probes']):
            self.groups[probe['layer']].append(j)
        self.vectors = {layer: torch.tensor(np.array([bundle['probes'][j]['vector'] for j in js]).T,
                       dtype=torch.float32, device='cuda') for layer, js in self.groups.items()}
        for layer, js in self.groups.items():
            def hook(module, args, output, layer=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                require(hidden.shape[-1] == bundle['width'], 'Wrong residual width')
                self.captured[layer] = (hidden[0].float() @ self.vectors[layer]).detach().cpu().numpy()
            self.handles.append(self.model.model.language_model.layers[layer].register_forward_hook(hook))
        self.runtime = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'tokenizers', 'numpy')}
        self.runtime['gpu'] = torch.cuda.get_device_name()

    def run(self, ids, cuts, chunk_size=None):
        torch = self.torch
        past, parts = None, []
        ranges = list(chunk_ranges(len(ids), cuts, chunk_size or self.spec['chunk_tokens']))
        with torch.inference_mode():
            for start, end in ranges:
                self.captured.clear()
                result = self.model(input_ids=torch.tensor([ids[start:end]], device='cuda'),
                                    past_key_values=past, use_cache=True, logits_to_keep=1, return_dict=True)
                past = result.past_key_values
                values = np.empty((end-start, 12), dtype=np.float32)
                for layer, js in self.groups.items():
                    values[:, js] = self.captured[layer]
                require(np.isfinite(values).all(), 'Nonfinite projections')
                parts.append(values)
        del result, past
        return np.concatenate(parts)

    def score(self, job, arrays, output, smoke=False):
        started = time.monotonic()
        ids, offsets, kind = arrays['token_ids'], arrays['offsets'], arrays['kind']
        encoded = self.tokenizer(job['rendered'], add_special_tokens=False, return_offsets_mapping=True)
        require(encoded['input_ids'] == ids.tolist(), 'Frozen token IDs differ from pinned Gemma tokenizer')
        require(encoded['offset_mapping'] == [tuple(x) for x in offsets.tolist()], 'Frozen tokenizer offsets differ')
        require(hashlib.sha256(job['rendered'].encode()).hexdigest() == job['rendered_sha256'], 'Rendered text changed')
        cuts = sentence_cuts(offsets, job['source_boundaries'])
        count = max(cuts)
        require(count <= self.spec['max_tokens'], 'Native context exceeded; truncation is forbidden')
        ids, offsets, kind = ids[:count], offsets[:count], kind[:count]
        scores = self.run(ids, cuts)
        checks = []
        # Re-run a genuine source prefix with its future completely absent.
        boundary = cuts[0]
        prefix = self.run(ids[:boundary], [boundary])
        error = float(np.max(np.abs(prefix - scores[:boundary])))
        rel = float(np.linalg.norm(prefix-scores[:boundary]) / max(np.linalg.norm(prefix), 1e-12))
        require(rel < 1e-6, f'Shorter-prefix replay differs: {rel}')
        checks.append({'check': 'future_removed_at_first_sentence', 'tokens': boundary,
                       'relative_l2': rel, 'max_abs_difference': error})
        if smoke:
            n = min(2304, count)
            direct = self.run(ids[:n], [n], chunk_size=n)
            cached = self.run(ids[:n], [n])
            rel = float(np.linalg.norm(direct-cached)/max(np.linalg.norm(direct), 1e-12))
            tolerance = float(self.spec.get('independent_prefix_tolerance', .02))
            require(rel < tolerance, f'Cached versus independent prefix forward differs: {rel}')
            checks.append({'check': 'independent_forward_vs_kv_chunks', 'tokens': n, 'relative_l2': rel,
                           'tolerance': tolerance})
        selected = np.flatnonzero((kind == 1) | (kind == 2))
        require(len(selected) > 0, 'No assistant replay tokens')
        buf = io.BytesIO()
        np.savez_compressed(buf, scores=scores[selected], offsets=offsets[selected], token_ids=ids[selected],
                            native_token_index=selected, response_char_start=np.array(0),
                            kind=kind[selected])
        blob = buf.getvalue()
        path = output / (job['id'] + '.npz')
        path.write_bytes(blob)
        receipt = {'schema_version': 1, 'id': job['id'], 'replay_identity': self.spec['identity'],
                   'probe_identity': self.spec['probe_identity'], 'input_sha256': job['input_sha256'],
                   'source_sha256': job['source_sha256'], 'rendered_sha256': job['rendered_sha256'],
                   'sha256': sha(path), 'prefix_tokens': count, 'assistant_replay_tokens': len(selected),
                   'sentence_boundary_count': len(cuts), 'sentence_cuts_sha256': digest(cuts),
                   'tokenizer_match': True, 'causal_checks': checks, 'runtime': self.runtime,
                   'seconds': time.monotonic()-started}
        atomic(path.with_suffix('.receipt.json'), receipt)
        del scores, prefix
        gc.collect()
        return blob, receipt
