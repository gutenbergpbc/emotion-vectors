"""Read-only tokenwise linear-probe capture on a causal language model.

Hooks observe block outputs before the final model norm. No interventions are
installed. Model/tokenizer ownership stays with the existing backend.
"""
from contextlib import ExitStack, contextmanager


@contextmanager
def capture_projections(layers, probes, vectors, sink):
    import torch

    with ExitStack() as stack:
        for probe in probes:
            emotion = probe['emotion']
            vector = torch.as_tensor(vectors[emotion], dtype=torch.float32)
            if vector.ndim != 1 or not torch.isfinite(vector).all() or vector.norm() == 0:
                raise ValueError(f'Invalid probe vector: {emotion}')

            def hook(module, inputs, output, emotion=emotion, vector=vector):
                h = output[0] if isinstance(output, tuple) else output
                if h.ndim != 3 or h.shape[0] != 1 or h.shape[-1] != vector.numel():
                    raise ValueError('Token replay requires batch one and a matching probe width')
                v = vector.to(h.device)
                projection = h[0].float() @ v
                cosine = projection / (h[0].float().norm(dim=-1) * v.norm()).clamp_min(1e-12)
                sink[emotion] = dict(projection=projection.cpu().numpy(),
                                     cosine=cosine.cpu().numpy())

            handle = layers[probe['layer_zero_based']].register_forward_hook(hook)
            stack.callback(handle.remove)
        yield


def score_rendered(model, tokenizer, text, probes, vectors, *, truncate_tokens=None,
                   response_char_start=None, chunk_size=8):
    import numpy as np
    import torch

    encoded = tokenizer(text, return_tensors='pt', return_offsets_mapping=True,
                        add_special_tokens=False, truncation=False)
    offsets = encoded.pop('offset_mapping')[0].numpy()
    if truncate_tokens is not None:
        encoded = {k: v[:, :truncate_tokens] for k, v in encoded.items()}
        offsets = offsets[:truncate_tokens]
    if encoded['input_ids'].shape[1] > model.config.max_position_embeddings:
        raise ValueError('Replay context exceeds the model limit; truncation is forbidden')
    token_ids = encoded['input_ids'][0].numpy()
    response_indices = np.flatnonzero(offsets[:, 1] > response_char_start)
    prompt_count = int(response_indices[0]) if len(response_indices) else len(token_ids)
    bounds = [(0, prompt_count)] + [(lo, min(lo + chunk_size, len(token_ids)))
                                   for lo in range(prompt_count, len(token_ids), chunk_size)]
    score_parts, cosine_parts = [], []
    cache = None
    for lo, hi in bounds:
        if lo == hi:
            continue
        sink = {}
        # Fixed prompt prefill, then fixed-size causal chunks. Cached activations
        # are never recomputed using a longer suffix or a future matrix shape.
        with capture_projections(model.model.layers, probes, vectors, sink), torch.inference_mode():
            result = model.model(input_ids=encoded['input_ids'][:, lo:hi].to(model.device),
                                 attention_mask=encoded['attention_mask'][:, :hi].to(model.device),
                                 past_key_values=cache, use_cache=True, return_dict=True)
            cache = result.past_key_values
        if set(sink) != {p['emotion'] for p in probes}:
            raise RuntimeError('A requested probe hook did not run')
        score_parts.append(np.stack([sink[p['emotion']]['projection'] for p in probes], axis=-1))
        cosine_parts.append(np.stack([sink[p['emotion']]['cosine'] for p in probes], axis=-1))
    scores = np.concatenate(score_parts)
    cosines = np.concatenate(cosine_parts)
    if not np.isfinite(scores).all() or not np.isfinite(cosines).all():
        raise ValueError('Nonfinite replay scores')
    return dict(scores=scores, cosines=cosines, token_ids=token_ids, offsets=offsets)
