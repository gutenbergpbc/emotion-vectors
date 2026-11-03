"""GPT-OSS 120B steering backend using the pinned MXFP4/H200 runtime."""
from contextlib import contextmanager


@contextmanager
def intervention(layer, vector, strength):
    import torch

    directions = {}

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[-1] != vector.numel():
            raise ValueError(f"Probe width {vector.numel()} != hidden width {hidden.shape[-1]}")
        if hidden.device not in directions:
            direction = vector.to(device=hidden.device, dtype=torch.float32)
            directions[hidden.device] = strength * direction / direction.norm()
        delta = directions[hidden.device] * hidden.float().norm(dim=-1, keepdim=True)
        steered = (hidden.float() + delta).to(hidden.dtype)
        return (steered,) + output[1:] if isinstance(output, tuple) else steered

    handle = layer.register_forward_hook(hook) if strength else None
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()


def make_backend(spec, root):
    import importlib.metadata
    import json
    import torch
    import kernels
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config, set_seed

    cfg = spec['model']
    runtime = cfg['runtime']
    for package, expected in runtime['packages'].items():
        actual = importlib.metadata.version(package).split('+')[0]
        if actual != expected:
            raise RuntimeError(f"{package} version {actual}, expected {expected}")

    original_get_kernel = kernels.get_kernel
    def pinned_get_kernel(repo_id, revision=None, **kwargs):
        if repo_id != runtime['kernel_repository']:
            raise ValueError(f"Unexpected Triton kernel repo: {repo_id}")
        return original_get_kernel(repo_id, revision=runtime['kernel_revision'], **kwargs)
    kernels.get_kernel = pinned_get_kernel
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg['name'], revision=cfg['revision'], torch_dtype=torch.bfloat16,
            device_map={'': 'cuda'}, attn_implementation=runtime['attention'],
            quantization_config=Mxfp4Config(dequantize=False),
        ).eval()
    finally:
        kernels.get_kernel = original_get_kernel
    tokenizer = AutoTokenizer.from_pretrained(cfg['name'], revision=cfg['revision'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if len(model.model.layers) != 36 or model.config.hidden_size != 2880:
        raise RuntimeError('Loaded checkpoint architecture differs from frozen GPT-OSS spec')
    if not model.is_quantized or model.config.quantization_config.dequantize:
        raise RuntimeError('GPT-OSS checkpoint did not load in original MXFP4 format')

    class Backend:
        hidden_size = 2880
        num_layers = 36

        def replay_chat_scores(self, messages, completion, probes, vectors, *, truncate_tokens=None):
            """Read-only replay of one saved native chat completion with its actual history."""
            from .probe_replay import score_rendered
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **cfg.get('template_kwargs', {}))
            if cfg.get('answer_prefix'):
                raise ValueError('Native chat replay cannot use a forced answer prefix')
            result = score_rendered(model, tokenizer, rendered + completion, probes, vectors,
                                    truncate_tokens=truncate_tokens, response_char_start=len(rendered))
            result['response_char_start'] = len(rendered)
            return result

        def replay_scores(self, prompt, response, probes, vectors, *, truncate_tokens=None):
            """Teacher-force a retained decoded answer with read-only probe hooks.

            Reconstructs the generation template, including a forced-final prefix
            when configured. Retained answers may have stripped whitespace, so
            this is a decoded-text replay, not recovery of original token IDs.
            """
            from .probe_replay import score_rendered
            rendered = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}], tokenize=False,
                add_generation_prompt=True, **cfg.get('template_kwargs', {}))
            rendered += cfg.get('answer_prefix', '')
            result = score_rendered(model, tokenizer, rendered + response, probes, vectors,
                                    truncate_tokens=truncate_tokens, response_char_start=len(rendered))
            result['response_char_start'] = len(rendered)
            return result

        def chat_completion(self, messages, probe, vector, *, strength, seed,
                            max_new_tokens, temperature, top_p, context_limit):
            """Native multi-turn chat for TerminalWrench, with no forced final prefix.

            The caller serializes calls: RNG state and layer hooks are process
            global. Return exact Harmony text as well as its final/analysis split.
            """
            from .terminal_wrench_server import parse_harmony
            if cfg.get('answer_prefix'):
                raise ValueError('Native terminal chat cannot use a forced-final answer prefix')
            prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, **cfg.get('template_kwargs', {}))
            tokens = tokenizer(prompt, return_tensors='pt', truncation=False).to('cuda')
            input_tokens = tokens['input_ids'].shape[1]
            limit = min(context_limit, model.config.max_position_embeddings)
            if input_tokens + max_new_tokens > limit:
                raise ValueError(f'context_length_exceeded: {input_tokens} input + '
                                 f'{max_new_tokens} output exceeds {limit}; no truncation applied')
            set_seed(seed)
            layer = model.model.layers[probe['layer_zero_based']] if probe else None
            context = intervention(layer, torch.as_tensor(vector, dtype=torch.float32), strength) if probe else _no_hook()
            with context, torch.inference_mode():
                output = model.generate(**tokens, max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    **({'temperature': temperature, 'top_p': top_p} if temperature > 0 else {}),
                    pad_token_id=tokenizer.pad_token_id, use_cache=True)
            generated = output[0, input_tokens:]
            eos = model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            ended = bool(len(generated) and generated[-1].item() in eos_ids)
            raw = tokenizer.decode(generated, skip_special_tokens=False)
            parsed = parse_harmony(raw, complete=ended)
            return dict(**parsed, raw_completion=raw, input_tokens=input_tokens,
                        generated_tokens=len(generated), finish_reason='stop' if ended else 'length')

        def sample_batch(self, tasks, probe, vector):
            import time
            set_seed(tasks[0]['seed'])
            tokenizer.padding_side = 'left'
            prompts = [tokenizer.apply_chat_template(
                task.get('messages') or [{'role': 'user', 'content': task['prompt']}],
                tokenize=False, add_generation_prompt=True,
                **cfg.get('template_kwargs', {}),
            ) for task in tasks]
            if cfg.get('answer_prefix'):
                prompts = [prompt + cfg['answer_prefix'] for prompt in prompts]
            tokens = tokenizer(prompts, return_tensors='pt', padding=True,
                               truncation=not spec.get('sales')).to('cuda')
            if spec.get('sales') and tokens['input_ids'].shape[1] + tasks[0]['max_new_tokens'] > model.config.max_position_embeddings:
                raise ValueError('Sales conversation exceeds model context; refusing silent truncation')
            layer = model.model.layers[probe['layer_zero_based']] if probe else None
            hook_context = intervention(layer, torch.as_tensor(vector, dtype=torch.float32),
                                        tasks[0]['strength']) if probe else _no_hook()
            if spec.get('sales'):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with hook_context, torch.inference_mode():
                outputs = model.generate(
                    **tokens, max_new_tokens=tasks[0]['max_new_tokens'], do_sample=True,
                    temperature=spec['generation']['temperature'],
                    top_p=spec['generation']['top_p'], pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            if spec.get('sales'):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            prefix = tokens['input_ids'].shape[1]
            eos = model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            results = []
            for output in outputs:
                generated = output[prefix:]
                stop = next((i for i, token in enumerate(generated.tolist()) if token in eos_ids), None)
                if stop is not None:
                    generated = generated[:stop + 1]
                results.append(dict(
                    response=tokenizer.decode(generated, skip_special_tokens=True).strip(),
                    generated_tokens=len(generated),
                    finish_reason='eos' if stop is not None else 'length',
                    input_tokens=prefix, sampling_seed=tasks[0]['seed'],
                    sampling_mode='shared_batch_seed_sample_ordered',
                ))
                if spec.get('sales'):
                    from .sales import parse_completion
                    raw = tokenizer.decode(generated, skip_special_tokens=False)
                    if stop is not None:
                        results[-1].update(parse_completion(raw, strict=False))
                    else:
                        results[-1]['raw_completion'] = raw
                    results[-1].update(batch_size=len(tasks), batch_elapsed_seconds=elapsed,
                        gpu_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                        gpu_peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
                if not results[-1]['response'] and results[-1]['finish_reason'] != 'invalid_channel':
                    raise RuntimeError('GPT-OSS generated an empty final-channel response')
            return results

        def release_memory(self):
            torch.cuda.empty_cache()

    return Backend()


@contextmanager
def _no_hook():
    yield
