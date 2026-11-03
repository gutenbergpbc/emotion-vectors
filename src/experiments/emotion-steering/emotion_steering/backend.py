"""One request at a time: stable per-sample RNG and exception-safe interventions."""

import importlib.metadata
import json
import re
from contextlib import contextmanager, nullcontext


class BackendValidationError(ValueError):
    def __init__(self, message, diagnostics):
        summary = [
            {k: v for k, v in row.items() if k != "raw_completion"}
            for row in diagnostics
        ]
        super().__init__(f"{message}: {json.dumps(summary, sort_keys=True)}")
        self.diagnostics = diagnostics


@contextmanager
def intervention(layer, vector, strength):
    if not strength:
        yield
        return
    import torch

    if vector.ndim != 1 or not torch.isfinite(vector).all() or vector.norm() <= 1e-8:
        raise ValueError("Invalid intervention direction")

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[-1] != vector.numel():
            raise ValueError("Probe/residual width mismatch")
        v = vector.to(device=h.device, dtype=torch.float32)
        changed = (
            h.float() + (strength * v / v.norm()) * h.float().norm(dim=-1, keepdim=True)
        ).to(h.dtype)
        return (changed,) + output[1:] if isinstance(output, tuple) else changed

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def split_completion(raw, text, model, finished):
    if model["family"] == "gpt_oss" and not model.get("answer_prefix"):
        marker = "<|channel|>final<|message|>"
        if marker in raw:
            reasoning, final = raw.split(marker, 1)
        elif raw.startswith("final<|message|>"):
            reasoning, final = "", raw[len("final<|message|>") :]
        else:
            return "", raw, "invalid_channel" if finished else "length"
        final = re.split(
            r"<\|(?:im_end|fim_suffix|ghissue|return|end|start)\|>", final
        )[0].strip()
        return (
            final,
            reasoning,
            ("eos" if final else "invalid_channel") if finished else "length",
        )
    if model["family"] == "glm":
        # The native GLM generation prompt already opens <think>. Decode with
        # special tokens retained so a tokenizer cannot erase the boundary.
        if "</think>" not in raw:
            return "", raw, "invalid_channel" if finished else "length"
        reasoning, final = raw.split("</think>", 1)
        final = re.split(r"<\|(?:endoftext|user|assistant|observation)\|>", final)[0]
        final = final.strip()
        return (
            final,
            reasoning.removeprefix("<think>").strip(),
            ("eos" if final else "invalid_channel") if finished else "length",
        )
    # Preserve reasoning separately; only the final portion is actionable in environments.
    if "</think>" in text:
        reasoning, text = text.split("</think>", 1)
        return text.strip(), reasoning, "eos" if finished else "length"
    return text.strip(), "", "eos" if finished else "length"


class HFBackend:
    def __init__(self, spec):
        import torch
        import transformers as tr

        self.spec, self.cfg = spec, spec["model"]
        cfg = self.cfg
        for package, version in cfg["packages"].items():
            actual = importlib.metadata.version(package).split("+")[0]
            if actual != version:
                raise ValueError(f"Expected {package} {version}, found {actual}")
        kwargs = {
            "revision": cfg["revision"],
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "attn_implementation": cfg["attention"],
            "trust_remote_code": False,
        }
        if cfg["family"] == "gpt_oss":
            import kernels

            original = kernels.get_kernel

            def pinned(repo_id, revision=None, **kw):
                if repo_id != cfg["kernel_repository"]:
                    raise ValueError("Unexpected kernel repository")
                return original(repo_id, revision=cfg["kernel_revision"], **kw)

            kernels.get_kernel = pinned
            kwargs["quantization_config"] = tr.Mxfp4Config(dequantize=False)
        try:
            self.model = (
                getattr(tr, cfg["class"]).from_pretrained(cfg["name"], **kwargs).eval()
            )
        finally:
            if cfg["family"] == "gpt_oss":
                kernels.get_kernel = original
        self.tokenizer = tr.AutoTokenizer.from_pretrained(
            cfg["name"], revision=cfg["revision"]
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        layers = self.model
        for part in cfg["layers_path"].split("."):
            layers = getattr(layers, part)
        self.layers = layers
        text_config = (
            self.model.config.get_text_config()
            if hasattr(self.model.config, "get_text_config")
            else self.model.config
        )
        if len(layers) != cfg["layers"] or text_config.hidden_size != cfg["width"]:
            raise ValueError("Loaded model architecture differs from preset")
        if cfg["family"] == "gpt_oss" and (
            not self.model.is_quantized
            or self.model.config.quantization_config.dequantize
        ):
            raise ValueError("GPT-OSS must retain original MXFP4 weights")
        self.max_context = text_config.max_position_embeddings

    def complete(self, messages, probe, strength, seed, max_tokens):
        import torch
        from transformers import set_seed

        cfg = self.cfg
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **cfg.get("template_kwargs", {}),
        ) + cfg.get("answer_prefix", "")
        tokens = self.tokenizer(
            prompt, return_tensors="pt", truncation=False, add_special_tokens=False
        )
        n = tokens["input_ids"].shape[1]
        if n + max_tokens > self.max_context:
            raise ValueError(
                f"Context overflow: {n}+{max_tokens}>{self.max_context}; no truncation"
            )
        tokens = tokens.to(self.model.device)
        set_seed(seed)
        context = (
            intervention(
                self.layers[probe["layer_zero_based"]],
                torch.tensor(probe["vector"]),
                strength,
            )
            if probe
            else nullcontext()
        )
        options = {
            "do_sample": self.spec["temperature"] > 0,
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
        }
        if self.spec["temperature"] > 0:
            options.update(
                temperature=self.spec["temperature"], top_p=self.spec["top_p"]
            )
        with context, torch.inference_mode():
            output = self.model.generate(**tokens, **options)[0, n:]
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        finished = bool(len(output) and output[-1].item() in eos)
        raw = self.tokenizer.decode(output, skip_special_tokens=False)
        text = self.tokenizer.decode(output, skip_special_tokens=True)
        final, reasoning, finish = split_completion(raw, text, cfg, finished)
        return {
            "response": final,
            "reasoning_text": reasoning,
            "raw_completion": raw,
            "finish_reason": finish,
            "input_tokens": n,
            "generated_tokens": len(output),
            "sampling_seed": seed,
        }


def lens_hook(vector, strength, request_tag=None):
    cached = {}

    def steer(ctx, hidden):
        import torch

        if hidden.ndim != 2 or hidden.shape[-1] != len(vector):
            raise ValueError("Lens must expose the full post-block residual")
        device = str(hidden.device)
        if device not in cached:
            v = torch.tensor(vector, device=hidden.device, dtype=torch.float32)
            if not torch.isfinite(v).all() or v.norm() <= 1e-8:
                raise ValueError("Invalid intervention direction")
            cached[device] = strength * v / v.norm()
        direction = cached[device]
        # All TP ranks receive the full replicated residual. Apply identically on every rank.
        full = hidden.float()
        changed = (full + direction * full.norm(dim=-1, keepdim=True)).to(hidden.dtype)
        # A small per-request receipt proves the hook actually ran on prefill and
        # decode; no large activations are retained or copied back to the client.
        ctx.saved["layer"] = ctx.layer_idx
        ctx.saved["strength"] = strength
        ctx.saved["request_tag"] = request_tag
        ctx.saved["tokens_steered"] = (
            ctx.saved.get("tokens_steered", 0) + hidden.shape[0]
        )
        if "first_delta_relative_norm" not in ctx.saved:
            before = full[0].norm().clamp_min(1e-12)
            ctx.saved["first_delta_relative_norm"] = float(
                (changed[0].float() - full[0]).norm() / before
            )
        return changed

    return steer


class LensBackend:
    def __init__(self, spec):
        from transformers import AutoConfig, AutoTokenizer, GenerationConfig
        from vllm import LLM

        self.spec, self.cfg = spec, spec["model"]
        c = self.cfg
        for package, version in c.get("packages", {}).items():
            actual = importlib.metadata.version(package).split("+")[0]
            if actual != version:
                raise ValueError(f"Expected {package} {version}, found {actual}")
        config = AutoConfig.from_pretrained(c["name"], revision=c["revision"])
        text_config = getattr(config, "text_config", config)
        if (text_config.num_hidden_layers, text_config.hidden_size) != (
            c["layers"],
            c["width"],
        ):
            raise ValueError("Loaded model architecture differs from preset")
        self.tokenizer = AutoTokenizer.from_pretrained(
            c.get("tokenizer_model", c["name"]),
            revision=c.get("tokenizer_revision", c["revision"]),
        )
        try:
            generation = GenerationConfig.from_pretrained(
                c["name"], revision=c["revision"]
            )
        except OSError:
            generation = GenerationConfig.from_model_config(config)
        eos = generation.eos_token_id or text_config.eos_token_id
        self.stop_token_ids = eos if isinstance(eos, list) else [eos]
        if self.tokenizer.eos_token_id not in self.stop_token_ids:
            self.stop_token_ids.append(self.tokenizer.eos_token_id)
        self.stop_token_ids = [
            token for token in self.stop_token_ids if token is not None
        ]
        options = (
            {"limit_mm_per_prompt": c["limit_mm_per_prompt"]}
            if c.get("limit_mm_per_prompt")
            else {}
        )
        self.llm = self.build_engine(
            model=c["name"],
            revision=c["revision"],
            tokenizer=c.get("tokenizer_model", c["name"]),
            tokenizer_revision=c.get("tokenizer_revision", c["revision"]),
            dtype="bfloat16",
            quantization=c["quantization"],
            tensor_parallel_size=c["tensor_parallel_size"],
            max_model_len=c["max_context"],
            max_num_seqs=spec.get("batch_size", 1),
            max_num_batched_tokens=8192,
            enforce_eager=True,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            skip_tokenizer_init=True,
            trust_remote_code=False,
            **options,
        )

    def build_engine(self, **options):
        from vllm import LLM
        return LLM(**options)

    def complete(self, messages, probe, strength, seed, max_tokens):
        return self.complete_batch(
            [
                {
                    "messages": messages,
                    "probe": probe,
                    "strength": strength,
                    "seed": seed,
                    "max_tokens": max_tokens,
                }
            ]
        )[0]

    def prepare_requests(self, requests):
        """Batch independent episodes; every request retains its own hook and RNG."""
        from vllm import SamplingParams
        from vllm_lens import Hook

        if not requests:
            return []
        prompts, parameters, prepared = [], [], []
        for request_index, request in enumerate(requests):
            probe, strength = request["probe"], request["strength"]
            prompt = self.tokenizer.apply_chat_template(
                request["messages"],
                tokenize=False,
                add_generation_prompt=True,
                **self.cfg.get("template_kwargs", {}),
            )
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(ids) + request["max_tokens"] > self.cfg["max_context"]:
                raise ValueError("Context overflow; no truncation")
            hooks = (
                [
                    Hook(
                        fn=lens_hook(
                            probe["vector"],
                            strength,
                            f"{request_index}:{request['seed']}",
                        ),
                        layer_indices=[probe["layer_zero_based"]],
                        pre=False,
                    )
                ]
                if probe and strength
                else []
            )
            prompts.append({"prompt_token_ids": ids})
            parameters.append(
                SamplingParams(
                    temperature=self.spec["temperature"],
                    top_p=self.spec["top_p"],
                    seed=request["seed"],
                    max_tokens=request["max_tokens"],
                    stop_token_ids=self.stop_token_ids,
                    extra_args={"apply_hooks": hooks},
                )
            )
            prepared.append((ids, bool(hooks)))
        return prompts, parameters, prepared

    def complete_batch(self, requests):
        if not requests:
            return []
        prompts, parameters, prepared = self.prepare_requests(requests)
        results = self.llm.generate(prompts, parameters, use_tqdm=False)
        return self.decode_results(requests, prepared, results)

    def decode_results(self, requests, prepared, results):
        if len(results) != len(requests):
            raise ValueError("Engine returned an incomplete completion batch")
        diagnostics = []
        for i, (request, (ids, _), result) in enumerate(
            zip(requests, prepared, results, strict=True)
        ):
            completion = result.outputs[0]
            diagnostics.append(
                {
                    "batch_index": i,
                    "request_id": getattr(result, "request_id", None),
                    "expected_strength": request["strength"],
                    "expected_layer": request["probe"]["layer_zero_based"]
                    if request["probe"]
                    else None,
                    "seed": request["seed"],
                    "prompt_tokens": len(ids),
                    "generated_tokens": len(completion.token_ids),
                    "token_tail": list(completion.token_ids[-8:]),
                    "finish_reason": completion.finish_reason,
                    "stop_reason": getattr(completion, "stop_reason", None),
                    "hook_results": getattr(result, "hook_results", None),
                    "raw_completion": self.tokenizer.decode(
                        completion.token_ids, skip_special_tokens=False
                    ),
                }
            )
        decoded = []
        for request_index, (request, (ids, hooked), result) in enumerate(
            zip(requests, prepared, results, strict=True)
        ):
            probe, strength = request["probe"], request["strength"]
            completion = result.outputs[0]
            if list(result.prompt_token_ids) != ids:
                raise ValueError(
                    "Engine changed or reordered the rendered prompt tokens"
                )
            steering_receipt = None
            if hooked:
                saved = getattr(result, "hook_results", None) or {}
                steering_receipt = saved.get("0", saved.get(0, saved))
                expected_tokens = len(ids) + len(completion.token_ids) - 1
                observed_tokens = steering_receipt.get("tokens_steered", -1)
                # Lens 1.2.1 _trim_activations documents one scheduled forward
                # after EOS; vLLM discards its output, but custom hooks still fire.
                # Permit exactly that extra terminal step, never missing tokens
                # or arbitrary count drift. Length-limited outputs get no allowance.
                terminal_stop = (
                    completion.finish_reason == "stop"
                    and bool(completion.token_ids)
                    and completion.token_ids[-1] in self.stop_token_ids
                )
                valid_count = observed_tokens == expected_tokens or (
                    terminal_stop and observed_tokens == expected_tokens + 1
                )
                if (
                    steering_receipt.get("layer") != probe["layer_zero_based"]
                    or steering_receipt.get("strength") != strength
                    or steering_receipt.get("request_tag")
                    != f"{request_index}:{request['seed']}"
                    or not valid_count
                ):
                    raise BackendValidationError(
                        "Missing or incomplete steering hook receipt", diagnostics
                    )
                steering_receipt = {
                    **steering_receipt,
                    "semantic_tokens_steered": expected_tokens,
                    "extra_terminal_forward_tokens": observed_tokens - expected_tokens,
                    "extra_forward_reason": "scheduler_forward_after_terminal_stop"
                    if observed_tokens > expected_tokens
                    else None,
                }
            elif getattr(result, "hook_results", None):
                raise BackendValidationError(
                    "Baseline unexpectedly received steering hooks", diagnostics
                )
            raw = self.tokenizer.decode(completion.token_ids, skip_special_tokens=False)
            text = self.tokenizer.decode(completion.token_ids, skip_special_tokens=True)
            final, reasoning, finish = split_completion(
                raw, text, self.cfg, completion.finish_reason == "stop"
            )
            decoded.append(
                {
                    "response": final,
                    "reasoning_text": reasoning,
                    "raw_completion": raw,
                    "finish_reason": finish,
                    "input_tokens": len(ids),
                    "generated_tokens": len(completion.token_ids),
                    "sampling_seed": request["seed"],
                    "engine_request_id": getattr(result, "request_id", None),
                    "steering_receipt": steering_receipt,
                }
            )
        return decoded


def make_backend(spec):
    model = spec["model"]
    use_lens = model.get("backend") == "vllm_lens" or model["family"] == "glm"
    return LensBackend(spec) if use_lens else HFBackend(spec)
