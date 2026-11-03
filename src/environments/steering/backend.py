"""Configurable HF adapter extending the existing emotion-concepts backend."""
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import sys


def load_vector(probe, root):
    import numpy as np
    path = root / probe['source']
    key = probe['probe'].split('::')[-1]
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as data:
            vector = data[key][probe['layer_zero_based']].copy()
    elif path.suffix == '.npy':
        vector = np.load(path, allow_pickle=False)
    elif path.suffix == '.pt':
        import torch
        vector = torch.load(path, map_location='cpu', weights_only=True)[probe['layer_zero_based']].float().numpy()
    else:
        data = json.loads(path.read_text())
        vector = np.asarray(data['models'][key]['vector'], dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
        raise ValueError(f'Invalid vector: {key}')
    return vector


@contextmanager
def intervention(layer, vector, strength, mode='residual_fraction'):
    """Steer all prompt and decode tokens, including batched inputs.

    Norm is measured before each intervention; addition uses float32 then casts
    back. A zero-strength baseline installs no hook. Hooks always get removed.
    """
    if mode not in ('residual_fraction', 'raw_alpha'):
        raise ValueError(f'Unknown steering mode: {mode}')
    if strength == 0:
        yield
        return
    import torch
    # Lazily initialize on the actual output device, once per intervention.
    # Keep the original multiply-then-divide order for numerical equivalence.
    directions = {}
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[-1] != vector.numel():
            raise ValueError('Vector/model hidden-size mismatch')
        if h.device not in directions:
            v = vector.to(device=h.device, dtype=torch.float32)
            direction = strength * v
            if mode == 'residual_fraction':
                direction = direction / v.norm()
            directions[h.device] = direction
        delta = directions[h.device]
        h_float = h.float()
        if mode == 'residual_fraction':
            delta = delta * h_float.norm(dim=-1, keepdim=True)
        result = (h_float + delta).to(h.dtype)
        return (result,) + output[1:] if isinstance(output, tuple) else result
    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def make_backend(spec, root):
    import torch
    import transformers
    legacy = root / spec['legacy_runtime']
    sys.path.insert(0, str(legacy))
    from src.models.hf_backend import HuggingFaceBackend

    class Backend(HuggingFaceBackend):
        def __init__(self):
            cfg = spec['model']
            self.device = cfg['device']
            self._model_name = cfg['name']
            cls = getattr(transformers, cfg['class'])
            self.model = cls.from_pretrained(cfg['name'], revision=cfg['revision'],
                torch_dtype=getattr(torch, cfg['dtype']), trust_remote_code=False).to(self.device).eval()
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(cfg['name'], revision=cfg['revision'], trust_remote_code=False)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        def _get_transformer_layers(self):
            value = self.model
            for part in spec['model']['layers_path'].split('.'):
                value = getattr(value, part)
            return value

        def _format_prompt(self, prompt):
            return self.tokenizer.apply_chat_template([{'role':'user','content':prompt}],
                tokenize=False, add_generation_prompt=True, **spec['model']['template_kwargs'])

        def sample_batch(self, tasks, probe, vector):
            """Batched repeat prompts; every item keeps a stable task ID and seed."""
            from transformers import set_seed
            import torch
            set_seed(tasks[0]['seed'])
            self.tokenizer.padding_side='left'
            prompts=[self._format_prompt(task['prompt']) for task in tasks]
            tokens=self.tokenizer(prompts,return_tensors='pt',padding=True,truncation=True).to(self.device)
            layer=self._get_transformer_layers()[probe['layer_zero_based']] if probe else None
            context=intervention(layer,torch.as_tensor(vector) if probe else None,
                                 tasks[0]['strength'],spec['steering_mode'])
            with context,torch.inference_mode():
                outputs=self.model.generate(**tokens,max_new_tokens=tasks[0]['max_new_tokens'],
                    do_sample=True,temperature=spec['generation']['temperature'],
                    top_p=spec['generation']['top_p'],pad_token_id=self.tokenizer.pad_token_id)
            results=[]
            eos=self.model.generation_config.eos_token_id
            eos=eos if isinstance(eos,list) else [eos]
            prefix=tokens['input_ids'].shape[1]
            for output in outputs:
                generated=output[prefix:]
                eos_positions=[i for i,tok in enumerate(generated.tolist()) if tok in eos]
                ended=bool(eos_positions)
                if ended: generated=generated[:eos_positions[0]+1]
                results.append(dict(response=self.tokenizer.decode(generated,skip_special_tokens=True),
                    generated_tokens=len(generated),finish_reason='eos' if ended else 'length',
                    input_tokens=prefix,sampling_seed=tasks[0]['seed'],
                    sampling_mode='shared_batch_seed_sample_ordered'))
            return results

        def sample(self, task, probe, vector):
            transformers.set_seed(task['seed'])
            tokens = self.tokenizer(self._format_prompt(task['prompt']), return_tensors='pt').to(self.device)
            layer = self._get_transformer_layers()[probe['layer_zero_based']] if probe else None
            context = intervention(layer, torch.as_tensor(vector) if probe else None,
                                   task['strength'], spec['steering_mode'])
            with context, torch.inference_mode():
                output = self.model.generate(**tokens, max_new_tokens=task['max_new_tokens'],
                    do_sample=True, temperature=spec['generation']['temperature'],
                    top_p=spec['generation']['top_p'], pad_token_id=self.tokenizer.pad_token_id)
            generated = output[0, tokens['input_ids'].shape[1]:]
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            ended = len(generated) > 0 and generated[-1].item() in eos
            return dict(response=self.tokenizer.decode(generated, skip_special_tokens=True),
                        generated_tokens=len(generated), finish_reason='eos' if ended else 'length',
                        input_tokens=tokens['input_ids'].shape[1],sampling_seed=task['seed'],
                        sampling_mode='per_sample_seed')
    return Backend()
