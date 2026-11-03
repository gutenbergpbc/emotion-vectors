from pathlib import Path
import re

from shared.io import digest, read, require, sha
from .data import validate_dataset

CANONICAL = Path(__file__).resolve().parents[2]
PROJECT = CANONICAL.parents[1]
DEFAULTS = {
    'schema_version': 1, 'experiment': 'emotion-probes', 'executor': 'local',
    'backend': 'vllm_lens', 'centering': 'within_contrast', 'dataset_contract': 'unified-twelve-v1',
    'dtype': 'bfloat16', 'quantization': 'fp8', 'max_model_len': 131072,
    'tensor_parallel_size': 8, 'max_num_batched_tokens': 8192,
    'gpu_memory_utilization': 0.90, 'shard_size': 16,
    'bootstrap_draws': 2000, 'bootstrap_seed': 20261009,
    'renderer': 'glm-native-preserved-streams-v1', 'reasoning_effort': 'max',
    'gpu': 'H200:8',
    'training_family': 'diversity',
    'repeatability_policy': 'strict',
}


def load_config(path):
    path = Path(path).resolve()
    supplied = read(path)
    allowed = set(DEFAULTS) | {'run_id', 'model', 'revision', 'layers', 'width', 'dataset', 'tokenizer_model', 'tokenizer_revision'}
    require(not (set(supplied) - allowed), f'Unknown config keys: {sorted(set(supplied) - allowed)}')
    cfg = {**DEFAULTS, **supplied}
    for key in ('run_id', 'model', 'revision', 'layers', 'width', 'dataset'):
        require(key in cfg, f'Missing config field: {key}')
    require(cfg['schema_version'] == 1 and cfg['experiment'] == 'emotion-probes', 'Unsupported config')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', cfg['run_id']), 'Unsafe run_id')
    require(cfg['executor'] in ('local', 'modal'), 'executor must be local or modal')
    require(cfg['backend'] in ('vllm_lens', 'synthetic'), 'Unsupported extraction backend')
    require(cfg['centering'] in ('within_contrast', 'none'), 'Unknown centering method')
    require(cfg['dataset_contract'] in ('unified-twelve-v1', 'custom'), 'Unknown dataset contract')
    require(cfg['training_family'] in ('diversity', 'story'), 'Unknown training family')
    require(cfg['repeatability_policy'] in ('strict', 'record_only'), 'Unknown repeatability policy')
    require(cfg['training_family'] != 'story' or cfg['dataset_contract'] == 'custom', 'Story training needs its own frozen dataset')
    require(cfg['renderer'] in (DEFAULTS['renderer'], 'gemma-native-preserved-streams-v2'), 'Unknown replay adapter')
    require(cfg['reasoning_effort'] in ('low', 'high', 'max'), 'Invalid reasoning effort')
    for key in ('layers', 'width', 'max_model_len', 'tensor_parallel_size', 'max_num_batched_tokens', 'shard_size', 'bootstrap_draws'):
        require(type(cfg[key]) is int and cfg[key] > 0, f'{key} must be a positive integer')
    require(0 < cfg['gpu_memory_utilization'] < 1, 'Invalid GPU memory fraction')
    require(cfg['dtype'] == 'bfloat16' and cfg['quantization'] in (None, 'fp8', 'modelopt_fp4'), 'Expected BF16, FP8 or NVFP4 checkpoint with BF16 residuals')
    cfg.setdefault('tokenizer_model', cfg['model'])
    cfg.setdefault('tokenizer_revision', cfg['revision'])
    if cfg['backend'] == 'synthetic':
        require(cfg['model'] == 'synthetic/demo' and cfg['executor'] == 'local', 'Synthetic fixtures must be explicitly labeled and local')
    else:
        require(re.fullmatch(r'[0-9a-f]{40}', cfg['revision']), 'Pin an immutable 40-character checkpoint revision')
        require(re.fullmatch(r'[0-9a-f]{40}', cfg['tokenizer_revision']), 'Pin tokenizer revision')
        if cfg['model'] == 'google/gemma-3-4b-it':
            require(cfg['quantization'] is None and cfg['renderer'] == 'gemma-native-preserved-streams-v2',
                    'Gemma requires BF16 weights and its native replay adapter')
            require((cfg['layers'], cfg['width']) == (34, 2560), 'Unexpected Gemma 3 4B architecture')
            require(cfg['tokenizer_model'] == cfg['model'] and cfg['tokenizer_revision'] == cfg['revision'],
                    'Gemma tokenizer must match the checkpoint')
            require(cfg['tensor_parallel_size'] == 1 and cfg['gpu'] == 'H100:1', 'Gemma allocation must be one H100')
        else:
            expected = {'zai-org/GLM-5.3': 'fp8', 'Inferact/GLM-5.3-NVFP4': 'modelopt_fp4',
                        'nvidia/GLM-5.2-NVFP4': 'modelopt_fp4'}
            require(cfg['model'] in expected and expected[cfg['model']] == cfg['quantization'], 'Unsupported checkpoint/precision combination')
            require(cfg['renderer'] == DEFAULTS['renderer'], 'GLM requires its native replay adapter')
            require(cfg['gpu'] == f"{'B200' if cfg['quantization'] == 'modelopt_fp4' else 'H200'}:{cfg['tensor_parallel_size']}",
                    'GPU allocation must match tensor parallelism and checkpoint hardware')
            require(cfg['repeatability_policy'] == 'strict' or cfg['quantization'] == 'modelopt_fp4',
                    'record_only repeatability is reserved for NVFP4 runs')
    cfg['dataset'] = str((path.parent / cfg['dataset']).resolve())
    return cfg


def source_hashes():
    files = list(Path(__file__).parent.glob('*.py')) + list((CANONICAL / 'shared').glob('*.py'))
    return {str(p.relative_to(CANONICAL)): sha(p) for p in sorted(files)}


def resolve(cfg):
    training, evaluation, split, hashes = validate_dataset(cfg['dataset'], cfg['dataset_contract'] == 'unified-twelve-v1', cfg['training_family'])
    spec = {**cfg, 'input_hashes': hashes, 'source_hashes': source_hashes(),
            'readout': 'post-block-full-residual/marked-assistant-token-mean/float32',
            'counts': {'training': len(training), 'eval': len(evaluation),
                       'calibration': sum(r['split'] == 'calibration' for r in split['rows']),
                       'heldout': sum(r['split'] == 'heldout' for r in split['rows'])}}
    if cfg['training_family'] == 'story':
        spec['training_readout'] = 'post-block-full-residual/raw-text-all-token-mean-including-special-tokens/float32'
    spec['identity'] = digest({k: v for k, v in spec.items() if k not in ('dataset', 'executor', 'run_id')})
    return spec


def run_root(cfg):
    return PROJECT / 'runs' / 'emotion-probes' / cfg['run_id']
