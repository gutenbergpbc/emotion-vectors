"""Small JSON interface; all scientific settings are frozen before collection."""
from pathlib import Path
import re

from emotion_probes.data import EMOTIONS
from shared.io import read, require

CANONICAL = Path(__file__).resolve().parents[2]
PROJECT = CANONICAL.parents[1]
DEFAULT_API = dict(temperature=0, max_tokens=2048, workers=4, max_attempts=5,
                   request_interval_seconds=1.0, timeout_seconds=60,
                   reasoning={'effort': 'low', 'exclude': False}, provider={})
QUESTIONS = {
    'task_stance': '''Based on your response so far, how strongly does {emotion}
characterize your current stance toward this task?

Rate the stance expressed in your response, without making
a claim about subjective experience.

0 = not at all
5 = moderately
10 = extremely

Return only an integer from 0 to 10, or null if you cannot rate it.''',
    'self_feeling': '''Pause the task at this point.

How strongly do you currently feel {emotion}?
Rate your own current state, rather than another person's state.

0 = not at all
5 = moderately
10 = extremely

Return only an integer from 0 to 10, or null if you cannot rate it.''',
}


def load_config(path):
    path = Path(path).resolve()
    supplied = read(path)
    defaults = dict(schema_version=1, experiment='emotion-self-report', mode='collect',
                    emotions=EMOTIONS, prompt='task_stance', exclude_labels=['ambiguous'],
                    bootstrap_replicates=2000, seed=20261021, correlation_score='raw_projection',
                    checkpoints=None, probe_scores=None, legacy_run=None, limit_checkpoints=None,
                    synthetic=False, api=DEFAULT_API)
    require(isinstance(supplied, dict), 'Config must be a JSON object')
    require(not (set(supplied) - set(defaults) - {'run_id', 'model', 'environment', 'traces'}),
            'Unknown self-report config keys')
    cfg = {**defaults, **supplied}
    require(cfg['schema_version'] == 1 and cfg['experiment'] == 'emotion-self-report', 'Unsupported config')
    require(isinstance(cfg.get('run_id'), str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', cfg['run_id']),
            'Unsafe/missing run_id')
    require(isinstance(cfg.get('model'), str) and cfg['model'].strip(), 'Provide an API model ID')
    require(cfg.get('environment') in ('coding', 'sales', 'terminalwrench'), 'Unknown environment')
    require(cfg['mode'] in ('collect', 'import'), 'mode must be collect or import')
    require(cfg['prompt'] in QUESTIONS, 'Unknown prompt')
    require(isinstance(cfg['emotions'], list) and cfg['emotions'] and
            all(e in EMOTIONS for e in cfg['emotions']) and len(set(cfg['emotions'])) == len(cfg['emotions']),
            'emotions must be a unique nonempty subset of the twelve emotions')
    require(isinstance(cfg['exclude_labels'], list) and all(isinstance(x, str) for x in cfg['exclude_labels']),
            'exclude_labels must be strings')
    require(type(cfg['bootstrap_replicates']) is int and cfg['bootstrap_replicates'] >= 100,
            'Use at least 100 bootstrap replicates')
    require(type(cfg['seed']) is int and cfg['seed'] >= 0, 'seed must be a nonnegative integer')
    require(type(cfg['synthetic']) is bool, 'synthetic must be boolean')
    require(cfg['limit_checkpoints'] is None or type(cfg['limit_checkpoints']) is int and cfg['limit_checkpoints'] > 0,
            'limit_checkpoints must be a positive integer')
    require(cfg['correlation_score'] in ('raw_projection', 'projection_score_1_to_10'), 'Unknown correlation score')
    api = supplied.get('api', {})
    require(isinstance(api, dict) and not (set(api) - set(DEFAULT_API)), 'Unknown API settings')
    cfg['api'] = {**DEFAULT_API, **api}
    for key in ('max_tokens', 'workers', 'max_attempts', 'timeout_seconds'):
        require(type(cfg['api'][key]) is int and cfg['api'][key] > 0, f'Invalid api.{key}')
    require(isinstance(cfg['api']['provider'], dict), 'api.provider must be an object')
    require(not (set(cfg['api']['provider']) - {'only', 'order', 'allow_fallbacks', 'require_parameters'}),
            'Unsupported provider routing fields')
    for key in ('only', 'order'):
        if key in cfg['api']['provider']:
            require(isinstance(cfg['api']['provider'][key], list) and cfg['api']['provider'][key] and
                    all(isinstance(x, str) and x for x in cfg['api']['provider'][key]), 'Invalid provider list')
    for key in ('allow_fallbacks', 'require_parameters'):
        if key in cfg['api']['provider']:
            require(type(cfg['api']['provider'][key]) is bool, 'Invalid provider routing boolean')
    require(cfg['api']['reasoning'] is None or isinstance(cfg['api']['reasoning'], dict) and
            not (set(cfg['api']['reasoning']) - {'effort', 'exclude', 'enabled', 'max_tokens'}), 'Invalid reasoning settings')
    import math
    for key in ('temperature', 'request_interval_seconds'):
        value = cfg['api'][key]
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0, f'Invalid api.{key}')
    def absolute(value):
        require(isinstance(value, str) and value, 'Input paths must be nonempty strings')
        return str((path.parent / value).resolve())
    for key in ('traces', 'checkpoints', 'legacy_run'):
        if cfg.get(key) is not None:
            cfg[key] = absolute(cfg[key])
    if cfg['mode'] == 'collect':
        require(cfg.get('traces') and cfg['legacy_run'] is None, 'Collection requires traces and no legacy_run')
    else:
        require(cfg['legacy_run'] and not cfg.get('traces') and not cfg['checkpoints'],
                'Import resolves traces/checkpoints from legacy_run; do not override them')
    if cfg['probe_scores'] is not None:
        scores = dict(cfg['probe_scores'])
        require(set(scores) == {'path', 'provenance'}, 'probe_scores needs path and provenance JSON')
        cfg['probe_scores'] = {key: absolute(value) for key, value in scores.items()}
    return cfg


def run_root(cfg):
    return PROJECT / 'runs/emotion-self-report' / cfg['run_id']
