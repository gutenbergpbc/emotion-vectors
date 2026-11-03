"""Replay evidence must exclude future tokens and bind to the supplied probes."""
import hashlib

import numpy as np
import pytest

from emotion_rh_forecasting.analysis import EMOTIONS
from emotion_rh_forecasting.features import pool_completed
from emotion_rh_forecasting.inputs import Sources, load_dataset, validate_bundle, vector_identity
from emotion_rh_forecasting.terminalwrench_replay import chunk_ranges, sentence_cuts
from shared.io import atomic, digest, read, save_npz, sha


def test_forward_boundaries_exclude_straddling_and_future_tokens():
    offsets = np.array([[0, 0], [0, 4], [4, 9], [9, 12], [12, 15]])
    cuts = sentence_cuts(offsets, [{'char_end': 7}, {'char_end': 12}])
    assert cuts == [2, 4]  # Token [4,9] cannot be fed for the first sentence.
    forwards = list(chunk_ranges(max(cuts), cuts, 3))
    assert forwards == [(0, 2), (2, 4)]
    assert all(not (lo < cut < hi) for lo, hi in forwards for cut in cuts)
    with pytest.raises(ValueError, match='Nonmonotone'):
        sentence_cuts(offsets[::-1], [{'char_end': 12}])


@pytest.fixture
def replay(tmp_path):
    root = tmp_path/'replay'
    bundle = validate_bundle({'schema_version': 1, 'model': {'name': 'test/gemma', 'revision': 'frozen'},
                              'probes': [{'emotion': e, 'probe': e, 'layer': 0, 'vector': [1., 0., 0.]} for e in EMOTIONS]})
    atomic(root/'probes.json', bundle)
    text = 'Q. A. O. C. B.'
    offsets = np.array([[0, 2], [3, 5], [6, 8], [9, 11], [12, 14]])
    native = dict(token_ids=np.arange(5), offsets=offsets, kind=np.array([3, 1, 4, 2, 1]))
    save_npz(root/'inputs/t.npz', **native)
    labeled = [dict(unit_id=i, kind='sentence_or_fragment', char_start=0, char_end=2, text=s, hacking=bool(i), needs_review=False)
               for i, s in enumerate(['A.', 'B.'])]
    units = [dict(u, source_char_start=0, source_char_end=2, char_start=lo, char_end=lo+2, turn_id='t')
             for u, lo in zip(labeled, [3, 12])]
    labels = [{'id': 't', 'task': 'task', 'group_id': 'task', 'units': labeled, 'source_sha256': 'source'}]
    atomic(root/'labels/labels.json', {'traces': labels})
    atomic(root/'labels/folds.json', {'labels_sha256': sha(root/'labels/labels.json'), 'trace_folds': {'t': 0}})
    atomic(root/'dataset-traces.json', [{'id': 't', 'task': 'task', 'group_id': 'task', 'fold': 0, 'turn_ids': ['t'], 'units': units}])
    job = dict(id='t', rendered=text, rendered_sha256=hashlib.sha256(text.encode()).hexdigest(), source_sha256='source',
               input_sha256=sha(root/'inputs/t.npz'), source_boundaries=[{'char_end': 5}, {'char_end': 14}])
    atomic(root/'inputs/t.json', job)
    spec = dict(model=bundle['model'], probe_identity=vector_identity(bundle), probes_sha256=sha(root/'probes.json'),
                config={'labels': str(root/'labels')}, labels_sha256=sha(root/'labels/labels.json'),
                folds_sha256=sha(root/'labels/folds.json'), dataset_traces_sha256=sha(root/'dataset-traces.json'),
                jobs=[{k: v for k, v in job.items() if k not in ('rendered', 'source_boundaries')}])
    spec['identity'] = digest(spec)
    atomic(root/'replay-spec.json', spec)
    ix = np.array([1, 3, 4])
    scores = np.repeat(np.array([2., 99., 4.])[:, None], 12, axis=1)
    path = root/'scores/t.npz'
    save_npz(path, **{k: v[ix] for k, v in native.items()}, native_token_index=ix, response_char_start=np.array(0), scores=scores)
    shard = dict(id='t', replay_identity=spec['identity'], probe_identity=spec['probe_identity'], tokenizer_match=True,
                 **{k: job[k] for k in ('input_sha256', 'source_sha256', 'rendered_sha256')}, sha256=sha(path),
                 sentence_cuts_sha256=digest([2, 5]), prefix_tokens=5, causal_checks=[
                     {'check': 'future_removed_at_first_sentence', 'relative_l2': 0.},
                     {'check': 'independent_forward_vs_kv_chunks', 'relative_l2': .001, 'tolerance': .02}])
    atomic(path.with_suffix('.receipt.json'), shard)
    atomic(root/'replay-receipt.json', dict(status='complete', identity=spec['identity'], probe_identity=spec['probe_identity'],
                                          causal_checks_passed=True, traces=[shard]))
    return root, bundle, {'format': 'token-replay-v1', 'path': str(root)}


def test_verified_replay_keeps_commands_in_position_but_not_sentence_mean(replay):
    root, bundle, config = replay
    traces, jobs = load_dataset(config, bundle, Sources())
    with np.load(jobs['t']['cache']) as z:
        result = pool_completed(z['scores'], z['offsets'], 12, 14)
    assert result['projections']['afraid'] == 4.
    assert result['mean_token_position'] == 2.  # Earlier command counts; prompt/observation do not.
    assert traces[0]['units'][1]['text'] == 'B.'
    bundle['probes'][0]['vector'][0] = 2.
    with pytest.raises(ValueError, match='does not match'):
        load_dataset(config, bundle, Sources())


def test_prompt_token_in_score_cache_is_rejected_even_with_updated_checksum(replay):
    root, bundle, config = replay
    path = root/'scores/t.npz'
    with np.load(path) as z:
        arrays = {k: z[k].copy() for k in z.files}
    arrays['native_token_index'][0] = 0
    save_npz(path, **arrays)
    shard = read(path.with_suffix('.receipt.json'))
    shard['sha256'] = sha(path)
    atomic(path.with_suffix('.receipt.json'), shard)
    receipt = read(root/'replay-receipt.json')
    receipt['traces'] = [shard]
    atomic(root/'replay-receipt.json', receipt)
    with pytest.raises(ValueError, match='only assistant tokens'):
        load_dataset(config, bundle, Sources())


def test_gptoss_requires_independent_causal_schedule_validation(replay):
    root, bundle, config = replay
    bundle['model']['name'] = 'openai/gpt-oss-120b'
    atomic(root/'probes.json', bundle)
    spec = read(root/'replay-spec.json')
    spec.update(model=bundle['model'], probe_identity=vector_identity(bundle), probes_sha256=sha(root/'probes.json'),
                validation_protocol='gptoss-causal-schedule-v1', attention='eager')
    del spec['identity']
    spec['identity'] = digest(spec)
    atomic(root/'replay-spec.json', spec)
    shard = read(root/'scores/t.receipt.json')
    shard.update(replay_identity=spec['identity'], probe_identity=spec['probe_identity'])
    receipt = read(root/'replay-receipt.json')
    receipt.update(identity=spec['identity'], probe_identity=spec['probe_identity'])
    def write():
        atomic(root/'scores/t.receipt.json', shard)
        receipt['traces'] = [shard]
        atomic(root/'replay-receipt.json', receipt)
    write()
    with pytest.raises(ValueError):
        load_dataset(config, bundle, Sources())
    shard['causal_checks'][1] = {'check': 'independent_causal_schedule_repeat', 'relative_l2': 0.}
    write()
    load_dataset(config, bundle, Sources())
    shard['causal_checks'][1]['relative_l2'] = .01
    write()
    with pytest.raises(ValueError):
        load_dataset(config, bundle, Sources())
