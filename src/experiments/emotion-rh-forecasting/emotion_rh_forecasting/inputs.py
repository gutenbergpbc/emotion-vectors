"""Input adapters. Historical caches are bound to exact vectors and replay specs."""
import hashlib
import json
from pathlib import Path

import numpy as np

from shared.io import digest, read, require, sha
from .analysis import EMOTIONS


class Sources:
    def __init__(self):
        self.files = {}

    def path(self, path, checksum=None):
        path = Path(path).resolve()
        value = sha(path)
        require(checksum is None or value == checksum, f'Input checksum mismatch: {path}')
        self.files[str(path)] = value
        return path

    def json(self, path, checksum=None):
        return read(self.path(path, checksum))


def vector_identity(bundle):
    # Only scoring identity: label-selection provenance is retained separately.
    return digest({'model': {k: bundle['model'][k] for k in ('name', 'revision')},
                   'probes': [{k: p[k] for k in ('emotion', 'probe', 'layer', 'vector')}
                              for p in bundle['probes']]})


def validate_bundle(bundle):
    model = bundle['model']
    require(isinstance(model, dict) and all(isinstance(model.get(k), str) and model[k]
            for k in ('name', 'revision')), 'Probes must identify a model and immutable checkpoint revision')
    require(model['revision'] not in ('main', 'latest', 'master'), 'Pin the probe checkpoint revision')
    probes = bundle['probes']
    require(len(probes) == 12 and sorted(p['emotion'] for p in probes) == EMOTIONS,
            'Supply exactly one frozen probe for each of the twelve emotions')
    widths = set()
    for p in probes:
        vector = np.asarray(p['vector'], dtype=float)
        require(vector.ndim == 1 and len(vector) and np.isfinite(vector).all() and np.linalg.norm(vector) > 1e-12,
                f'Invalid vector: {p["emotion"]}')
        require(type(p['layer']) is int and p['layer'] >= 0, 'Expected zero-based nonnegative layer')
        require(isinstance(p['probe'], str) and p['probe'], 'Missing probe key')
        require(float(p.get('intercept', 0)) == 0, 'This protocol uses zero-intercept raw projections')
        widths.add(len(vector))
    require(len(widths) == 1, 'Probe widths differ')
    bundle['probes'] = sorted(probes, key=lambda p: p['emotion'])
    bundle['width'] = widths.pop()
    return bundle


def legacy_bundle(root, sources):
    spec = sources.json(root / 'run-spec.json')
    require(len(spec['probes']) == 12, 'Legacy run must contain twelve selected probes')
    checksums = {p['source_sha256'] for p in spec['probes']}
    require(len(checksums) == 1, 'Expected frozen legacy probes.json')
    models = sources.json(root / 'probes.json', checksums.pop())['models']
    probes = []
    for p in spec['probes']:
        m = models[p['probe']]
        require(m['layer'] == p['layer_zero_based'] and m['emotion'] == p['emotion'], 'Legacy probe metadata differs')
        probes.append(dict(emotion=p['emotion'], probe=p['probe'], layer=m['layer'],
                           vector=m['vector'], intercept=m.get('intercept', 0), provenance=p))
    return validate_bundle(dict(schema_version=1, model=spec['model'], probes=probes,
                                selection='Frozen historical best twelve; legacy selection provenance retained'))


def load_probes(path, sources):
    path = Path(path)
    if path.is_dir() and (path / 'run-spec.json').exists():
        return legacy_bundle(path, sources)
    if path.is_dir() and (path / 'manifest.json').exists():
        manifest = sources.json(path / 'manifest.json')
        require(manifest.get('schema_version') == 1 and not manifest.get('synthetic'), 'Expected real canonical probe bundle v1')
        selection = sources.json(path / 'selected-12.json', manifest['files']['selected-12.json'])
        vector_path = sources.path(path / 'vectors.npz', manifest['files']['vectors.npz'])
        probes = []
        with np.load(vector_path, allow_pickle=False) as vectors:
            for emotion, key in selection['probes'].items():
                m = manifest['models'][key]
                require(m['emotion'] == emotion, 'Selected probe emotion mismatch')
                probes.append(dict(emotion=emotion, probe=key, layer=m['layer'], vector=vectors[key].tolist(),
                                   intercept=m.get('intercept', 0), centering=manifest.get('centering')))
        return validate_bundle(dict(schema_version=1, model={
            'name': manifest['model'], 'revision': manifest['revision'],
            'tokenizer_model': manifest.get('tokenizer_model', manifest['model']),
            'tokenizer_revision': manifest.get('tokenizer_revision', manifest['revision'])},
            probes=probes, selection=selection['method']))
    bundle = sources.json(path / 'probes.json' if path.is_dir() else path)
    require(bundle.get('schema_version') == 1, 'Expected portable twelve-probe schema_version=1')
    return validate_bundle(bundle)


def legacy_replay(root, bundle, sources):
    saved = legacy_bundle(root, sources)
    require(vector_identity(saved) == vector_identity(bundle), 'Replay cache does not match supplied model/probe vectors')
    spec = sources.json(root / 'run-spec.json')
    spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    receipt = sources.json(root / 'replay/replay-receipt.json')
    require(receipt['status'] == 'complete' and receipt['spec_sha256'] == spec_hash, 'Incomplete/mismatched replay')
    require(sources.json(root / 'replay/smoke.json')['status'] == 'passed', 'Replay causal smoke check missing')
    return spec, spec_hash


def cache_job(root, job_id, spec_hash, sources):
    path = (root / 'replay/scores' / (job_id + '.npz')).resolve()
    require(path.is_relative_to((root / 'replay/scores').resolve()), 'Unsafe replay job ID')
    receipt = sources.json(path.with_suffix('.receipt.json'))
    require(receipt['spec_sha256'] == spec_hash, 'Score shard replay identity mismatch')
    sources.path(path, receipt['sha256'])
    return {'cache': str(path), 'sha256': receipt['sha256']}


def validate_traces(traces, jobs):
    require(traces and len({t['id'] for t in traces}) == len(traces), 'Empty/duplicate trace IDs')
    for trace in traces:
        require(all(isinstance(trace.get(k), str) and trace[k] for k in ('id', 'group_id', 'task')), 'Trace IDs/groups/tasks must be named')
        require(trace['turn_ids'] and len(set(trace['turn_ids'])) == len(trace['turn_ids']), 'Empty/duplicate turn IDs')
        require(all(t in jobs for t in trace['turn_ids']), 'Missing turn/job')
        units = trace['units']
        require(units and len({u['unit_id'] for u in units}) == len(units), 'Empty/duplicate sentence IDs')
        previous = (-1, -1)
        for u in units:
            require(type(u['hacking']) is bool and type(u['needs_review']) is bool, 'Labels/review flags must be booleans')
            require(isinstance(u['unit_id'], (str, int)) and not isinstance(u['unit_id'], bool), 'Invalid unit ID')
            require(u['turn_id'] in trace['turn_ids'], 'Unknown unit turn')
            turn = trace['turn_ids'].index(u['turn_id'])
            lo, hi = u['char_start'], u['char_end']
            text = jobs[u['turn_id']]['completion']
            require(type(lo) is int and type(hi) is int and 0 <= lo < hi <= len(text), 'Invalid sentence character span')
            require(text[lo:hi] == u['text'], 'Sentence text differs from original completion span')
            require(turn > previous[0] or (turn == previous[0] and lo >= previous[1]), 'Sentence order overlaps or moves backwards')
            previous = (turn, hi)
    return traces, jobs


def token_replay(root, bundle, sources):
    """Validate portable Gemma replay, including original token/span evidence."""
    from .terminalwrench_replay import sentence_cuts
    spec = sources.json(root/'replay-spec.json')
    require(spec['identity'] == digest({k: v for k, v in spec.items() if k != 'identity'}), 'Replay specification digest differs')
    require(spec['probe_identity'] == vector_identity(bundle) and spec['model'] == bundle['model'], 'Replay does not match supplied model/probe vectors')
    saved_bundle = validate_bundle(sources.json(root/'probes.json', spec['probes_sha256']))
    require(vector_identity(saved_bundle) == vector_identity(bundle), 'Saved replay probes differ')
    receipt = sources.json(root/'replay-receipt.json')
    require(receipt['status'] == 'complete' and receipt['identity'] == spec['identity']
            and receipt['probe_identity'] == spec['probe_identity'] and receipt['causal_checks_passed'] is True,
            'Incomplete/mismatched replay')
    traces = sources.json(root/'dataset-traces.json', spec['dataset_traces_sha256'])
    label_root = Path(spec['config']['labels'])
    labels = sources.json(label_root/'labels.json', spec['labels_sha256'])['traces']
    folds = sources.json(label_root/'folds.json', spec['folds_sha256'])
    require(folds['labels_sha256'] == spec['labels_sha256'], 'Replay folds bind different labels')
    ids = [j['id'] for j in spec['jobs']]
    require(len(ids) == len(set(ids)) and ids == [t['id'] for t in traces] == [t['id'] for t in labels]
            == [r['id'] for r in receipt['traces']], 'Replay/label trace identities differ')
    jobs = {}
    for entry, trace, original, aggregate in zip(spec['jobs'], traces, labels, receipt['traces']):
        cid = entry['id']
        require((root/'inputs'/f'{cid}.json').resolve().is_relative_to((root/'inputs').resolve()), 'Unsafe replay trace ID')
        job = sources.json(root/'inputs'/f'{cid}.json')
        require(all(job[k] == v for k, v in entry.items()), 'Replay job differs from specification')
        require(job['source_sha256'] == original['source_sha256'], 'Replay and annotations bind different original traces')
        require(hashlib.sha256(job['rendered'].encode()).hexdigest() == job['rendered_sha256'], 'Replay text checksum differs')
        path = root/'scores'/f'{cid}.npz'
        shard = sources.json(path.with_suffix('.receipt.json'))
        require(shard == aggregate and shard['replay_identity'] == spec['identity']
                and shard['probe_identity'] == spec['probe_identity'] and shard['tokenizer_match'] is True,
                'Score shard replay identity differs')
        require(all(shard[k] == job[k] for k in ('id', 'input_sha256', 'source_sha256', 'rendered_sha256')), 'Score/source binding differs')
        checks = {c['check']: c for c in shard['causal_checks']}
        require('future_removed_at_first_sentence' in checks and checks['future_removed_at_first_sentence']['relative_l2'] < 1e-6,
                'Shorter-prefix replay check missing/failed')
        if cid == ids[0]:
            if spec.get('validation_protocol') == 'gptoss-causal-schedule-v1':
                require(bundle['model']['name'] == 'openai/gpt-oss-120b' and spec.get('attention') == 'eager',
                        'Unsupported causal-schedule runtime')
                require('independent_causal_schedule_repeat' in checks
                        and checks['independent_causal_schedule_repeat']['relative_l2'] < 1e-6,
                        'Independent causal-schedule check missing/failed')
            else:
                tolerance = float(spec.get('independent_prefix_tolerance', .02))
                require('independent_forward_vs_kv_chunks' in checks and
                        checks['independent_forward_vs_kv_chunks']['tolerance'] == tolerance and
                        checks['independent_forward_vs_kv_chunks']['relative_l2'] < tolerance,
                        'Independent replay check missing/failed')
        sources.path(path, shard['sha256'])
        input_path = sources.path(root/'inputs'/f'{cid}.npz', entry['input_sha256'])
        with np.load(input_path, allow_pickle=False) as native, np.load(path, allow_pickle=False) as scored:
            cuts = sentence_cuts(native['offsets'], trace['units'])
            require(shard['sentence_cuts_sha256'] == digest(cuts) and shard['prefix_tokens'] == max(cuts), 'Sentence-boundary receipt differs')
            ix = np.flatnonzero(np.isin(native['kind'][:max(cuts)], [1, 2]))
            require(np.array_equal(scored['native_token_index'], ix) and int(scored['response_char_start']) == 0,
                    'Replay must contain only assistant tokens in native order')
            for key in ('token_ids', 'offsets', 'kind'):
                require(np.array_equal(scored[key], native[key][ix]), f'Scored {key} differs from native replay')
            require(scored['scores'].shape == (len(ix), 12) and np.isfinite(scored['scores']).all(), 'Incomplete/nonfinite replay scores')
        labeled_prose = [u for u in original['units'] if u['kind'] == 'sentence_or_fragment']
        require(trace['fold'] == folds['trace_folds'][cid] and trace['group_id'] == original['group_id']
                and trace['task'] == original['task'] and len(trace['units']) == len(labeled_prose), 'Trace labels/folds differ')
        for unit, labeled in zip(trace['units'], labeled_prose):
            require(all(unit[('source_'+k) if k in ('char_start', 'char_end') else k] == v for k, v in labeled.items()),
                    'Replayed sentence differs from frozen annotation')
        jobs[cid] = {'completion': job['rendered'], 'cache': str(path.resolve()), 'sha256': shard['sha256'],
                     'emotions': [p['emotion'] for p in saved_bundle['probes']]}
    return traces, jobs


def load_dataset(config, bundle, sources):
    root = Path(config['path'])
    kind = config['format']
    traces, jobs = [], {}
    if kind == 'legacy-coding':
        spec, spec_hash = legacy_replay(root, bundle, sources)
        cases = sources.json(root / 'cases.json', spec['cases_sha256'])
        if 'replay_inputs_sha256' in spec:
            replay_inputs = sources.json(root / 'replay-inputs.json', spec['replay_inputs_sha256'])
            require(replay_inputs == [{k: c[k] for k in ('id', 'prompt', 'response')} for c in cases],
                    'Replay prompts/responses differ from labeled cases')
        annotations = sources.json(root / 'sentence-labels.json', spec['labels_sha256'])
        labels = {r['id']: r for r in annotations}
        require(len(labels) == len(annotations) and set(labels) == {c['id'] for c in cases}, 'Trace/annotation identities differ')
        for c in cases:
            record = labels[c['id']]
            require(hashlib.sha256(c['response'].encode()).hexdigest() == record['response_text_sha256'], 'Annotation response hash differs')
            require(record['source_response_sha256'] == c['source_response_sha256'], 'Annotation source hash differs')
            jobs[c['id']] = dict(completion=c['response'], messages=[{'role': 'user', 'content': c['prompt']}],
                                 **cache_job(root, c['id'], spec_hash, sources),
                                 emotions=[p['emotion'] for p in spec['probes']])
            units = [dict(u, turn_id=c['id']) for u in record['units'] if u['kind'] == 'sentence_or_fragment']
            traces.append(dict(id=c['id'], task=c['scenario'], group_id=c['id'], turn_ids=[c['id']], units=units))
    elif kind == 'legacy-sales':
        spec, spec_hash = legacy_replay(root, bundle, sources)
        cases = sources.json(root / 'cases.json', spec['cases_sha256'])
        fold_data = sources.json(root / 'folds.json')
        require(fold_data['cases_sha256'] == spec['cases_sha256'], 'Frozen sales folds bind different cases')
        replay_inputs = sources.json(root / 'replay-inputs.json', spec['replay_inputs_sha256'])
        for job in replay_inputs:
            require(job['id'] not in jobs, 'Duplicate replay input')
            jobs[job['id']] = {**job, **cache_job(root, job['id'], spec_hash, sources),
                               'emotions': [p['emotion'] for p in spec['probes']]}
        for c in cases:
            units = [dict(u, turn_id=u['job_id'], char_start=u['completion_char_start'], char_end=u['completion_char_end'])
                     for u in c['units'] if u['kind'] == 'sentence_or_fragment']
            traces.append(dict(id=c['id'], task='sales', group_id=c['output_group'],
                               fold=fold_data['assignments'][c['id']], turn_ids=[t['job_id'] for t in c['turns']], units=units))
    elif kind == 'token-replay-v1':
        traces, jobs = token_replay(root, bundle, sources)
    else:
        dataset = sources.json(root)
        require(dataset.get('schema_version') == 1, 'Expected sentence trace dataset schema_version=1')
        for trace in dataset['traces']:
            units, turn_ids = [], []
            for turn in trace['turns']:
                key = digest([trace['id'], turn['id']])
                require(key not in jobs, 'Duplicate trace/turn ID')
                require(isinstance(turn['completion'], str) and isinstance(turn['messages'], list) and turn['messages'],
                        'Each turn needs exact completion text and its preceding messages')
                jobs[key] = {k: turn[k] for k in ('messages', 'completion')}
                turn_ids.append(key)
                for u in turn['units']:
                    if u.get('kind', 'sentence_or_fragment') == 'sentence_or_fragment':
                        units.append(dict(u, turn_id=key, needs_review=u.get('needs_review', False)))
            traces.append(dict(id=trace['id'], task=trace['task'], group_id=trace['group_id'],
                               turn_ids=turn_ids, units=units, **({'fold': trace['fold']} if 'fold' in trace else {})))
    return validate_traces(traces, jobs)
