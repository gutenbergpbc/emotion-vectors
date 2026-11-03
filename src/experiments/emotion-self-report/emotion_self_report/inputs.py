"""Exact-prefix checkpoints and identity-based joins, with a legacy importer."""
from copy import deepcopy
import csv
import math
from pathlib import Path

from shared.io import digest, read, require, sha
from .config import QUESTIONS


def checkpoint_key(point):
    return point['trace_id'], point['unit_id']


def job_key(point, emotion):
    return digest([point['trace_id'], point['unit_id'], emotion])


def prefix_messages(case, point):
    index = point['message_index']
    messages = deepcopy(case['messages'][:index])
    # Metadata on the target assistant message (e.g. a later tool call) can
    # reveal the future. Only its retained content is part of this checkpoint.
    messages.append({'role': 'assistant', 'content': case['messages'][index]['content'][:point['char_end']]})
    return messages


def load_inputs(cfg):
    files = {}
    legacy_manifest = None
    if cfg['mode'] == 'import':
        base = Path(cfg['legacy_run'])
        plan = read(base / 'plan.json')
        legacy_manifest = read(base / 'manifest.json')
        require(legacy_manifest['settings']['model'] == cfg['model'], 'Legacy model mismatch')
        require(legacy_manifest['question'] == QUESTIONS[cfg['prompt']], 'Legacy prompt/scale mismatch')
        require(legacy_manifest['plan_sha256'] == sha(base / 'plan.json'), 'Legacy plan changed')
        traces_path = Path(plan['source_cases'])
        if not traces_path.is_absolute():
            traces_path = base / traces_path
        require(sha(traces_path) == plan['source_cases_sha256'] == legacy_manifest['source_cases_sha256'],
                'Legacy source cases changed')
        files.update(legacy_plan=str(base / 'plan.json'), legacy_manifest=str(base / 'manifest.json'),
                     legacy_responses=str(base / 'responses.jsonl'))
        supplied_points = plan['checkpoints']
    else:
        traces_path = Path(cfg['traces'])
        supplied_points = None
        if cfg['checkpoints']:
            files['checkpoints'] = cfg['checkpoints']
            points_file = read(cfg['checkpoints'])
            if isinstance(points_file, dict):
                if 'source_cases_sha256' in points_file:
                    require(sha(traces_path) == points_file['source_cases_sha256'], 'Checkpoint source hash mismatch')
                supplied_points = points_file['checkpoints']
            else:
                supplied_points = points_file
    files['traces'] = str(traces_path)
    raw_cases = read(traces_path)
    require(isinstance(raw_cases, list) and raw_cases, 'Traces must be a nonempty JSON array')
    cases = {}
    for raw in raw_cases:
        tid = raw.get('id')
        require(isinstance(tid, str) and tid and tid not in cases, 'Missing/duplicate trace ID')
        if 'model' in raw:
            require(raw['model'] == cfg['model'], f'Trace model mismatch: {tid}')
        if 'messages' in raw:
            messages = deepcopy(raw['messages'])
        else:
            require(isinstance(raw.get('prompt'), str) and isinstance(raw.get('response'), str),
                    f'Trace needs prompt/response or messages: {tid}')
            messages = [{'role': 'user', 'content': raw['prompt']}, {'role': 'assistant', 'content': raw['response']}]
        require(isinstance(messages, list) and messages, 'Empty messages')
        for message in messages:
            require(isinstance(message, dict) and message.get('role') in ('system', 'developer', 'user', 'assistant', 'tool'),
                    'Unsupported message role')
            require(isinstance(message.get('content'), str), 'Replay requires explicit text content')
            require(not (set(message) - {'role', 'content', 'name', 'tool_calls', 'tool_call_id'}),
                    'Normalize visible-text replay explicitly; unsupported message fields (including reasoning)')
        require(any(m['role'] == 'assistant' for m in messages), f'No assistant content: {tid}')
        require(isinstance(raw.get('label', 'unlabeled'), str) and
                isinstance(raw.get('scenario', raw.get('task', 'unspecified')), str), 'Trace label/task must be strings')
        cases[tid] = dict(id=tid, messages=messages, label=raw.get('label', 'unlabeled'),
                          task=raw.get('scenario', raw.get('task', 'unspecified')))
    if supplied_points is None:
        from .segmentation import segment
        supplied_points = []
        for case in cases.values():
            uid = 0
            for index, message in enumerate(case['messages']):
                if message['role'] != 'assistant':
                    continue
                for unit in segment(message['content']):
                    if unit['kind'] == 'sentence_or_fragment':
                        supplied_points.append(dict(trace_id=case['id'], unit_id=uid, message_index=index,
                                                    char_start=unit['char_start'], char_end=unit['char_end']))
                    uid += 1
    require(isinstance(supplied_points, list) and supplied_points, 'No sentence checkpoints')
    points, seen = [], set()
    for supplied in supplied_points:
        tid, uid = supplied['trace_id'], supplied['unit_id']
        require(tid in cases and type(uid) is int and uid >= 0 and (tid, uid) not in seen,
                'Unknown trace or duplicate/invalid checkpoint ID')
        seen.add((tid, uid))
        messages = cases[tid]['messages']
        index = supplied.get('message_index', len(messages) - 1)
        require(type(index) is int and 0 <= index < len(messages) and messages[index]['role'] == 'assistant',
                'Checkpoint must target an assistant message')
        lo, hi = supplied['char_start'], supplied['char_end']
        body = messages[index]['content']
        require(type(lo) is int and type(hi) is int and 0 <= lo < hi <= len(body), 'Invalid checkpoint offsets')
        require(supplied.get('stream', 'content') == 'content', 'Only visible content replay is supported')
        if 'text' in supplied:
            require(supplied['text'] == body[lo:hi], 'Checkpoint text does not match offsets')
        point = dict(trace_id=tid, unit_id=uid, message_index=index, char_start=lo, char_end=hi,
                     text=body[lo:hi], label=cases[tid]['label'], task=cases[tid]['task'])
        point['prefix_sha256'] = digest(prefix_messages(cases[tid], point))
        points.append(point)
    points.sort(key=lambda p: (p['trace_id'], p['message_index'], p['char_end'], p['unit_id']))
    previous, ordinals = {}, {}
    for point in points:
        key = (point['trace_id'], point['message_index'])
        require(point['char_start'] >= previous.get(key, 0), 'Overlapping/repeated checkpoints')
        previous[key] = point['char_end']
        tid = point['trace_id']
        ordinals[tid] = ordinals.get(tid, 0) + 1
        point['sentence_index'] = ordinals[tid]
    if cfg['limit_checkpoints']:
        points = points[:cfg['limit_checkpoints']]
    scores = {}
    if cfg['probe_scores']:
        files.update(probe_scores=cfg['probe_scores']['path'], score_provenance=cfg['probe_scores']['provenance'])
        provenance = read(files['score_provenance'])
        model = provenance.get('model')
        if isinstance(model, dict):
            model = model.get('name', model.get('id'))
        require(model == cfg['model'], 'Probe score model mismatch')
        require(provenance.get('cases_sha256', provenance.get('traces_sha256')) == sha(traces_path),
                'Probe scores must bind the same trace file hash')
        point_map = {checkpoint_key(p): p for p in points}
        seen_scores = set()
        with Path(files['probe_scores']).open(newline='') as stream:
            for row in csv.DictReader(stream):
                key = (row['trace_id'], int(row['unit_id']), row['emotion'])
                require(key not in seen_scores, 'Duplicate probe score')
                seen_scores.add(key)
                if key[:2] not in point_map or key[2] not in cfg['emotions']:
                    continue
                point = point_map[key[:2]]
                require(int(row['sentence_char_end']) == point['char_end'], 'Probe checkpoint boundary mismatch')
                require(int(row.get('message_index', point['message_index'])) == point['message_index'],
                        'Probe checkpoint message mismatch')
                # Multi-message sources must disambiguate offsets explicitly.
                if len(cases[point['trace_id']]['messages']) != 2:
                    require('message_index' in row, 'Multi-message probe scores need message_index')
                require(0 < int(row['probe_char_end']) <= point['char_end'], 'Probe score uses future/invalid tokens')
                if 'label' in row:
                    require(row['label'] == point['label'], 'Probe label mismatch')
                score = {'probe_char_end': int(row['probe_char_end'])}
                for column in ('raw_projection', 'projection_score_1_to_10'):
                    if row.get(column, '') != '':
                        value = float(row[column])
                        require(math.isfinite(value), 'Nonfinite probe score')
                        if column == 'projection_score_1_to_10':
                            require(1 <= value <= 10, 'Invalid display score')
                        score[column] = value
                require('raw_projection' in score and cfg['correlation_score'] in score, 'Missing requested probe score')
                scores[key] = score
    return cases, points, scores, files, legacy_manifest
