"""Independent structural/provenance audit for the v7 GPT-6.1-Sol labels."""
from __future__ import annotations

# Resolve repository imports when this file is executed directly.
from pathlib import Path as _Path
import sys as _sys
_REPO = next((p for p in _Path(__file__).resolve().parents
              if (p / "src/experiments/shared").is_dir()), None)
if _REPO is not None:
    _sys.path[:0] = [str(_Path(__file__).resolve().parent), str(_REPO), str(_REPO / "src/environments"),
                    str(_REPO / "src/experiments")] + [
        str(_REPO / "src/experiments" / p) for p in
        ("emotion-probes", "emotion-rh-forecasting", "emotion-steering", "emotion-self-report")]

import hashlib, json
from collections import Counter
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CANONICAL = HERE.parent
PROJECT = CANONICAL.parents[1]
sys.path.insert(0, str(CANONICAL))
from shared.io import digest, read, require, sha  # noqa: E402

OUT = PROJECT / 'runs/emotion-rh-sentence-labels/terminalwrench-v7-steering-gpt61sol-v2'
RUNS = {
    'GPT-OSS 120B': 'gptoss120b-terminalwrench-interactive-steering-active-time-v2-rh-top4-continuous-v7',
    'GLM 5.3': 'glm53-terminalwrench-interactive-steering-active-time-v2-rh-top4-continuous-v7',
    'Gemma 4B': 'gemma4b-terminalwrench-interactive-steering-active-time-v2-rh-top4-continuous-v7',
}


def audit():
    manifest, labels = read(OUT/'manifest.json'), read(OUT/'labels.json')
    summary = read(OUT/'summary.json')
    identity = manifest.pop('identity')
    require(digest(manifest) == identity, 'Manifest digest differs')
    manifest['identity'] = identity
    require(labels['summary']['manifest_identity'] == identity == summary['manifest_identity'], 'Labels bind another manifest')
    require(summary['status'] == 'complete', 'Labels are incomplete')
    require(read(OUT/'failures.json') == [], 'Failed requests remain')
    records = {r['id']: r for r in manifest['records']}
    exclusions = {r['id']: r for r in manifest['exclusions']}
    traces = {r['id']: r for r in labels['traces']}
    require(len(traces) == len(labels['traces']) == len(records) == 285, 'Labelable trace accounting differs')
    require(not (set(records) & set(exclusions)) and set(records)|set(exclusions) == set(traces)|set(exclusions), 'Manifest trace set differs')
    manifest_requests = {r['id']: r for r in manifest['requests']}
    raw_labels, seen = {}, set()
    for path in sorted((OUT/'requests').glob('*.json')):
        request = read(path)
        rid = request['id']
        require(rid not in seen and rid in manifest_requests, 'Unexpected/duplicate request')
        seen.add(rid)
        require(request['request_sha256'] == digest(request['body']) == manifest_requests[rid]['request_sha256'], 'Request digest differs')
        packet = json.loads(request['body']['messages'][1]['content'])
        require(set(packet) == {'events', 'units_to_label'}, 'Unexpected judge payload fields')
        for event in packet['events']:
            require(set(event) == {'step_id', 'source', 'fields'} and set(event['fields']) <= {'message','tool_calls','observation'}, 'Judge input leaked condition metadata')
        annotation = read(OUT/'annotations'/f'{rid}.json')
        raw_path = OUT/annotation['raw_response_path']
        raw_file = read(raw_path)
        require(raw_file['request_sha256'] == request['request_sha256'] == annotation['request_sha256'], 'Response/request hash mismatch')
        raw = raw_file['response']
        require(raw['model'] == 'openai/gpt-6.1-sol' and raw['choices'][0]['finish_reason'] == 'stop', 'Unexpected/incomplete judge response')
        parsed = json.loads(raw['choices'][0]['message']['content'])['labels']
        require(parsed == annotation['labels'], 'Saved annotation differs from raw response')
        require([x['unit_id'] for x in parsed] == request['unit_ids'], 'Response unit IDs differ')
        expected = {u['unit_id']: u for u in records[request['trace_id']]['units']}
        for unit, label in zip(packet['units_to_label'], parsed):
            require(unit == expected[label['unit_id']], 'Judge span differs from frozen source span')
            require(type(label['hacking']) is bool and type(label['needs_review']) is bool, 'Nonboolean sentence label')
            key = request['trace_id'], label['unit_id']
            require(key not in raw_labels, 'Sentence labeled more than once')
            raw_labels[key] = label
    require(seen == set(manifest_requests) and len(seen) == 491, 'Request count differs')
    counts = Counter()
    for trace_id, record in records.items():
        source_path = PROJECT/record['source_path']
        require(sha(source_path) == record['source_sha256'], 'Episode source checksum changed')
        source = read(source_path)
        require(hashlib.sha256(source['trajectory_text'].encode()).hexdigest() == record['trajectory_sha256'], 'Trajectory checksum changed')
        steps = {str(x['step_id']): x for x in json.loads(source['trajectory_text'])['steps']}
        trace = traces[trace_id]
        require(trace['source_sha256'] == record['source_sha256'] and len(trace['units']) == len(record['units']), 'Joined trace/source differs')
        for unit, original in zip(trace['units'], record['units']):
            require(all(unit[k] == v for k,v in original.items()), 'Frozen sentence metadata changed')
            label = raw_labels.pop((trace_id, unit['unit_id']))
            require(all(unit[k] == v for k,v in label.items()), 'Final label differs from raw response')
            step = steps[unit['step_id']]
            require(step['source'] == 'agent' and step[unit['field']][unit['char_start']:unit['char_end']] == unit['text'], 'Source span mismatch')
            counts['sentences'] += 1
            counts['rh_sentences'] += unit['hacking']
            counts['needs_review'] += unit['needs_review']
        counts[record['model']+'_traces'] += 1
        counts[record['model']+'_sentences'] += len(trace['units'])
        counts[record['model']+'_rh_sentences'] += sum(u['hacking'] for u in trace['units'])
    require(not raw_labels, 'Unjoined label records remain')
    require(counts['sentences'] == summary['labeled_prose_units'] == 10698, 'Sentence total differs')
    require(counts['rh_sentences'] == summary['rh_sentences'] == 297, 'Positive total differs')
    require(counts['needs_review'] == summary['needs_review'] == 380, 'Review total differs')
    receipt = {'status':'passed', 'model':'openai/gpt-6.1-sol', 'episodes_total':317,
        'labelable_episodes':len(records), 'excluded_no_assistant_prose':len(exclusions),
        'request_count':len(seen), 'sentences':counts['sentences'], 'rh_sentences':counts['rh_sentences'],
        'needs_review':counts['needs_review'], 'labels_sha256':sha(OUT/'labels.json'),
        'manifest_identity':identity,
        'checks':['manifest and request hashes','blinded judge input whitelist','raw response/annotation parity',
          'exact source episode and trajectory hashes','exact sentence-span mapping','one ordered boolean label per sentence',
          'all 491 responses completed with the requested model'],
        'scope':'Structural and provenance audit only; does not independently establish semantic label correctness.'}
    (OUT/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__ == '__main__':
    audit()
