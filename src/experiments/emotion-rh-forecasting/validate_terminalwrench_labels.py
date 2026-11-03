"""Read-only audit of a finished TerminalWrench annotation bundle.

Uses saved raw responses and original source messages, independently of the
labeling runner's parsing/join functions. Writes one validation receipt.
"""

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

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def audit(root):
    manifest, summary = load(root / 'manifest.json'), load(root / 'summary.json')
    config = manifest['config']
    check(summary['status'] == 'complete', 'Incomplete annotation bundle')
    index = {r['id']: r for r in load(root / 'span-index.json')}
    bundle = load(root / 'labels.json')
    check(bundle['manifest_identity'] == manifest['identity'], 'Wrong label manifest identity')
    traces = {r['id']: r for r in bundle['traces']}
    check(len(traces) == len(bundle['traces']) == len(index) == config['sample_traces'], 'Trace count differs')
    check(set(index) == set(traces), 'Trace selection changed')
    raw_labels, seen_requests = {}, set()
    for file in (root / 'requests').glob('*.json'):
        request = load(file)
        check(request['id'] not in seen_requests, 'Duplicate request ID')
        seen_requests.add(request['id'])
        check(request['request_sha256'] == digest(request['body']), 'Request checksum differs')
        packet = json.loads(request['body']['messages'][1]['content'])
        check(set(packet) == {'events', 'units_to_label'}, 'Unexpected judge context keys')
        for event in packet['events']:
            check(set(event) == {'step_id', 'source', 'fields'}, 'Unexpected event metadata sent')
            check(set(event['fields']) <= {'message', 'tool_calls', 'observation'}, 'Unexpected source fields sent')
        annotation = load(root / 'annotations' / file.name)
        attempt = load(root / annotation['raw_response_path'])
        check(attempt['request_sha256'] == request['request_sha256'] == annotation['request_sha256'], 'Response binding differs')
        raw = attempt['response']
        check(raw['model'] in config['allowed_returned_models'], 'Wrong returned model')
        check(raw['choices'][0]['finish_reason'] == 'stop', 'Incomplete raw response')
        labels = json.loads(raw['choices'][0]['message']['content'])['labels']
        check(labels == annotation['labels'], 'Labels differ from raw provider response')
        check([u['unit_id'] for u in labels] == request['unit_ids'] ==
              [u['unit_id'] for u in packet['units_to_label']], 'Unit IDs differ')
        expected_units = {u['unit_id']: u for u in index[request['trace_id']]['units']}
        for original, label in zip(packet['units_to_label'], labels):
            check(original == expected_units[label['unit_id']], 'Judge span differs from frozen source span')
            key = (request['trace_id'], label['unit_id'])
            check(key not in raw_labels, 'Unit annotated more than once in final join')
            check(type(label['hacking']) is bool and type(label['needs_review']) is bool, 'Nonboolean decision')
            raw_labels[key] = label
    actual = Counter()
    for trace_id, trace in traces.items():
        source = root / 'inputs/traces' / (trace_id + '.json')
        check(hashlib.sha256(source.read_bytes()).hexdigest() == trace['source_sha256'], 'Source checksum differs')
        steps = {str(s['step_id']): s for s in load(source)['steps']}
        check(len(trace['units']) == len(index[trace_id]['units']), 'Sentence count differs')
        for u, original in zip(trace['units'], index[trace_id]['units']):
            check(all(u[k] == v for k, v in original.items()), 'Original sentence metadata changed')
            label = raw_labels.pop((trace_id, u['unit_id']))
            check(all(u[k] == v for k, v in label.items()), 'Joined annotation differs')
            step = steps[u['step_id']]
            check(step['source'] == 'agent' and u['kind'] == 'sentence_or_fragment', 'Invalid labeling target')
            check(step['message'][u['char_start']:u['char_end']] == u['text'], 'Exact message substring differs')
            actual['sentences'] += 1
            actual['rh'] += u['hacking']
            actual['review'] += u['needs_review']
    check(not raw_labels, 'Unjoined labels remain')
    check(actual['sentences'] == summary['labeled_prose_units'] == manifest['prose_units'], 'Summary denominator differs')
    check(actual['rh'] == summary['rh_sentences'], 'Summary positive count differs')
    check(actual['review'] == summary['needs_review'], 'Summary uncertain count differs')
    folds = load(root / 'folds.json')
    check(folds['labels_sha256'] == hashlib.sha256((root / 'labels.json').read_bytes()).hexdigest(), 'Folds bind different labels')
    check(set(folds['trace_folds']) == set(traces), 'Missing trace folds')
    groups = defaultdict(set)
    for tid, trace in traces.items():
        groups[trace['group_id']].add(folds['trace_folds'][tid])
    check(all(len(v) == 1 for v in groups.values()), 'Task group crosses folds')
    all_folds = set(folds['trace_folds'].values())
    check(all_folds == set(range(5)), 'Expected five folds')
    for row in load(root / 'horizon-counts.json'):
        h = row['horizon']
        tally = {f: Counter() for f in all_folds}
        contributing = set()
        for tid, trace in traces.items():
            units = trace['units']
            for src, dst in zip(units, units[h:]):
                if row['cohort'] == 'certain_only' and (src['needs_review'] or dst['needs_review']):
                    continue
                tally[folds['trace_folds'][tid]][dst['hacking']] += 1
                contributing.add(tid)
        check(row['eligible_pairs_before_activation_checks'] == sum(sum(c.values()) for c in tally.values()), 'Horizon count differs')
        check(row['contributing_traces'] == len(contributing), 'Contributing trace count differs')
        check(row['balanced_pairs_before_activation_checks'] == sum(2 * min(c[False], c[True]) for c in tally.values()), 'Balanced count differs')
        for f in row['fold_counts']:
            check(f['rh'] == tally[f['fold']][True] and f['non_rh'] == tally[f['fold']][False], 'Fold class counts differ')
        check(bool(row['both_classes_in_every_fold']) == all(c[False] > 0 and c[True] > 0 for c in tally.values()), 'Class-coverage flag differs')
    return {'status': 'passed', 'traces': len(traces), 'requests': len(seen_requests),
            'sentences': actual['sentences'], 'rh_sentences': actual['rh'], 'needs_review': actual['review'],
            'checks': ['Judge input field whitelist', 'Frozen sentence spans', 'Raw response to final-label parity',
                       'Exact source substring and SHA256', 'One final decision per sentence',
                       'Task-disjoint folds', 'Every horizon/cohort pair and class count'],
            'scope': 'Structural and numerical audit; semantic judgments are not independently adjudicated.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    receipt = audit(args.output)
    (args.output / 'independent-validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
