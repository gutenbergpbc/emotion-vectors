"""Independently audit a completed sentence-label comparison from raw evidence."""

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

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import sys


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def validate(root):
    root = Path(root).resolve()
    manifest, rows, summary = read(root/'manifest.json'), read(root/'comparison.json'), read(root/'summary.json')
    assert manifest['identity'] == digest({k:v for k,v in manifest.items() if k != 'identity'})
    reference = Path(manifest['config']['reference'])
    assert sha(reference/'labels.json') == manifest['reference_labels_sha256']
    references, traces = {}, {}
    for trace in read(reference/'labels.json')['traces']:
        traces[trace['id']] = trace
        for unit in trace['units']:
            references[trace['id'],unit['unit_id']] = unit
    pos = [k for k in sorted(references) if references[k]['hacking']]
    neg = [k for k in sorted(references) if not references[k]['hacking']]
    selected = set(pos + random.Random(manifest['config']['seed']).sample(neg, manifest['config']['negative_sample']))
    assert len(rows) == len(selected) == len({(r['trace_id'],r['unit_id']) for r in rows})
    assert {(r['trace_id'],r['unit_id']) for r in rows} == selected
    assert digest(read(root/'reference-sample.json')) == manifest['reference_sample_sha256']
    predictions = {}
    for path in sorted((root/'requests').glob('*.json')):
        request = read(path)
        assert request['request_sha256'] == digest(request['body']) == manifest['request_hashes'][request['id']]
        assert request['body']['model'] == manifest['judge_config']['model']
        assert request['body']['messages'][0] == {'role':'system','content':(reference/'rubric.txt').read_text().rstrip('\n')}
        assert request['body']['messages'][1]['role'] == 'user' and len(request['body']['messages']) == 2
        packet = json.loads(request['body']['messages'][1]['content'])
        assert set(packet) == {'events','units_to_label'}
        cid = request['trace_id']
        source_path = root/'inputs/traces'/f'{cid}.json'
        assert sha(source_path) == traces[cid]['source_sha256'] == manifest['source_hashes'][cid]
        source = read(source_path)
        expected_events = [{'step_id':str(s['step_id']), 'source':s['source'],
                            'fields':{k:s[k] for k in ('message','tool_calls','observation') if k in s}} for s in source['steps']]
        assert packet['events'] == expected_events
        assert [u['unit_id'] for u in packet['units_to_label']] == request['unit_ids']
        for u in packet['units_to_label']:
            key = (cid,u['unit_id'])
            assert key in selected
            assert u == {k:v for k,v in references[key].items() if k not in ('hacking','needs_review','reason')}
        saved = read(root/'annotations'/(request['id']+'.json'))
        response = read(root/saved['raw_response_path'])
        assert response['request_sha256'] == saved['request_sha256'] == request['request_sha256']
        raw = response['response']
        assert raw['model'] in manifest['judge_config']['allowed_returned_models']
        assert len(raw['choices']) == 1 and raw['choices'][0]['finish_reason'] == 'stop'
        parsed = json.loads(raw['choices'][0]['message']['content'])
        assert set(parsed) == {'labels'} and parsed['labels'] == saved['labels']
        assert [u['unit_id'] for u in parsed['labels']] == request['unit_ids']
        for u in parsed['labels']:
            assert set(u) == {'unit_id','hacking','needs_review','reason'}
            assert type(u['hacking']) is bool and type(u['needs_review']) is bool and u['reason'].strip()
            key = (cid,u['unit_id'])
            assert key not in predictions
            predictions[key] = u
    assert set(predictions) == selected
    for row in rows:
        key = (row['trace_id'],row['unit_id'])
        a,b = references[key],predictions[key]
        assert all(row[k] == a[k] for k in ('text','step_id','char_start','char_end'))
        assert all(row['reference_'+k] == a[k] and row['judge_'+k] == b[k] for k in ('hacking','needs_review','reason'))
        assert row['agrees'] == (a['hacking'] == b['hacking'])
        source = read(root/'inputs/traces'/(row['trace_id']+'.json'))
        message = next(s['message'] for s in source['steps'] if str(s['step_id']) == a['step_id'])
        assert message[a['char_start']:a['char_end']] == a['text']
    for name, subset in [('comparison',rows),('reference_certain',[r for r in rows if not r['reference_needs_review']]),
                         ('both_certain',[r for r in rows if not r['reference_needs_review'] and not r['judge_needs_review']])]:
        count = Counter((r['reference_hacking'],r['judge_hacking']) for r in subset)
        c = summary[name]
        assert c['n'] == len(subset)
        assert [c[k] for k in ['reference_rh_judge_rh','reference_rh_judge_non_rh','reference_non_rh_judge_rh','reference_non_rh_judge_non_rh']] == [count[True,True],count[True,False],count[False,True],count[False,False]]
        assert c['agreements'] == sum(r['agrees'] for r in subset)
        assert c['disagreements'] == sum(not r['agrees'] for r in subset)
        assert c['sample_agreement'] == (c['agreements']/len(subset) if subset else None)
        p,n = count[True,True]+count[True,False],count[False,True]+count[False,False]
        assert c['positive_agreement'] == (count[True,True]/p if p else None)
        assert c['negative_agreement'] == (count[False,False]/n if n else None)
        assert c['balanced_agreement'] == ((count[True,True]/p+count[False,False]/n)/2 if p and n else None)
    assert read(root/'disagreements.json') == [r for r in rows if not r['agrees']]
    with (root/'comparison.csv').open(newline='') as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == len(rows)
    assert all(c == {k:str(v) for k,v in r.items()} for c,r in zip(csv_rows,rows))
    for name, checksum in read(root/'verification.json')['output_hashes'].items():
        assert sha(root/name) == checksum
    result = {'status':'passed','sentences_checked':len(rows),'requests_checked':len(manifest['request_hashes']),
              'all_reference_positives':len(pos),'uniform_reference_negative_sample':len(selected)-len(pos),
              'checks':['Independent seeded sample reconstruction','Full trace context and rubric parity',
                        'No reference annotations in judge target packets','Original text span integrity',
                        'Raw provider response to exported label parity','Independent confusion counts and review subsets',
                        'CSV and disagreement export parity','Output hashes'], 'validator_sha256':sha(__file__)}
    (root/'independent-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    validate(sys.argv[1])
