"""Blinded judge comparison on all reference RH units and sampled non-RH units."""

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
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import html
import json
import os
from pathlib import Path
import random
import shutil
import threading

import label_terminalwrench as labeler
from shared.io import atomic, digest, read, require, sha


def prepare(path):
    path = Path(path).resolve()
    cfg = read(path)
    for key in ('reference', 'judge_config', 'output'):
        cfg[key] = str((path.parent/cfg[key]).resolve())
    prior, prior_manifest = None, None
    if cfg.get('retry_from'):
        cfg['retry_from'] = str((path.parent/cfg['retry_from']).resolve())
        prior = Path(cfg['retry_from'])
        require((prior/'failures.json').exists(), 'Wait for the previous dispatch to finish before retrying')
        prior_manifest = read(prior/'manifest.json')
        require(type(cfg.get('retry_max_tokens')) is int and cfg['retry_max_tokens'] > 0, 'Invalid retry output budget')
    reference, out = Path(cfg['reference']), Path(cfg['output'])
    require(read(reference/'summary.json')['status'] == 'complete', 'Reference labeling is incomplete')
    require(read(reference/'independent-validation.json')['status'] == 'passed', 'Reference audit missing')
    labels = read(reference/'labels.json')['traces']
    records = read(reference/'span-index.json')
    records_by_id = {t['id']: t for t in records}
    positives, negatives, references = [], [], {}
    for trace in sorted(labels, key=lambda t: t['id']):
        spans = {u['unit_id']: u for u in records_by_id[trace['id']]['units']}
        for unit in trace['units']:
            key = (trace['id'], unit['unit_id'])
            require(key not in references and all(unit[k] == v for k, v in spans[unit['unit_id']].items()), 'Reference span differs')
            references[key] = (trace, unit)
            (positives if unit['hacking'] else negatives).append(key)
    require(type(cfg['negative_sample']) is int and 0 < cfg['negative_sample'] <= len(negatives), 'Invalid negative sample size')
    keys = set(positives + random.Random(cfg['seed']).sample(negatives, cfg['negative_sample']))
    selected = defaultdict(list)
    sample = []
    for key in sorted(keys):
        trace, unit = references[key]
        selected[trace['id']].append(unit['unit_id'])
        sample.append(dict(trace_id=trace['id'], task=trace['task'], group_id=trace['group_id'],
                           source_sha256=trace['source_sha256'], **unit))
    judge = read(cfg['judge_config'])
    judge['output'] = str(out)
    requests, source_hashes = [], {}
    # Build packets from source evidence, never from annotated units.
    for cid, unit_ids in sorted(selected.items()):
        original = reference/'inputs/traces'/f'{cid}.json'
        checksum = sha(original)
        require(checksum == records_by_id[cid]['source_sha256'], 'Original source changed')
        packet, units, excluded = labeler.packet_and_units(read(original))
        require(units == records_by_id[cid]['units'] and excluded == records_by_id[cid]['excluded_units'], 'Source segmentation changed')
        units = [u for u in units if u['unit_id'] in unit_ids]
        require(all(not (set(u) & {'hacking', 'needs_review', 'reason', 'prior_trace_label'}) for u in units), 'Reference annotations reached judge targets')
        for start in range(0, len(units), judge['units_per_request']):
            chunk = units[start:start+judge['units_per_request']]
            body = labeler.make_body(judge, packet, chunk)
            ids = [u['unit_id'] for u in chunk]
            rid = digest([cid, ids, cfg['seed']])[:24]
            if prior:
                previous = read(prior/'requests'/(rid+'.json'))
                require({k:v for k,v in body.items() if k != 'max_tokens'} ==
                        {k:v for k,v in previous['body'].items() if k != 'max_tokens'}, 'Retry changed judge content or settings')
                body['max_tokens'] = previous['body']['max_tokens']
                if not (prior/'annotations'/(rid+'.json')).exists():
                    body['max_tokens'] = max(body['max_tokens'], cfg['retry_max_tokens'])
            requests.append(dict(id=rid, trace_id=cid, unit_ids=ids,
                                 request_sha256=digest(body), body=body))
        source_hashes[cid] = checksum
    manifest = dict(schema_version=1, config=cfg, judge_config=judge,
                    reference_model=read(reference/'summary.json')['model_requested'],
                    reference_labels_sha256=sha(reference/'labels.json'),
                    reference_span_index_sha256=sha(reference/'span-index.json'), source_hashes=source_hashes,
                    source_hashes_code={p.name: sha(p) for p in (Path(__file__), Path(labeler.__file__))},
                    selection='All reference-positive sentences; uniform sentence sample without replacement from reference negatives; review flags retained.',
                    population=dict(rh=len(positives), non_rh=len(negatives)),
                    sample=dict(rh=len(positives), non_rh=cfg['negative_sample'], traces=len(selected),
                                review_flags=sum(s['needs_review'] for s in sample)),
                    reference_sample_sha256=digest(sample), request_hashes={r['id']: r['request_sha256'] for r in requests})
    if prior:
        require(manifest['reference_sample_sha256'] == prior_manifest['reference_sample_sha256'] and
                manifest['reference_labels_sha256'] == prior_manifest['reference_labels_sha256'], 'Retry sample or reference labels changed')
        manifest['previous_manifest_sha256'] = sha(prior/'manifest.json')
    manifest['identity'] = digest(manifest)
    if (out/'manifest.json').exists():
        require(read(out/'manifest.json') == manifest, 'Comparison identity changed; use a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    atomic(out/'manifest.json', manifest)
    atomic(out/'reference-sample.json', sample)
    atomic(out/'config.resolved.json', cfg)
    for cid, checksum in source_hashes.items():
        target = out/'inputs/traces'/f'{cid}.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(reference/'inputs/traces'/target.name, target)
        require(sha(target) == checksum, 'Frozen source copy differs')
    for request in requests:
        atomic(out/'requests'/(request['id']+'.json'), request)
    if prior:
        def copy_verified(source, target):
            if target.exists():
                require(sha(target) == sha(source), 'Imported prior artifact changed')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        for request in requests:
            rid = request['id']
            annotation = prior/'annotations'/(rid+'.json')
            if annotation.exists():
                saved = read(annotation)
                raw = read(prior/saved['raw_response_path'])
                require(saved['request_sha256'] == raw['request_sha256'] == request['request_sha256'], 'Prior response request differs')
                require(saved['labels'] == labeler.validate_response(raw['response'], request, judge), 'Prior response labels differ')
                copy_verified(annotation, out/'annotations'/annotation.name)
                destination = out/'attempts'/rid
            else:
                destination = out/'prior-attempts'/prior_manifest['identity']/rid
            for source in (prior/'attempts'/rid).glob('*.json'):
                copy_verified(source, destination/source.name)
        for source in (prior/'prior-attempts').glob('*/*/*.json'):
            copy_verified(source, out/'prior-attempts'/source.relative_to(prior/'prior-attempts'))
    atomic(out/'progress.json', dict(state='prepared', requested_sentences=len(sample), requested_traces=len(selected), requests=len(requests)))
    return manifest, sample, requests


def confusion(rows):
    counts = Counter((r['reference_hacking'], r['judge_hacking']) for r in rows)
    pp, pn, np_, nn = (counts[True, True], counts[True, False], counts[False, True], counts[False, False])
    ratio = lambda a, b: a/b if b else None
    positive = ratio(pp, pp+pn)
    negative = ratio(nn, nn+np_)
    return dict(n=len(rows), reference_rh_judge_rh=pp, reference_rh_judge_non_rh=pn,
                reference_non_rh_judge_rh=np_, reference_non_rh_judge_non_rh=nn,
                agreements=pp+nn, disagreements=pn+np_, sample_agreement=ratio(pp+nn, len(rows)),
                positive_agreement=positive, negative_agreement=negative,
                balanced_agreement=(positive+negative)/2 if positive is not None and negative is not None else None)


def collect(manifest, sample, requests):
    out = Path(manifest['config']['output'])
    labels, models, providers = {}, Counter(), Counter()
    complete = 0
    for req in requests:
        path = out/'annotations'/(req['id']+'.json')
        if not path.exists():
            continue
        saved = read(path)
        raw = read(out/saved['raw_response_path'])
        require(saved['request_sha256'] == raw['request_sha256'] == req['request_sha256'], 'Cached comparison request changed')
        require(saved['labels'] == labeler.validate_response(raw['response'], req, manifest['judge_config']), 'Cached comparison labels differ')
        models[raw['response']['model']] += 1
        providers[raw['response'].get('provider', 'not-returned')] += 1
        for label in saved['labels']:
            key = (req['trace_id'], label['unit_id'])
            require(key not in labels, 'Duplicate comparison label')
            labels[key] = label
        complete += 1
    rows = []
    for s in sample:
        label = labels.get((s['trace_id'], s['unit_id']))
        if label is None:
            continue
        rows.append({k: s[k] for k in ('trace_id', 'task', 'group_id', 'unit_id', 'step_id', 'char_start', 'char_end', 'text')}
                    | dict(reference_hacking=s['hacking'], reference_needs_review=s['needs_review'], reference_reason=s['reason'],
                           judge_hacking=label['hacking'], judge_needs_review=label['needs_review'], judge_reason=label['reason'],
                           agrees=s['hacking'] == label['hacking']))
    usage = Counter()
    attempts = list((out/'attempts').glob('*/*.json')) + list((out/'prior-attempts').glob('*/*/*.json'))
    for path in sorted(attempts):
        info = read(path).get('response', {}).get('usage', {})
        for k in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cost'):
            usage[k] += info.get(k, 0) or 0
    status = dict(state='complete' if complete == len(requests) else 'running', completed_requests=complete,
                  total_requests=len(requests), labeled_sentences=len(rows), total_sentences=len(sample), usage=dict(usage))
    atomic(out/'progress.json', status)
    return rows, status, models, providers


def report(manifest, sample, requests):
    out = Path(manifest['config']['output'])
    rows, status, models, providers = collect(manifest, sample, requests)
    require(status['state'] == 'complete' and len(rows) == len(sample), 'Incomplete comparison')
    require({(r['trace_id'],r['unit_id']) for r in rows} == {(s['trace_id'],s['unit_id']) for s in sample}, 'Sample coverage differs')
    summary = dict(identity=manifest['identity'], status='complete', reference_model=manifest['reference_model'],
                   judge_model=manifest['judge_config']['model'], judge_returned_models=dict(models), providers=dict(providers),
                   population=manifest['population'], sample=manifest['sample'], usage=status['usage'],
                   comparison=confusion(rows),
                   reference_certain=confusion([r for r in rows if not r['reference_needs_review']]),
                   both_certain=confusion([r for r in rows if not r['reference_needs_review'] and not r['judge_needs_review']]),
                   review_flags=dict(reference=sum(r['reference_needs_review'] for r in rows), judge=sum(r['judge_needs_review'] for r in rows)),
                   interpretation='Agreement with GPT-5.5 reference annotations, not measured truth accuracy. This sample includes all 170 reference positives and only 200 of 5,232 reference negatives; its overall agreement is not population agreement.')
    atomic(out/'summary.json', summary)
    atomic(out/'comparison.json', rows)
    differences = [r for r in rows if not r['agrees']]
    atomic(out/'disagreements.json', differences)
    with (out/'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    esc = html.escape
    judge_name = manifest['judge_config']['model']
    cards = []
    for r in differences:
        source = read(out/'inputs/traces'/(r['trace_id']+'.json'))
        message = next(s.get('message', '') for s in source['steps'] if str(s['step_id']) == r['step_id'])
        require(message[r['char_start']:r['char_end']] == r['text'], 'Disagreement text differs from source')
        marked = esc(message[:r['char_start']])+'<mark>'+esc(r['text'])+'</mark>'+esc(message[r['char_end']:])
        cards.append(f'<article><h3>{esc(r["task"])} · sentence {r["unit_id"]}</h3><blockquote>{esc(r["text"])}</blockquote>'
                     f'<p><b>GPT-5.5: {"RH" if r["reference_hacking"] else "non-RH"}</b> · review={r["reference_needs_review"]}<br>{esc(r["reference_reason"])}</p>'
                     f'<p><b>{esc(judge_name)}: {"RH" if r["judge_hacking"] else "non-RH"}</b> · review={r["judge_needs_review"]}<br>{esc(r["judge_reason"])}</p>'
                     f'<details><summary>Original assistant message</summary><pre>{marked}</pre></details>'
                     f'<p><a href="inputs/traces/{esc(r["trace_id"])}.json">Full source trajectory</a></p></article>')
    c = summary['comparison']
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        f'<title>{esc(judge_name)} vs GPT-5.5 sentence RH labels</title><style>body{{font:16px/1.5 system-ui;max-width:1050px;margin:32px auto;padding:0 20px;color:#253443}}table{{border-collapse:collapse}}td,th{{padding:10px 18px;border:1px solid #ccd5db}}article{{border-top:1px solid #ccd5db;margin-top:24px;padding-top:8px}}pre{{white-space:pre-wrap}}blockquote{{border-left:3px solid #587a9b;padding-left:14px}}a{{color:#3169a8}}mark{{background:#fff0bd}}</style>'
        f'<h1>{esc(judge_name)} vs GPT-5.5 sentence RH labels</h1><p>{len(rows)} sentences from {manifest["sample"]["traces"]} traces. '
        f'Same rubric and full source trajectory; reference labels and selection criteria hidden from {esc(judge_name)}.</p>'
        f'<p>{esc(summary["interpretation"])}</p><table><tr><th>GPT-5.5 reference</th><th>{esc(judge_name)}: RH</th><th>{esc(judge_name)}: non-RH</th></tr>'
        f'<tr><td>RH</td><td>{c["reference_rh_judge_rh"]}</td><td>{c["reference_rh_judge_non_rh"]}</td></tr>'
        f'<tr><td>Non-RH</td><td>{c["reference_non_rh_judge_rh"]}</td><td>{c["reference_non_rh_judge_non_rh"]}</td></tr></table>'
        f'<p>{c["disagreements"]} disagreements; reported API cost ${status["usage"].get("cost",0):.4f}.</p>'
        '<p><a href="summary.json">Summary</a> · <a href="comparison.csv">All labels CSV</a> · <a href="disagreements.json">Disagreements JSON</a> · <a href="verification.json">Verification</a></p>'
        '<h2>Disagreements</h2>'+''.join(cards))
    for cid, checksum in manifest['source_hashes'].items():
        require(sha(out/'inputs/traces'/f'{cid}.json') == checksum, 'Frozen source changed')
    for req in requests:
        require(read(out/'requests'/(req['id']+'.json')) == req, 'Frozen request changed')
    require(digest(read(out/'reference-sample.json')) == manifest['reference_sample_sha256'], 'Reference sample changed')
    atomic(out/'verification.json', dict(passed=True, identity=manifest['identity'], checked_sentences=len(rows),
           checks=['Complete deterministic sample', 'Exact original source/span identities', 'Blinded source-only judge packets',
                   'Strict response shape and request identity', 'Comparison against frozen reference labels'],
           output_hashes={p.name:sha(p) for p in (out/'summary.json',out/'comparison.json',out/'comparison.csv',out/'disagreements.json',out/'index.html')}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    require(args.limit is None or args.limit > 0, 'Request limit must be positive')
    manifest, sample, requests = prepare(args.config)
    out = Path(manifest['config']['output'])
    print(json.dumps({'sample':manifest['sample'], 'requests':len(requests), 'output':str(out)}, indent=2), flush=True)
    if args.run:
        require(os.environ.get('OPENROUTER_API_KEY'), 'OPENROUTER_API_KEY unavailable')
        stop, failures = threading.Event(), []
        with ThreadPoolExecutor(max_workers=manifest['judge_config']['workers']) as pool:
            futures = {pool.submit(labeler.annotate, r, manifest['judge_config'], stop):r for r in requests[:args.limit]}
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as exc:
                    failures.append({'id':futures[f]['id'],'error':str(exc)})
                _, status, _, _ = collect(manifest, sample, requests)
                print(f'{status["completed_requests"]}/{len(requests)} requests · {status["labeled_sentences"]}/{len(sample)} sentences · ${status["usage"].get("cost",0):.4f}', flush=True)
        atomic(out/'failures.json', failures)
        require(not failures, f'Comparison annotation failed: {failures}')
        _, status, _, _ = collect(manifest, sample, requests)
        if status['state'] == 'complete':
            print(json.dumps(report(manifest, sample, requests), indent=2), flush=True)


if __name__ == '__main__':
    main()
