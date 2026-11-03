"""Blindly relabel the frozen MiMo/GPT-5.5 disagreement sentences with a third judge."""

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
import json
import os
from pathlib import Path
import shutil
import threading

import label_terminalwrench as labeler
from shared.io import atomic, digest, read, require, sha


def prepare(config_path):
    config_path = Path(config_path).resolve()
    cfg = read(config_path)
    for key in ('comparison', 'audit', 'output'):
        cfg[key] = str((config_path.parent / cfg[key]).resolve())
    comparison, out = Path(cfg['comparison']), Path(cfg['output'])
    require(read(comparison / 'verification.json')['passed'], 'Source comparison not verified')
    source = read(comparison / 'disagreements.json')
    require(len(source) == 80 and all(r['reference_hacking'] and not r['judge_hacking'] for r in source),
            'Unexpected source disagreement selection')
    with Path(cfg['audit']).open(newline='') as f:
        audit = {(r['trace_id'], int(r['unit_id'])): r['audit_verdict'] for r in csv.DictReader(f)}
    require(set(audit) == {(r['trace_id'], r['unit_id']) for r in source}, 'Audit sample differs')
    if cfg.get('selected_trace_ids'):
        selected_ids = set(cfg['selected_trace_ids'])
        require(len(selected_ids) == len(cfg['selected_trace_ids']) and
                selected_ids <= {r['trace_id'] for r in source}, 'Invalid selected trace IDs')
        source = [r for r in source if r['trace_id'] in selected_ids]
        audit = {key: value for key, value in audit.items() if key[0] in selected_ids}
    selected = defaultdict(list)
    for row in source:
        selected[row['trace_id']].append(row)
    requests, source_hashes = [], {}
    for cid, rows in sorted(selected.items()):
        original = comparison / 'inputs/traces' / f'{cid}.json'
        packet, units, _ = labeler.packet_and_units(read(original))
        by_id = {u['unit_id']: u for u in units}
        ids = sorted(r['unit_id'] for r in rows)
        for row in rows:
            unit = by_id[row['unit_id']]
            require(all(row[k] == unit[k] for k in ('step_id', 'char_start', 'char_end', 'text')),
                    'Source span does not match disagreement')
        targets = [by_id[uid] for uid in ids]
        body = labeler.make_body(cfg, packet, targets)
        body['max_tokens'] = max(body['max_tokens'], 16384)
        rid = digest([cid, ids, sha(original), cfg['model']])[:24]
        requests.append(dict(id=rid, trace_id=cid, unit_ids=ids,
                             request_sha256=digest(body), body=body))
        source_hashes[cid] = sha(original)
    manifest = dict(schema_version=1, config=cfg, source_comparison_sha256=sha(comparison / 'disagreements.json'),
                    source_audit_sha256=sha(cfg['audit']), selection='Exactly the 80 MiMo-negative/GPT-5.5-positive sentences; labels hidden from judge.',
                    sample=dict(sentences=len(source), traces=len(selected)), source_hashes=source_hashes,
                    source_code_sha256=sha(__file__), rubric_sha256=digest(labeler.RUBRIC),
                    request_hashes={r['id']: r['request_sha256'] for r in requests})
    manifest['identity'] = digest(manifest)
    if (out / 'manifest.json').exists():
        require(read(out / 'manifest.json') == manifest, 'Run identity changed')
    out.mkdir(parents=True, exist_ok=True)
    atomic(out / 'manifest.json', manifest)
    for cid, checksum in source_hashes.items():
        target = out / 'inputs/traces' / f'{cid}.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(comparison / 'inputs/traces' / target.name, target)
        require(sha(target) == checksum, 'Source copy changed')
    for request in requests:
        atomic(out / 'requests' / f'{request["id"]}.json', request)
    return cfg, manifest, source, audit, requests


def collect(cfg, manifest, source, audit, requests):
    out = Path(cfg['output'])
    answers, returned, providers, usage = {}, Counter(), Counter(), Counter()
    for req in requests:
        path = out / 'annotations' / f'{req["id"]}.json'
        if not path.exists():
            continue
        saved = labeler.annotate(req, cfg, threading.Event())
        raw = read(out / saved['raw_response_path'])['response']
        returned[raw['model']] += 1
        providers[raw.get('provider', 'not-returned')] += 1
        for label in saved['labels']:
            key = (req['trace_id'], label['unit_id'])
            require(key not in answers, 'Duplicate adjudication')
            answers[key] = label
    for path in (out / 'attempts').glob('*/*.json'):
        info = read(path).get('response', {}).get('usage', {})
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cost'):
            usage[key] += info.get(key, 0) or 0
    progress = dict(state='complete' if len(answers) == len(source) else 'running', labeled=len(answers),
                    total=len(source), completed_requests=sum(returned.values()), total_requests=len(requests),
                    usage=dict(usage))
    atomic(out / 'progress.json', progress)
    if progress['state'] != 'complete':
        return progress
    rows = []
    for row in source:
        key = (row['trace_id'], row['unit_id'])
        answer = answers[key]
        rows.append({k: row[k] for k in ('task', 'trace_id', 'unit_id', 'step_id', 'text', 'reference_hacking',
                                          'reference_needs_review', 'judge_hacking', 'judge_needs_review')}
                    | dict(audit_verdict=audit[key], sol_hacking=answer['hacking'],
                           sol_needs_review=answer['needs_review'], sol_reason=answer['reason']))
    counts = Counter((r['audit_verdict'], r['sol_hacking']) for r in rows)
    summary = dict(identity=manifest['identity'], status='complete', model_requested=cfg['model'],
                   models_returned=dict(returned), providers=dict(providers), sample=manifest['sample'],
                   usage=dict(usage), sol_rh=sum(r['sol_hacking'] for r in rows),
                   sol_non_rh=sum(not r['sol_hacking'] for r in rows),
                   sol_needs_review=sum(r['sol_needs_review'] for r in rows),
                   by_audit_category={category: dict(n=sum(r['audit_verdict'] == category for r in rows),
                                                    sol_rh=counts[category, True], sol_non_rh=counts[category, False])
                                      for category in ('clear_rh', 'debatable', 'likely_non_rh')},
                   interpretation='Blinded third-model judgment of a disagreement-selected sample, not ground truth or population accuracy.')
    atomic(out / 'labels.json', rows)
    atomic(out / 'summary.json', summary)
    with (out / 'labels.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for cid, checksum in manifest['source_hashes'].items():
        require(sha(out / 'inputs/traces' / f'{cid}.json') == checksum, 'Frozen source changed')
    require(sha(cfg['comparison'] + '/disagreements.json') == manifest['source_comparison_sha256'],
            'Source disagreement changed')
    atomic(out / 'verification.json', dict(passed=True, identity=manifest['identity'], checked_sentences=len(rows),
                                           source_sha256=manifest['source_comparison_sha256'],
                                           labels_sha256=sha(out / 'labels.json')))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    cfg, manifest, source, audit, requests = prepare(args.config)
    print(json.dumps({'sample': manifest['sample'], 'requests': len(requests), 'output': cfg['output']}, indent=2), flush=True)
    if not args.run:
        return
    require(os.environ.get('OPENROUTER_API_KEY'), 'OPENROUTER_API_KEY unavailable')
    stop, failures = threading.Event(), []
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        futures = {pool.submit(labeler.annotate, r, cfg, stop): r for r in requests[:args.limit]}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failures.append(dict(id=futures[future]['id'], error=str(exc)))
            progress = collect(cfg, manifest, source, audit, requests)
            print(f'{progress.get("labeled", 80)}/{len(source)} sentences, '
                  f'{progress.get("completed_requests", len(requests))}/{len(requests)} requests, '
                  f'${progress["usage"].get("cost", 0):.4f}', flush=True)
    atomic(Path(cfg['output']) / 'failures.json', failures)
    require(not failures, f'Adjudication failed: {failures}')
    progress = collect(cfg, manifest, source, audit, requests)
    if progress.get('status') == 'complete':
        print(json.dumps(progress, indent=2), flush=True)


if __name__ == '__main__':
    main()
