"""Validate and materialize the 5,000 GPT-6.1 Sol sentence judgments."""
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


import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE.parent), str(HERE)]
import label_goodfire_glm52_gpt61sol_5k as judge  # noqa: E402
from shared.io import atomic, read, require, sha  # noqa: E402

OUT = judge.OUT


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out
    judge.OUT = OUT
    selection = read(OUT / 'selection.json')
    jobs = judge.prepare()
    manifest = read(OUT / 'judge-manifest.json')
    require(manifest['request_sha256'] == [j['request_sha256'] for j in jobs] and
            manifest['selection_sha256'] == sha(OUT / 'selection.json'), 'Judge inputs changed')
    proposed = {}
    cost = 0.0
    for job in jobs:
        saved = read(OUT / 'annotations' / f"{job['id']}.json")
        require(saved['request_sha256'] == job['request_sha256'] and
                saved['source_sha256'] == job['source_sha256'] and
                saved['model'] == judge.MODEL, 'Judge receipt changed')
        judge.base.validate_labels({'labels': saved['labels']}, job['units'])
        cost += (saved.get('usage') or {}).get('cost', 0) or 0
        for label in saved['labels']:
            key = (job['row_index'], label['unit_id'])
            require(key not in proposed, 'Duplicate sentence judgment')
            proposed[key] = label
    require(len(proposed) == 5000, 'Missing judgments')
    audit = read(OUT / 'manual-audit-overrides.json')
    audit_by_key = {}
    for override in audit['overrides']:
        key = (override['row_index'], override['unit_id'])
        require(key not in audit_by_key and proposed[key]['hacking'] and
                not proposed[key]['needs_review'] and override['needs_review'],
                'Invalid audit override')
        audit_by_key[key] = override

    meta = {t['row_index']: t for t in selection['traces']}
    rows, traces, by_trace = [], [], defaultdict(list)
    with (OUT / 'selected-sentences.jsonl').open() as stream:
        for line in stream:
            unit = json.loads(line)
            key = (unit['row_index'], unit['sentence_index'])
            label = proposed[key]
            require(unit['source_line_sha256'] == meta[unit['row_index']]['source_line_sha256'],
                    'Source identity changed')
            row = {**unit, 'unit_id': unit['sentence_index'], 'hacking': label['hacking'],
                   'needs_review': audit_by_key.get(key, label)['needs_review'],
                   'reason': audit_by_key.get(key, label)['reason'],
                   'label_method': 'gpt61sol_judge'}
            if key in audit_by_key:
                row['label_method'] = 'manual_review_flag'
                row['gpt61sol_reason'] = label['reason']
            rows.append(row)
            by_trace[unit['row_index']].append(row)
    require(len(rows) == 5000 and len(by_trace) == len(selection['traces']), 'Selection changed')
    for row_index in sorted(by_trace):
        m = meta[row_index]
        traces.append({'id': m['trace_id'], 'row_index': row_index, 'task': m['task'],
                       'source_line_sha256': m['source_line_sha256'], 'units': by_trace[row_index]})
    counts = Counter((r['hacking'], r['needs_review']) for r in rows)
    by_task = [{'row_index': t['row_index'], 'task': t['task'],
                'rh_certain': sum(u['hacking'] and not u['needs_review'] for u in t['units']),
                'rh_flagged': sum(u['hacking'] and u['needs_review'] for u in t['units']),
                'negative_certain': sum(not u['hacking'] and not u['needs_review'] for u in t['units'])}
               for t in traces]
    summary = {'status': 'complete', 'model': judge.MODEL,
               'traces': len(traces), 'tasks': len({t['task'] for t in traces}),
               'sentences': 5000, 'rh_certain': counts[(True, False)],
               'rh_flagged': counts[(True, True)],
               'negative_certain': counts[(False, False)],
               'negative_flagged': counts[(False, True)],
               'needs_review': counts[(True, True)] + counts[(False, True)],
               'cost_usd': round(cost, 6), 'jobs': len(jobs),
               'selection_sha256': sha(OUT / 'selection.json'),
               'judge_manifest_sha256': sha(OUT / 'judge-manifest.json'),
               'manual_audit_sha256': sha(OUT / 'manual-audit-overrides.json'),
               'source_index_sha256': selection['source_index_sha256'],
               'method': 'One GPT-6.1 Sol judgment per selected assistant prose sentence. Source-positive passage votes selected the cohort but are not sentence labels. Flagged judgments remain unresolved for the primary plot.',
               'caveat': f'Outcome-enriched selection from {len({t["task"] for t in traces})} tasks; the 5,000 sentences are not a random sample of the 562-trace archive. Multiple RH sentences can describe one episode.',
               'by_task': by_task}
    atomic(OUT / 'labels.json', {'summary': summary, 'traces': traces})
    atomic(OUT / 'summary.json', summary)
    with (OUT / 'sentences.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    positives = [r for r in rows if r['hacking']]
    with (OUT / 'positive-candidates.jsonl').open('w') as stream:
        for row in positives:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    page = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>GPT-6.1 Sol 5k RH candidates</title>',
            '<style>body{font:16px/1.5 system-ui;max-width:1050px;margin:2rem auto;color:#233}article{border-top:1px solid #bbb;padding:1rem 0}small{color:#666}</style>',
            f'<h1>GPT-6.1 Sol reward-hack sentence candidates</h1><p>{len(positives)} positive calls '
            f'({counts[(True, False)]} unflagged, {counts[(True, True)]} flagged). '
            'Flagged calls remain outside the primary analysis.</p>']
    for r in positives:
        page.append(f'<article><h3>{html.escape(r["task"])} · trace row {r["row_index"]} · '
                    f'sentence {r["sentence_index"]} {"(flagged)" if r["needs_review"] else ""}</h3>'
                    f'<p>{html.escape(r["text"])}</p><small>{html.escape(r["reason"])}</small></article>')
    page.append('</html>')
    (OUT / 'review.html').write_text('\n'.join(page))
    print(json.dumps({k: v for k, v in summary.items() if k != 'by_task'}, indent=2))


if __name__ == '__main__':
    main()
