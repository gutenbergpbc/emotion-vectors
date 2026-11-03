"""Label a frozen, scenario-balanced Gemma 3 4B sales trace cohort with GPT-5.5."""
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random

import label_glm_hack_traces as base


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'runs/emotion-rh-traces/gemma3-4b-sales-openrouter-v1/cohort-finalized-150.json'
OUT = ROOT / 'runs/emotion-rh-sentence-labels/gemma3-4b-sales-gpt55-v1'
MODEL = 'openai/gpt-5.5'
TRACE_MODEL = 'google/gemma-3-4b-it'
SEED = 20261022
TARGET_PROSE = 1200


def select() -> dict:
    cohort = base.read_source(SOURCE)
    assert len(cohort['traces']) == 150
    grouped = defaultdict(list)
    for item in cohort['traces']:
        trace = base.read_source(ROOT / item['path'], item['sha256'])
        assert trace['id'] == item['id'] and trace['model_requested'] == TRACE_MODEL
        assert trace['status'] == 'complete'
        prose = sum(unit['kind'] == 'sentence_or_fragment'
                    for turn in trace['turns'] for field in ('reasoning_text', 'response')
                    for unit in base.segment(turn.get(field) or ''))
        if prose:
            grouped[trace['scenario']].append({**item, 'dataset': 'sales',
                                               'scenario': trace['scenario'], 'prose_units': prose})
    assert set(grouped) == {'hostile', 'neutral', 'appreciative'}
    rng = random.Random(SEED)
    for group in grouped.values():
        group.sort(key=lambda row: row['id'])
        rng.shuffle(group)
    selected, total = [], 0
    for index in range(min(map(len, grouped.values()))):
        for scenario in sorted(grouped):
            item = grouped[scenario][index]
            selected.append(item)
            total += item['prose_units']
            if total >= TARGET_PROSE:
                break
        if total >= TARGET_PROSE:
            break
    assert total >= TARGET_PROSE and len({item['id'] for item in selected}) == len(selected)
    return {'model': MODEL, 'trace_model': TRACE_MODEL, 'seed': SEED,
            'target_prose_units': TARGET_PROSE, 'selected_prose_units': total,
            'source_cohort_sha256': base.sha(SOURCE.read_bytes()),
            'selection_rule': 'Seeded shuffle within each of three sales scenarios; round-robin complete traces with at least one prose unit until >=1200 prose units. No RH outcome or judge label used.',
            'selected': selected}


def summarize(cases: list[dict], selection: dict) -> dict:
    traces = []
    for case in cases:
        labels = []
        for chunk in case['chunks']:
            key = case['id'].replace('/', '__') + f"__{chunk['start']:04d}"
            path = OUT / 'annotations' / f'{key}.json'
            if not path.exists():
                break
            result = base.read_source(path)
            assert result['request_sha256'] == chunk['request_sha256']
            labels.extend(base.validate_labels({'labels': result['labels']}, chunk['units']))
        if len(labels) == len(case['units']):
            traces.append({'id': case['id'], 'dataset': 'sales', 'scenario': case['scenario'],
                           'source_path': case['path'], 'source_sha256': case['sha256'],
                           'units': [{**unit, **{k: labels[i][k] for k in ('hacking', 'needs_review', 'reason')}}
                                     for i, unit in enumerate(case['units'])]})
    counts = Counter()
    for trace in traces:
        counts[(trace['scenario'], 'traces')] += 1
        for unit in trace['units']:
            counts[(trace['scenario'], 'all_units')] += 1
            if unit['kind'] == 'sentence_or_fragment':
                counts[(trace['scenario'], 'prose_units')] += 1
                counts[(trace['scenario'], 'rh_prose_units')] += unit['hacking']
                counts[(trace['scenario'], 'review_prose_units')] += unit['needs_review']
    by_scenario = {s: {k: counts[(s, k)] for k in
                       ('traces', 'all_units', 'prose_units', 'rh_prose_units', 'review_prose_units')}
                   for s in sorted({case['scenario'] for case in cases})}
    billed = {}
    for path in (OUT / 'attempts').rglob('*.json'):
        response = base.read_source(path).get('response') or {}
        billed[response.get('id') or str(path)] = (response.get('usage') or {}).get('cost', 0) or 0
    summary = {'status': 'complete' if len(traces) == len(cases) else 'partial',
               'judge_model': MODEL, 'trace_model': TRACE_MODEL,
               'expected_traces': len(cases), 'labeled_traces': len(traces),
               'expected_prose_units': selection['selected_prose_units'],
               'labeled_prose_units': sum(x['prose_units'] for x in by_scenario.values()),
               'rh_prose_units': sum(x['rh_prose_units'] for x in by_scenario.values()),
               'review_prose_units': sum(x['review_prose_units'] for x in by_scenario.values()),
               'by_scenario': by_scenario,
               'reported_api_cost_usd_including_retries': round(sum(billed.values()), 6),
               'rubric_sha256': base.sha(base.RUBRIC.encode()),
               'method': 'Full-trace GPT-5.5 judgments of every assistant-output unit. Prose units exclude code blocks and headings. Selection independent of RH outcomes.',
               'caveat': 'Model judgments, not human ground truth; sales traces include instructed emotional styles and the source sample is not prevalence-representative.'}
    base.write(OUT / 'summary.json', summary)
    base.write(OUT / 'labels.json', {'summary': summary, 'traces': traces})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--limit-requests', type=int)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    base.OUT = OUT
    base.MODEL = MODEL
    selection = select()
    selection_path = OUT / 'selection.json'
    if selection_path.exists():
        assert base.read_source(selection_path) == selection, 'Frozen selection changed'
    else:
        base.write(selection_path, selection)
    rubric = OUT / 'rubric.txt'
    if rubric.exists():
        assert rubric.read_text() == base.RUBRIC + '\n'
    else:
        rubric.write_text(base.RUBRIC + '\n')
    cases = base.prepare(selection['selected'], expected_model=TRACE_MODEL)
    assert sum(u['kind'] == 'sentence_or_fragment' for c in cases for u in c['units']) == selection['selected_prose_units']
    manifest = {'model': MODEL, 'trace_model': TRACE_MODEL,
                'selection_sha256': base.sha(selection_path.read_bytes()),
                'rubric_sha256': base.sha(base.RUBRIC.encode()),
                'traces': len(cases), 'prose_units': selection['selected_prose_units'],
                'all_units': sum(len(c['units']) for c in cases),
                'requests': [{'trace_id': c['id'], 'start': ch['start'],
                              'request_sha256': ch['request_sha256']}
                             for c in cases for ch in c['chunks']]}
    manifest_path = OUT / 'manifest.json'
    if manifest_path.exists():
        assert base.read_source(manifest_path) == manifest, 'Frozen requests changed'
    else:
        base.write(manifest_path, manifest)
    print(json.dumps({k: manifest[k] for k in ('traces', 'prose_units', 'all_units')}
                     | {'requests': len(manifest['requests'])}, indent=2), flush=True)
    if args.prepare_only:
        return
    jobs = [(case, chunk) for case in cases for chunk in case['chunks']]
    if args.limit_requests is not None:
        jobs = jobs[:args.limit_requests]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(base.label_chunk, case, chunk): (case['id'], chunk['start'])
                   for case, chunk in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                failures.append({'job': futures[future], 'error': str(exc)})
                print(f"FAILED {futures[future]}: {exc}", flush=True)
            if index % 10 == 0 or index == len(jobs):
                print(f'{index}/{len(jobs)} requests finished; {len(failures)} failed', flush=True)
    base.write(OUT / 'failures.json', failures)
    summary = summarize(cases, selection)
    print(json.dumps(summary, indent=2), flush=True)
    if failures or (args.limit_requests is None and summary['status'] != 'complete'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
