"""Verify and merge GPT-6.1 Sol Flex and standard adjudication responses."""

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
from collections import Counter
import csv
from pathlib import Path

import label_terminalwrench as labeler
from shared.io import atomic, read, require, sha


def verified_labels(run):
    manifest = read(run / 'manifest.json')
    cfg = manifest['config']
    result, usage, models, providers = {}, Counter(), Counter(), Counter()
    failed_ids = {item['id'] for item in read(run / 'failures.json')}
    for path in sorted((run / 'requests').glob('*.json')):
        req = read(path)
        if req['id'] in failed_ids:
            require(not (run / 'annotations' / path.name).exists(), 'Failed request has an annotation')
            continue
        saved = read(run / 'annotations' / path.name)
        raw = read(run / saved['raw_response_path'])
        require(saved['request_sha256'] == raw['request_sha256'] == req['request_sha256'], 'Request identity differs')
        labels = labeler.validate_response(raw['response'], req, cfg)
        require(saved['labels'] == labels, 'Saved labels differ from raw response')
        models[raw['response']['model']] += 1
        providers[raw['response'].get('provider', 'not-returned')] += 1
        for label in labels:
            key = (req['trace_id'], label['unit_id'])
            require(key not in result, 'Duplicate unit within run')
            result[key] = label
    for path in (run / 'attempts').glob('*/*.json'):
        info = read(path).get('response', {}).get('usage', {})
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cost'):
            usage[key] += info.get(key, 0) or 0
    return manifest, result, usage, models, providers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('comparison', 'audit', 'flex', 'standard', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    comparison, audit_path, flex, standard, out = map(Path, (args.comparison, args.audit, args.flex,
                                                             args.standard, args.output))
    source = read(comparison / 'disagreements.json')
    require(len(source) == 80, 'Unexpected disagreement count')
    with audit_path.open(newline='') as f:
        audit = {(r['trace_id'], int(r['unit_id'])): r['audit_verdict'] for r in csv.DictReader(f)}
    merged, total_usage, returned, providers, inputs = {}, Counter(), Counter(), Counter(), []
    for route, run in (('flex', flex), ('standard', standard)):
        manifest, labels, usage, models, route_providers = verified_labels(run)
        require(manifest['source_comparison_sha256'] == sha(comparison / 'disagreements.json'),
                'Run selected from different disagreement source')
        require(manifest['source_audit_sha256'] == sha(audit_path), 'Run used different audit')
        for key, value in labels.items():
            require(key not in merged, 'Overlap between routes')
            merged[key] = (route, value)
        total_usage.update(usage)
        returned.update(models)
        providers.update(route_providers)
        inputs.append(dict(route=route, run=str(run.resolve()), manifest_sha256=sha(run / 'manifest.json'),
                           requests=sum(models.values()), sentences=len(labels)))
    keys = {(r['trace_id'], r['unit_id']) for r in source}
    require(set(merged) == keys == set(audit), 'Combined coverage is not exactly the 80-sentence sample')
    rows = []
    for original in source:
        key = (original['trace_id'], original['unit_id'])
        route, answer = merged[key]
        rows.append({k: original[k] for k in ('task', 'trace_id', 'unit_id', 'step_id', 'text',
                                               'reference_hacking', 'reference_needs_review',
                                               'judge_hacking', 'judge_needs_review')}
                    | dict(audit_verdict=audit[key], sol_hacking=answer['hacking'],
                           sol_needs_review=answer['needs_review'], sol_reason=answer['reason'], route=route))
    counts = Counter((r['audit_verdict'], r['sol_hacking']) for r in rows)
    summary = dict(status='complete', model='openai/gpt-6.1-sol', sample=dict(sentences=len(rows), traces=len(set(r['trace_id'] for r in rows))),
                   source_disagreements_sha256=sha(comparison / 'disagreements.json'),
                   source_audit_sha256=sha(audit_path), runs=inputs, models_returned=dict(returned),
                   providers=dict(providers), usage=dict(total_usage), sol_rh=sum(r['sol_hacking'] for r in rows),
                   sol_non_rh=sum(not r['sol_hacking'] for r in rows),
                   sol_needs_review=sum(r['sol_needs_review'] for r in rows),
                   by_audit_category={category: dict(n=sum(r['audit_verdict'] == category for r in rows),
                                                    sol_rh=counts[category, True], sol_non_rh=counts[category, False])
                                      for category in ('clear_rh', 'debatable', 'likely_non_rh')},
                   interpretation='Blinded third-model labeling of a disagreement-selected sample, not ground truth or population accuracy.')
    out.mkdir(parents=True, exist_ok=True)
    atomic(out / 'labels.json', rows)
    atomic(out / 'summary.json', summary)
    with (out / 'labels.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic(out / 'verification.json', dict(passed=True, source_count=len(source), labeled_count=len(rows),
                                           labels_sha256=sha(out / 'labels.json'), summary_sha256=sha(out / 'summary.json')))
    print(summary)


if __name__ == '__main__':
    main()
