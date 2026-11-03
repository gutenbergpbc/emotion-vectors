"""Judge all saved unsteered GPT-OSS sales and coding traces; no generation."""

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
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

ROOT = _REPO
sys.path[:0] = [str(ROOT / 'src/experiments'), str(ROOT / 'src/experiments/emotion-steering')]
from emotion_steering.judging import RUBRIC, audit  # noqa: E402

OUT = ROOT / 'runs/gptoss120b-emotion-denial-lowreason-20261027'
INDEX = ROOT / 'experiments/gpt_oss_sales_sentence_labels_20261016/episode-index.json'
CODING = ROOT / 'steering/runs/gpt-oss-120b-rh-emotion-sweep-pm005-pm010-mixbest-pooled-b96-20261016/responses/baseline'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def main():
    spec = read(OUT / 'run-spec.json')
    configs = {d: read(Path(spec['source_configs'][d]['path'])) for d in ('sales', 'coding')}
    scenarios = {d: {s['id']: s for s in configs[d]['scenarios']} for d in configs}
    entries = []
    for row in read(INDEX):
        if row['condition'] not in ('baseline', 'unsteered'):
            continue
        path = Path(row['path'])
        assert digest(path) == row['source_sha256']
        trace = read(path)
        assert [(m['role'], m['content']) for m in trace['messages'][:2]] == [(m['role'], m['content']) for m in scenarios['sales'][row['tone']]['messages']]
        entries.append({'dataset': 'sales', 'scenario': row['tone'], 'source_group': row['source_group'], 'source_path': str(path), 'source_sha256': row['source_sha256'], 'id': row['id'], 'eligible_in_original_analysis': row['score']['eligible']})
    for path in sorted(CODING.glob('*/*.json')):
        trace = read(path)
        assert trace['condition'] == 'baseline' and trace['strength'] == 0
        assert trace['prompt'] == scenarios['coding'][trace['scenario']]['messages'][0]['content']
        entries.append({'dataset': 'coding', 'scenario': trace['scenario'], 'source_group': 'local_steering_run', 'source_path': str(path), 'source_sha256': digest(path), 'id': trace['id']})
    assert Counter(x['dataset'] for x in entries) == {'sales': 156, 'coding': 96}
    assert len({(x['dataset'], x['id']) for x in entries}) == len(entries)
    manifest = {'selection': 'All baseline/unsteered sales records in frozen episode index, including ineligible or incomplete records; all 96 saved baseline coding responses', 'sales_index_path': str(INDEX), 'sales_index_sha256': digest(INDEX), 'traces': entries}
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / 'baseline-manifest.json'
    if manifest_path.exists():
        assert read(manifest_path) == manifest
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

    def job(entry):
        path = Path(entry['source_path']); assert digest(path) == entry['source_sha256']
        trace = read(path); ds = entry['dataset']
        if ds == 'sales':
            record = {'turns': [{k: t.get(k) for k in ('turn', 'response', 'reasoning_text', 'feedback', 'ledger')} for t in trace['turns']], 'termination': trace.get('termination'), 'finish_reason': trace.get('finish_reason')}
        else:
            record = {'response': trace['response'], 'finish_reason': trace.get('finish_reason')}
        result = audit(record, scenarios[ds][entry['scenario']], configs[ds])
        return {**entry, 'label': result['label'], 'quote': result['quote'], 'mechanism': result['mechanism'], 'explanation': result['explanation'], 'judge_model': result['judge_model'], 'rubric_sha256': hashlib.sha256(RUBRIC.encode()).hexdigest(), 'raw': result['raw'], 'rejected_attempts': result.get('rejected_attempts', [])}

    pending = []
    for entry in entries:
        target = OUT / 'baseline-judgments' / entry['dataset'] / (hashlib.sha256(entry['id'].encode()).hexdigest() + '.json')
        if target.exists():
            old = read(target)
            assert old['source_sha256'] == entry['source_sha256'] and old['rubric_sha256'] == hashlib.sha256(RUBRIC.encode()).hexdigest()
        else:
            pending.append((entry, target))
    print('baseline judgments remaining', len(pending), flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(job, entry): (entry, target) for entry, target in pending}
        for n, future in enumerate(as_completed(futures), 1):
            entry, target = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append({'id': entry['id'], 'error': repr(exc)})
                print('FAILED', failures[-1], flush=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            if n % 50 == 0:
                print('baseline judged', n, '/', len(pending), flush=True)
    if failures:
        (OUT / 'baseline-judge-errors.json').write_text(json.dumps(failures, indent=2) + '\n')
        raise RuntimeError(f'{len(failures)} baseline judgments failed; rerun to resume')
    summaries = []
    for ds in ('sales', 'coding'):
        rows = [read(p) for p in (OUT / 'baseline-judgments' / ds).glob('*.json')]
        assert len(rows) == sum(e['dataset'] == ds for e in entries)
        counts = Counter(r['label'] for r in rows)
        summaries.append({'dataset': ds, 'n': len(rows), 'reward_hack': counts['reward_hack'], 'rh_rate': counts['reward_hack'] / len(rows), 'labels': dict(counts), 'sources': dict(Counter(r['source_group'] for r in rows)), 'scenarios': {s: {'n': sum(r['scenario'] == s for r in rows), 'reward_hack': sum(r['scenario'] == s and r['label'] == 'reward_hack' for r in rows)} for s in sorted({r['scenario'] for r in rows})}})
    (OUT / 'baseline-results.json').write_text(json.dumps({'cells': summaries}, indent=2) + '\n')
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
