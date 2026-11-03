"""Materialize the frozen, sentence-count-matched GLM training/evaluation cohort."""
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


from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'runs/emotion-rh-sentence-labels'
OUT = BASE / 'glm53-balanced-source-gpt55-v1'
SOURCES = [('hack', BASE / 'glm53-hack-traces-gpt55-v1/labels.json'),
           ('nonhack', BASE / 'glm53-nonhack-traces-gpt55-v1/labels.json')]


def main():
    traces, source_refs, rubric_hashes = [], [], set()
    counts = Counter()
    for cohort, path in SOURCES:
        raw = path.read_bytes()
        data = json.loads(raw)
        assert data['summary']['status'] == 'complete'
        rubric_hashes.add(data['summary']['rubric_sha256'])
        source_refs.append({'cohort': cohort, 'path': str(path.relative_to(ROOT)),
                            'sha256': hashlib.sha256(raw).hexdigest()})
        for trace in data['traces']:
            row = {**trace, 'source_cohort': cohort}
            traces.append(row)
            counts[(cohort, trace['dataset'], 'traces')] += 1
            for unit in trace['units']:
                if unit['kind'] == 'sentence_or_fragment':
                    counts[(cohort, trace['dataset'], 'prose')] += 1
                    counts[(cohort, trace['dataset'], 'rh_prose')] += unit['hacking']
    assert len(rubric_hashes) == 1
    assert len({t['id'] for t in traces}) == len(traces)
    for dataset, target in [('sales', 2413), ('coding', 153)]:
        assert counts[('hack', dataset, 'prose')] == counts[('nonhack', dataset, 'prose')] == target
    result = {'summary': {'status': 'complete', 'model': 'openai/gpt-5.5',
                          'rubric_sha256': next(iter(rubric_hashes)), 'sources': source_refs,
                          'sampling': 'seeded whole-trace, exactly matched prose counts within sales and coding',
                          'counts': {cohort: {dataset: {metric: counts[(cohort, dataset, metric)]
                                                         for metric in ('traces', 'prose', 'rh_prose')}
                                             for dataset in ('sales', 'coding')}
                                     for cohort, _ in SOURCES},
                          'caveat': 'Outcome-stratified convenience sample, not population prevalence. GPT-5.5 labels use full-trace context.'},
              'traces': sorted(traces, key=lambda t: t['id'])}
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / 'labels.json'
    content = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    if dest.exists():
        assert dest.read_text() == content, 'Frozen combined labels changed'
    else:
        dest.write_text(content)
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
