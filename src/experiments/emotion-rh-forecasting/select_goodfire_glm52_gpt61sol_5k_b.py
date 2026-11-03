"""Freeze a second non-overlapping 5,000-sentence Goodfire GLM 5.2 cohort."""
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


from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path('/scratch/model-forensics/goodfire/deepswe-glm-5.2-fp8')
PILOT = ROOT / 'runs/emotion-rh-sentence-labels/goodfire-glm52-deepswe-gpt61sol-pilot-v1'
FIRST = ROOT / 'runs/emotion-rh-sentence-labels/goodfire-glm52-deepswe-gpt61sol-5k-v1'
OUT = ROOT / 'runs/emotion-rh-sentence-labels/goodfire-glm52-deepswe-gpt61sol-5k-b-v1'


def main():
    pilot = json.loads((PILOT / 'labels.json').read_text())
    first = json.loads((FIRST / 'selection.json').read_text())
    used_rows = {t['row_index'] for t in pilot['traces']} | {t['row_index'] for t in first['traces']}
    used_tasks = {t['task'] for t in pilot['traces']} | {t['task'] for t in first['traces']}
    summary = json.loads((SOURCE / 'sentence-index-summary.json').read_text())
    by_task = defaultdict(list)
    for row in summary['per_trace']:
        if row['row_index'] not in used_rows and row['positive_passage_sentences'] >= 125:
            by_task[row['task']].append(row)
    first_per_task = [min(rows, key=lambda r: (r['sentences'], r['row_index']))
                      for rows in by_task.values()]
    require_40 = 40 - len(first_per_task)
    assert 0 <= require_40 <= 20
    first_ids = {r['row_index'] for r in first_per_task}
    extras = sorted((r for rows in by_task.values() for r in rows if r['row_index'] not in first_ids),
                    key=lambda r: (r['sentences'], r['row_index']))[:require_40]
    chosen = sorted(first_per_task + extras, key=lambda r: r['row_index'])
    assert len(chosen) == 40 and len({r['row_index'] for r in chosen}) == 40
    wanted = {r['row_index'] for r in chosen}
    passages = {}
    with (SOURCE / 'judge_passages.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['row_index'] in wanted and row['consensus'] == 'positive' and row['channel'] in ('reasoning', 'content'):
                passages[(row['row_index'], row['node_index'], row['channel'])] = row
    by_trace = defaultdict(lambda: defaultdict(list))
    with (SOURCE / 'sentence-index.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['row_index'] in wanted and row['passage_consensus'] == 'positive':
                key = (row['node_index'], row['channel'])
                assert (row['row_index'], *key) in passages
                by_trace[row['row_index']][key].append(row)
    line_hashes = {}
    with (SOURCE / 'traces.jsonl').open('rb') as stream:
        for index, raw in enumerate(stream):
            if index in wanted:
                line_hashes[index] = hashlib.sha256(raw).hexdigest()
    selected, trace_meta = [], []
    for trace in chosen:
        row_index = trace['row_index']
        queues = []
        for key, units in sorted(by_trace[row_index].items()):
            passage = passages[(row_index, *key)]
            units.sort(key=lambda u: u['sentence_index'])
            quotes = [q.strip() for q in passage['carrier_quotes']
                      if isinstance(q, str) and len(q.strip()) >= 16]
            matches = [i for i, unit in enumerate(units)
                       if any(q in unit['text'] or (len(unit['text']) >= 16 and unit['text'] in q)
                              for q in quotes)]
            center = matches[0] if matches else len(units) // 2
            queues.append([unit for _, unit in sorted(enumerate(units),
                          key=lambda pair: (abs(pair[0] - center), pair[1]['sentence_index']))])
        picked = []
        offset = 0
        while len(picked) < 125:
            progressed = False
            for queue in queues:
                if offset < len(queue):
                    picked.append(queue[offset])
                    progressed = True
                    if len(picked) == 125:
                        break
            if not progressed:
                raise RuntimeError(f'Insufficient source-positive sentences: {row_index}')
            offset += 1
        assert len({u['sentence_index'] for u in picked}) == 125
        picked.sort(key=lambda u: u['sentence_index'])
        selected.extend({**u, 'source_line_sha256': line_hashes[row_index]} for u in picked)
        trace_meta.append({**trace, 'source_line_sha256': line_hashes[row_index],
                           'selected_sentences': 125,
                           'selected_passages': len({(u['node_index'], u['channel']) for u in picked})})
    assert len(selected) == 5000
    selected.sort(key=lambda u: (u['row_index'], u['sentence_index']))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'selection.json').write_text(json.dumps({
        'source': 'Goodfire/reward-hack-data/deepswe-glm-5.2-fp8',
        'source_revision': summary['source_revision'],
        'source_index_sha256': summary['index_sha256'],
        'source_traces_sha256': summary['source_sha256'],
        'rule': 'Exclude all pilot and first-5k traces. From remaining traces with >=125 source-positive prose sentences, choose shortest trace per task, then shortest additional traces to reach 40; select 125 spans per trace round-robin near carrier quotes.',
        'traces': trace_meta, 'sentences': 5000,
        'previous_selection_sha256': hashlib.sha256((FIRST / 'selection.json').read_bytes()).hexdigest(),
    }, ensure_ascii=False, indent=2) + '\n')
    with (OUT / 'selected-sentences.jsonl').open('w') as stream:
        for unit in selected:
            stream.write(json.dumps(unit, ensure_ascii=False) + '\n')
    print(json.dumps({'traces': len(chosen), 'tasks': len({t['task'] for t in chosen}),
                      'new_tasks': len({t['task'] for t in chosen if t['task'] not in used_tasks}),
                      'sentences': len(selected),
                      'selected_passages': sum(t['selected_passages'] for t in trace_meta),
                      'source_trace_sentences': sum(t['sentences'] for t in trace_meta)}, indent=2))


if __name__ == '__main__':
    main()
