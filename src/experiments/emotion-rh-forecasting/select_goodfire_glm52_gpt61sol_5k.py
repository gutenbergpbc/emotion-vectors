"""Freeze 5,000 source-positive-passage sentences from 50 new DeepSWE tasks."""
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
OLD = ROOT / 'runs/emotion-rh-sentence-labels/goodfire-glm52-deepswe-gpt61sol-pilot-v1'
OUT = ROOT / 'runs/emotion-rh-sentence-labels/goodfire-glm52-deepswe-gpt61sol-5k-v1'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    used = {t['task'] for t in json.loads((OLD / 'labels.json').read_text())['traces']}
    summary = json.loads((SOURCE / 'sentence-index-summary.json').read_text())
    by_task = defaultdict(list)
    for row in summary['per_trace']:
        if row['task'] not in used and row['positive_passage_sentences'] >= 100:
            by_task[row['task']].append(row)
    # One trace per task; favor shorter traces to keep later activation replay feasible.
    chosen = sorted((min(rows, key=lambda r: (r['sentences'], r['row_index']))
                     for rows in by_task.values()), key=lambda r: (r['sentences'], r['row_index']))[:50]
    assert len(chosen) == 50 and len({r['task'] for r in chosen}) == 50
    wanted = {r['row_index'] for r in chosen}
    passage_rows = {}
    with (SOURCE / 'judge_passages.jsonl').open() as stream:
        for line in stream:
            p = json.loads(line)
            if p['row_index'] in wanted and p['consensus'] == 'positive' and p['channel'] in ('reasoning', 'content'):
                passage_rows[(p['row_index'], p['node_index'], p['channel'])] = p
    by_trace = defaultdict(lambda: defaultdict(list))
    with (SOURCE / 'sentence-index.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['row_index'] in wanted and row['passage_consensus'] == 'positive':
                key = (row['node_index'], row['channel'])
                assert (row['row_index'], *key) in passage_rows
                by_trace[row['row_index']][key].append(row)
    line_hashes = {}
    with (SOURCE / 'traces.jsonl').open('rb') as stream:
        for i, raw in enumerate(stream):
            if i in wanted:
                line_hashes[i] = digest(raw)

    selected = []
    trace_meta = []
    for trace in chosen:
        row_index = trace['row_index']
        queues = []
        for key, units in sorted(by_trace[row_index].items()):
            passage = passage_rows[(row_index, *key)]
            units.sort(key=lambda u: u['sentence_index'])
            quotes = [q.strip() for q in passage['carrier_quotes'] if isinstance(q, str) and len(q.strip()) >= 16]
            matches = [i for i, unit in enumerate(units)
                       if any(q in unit['text'] or (len(unit['text']) >= 16 and unit['text'] in q)
                              for q in quotes)]
            center = matches[0] if matches else len(units) // 2
            queues.append([unit for _, unit in sorted(enumerate(units),
                          key=lambda pair: (abs(pair[0] - center), pair[1]['sentence_index']))])
        picked = []
        offset = 0
        while len(picked) < 100:
            progressed = False
            for queue in queues:
                if offset < len(queue):
                    picked.append(queue[offset])
                    progressed = True
                    if len(picked) == 100:
                        break
            if not progressed:
                raise RuntimeError(f'Insufficient positive-passage sentences: {row_index}')
            offset += 1
        assert len({x['sentence_index'] for x in picked}) == 100
        picked.sort(key=lambda u: u['sentence_index'])
        for unit in picked:
            selected.append({**unit, 'source_line_sha256': line_hashes[row_index]})
        trace_meta.append({**trace, 'source_line_sha256': line_hashes[row_index],
                           'selected_sentences': 100,
                           'selected_passages': len({(u['node_index'], u['channel']) for u in picked})})
    assert len(selected) == 5000 and len({(u['row_index'], u['sentence_index']) for u in selected}) == 5000
    selected.sort(key=lambda u: (u['row_index'], u['sentence_index']))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'selection.json').write_text(json.dumps({
        'source': 'Goodfire/reward-hack-data/deepswe-glm-5.2-fp8',
        'source_revision': summary['source_revision'],
        'source_index_sha256': summary['index_sha256'],
        'source_traces_sha256': summary['source_sha256'],
        'rule': '50 shortest distinct previously unused tasks with >=100 source-positive prose sentences; one trace per task, 100 spans per trace, round-robin near verified carrier quotes across positive passages.',
        'traces': trace_meta, 'sentences': 5000,
    }, ensure_ascii=False, indent=2) + '\n')
    with (OUT / 'selected-sentences.jsonl').open('w') as stream:
        for unit in selected:
            stream.write(json.dumps(unit, ensure_ascii=False) + '\n')
    print(json.dumps({'traces': len(chosen), 'sentences': len(selected),
                      'selected_passages': sum(t['selected_passages'] for t in trace_meta),
                      'source_trace_sentences': sum(t['sentences'] for t in trace_meta)}, indent=2))


if __name__ == '__main__':
    main()
