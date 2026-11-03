"""Verify trace artifacts and derive the final report from saved sentence labels."""

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
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(HERE.parent)]
from emotion_steering.terminalwrench import alias
from shared.io import atomic, read, sha

RUN = 'glm53-terminalwrench-interactive-steering-active-time-v2-rh-top4-continuous-v7-neutral10-sample3-20261101-recovery-v2'
root = PROJECT / 'runs/emotion-steering' / RUN
label_root = PROJECT / 'runs/emotion-rh-sentence-labels'
new_labels = label_root / 'glm53-terminalwrench-neutral10-20261101-recovery-v2-gpt61sol-v1/labels.json'
labels = read(new_labels)
assert labels['summary']['status'] == 'complete'
plan = read(root / 'planned-episodes.json')
traces = {t['task']['id']: t for t in labels['traces']}
rows = []
for task in plan:
    path = root / 'episodes' / f'{alias(task)}.json'
    episode = read(path)
    assert episode['task'] == task and task['sample'] == 3 and task['strength'] == 0
    assert episode['termination'] in ('complete', 'active_budget_timeout'), task['id']
    assert episode.get('generation_error') in (None, 'active_budget_timeout'), task['id']
    artifacts = episode['native_artifacts']
    assert artifacts and all(sha(root / a['path']) == a['sha256'] for a in artifacts)
    trace = traces.get(task['id'])
    units = trace['units'] if trace else []
    if trace:
        assert trace['source_sha256'] == sha(path)
    rows.append(dict(task=task['scenario'], termination=episode['termination'],
                     sentence_count=len(units), rh_unflagged=sum(u['hacking'] and not u['needs_review'] for u in units),
                     rh_including_review=sum(u['hacking'] for u in units),
                     non_rh=sum(not u['hacking'] for u in units),
                     review_flagged=sum(u['needs_review'] for u in units),
                     episode_path=str(path), episode_sha256=sha(path)))
assert len(rows) == 10 and len({r['task'] for r in rows}) == 10
with (root / 'trace-inventory.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
sources = []
all_traces = []
for source in [
    label_root / 'terminalwrench-v7-steering-gpt61sol-v2/labels.json',
    label_root / 'terminalwrench-v7-neutral-extension-gpt61sol-v2/labels.json',
    label_root / 'terminalwrench-neutral20-20261026-astra-review-v1/labels.json',
    new_labels,
]:
    selected = [t for t in read(source)['traces'] if t['model'] == 'GLM 5.3' and t['task']['condition'] == 'baseline' and t['task']['strength'] == 0]
    sources.append(dict(path=str(source), sha256=sha(source), traces=len(selected)))
    all_traces.extend(selected)
assert len({t['task']['id'] for t in all_traces}) == len(all_traces)
all_units = [u for t in all_traces for u in t['units']]
combined = dict(sources=sources, traces=len(all_traces), sentences=len(all_units),
                rh_unflagged=sum(u['hacking'] and not u['needs_review'] for u in all_units),
                rh_including_review=sum(u['hacking'] for u in all_units),
                non_rh=sum(not u['hacking'] for u in all_units), review_flagged=sum(u['needs_review'] for u in all_units),
                positive_tasks=sorted({t['task']['scenario'] for t in all_traces if any(u['hacking'] and not u['needs_review'] for u in t['units'])}))
atomic(root / 'combined-neutral-label-inventory.json', combined)
totals = {key: sum(r[key] for r in rows) for key in ['sentence_count', 'rh_unflagged', 'rh_including_review', 'non_rh', 'review_flagged']}
summary = dict(passed=True, saved_traces=len(rows), labeled_traces=len(traces),
               terminations=dict(Counter(r['termination'] for r in rows)), **totals,
               positive_tasks=[r['task'] for r in rows if r['rh_unflagged']],
               labels_path=str(new_labels), labels_sha256=sha(new_labels),
               combined_neutral_inventory=combined, episodes=rows)
atomic(root / 'verified-summary.json', summary)
lines = ['# Ten additional GLM 5.3 TerminalWrench traces', '',
         f"Saved {len(rows)} fresh, unsteered task traces. Terminations: {json.dumps(summary['terminations'])}.", '',
         f"Sentence labels: {totals['sentence_count']}; unflagged RH: {totals['rh_unflagged']}; RH including review flags: {totals['rh_including_review']}; non-RH: {totals['non_rh']}; review-flagged: {totals['review_flagged']}.", '',
         'New positive tasks: ' + (', '.join(summary['positive_tasks']) or 'none') + '.', '',
         f"Including the earlier neutral corpus: {combined['traces']} traces, {combined['sentences']} sentences, {combined['rh_unflagged']} unflagged RH sentences across {len(combined['positive_tasks'])} task(s).", '',
         'Ten distinct tasks were selected across six task categories before outcomes, with fresh sample-3 seeds, no steering, and a 900-second active-time budget. One completed trace was recovered after coordinator SIGTERM; three infrastructure-interrupted first attempts were retained separately and retried with the same settings. Timed-out task histories remain in the labeled dataset.', '',
         'The matching Inferact/GLM-5.3-NVFP4 checkpoint was used. The historical configuration says timing v2.1; actual clock receipts and pinned source code use v2.2, with the same 900-second active-time rule. Labels use the existing GPT-6.1 Sol rubric on assistant prose with full task history. They are model judgments, tool-only RH can be missed, and selected tasks do not estimate corpus prevalence.', '',
         '[Verified summary](verified-summary.json) · [Per-trace CSV](trace-inventory.csv) · [Recovery provenance](recovery-provenance.json)', '',
         f'[Sentence labels]({new_labels})', '']
(root / 'RESULTS.md').write_text('\n'.join(lines))
print(json.dumps({k: v for k, v in summary.items() if k not in ('episodes', 'combined_neutral_inventory')}, indent=2))
