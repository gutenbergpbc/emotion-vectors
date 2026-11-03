"""Select descriptive upward divergences from saved token-level replay scores."""

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

import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = _REPO / 'runs/emotion-traces/spikes'
HERE.mkdir(parents=True, exist_ok=True)
ROOT = _REPO
ADAPTER = ROOT / 'src/experiments/emotion-traces/task-averages/build_data.py'
ns = {'__file__': str(ADAPTER), '__name__': 'saved_replay_adapters'}
exec(compile(ADAPTER.read_text().split('def trajectory(')[0], str(ADAPTER), 'exec'), ns)
entries, emotions = ns['entries'], ns['EM']
if '--include-new-scores' in sys.argv:
    from load_missing import append_entries
    entries = append_entries(HERE, ns)
allowed = {'hostile', 'ashamed', 'sad', 'guilty', 'calm', 'anxious'}
models = ['Gemma 4B', 'Gemma 27B', 'GPT-OSS 120B', 'GLM 5.3']
tasks = ['TerminalWrench', 'Sales', 'Coding']
coverage, selected, audit, plotrows = [], [], [], []


def curve(e, event, window, smooth):
    """Causal trailing mean; center on the first half of this display window."""
    if event < window:
        return None
    start = max(0, event - window - smooth - 50)
    values = e['scores'][start:event].astype(float)
    mask = e['mask'][start:event]
    baseline = mask[-window:-window // 2]
    if not baseline.any():
        return None
    center = values[-window:-window // 2][baseline].mean(axis=0)
    sums = np.vstack([np.zeros(12), np.cumsum(np.where(mask[:, None], values - center, 0), axis=0)])
    counts = np.r_[0, np.cumsum(mask)]
    i = np.arange(len(values)) + 1
    lo = np.maximum(0, i - smooth)
    n = counts[i] - counts[lo]
    y = np.divide(sums[i] - sums[lo], n[:, None], out=np.full(values.shape, np.nan), where=n[:, None] > 0)
    valid = np.isfinite(y[:, 0])
    previous = np.maximum.accumulate(np.where(valid, np.arange(len(y)), -1))
    y[previous >= 0] = y[previous[previous >= 0]]
    y = y[-window:]
    return y if np.isfinite(y).all() else None


for model in models:
    for task in tasks:
        group = [e for e in entries if e['model'] == model and e['task'] == task]
        hacks = [e for e in group if e['event'] is not None]
        controls = [e for e in group if e['event'] is None]
        window = min(128, int(np.ceil(np.median([e['event'] for e in hacks])))) if hacks else 128
        smooth = max(2, min(32, round(window / 4)))
        late = max(2, round(window / 8))
        candidates = []
        scanned = 0
        scanned_ids = set()
        for h in hacks:
            hy = curve(h, h['event'], window, smooth)
            if hy is None:
                continue
            cc, cm = [], []
            for c in controls:
                if task == 'TerminalWrench' and c['source'] != h['source']:
                    continue
                if model == 'Gemma 4B' and task == 'Coding' and c['subtask'] != h['subtask']:
                    continue
                minimum_event = window
                if model == 'Gemma 4B' and task == 'Coding':
                    retained = np.flatnonzero(c['mask'])
                    if not len(retained):
                        continue
                    minimum_event += int(retained[0])
                event = min(len(c['scores']), max(minimum_event, round(h['event'] / len(h['scores']) * len(c['scores']))))
                cy = curve(c, event, window, smooth)
                if cy is not None:
                    cc.append(cy)
                    cm.append({'trace_id': c['id'], 'source': c['source'], 'pseudo_event': event,
                               'subtask': c['subtask']})
            if not cc:
                continue
            scanned += 1
            scanned_ids.add((h['source'], h['id']))
            average = np.mean(cc, axis=0)
            for k, emotion in enumerate(emotions):
                if emotion not in allowed:
                    continue
                y, c = hy[:, k], average[:, k]
                prior = y[:-2 * late]
                scale = max(float(np.ptp(np.r_[y, c])), 1e-12)
                rise = float(y[-late:].mean() - y[:window // 2].mean())
                gap = float(np.min(y[-late:] - c[-late:]))
                new_peak = float(np.mean(y[-late:]) - np.max(prior))
                strength = min(rise, gap) / scale
                passes = rise > .20 * scale and gap > .20 * scale and new_peak > 0
                record = dict(model=model, task=task, emotion=emotion, hack_id=h['id'],
                              source=h['source'], subtask=h['subtask'], event=h['event'],
                              window_tokens=window, smoothing_tokens=smooth, final_tokens=late,
                              n_controls=len(cc), rise=rise, minimum_late_control_gap=gap,
                              late_mean_over_earlier_max=new_peak, strength=strength, passes=bool(passes))
                if 'anchor_unit_id' in h:
                    record.update(anchor_unit_id=h['anchor_unit_id'], anchor_text=h['anchor_text'],
                                  first_rh_event=h['first_rh_event'], rh_passage_number=h['rh_passage_number'],
                                  tokens_after_first_rh_label=h['event']-h['first_rh_event'],
                                  alignment='selected RH-labeled passage')
                audit.append(record)
                if passes:
                    candidates.append((strength, record, y.copy(), c.copy(), cm))
        status = dict(model=model, task=task, labeled_rh_traces=len({(h['source'], h['id']) for h in hacks}),
                      labeled_nonrh_controls=len(controls), window_tokens=window,
                      scanned_traces=len(scanned_ids), scanned_rh_passages=scanned,
                      qualifying_emotion_trace_pairs=len(candidates))
        if candidates:
            _, r, y, c, cm = max(candidates, key=lambda x: (x[0], x[1]['hack_id'], x[1]['emotion']))
            slug = model.lower().replace(' ', '-').replace('.', '-') + '-' + task.lower()
            r = {**r, 'slug': slug, 'control_audit': cm}
            selected.append(r)
            for x, hval, cval in zip(range(-window, 0), y, c):
                plotrows.append(dict(slug=slug, token=x, rh_projection=float(hval), control_projection=float(cval)))
            status['status'] = 'selected'
            print('SELECT', model, task, r['emotion'], r['hack_id'], 'controls', len(cm), 'strength', round(r['strength'], 3), flush=True)
        else:
            status['status'] = 'unavailable' if not hacks or not controls else 'no_qualifying_rise'
            print('GAP', status, flush=True)
        coverage.append(status)

if '--stronger-coding' in sys.argv:
    replacement = json.loads((HERE / 'stronger-coding-review/candidates.json').read_text())[0]
    assert replacement['clear_separation'] and replacement['n_controls'] == 5
    chosen = {k: v for k, v in replacement.items() if k not in {'rh', 'control'}}
    selected = [chosen if r['slug'] == chosen['slug'] else r for r in selected]
    plotrows = [p for p in plotrows if p['slug'] != chosen['slug']]
    plotrows += [dict(slug=chosen['slug'], token=x, rh_projection=h, control_projection=c)
                 for x, h, c in zip(range(-chosen['window_tokens'], 0), replacement['rh'], replacement['control'])]
    order = {r['slug']: i for i, r in enumerate(selected)}
    plotrows.sort(key=lambda p: (order[p['slug']], p['token']))
    for status in coverage:
        if (status['model'], status['task']) == ('Gemma 4B', 'Coding'):
            status['expanded_passage_search'] = 'stronger-coding-review/audit.json'
            status['selected_passage_number'] = chosen['rh_passage_number']
    print('REPLACE Gemma 4B Coding', chosen['emotion'], chosen['hack_id'], 'later label offset', chosen['tokens_after_first_rh_label'], flush=True)

for filename, data in [('selection.json', selected), ('coverage.json', coverage), ('candidate-audit.json', audit)]:
    (HERE / filename).write_text(json.dumps(data, indent=2) + '\n')
with (HERE / 'plot-data.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(plotrows[0]))
    writer.writeheader()
    writer.writerows(plotrows)
(HERE / 'analysis-provenance.json').write_text(json.dumps({
    'adapter': str(ADAPTER.relative_to(ROOT)),
    'adapter_sha256': hashlib.sha256(ADAPTER.read_bytes()).hexdigest(),
    'include_new_scores': '--include-new-scores' in sys.argv,
    'stronger_coding_reselection': ('Search all saved RH-labeled code-line passages in the 33 same-prompt coding responses; keep the same five controls, 91-token window, 23-token causal smoothing and centering. Require the original upward-spike rule, final-window gap >50% of joint plotted range, control range <40% of joint range, and control rise <30% of RH rise. Prefer first onset among qualifying clear examples, then maximize the original strength statistic. Selected passage is 177 tokens after the first RH label.' if '--stronger-coding' in sys.argv else None),
    'new_score_roots': ([str(p.relative_to(ROOT)) for p in sorted((HERE / 'replays').iterdir())]
                        if '--include-new-scores' in sys.argv else []),
    'selection': 'Exploratory maximum over the six requested emotions and eligible RH traces in each model/dataset cell. GLM TerminalWrench also scans all nine reviewed RH-labeled passages within its one positive trace; the selected passage is explicitly recorded. The original first-onset GLM scan failed the upward-spike rule.',
    'window': 'Minimum of 128 and the ceiling of the cell median pre-onset token count, determined before scanning scores.',
    'smoothing': 'Causal trailing mean over one quarter of the window, capped at 32; exclude punctuation-only and whitespace-only tokens; carry forward across empty windows.',
    'centering': 'Subtract each trajectory’s mean retained-token raw projection in the first half of its display window. Same rule for RH and controls.',
    'controls': 'Every eligible sentence-negative control in the same model/dataset; TerminalWrench also requires the same replay source. New Gemma 4B coding controls are independently sampled expansion episodes and match the RH task prompt (impossible_sort). Each episode contributes once. Pseudo-event at the same fractional position as the RH onset, constrained to a full window. New Gemma coding control pseudo-events also ensure the display window begins at or after the first retained token, preventing punctuation-only leading tokens from excluding an otherwise valid control; this uses token masks, not probe scores.',
    'criterion': 'Final one-eighth window mean rises by >20% of joint plotted range over early RH mean; every final-window point exceeds control average by >20% of that range; late RH mean exceeds its earlier maximum. Rank by min(rise, minimum late control gap) / joint range.',
    'scope': 'Selected examples; not evidence of typicality or validated onset prediction. Controls are sentence-label negatives, not necessarily clean final sales ledgers. Per-panel units and scales differ.',
}, indent=2) + '\n')
