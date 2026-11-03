"""Search saved Gemma coding passages for large separation and flat controls."""

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

import ast
import functools
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = _REPO
OUT = HERE / 'stronger-coding-review'
OUT.mkdir(exist_ok=True)
EM = ['afraid', 'anxious', 'appreciative', 'ashamed', 'calm', 'desperate', 'excited', 'frustrated', 'guilty', 'hostile', 'sad', 'surprised']
ALLOWED = {'hostile', 'ashamed', 'sad', 'guilty', 'calm', 'anxious'}
tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-4b-it', revision='093f9f388b31de276ce2de164bdc2081324b9767')


@functools.lru_cache(None)
def keep_id(i):
    s = tokenizer.decode([int(i)], skip_special_tokens=True)
    return bool(s) and any(not (c.isspace() or unicodedata.category(c).startswith('P')) for c in s)


node = next(n for n in ast.parse((HERE / 'select.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'curve')
scope = {'np': np}
exec(compile(ast.Module(body=[node], type_ignores=[]), 'saved_curve', 'exec'), scope)
curve = scope['curve']
controls = []
folder = HERE / 'replays/gemma4b-coding-controls-v1'
for job in json.loads((folder / 'replay-spec.json').read_text())['jobs']:
    j = json.loads((folder / 'inputs' / f"{job['id']}.json").read_text())
    values = np.load(folder / 'scores' / f"{job['id']}.npz")['scores']
    start = len(values) - j['output_tokens']
    controls.append(dict(id=j['trace_id'], subtask=j['subtask'], scores=values[start:],
                         mask=np.array([keep_id(t) for t in j['token_ids'][start:]], dtype=bool)))

source = ROOT / 'runs/emotion-rh-replay/gemma3-4b-coding-lines-best12-v1'
window, smooth, late = 91, 23, 11
audit, candidates, seen = [], [], set()
for job in json.loads((source / 'replay-spec.json').read_text())['jobs']:
    j = json.loads((source / 'inputs' / f"{job['id']}.json").read_text())
    if j['trace_id'].split('/')[1] != 'impossible_sort':
        continue
    values = np.load(source / 'scores' / f"{job['id']}.npz")['scores']
    start = len(values) - j['output_tokens']
    h = dict(scores=values[start:], mask=np.array([keep_id(t) for t in j['token_ids'][start:]], dtype=bool))
    units = [u for u in j['units'] if u['hacking'] and not u['needs_review'] and u.get('token_indices')]
    first = min(min(u['token_indices']) for u in units) - start
    for number, unit in enumerate(units, 1):
        event = min(unit['token_indices']) - start
        if (j['trace_id'], event) in seen:
            continue
        seen.add((j['trace_id'], event))
        hy = curve(h, event, window, smooth)
        if hy is None:
            continue
        cc, meta = [], []
        for c in controls:
            ev = min(len(c['scores']), max(window + int(np.flatnonzero(c['mask'])[0]), round(event / len(h['scores']) * len(c['scores']))))
            cy = curve(c, ev, window, smooth)
            if cy is not None:
                cc.append(cy)
                meta.append(dict(trace_id=c['id'], source='gemma4b-coding-controls-v1', pseudo_event=ev, subtask=c['subtask']))
        assert len(cc) == 5
        average = np.mean(cc, axis=0)
        for k, emotion in enumerate(EM):
            if emotion not in ALLOWED:
                continue
            y, c = hy[:, k], average[:, k]
            span = max(float(np.ptp(np.r_[y, c])), 1e-12)
            rise = float(y[-late:].mean() - y[:window//2].mean())
            gap = float(np.min(y[-late:] - c[-late:]))
            peak = float(y[-late:].mean() - y[:-2*late].max())
            crange = float(np.ptp(c))
            cr = float(c[-late:].mean() - c[:window//2].mean())
            passes = rise > .2*span and gap > .2*span and peak > 0
            clear = passes and gap > .5*span and crange < .4*span and cr < .3*rise
            strength = min(rise, gap) / span
            r = dict(model='Gemma 4B', task='Coding', emotion=emotion, hack_id=j['trace_id'],
                     source=source.name, subtask='impossible_sort', event=event,
                     window_tokens=window, smoothing_tokens=smooth, final_tokens=late,
                     n_controls=5, rise=rise, minimum_late_control_gap=gap,
                     late_mean_over_earlier_max=peak, strength=strength, passes=bool(passes),
                     clear_separation=bool(clear), control_range_fraction=crange/span,
                     late_gap_fraction=gap/span, control_rise=cr,
                     anchor_unit_id=unit['unit_id'], anchor_text=unit['text'], first_rh_event=first,
                     rh_passage_number=number, tokens_after_first_rh_label=event-first,
                     alignment='first RH onset' if event == first else 'selected RH-labeled passage',
                     slug='gemma-4b-coding', control_audit=meta)
            audit.append(r)
            if passes:
                candidates.append((r, y.copy(), c.copy()))
candidates.sort(key=lambda x: (not x[0]['clear_separation'], x[0]['tokens_after_first_rh_label'] != 0,
                               -x[0]['strength'], x[0]['control_range_fraction']))
(OUT / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
top = [{**r, 'rh': y.tolist(), 'control': c.tolist()} for r, y, c in candidates[:12]]
(OUT / 'candidates.json').write_text(json.dumps(top, indent=2) + '\n')
print('scanned', len(audit), 'qualifying', len(candidates), 'clear', sum(r['clear_separation'] for r, _, _ in candidates), flush=True)
for r, y, c in candidates[:8]:
    print({k:r[k] for k in ['hack_id','emotion','rh_passage_number','tokens_after_first_rh_label','strength','late_gap_fraction','control_range_fraction','clear_separation']}, flush=True)
