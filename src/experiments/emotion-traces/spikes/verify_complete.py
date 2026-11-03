"""Validate the completed grid, new replay receipts, and frozen original curves."""

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

import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = _REPO
rows = json.loads((HERE / 'selection.json').read_text())
expected = {(m, d) for m in ['Gemma 4B', 'Gemma 27B', 'GPT-OSS 120B', 'GLM 5.3']
            for d in ['TerminalWrench', 'Sales', 'Coding']}
assert len(rows) == 12 and {(r['model'], r['task']) for r in rows} == expected
glm = next(r for r in rows if (r['model'], r['task']) == ('GLM 5.3', 'TerminalWrench'))
assert glm['rh_passage_number'] == 2 and glm['tokens_after_first_rh_label'] == 28 and glm['n_controls'] == 19
gemma = next(r for r in rows if (r['model'], r['task']) == ('Gemma 4B', 'Coding'))
assert gemma['n_controls'] == 5 and all(c['subtask'] == gemma['subtask'] for c in gemma['control_audit'])
if gemma.get('clear_separation'):
    assert gemma['late_gap_fraction'] > .5 and gemma['control_range_fraction'] < .4
    assert gemma['control_rise'] < .3 * gemma['rise']
    assert gemma['tokens_after_first_rh_label'] == 177
curves = list(csv.DictReader((HERE / 'plot-data.csv').open()))
old = list(csv.DictReader((HERE / 'initial-ten-plot-data.csv').open()))
for r in rows:
    d = [x for x in curves if x['slug'] == r['slug']]
    assert len(d) == r['window_tokens']
    h = np.array([float(x['rh_projection']) for x in d])
    c = np.array([float(x['control_projection']) for x in d])
    assert np.isfinite(np.r_[h, c]).all()
    n = r['final_tokens']
    scale = np.ptp(np.r_[h, c])
    assert h[-n:].mean() - h[:len(h)//2].mean() > .2 * scale
    assert np.min(h[-n:] - c[-n:]) > .2 * scale
    assert h[-n:].mean() > h[:-2*n].max()
    assert len(r['control_audit']) == r['n_controls']
    for ext in ['png', 'svg']:
        assert (HERE / f"{r['slug']}.{ext}").is_file()
for initial in json.loads((HERE / 'initial-ten-selection.json').read_text()):
    final = next(r for r in rows if r['slug'] == initial['slug'])
    assert initial == final, f"Original selection changed: {initial['slug']}"
    assert [x for x in old if x['slug'] == initial['slug']] == [x for x in curves if x['slug'] == initial['slug']]
if gemma.get('clear_separation'):
    previous = json.loads((HERE / 'stronger-coding-review/previous-selection.json').read_text())
    previous_curves = list(csv.DictReader((HERE / 'stronger-coding-review/previous-plot-data.csv').open()))
    for r in previous:
        if r['slug'] == 'gemma-4b-coding':
            continue
        assert r == next(x for x in rows if x['slug'] == r['slug'])
        assert [x for x in previous_curves if x['slug'] == r['slug']] == [x for x in curves if x['slug'] == r['slug']]
shards = 0
for p in sorted((HERE / 'replays').iterdir()):
    spec = json.loads((p / 'replay-spec.json').read_text())
    assert hashlib.sha256((p / 'probes.json').read_bytes()).hexdigest() == spec['probes_sha256']
    for source in spec['label_sources']:
        assert hashlib.sha256((ROOT / source['path']).read_bytes()).hexdigest() == source['sha256']
    for j in spec['jobs']:
        inp = p / 'inputs' / f"{j['id']}.json"
        assert hashlib.sha256(inp.read_bytes()).hexdigest() == j['sha256']
        score = p / 'scores' / f"{j['id']}.npz"
        receipt = json.loads(score.with_suffix('.receipt.json').read_text())
        assert hashlib.sha256(score.read_bytes()).hexdigest() == receipt['sha256']
        assert receipt['identity'] == spec['identity'] and receipt['job_sha256'] == j['sha256']
        values = np.load(score)['scores']
        assert values.shape == (j['tokens'], 12) and np.isfinite(values).all()
        shards += 1
assert shards == 25
with zipfile.ZipFile(HERE / 'figures-and-data.zip') as z:
    assert z.testzip() is None
result = dict(status='passed', model_dataset_cells=12, new_score_shards=shards,
              original_ten_selections_and_curves='unchanged', all_twelve_spike_checks='passed',
              glm_terminalwrench_alignment='second RH-labeled passage, 28 tokens after first label',
              gemma_coding_alignment=gemma.get('alignment', 'first RH onset'),
              gemma_coding_tokens_after_first_label=gemma.get('tokens_after_first_rh_label', 0),
              finite_score_matrices='passed', receipt_input_and_score_hashes='passed', zip_integrity='passed')
(HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
