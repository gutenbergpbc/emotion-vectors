"""Render selected saved trajectories without re-running the selection."""

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
import html
import json
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / '_kit'))
import figkit as fk

rows = json.loads((HERE / 'selection.json').read_text())
data = list(csv.DictReader((HERE / 'plot-data.csv').open()))
models = ['Gemma 4B', 'Gemma 27B', 'GPT-OSS 120B', 'GLM 5.3']
tasks = ['TerminalWrench', 'Sales', 'Coding']
colors = dict(guilty='#5a719c', sad='#8065ad', anxious='#698442', calm='#257e82', hostile='#cf6252', ashamed='#aa7c2d')
missing = {
    ('GLM 5.3', 'TerminalWrench'): 'Scored archive: 7 non-hacking traces.\nRH-positive traces need probe replay.',
    ('Gemma 4B', 'Coding'): 'Original scored cohort: no non-hacking controls.\n5 labeled expansion controls need probe replay.',
}
fk.use()


def draw(ax, r):
    d = [v for v in data if v['slug'] == r['slug']]
    x = np.array([int(v['token']) for v in d])
    h = np.array([float(v['rh_projection']) for v in d])
    c = np.array([float(v['control_projection']) for v in d])
    assert np.isfinite(np.r_[h, c]).all() and len(x) == r['window_tokens']
    assert np.min(h[-r['final_tokens']:] - c[-r['final_tokens']:]) > 0
    color = colors[r['emotion']]
    ax.plot(x, h, color=color, lw=2.4)
    ax.plot(x, c, color=color, lw=1.9, ls=(0, (4, 2)), alpha=.8)
    ax.axhline(0, color='#bdc8cb', lw=.7)
    ax.axvline(0, color='#8e9da5', lw=.9, ls=(0, (3, 3)))
    span = max(np.ptp(np.r_[h, c]), 1e-9)
    ax.set_ylim(min(h.min(), c.min()) - .13 * span, max(h.max(), c.max()) + .13 * span)
    ax.set_xlim(x[0], .29 * len(x))
    ax.set_xticks([x[0], -len(x) // 2, 0])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(axis='y', color='#e4eaec', lw=.7)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    title = f"{r['model']} × {r['task']}: {r['emotion'].title()}"
    ax.set_title(title, loc='left', fontsize=12, weight='semibold', pad=10)
    ax.set_xlabel('Tokens before selected RH passage' if r.get('tokens_after_first_rh_label', 0) else 'Tokens before RH-labeled text', fontsize=10)
    ax.set_ylabel('Centered projection', fontsize=10)
    ax.tick_params(labelsize=10)
    label_x = .025 * len(x)
    ax.text(label_x, h[-1], 'RH trace', color=color, fontsize=10, weight='semibold', va='center')
    ax.text(label_x, c[-1], 'Control avg.', color=color, fontsize=10, weight='semibold', va='center')


for r in rows:
    fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
    draw(ax, r)
    for ext in ['png', 'svg']:
        fk.save(fig, HERE / f"{r['slug']}.{ext}", words=35, title_words=9, min_pt=10)
    plt.close(fig)

with PdfPages(HERE / 'all-plots.pdf') as pdf:
    for task in tasks:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.8), layout='constrained')
        for model, ax in zip(models, axes.flat):
            r = next((r for r in rows if r['model'] == model and r['task'] == task), None)
            if r:
                draw(ax, r)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_axis_off()
                ax.text(.04, .88, model, transform=ax.transAxes, fontsize=12, weight='semibold', color='#52616d')
                ax.text(.04, .58, 'Additional scoring needed', transform=ax.transAxes, fontsize=13, color='#52616d')
                note = ('7 scored traces; all non-hacking.\nRH traces await scoring.' if model == 'GLM 5.3'
                        else 'No scored non-hacking controls.\n5 labeled controls await scoring.')
                ax.text(.04, .39, note, transform=ax.transAxes, fontsize=10, color='#6b7983', linespacing=1.6)
        for ext in ['png', 'svg']:
            fk.save(fig, HERE / f'{task.lower()}.{ext}', max_panels=4, words=110, title_words=6, min_pt=10)
        pdf.savefig(fig)
        plt.close(fig)

later = [r for r in rows if r.get('tokens_after_first_rh_label', 0)]
alignment = '; '.join(f"{r['model']} {r['task']} uses a passage {r['tokens_after_first_rh_label']} tokens after its first RH label" for r in later)
caption = ('Solid lines are selected RH traces; dashed lines average eligible non-hacking controls from the same model and dataset. '
           f"Zero is the first RH-labeled text in {len(rows)-len(later)} panels; {alignment}. "
           'Controls use proportional pseudo-events. Each trajectory is independently centered, '
           'with causal smoothing; windows range from 17 to 128 tokens and y-scales differ. These are selected illustrations, not typical effects or a validated detector.')
sections = []
for task in tasks:
    table = []
    for model in models:
        r = next((r for r in rows if (r['model'], r['task']) == (model, task)), None)
        if r:
            table.append(f"<tr><td>{model}</td><td>{r['emotion'].title()}</td><td>{r['n_controls']}</td><td>{r['window_tokens']}</td>"
                         f"<td><a href='{r['slug']}.png'>PNG</a> · <a href='{r['slug']}.svg'>SVG</a></td><td><code>{html.escape(r['hack_id'])}</code></td></tr>")
        else:
            table.append(f"<tr><td>{model}</td><td colspan='5'>{html.escape(missing[(model, task)]).replace(chr(10), ' ')}</td></tr>")
    sections.append(f"<section><h2>{task}</h2><img src='{task.lower()}.png' alt='{task} trace comparisons'>"
                    "<table><tr><th>Model</th><th>Emotion</th><th>Controls</th><th>Tokens</th><th>Download</th><th>RH trace</th></tr>" + ''.join(table) + '</table></section>')
coverage_note = (f'{len(rows)} supported plots across 12 requested combinations.' +
                 (' All combinations are filled.' if len(rows) == 12 else ' The remaining combinations require additional probe scoring.'))
page = ("<!doctype html><meta charset='utf-8'><title>Emotion spikes by model and dataset</title>"
        "<style>body{font:16px/1.55 system-ui;color:#293541;max-width:1280px;margin:32px auto;padding:0 22px}"
        "img{width:100%}section{margin:42px 0}table{border-collapse:collapse;width:100%;font-size:13px}"
        "td,th{text-align:left;border-bottom:1px solid #d9e0e4;padding:10px}code{overflow-wrap:anywhere}a{color:#356893}</style>"
        "<h1>Emotion spikes by model and dataset</h1><p>" + coverage_note + "</p><p>" + caption + "</p>"
        "<p><a href='all-plots.pdf'>Three-page PDF</a> · <a href='figures-and-data.zip'>Figures and data ZIP</a> · "
        "<a href='selection.json'>Selected traces and control IDs</a> · <a href='analysis-provenance.json'>Method</a></p>" + ''.join(sections) +
        "<p>Controls are negative under the saved sentence-level RH labels; sales controls are not additionally filtered for final-ledger validity. "
        "GLM 5.3 uses NVFP4 replay projections. Curves stop before the selected labeled passage; panels aligned to later passages can include earlier RH-labeled text.</p>")
(HERE / 'index.html').write_text(page)
with zipfile.ZipFile(HERE / 'figures-and-data.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(HERE.iterdir()):
        if p.suffix in {'.png', '.svg', '.pdf', '.csv', '.json', '.py', '.html'}:
            z.write(p, p.name)
    for p in sorted((HERE / '_kit').rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            z.write(p, str(p.relative_to(HERE)))
    for p in sorted((HERE / 'replays').rglob('*.json')):
        if p.name in {'replay-spec.json', 'progress.json'} or p.name.endswith('.receipt.json'):
            z.write(p, str(p.relative_to(HERE)))
    for p in sorted((HERE / 'stronger-coding-review').glob('*')):
        if p.suffix in {'.json', '.csv'}:
            z.write(p, str(p.relative_to(HERE)))
print('Rendered', len(rows), 'plots; three dataset pages and PDF; downloadable archive.')
