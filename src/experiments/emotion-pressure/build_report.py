"""Render all emotion results as a standalone interactive HTML and twelve-page PDF."""

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
import json
from pathlib import Path
import sys
import textwrap

import numpy as np

HERE = Path(__file__).resolve().parent
SLUGS = ('gemma-3-27b-it', 'gpt-oss-120b')
MODELS = ('Gemma 3 27B', 'GPT-OSS 120B')
EMOTIONS = 'desperate frustrated anxious afraid calm appreciative hostile excited sad surprised guilty ashamed'.split()
TITLES = dict(zip(EMOTIONS, ['Desperation', 'Frustration', 'Anxiety', 'Fear', 'Calmness',
                            'Appreciation', 'Hostility', 'Excitement', 'Sadness', 'Surprise', 'Guilt', 'Shame']))
COLORS = ['#405e70', '#ce704c', '#258c83']
LABELS = ['No instruction', 'Named emotions', 'General emotions']


def build(root, slugs=SLUGS):
    output = root / ('viewer' if len(slugs) == 2 else slugs[0] + '-viewer')
    output.mkdir(exist_ok=True)
    frozen = json.loads((HERE / 'frozen-inputs.json').read_text())
    verification_root = root if len(slugs) == 2 else root / slugs[0]
    verification = json.loads((verification_root / 'independent-verification.json').read_text())
    summaries = {s: json.loads((root / s / 'summary.json').read_text()) for s in slugs}
    data = {'models': [], 'emotions': EMOTIONS, 'titles': TITLES,
            'conditions': frozen['conditions'], 'verification': verification,
            'scenario_titles': {s['id']: s['title'] for s in frozen['fixtures']['sets']}}
    for slug in slugs:
        name = MODELS[SLUGS.index(slug)]
        data['models'].append({'slug': slug, 'name': name, 'summary': summaries[slug],
                               'probes': {e: {k: v for k, v in frozen['emotion_probes'][slug][e].items() if k != 'vector'} for e in EMOTIONS}})
    (output / 'plot-data.json').write_text(json.dumps(data, indent=2) + '\n')
    template = (HERE / 'viewer.html').read_text()
    payload = json.dumps(data).replace('</', '<\\/')
    (output / 'index.html').write_text(template.replace('__DATA__', payload))
    pdf(root, output, data)
    print('Viewer:', output / 'index.html')


def pdf(root, output, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import MaxNLocator
    sys.path.insert(0, str(HERE / '_kit'))
    import figkit as fk
    fk.use()
    plt.rcParams.update({'figure.constrained_layout.use': False, 'font.size': 11,
                         'axes.grid': False, 'savefig.dpi': 170})
    x = np.arange(4)
    with PdfPages(output / 'all-emotions.pdf') as pdf_file:
        pdf_file.infodict().update(Title='Emotion resistance pressure sweep — November 2, 2026',
                                   Subject='Fresh measurements using two new resistance prompts, all twelve frozen emotion probes')
        cover = plt.figure(figsize=(10, 6.8))
        cover.text(.08, .91, 'Emotion resistance under pressure', fontsize=23, weight='semibold', color='#253541')
        cover.text(.08, .85, ' and '.join(m['name'] for m in data['models']) + ' · November 2, 2026', fontsize=12, color='#61717d')
        for y, label, condition in [(.74, 'Named emotions', 'named_emotions'), (.49, 'General emotions', 'general_emotions')]:
            cover.text(.08, y, label, fontsize=12, weight='semibold')
            cover.text(.08, y-.045, textwrap.fill(data['conditions'][condition], 113),
                       fontsize=10, va='top', linespacing=1.6)
        cover.text(.08, .23, f"20 matched scripted conversations × 4 pressure levels × 3 instructions × {len(data['models'])} model(s)", fontsize=11)
        cover.text(.08, .18, 'All twelve probes are frozen. The following pages show mean raw projection ±1 sample SD.\n'
                   'Readout: final assistant-prefix token, each probe’s recorded layer; no generation or steering.\n'
                   'Model-specific raw scores are not calibrated emotion intensities. Instructions target desperation.\n'
                   'The companion HTML includes within-condition changes and paired bootstrap comparisons.',
                   fontsize=10, va='top', color='#61717d', linespacing=1.6)
        pdf_file.savefig(cover)
        plt.close(cover)
        for e in EMOTIONS:
            fig, axes = plt.subplots(1, len(data['models']), figsize=(10 if len(data['models']) == 2 else 7, 6.8))
            axes = np.atleast_1d(axes)
            fig.subplots_adjust(left=.12 if len(data['models']) == 2 else .17,
                                right=.97, bottom=.28, top=.77, wspace=.48)
            fig.text(.07, .94, TITLES[e], fontsize=22, weight='semibold', color='#253541')
            fig.text(.07, .885, 'New resistance prompts · November 2, 2026', fontsize=11, color='#61717d')
            for ax, model in zip(axes, data['models']):
                d = model['summary']['probes'][e]
                mean, sd = np.asarray(d['mean']), np.asarray(d['sample_sd'])
                for i in range(3):
                    ax.plot(x, mean[i], 'o-', lw=2, markersize=4, color=COLORS[i], label=LABELS[i])
                    ax.fill_between(x, mean[i]-sd[i], mean[i]+sd[i], color=COLORS[i], alpha=.10, linewidth=0)
                ax.set_xticks(x, ['Neutral', 'Some', 'High', 'Extreme'])
                ax.set_xlabel('Prompt pressure')
                ax.set_ylabel('Raw probe projection')
                ax.set_title(model['name'], loc='left', fontsize=13, weight='semibold', pad=13)
                ax.set_xlim(-.08, 4.72)
                ax.yaxis.set_major_locator(MaxNLocator(5))
                ax.spines[['top', 'right']].set_visible(False)
                ax.grid(axis='y', alpha=.15)
                with plt.rc_context({'font.size': 8.5}):
                    fk.label_lines(ax)
                p = model['probes'][e]
                fig.text(.12 if model['slug'] == SLUGS[0] else .63, .12,
                         f"{p['probe_id']} · layer {p['layer']}\n{p['training_family']}", fontsize=8, color='#61717d')
            fig.text(.07, .045, 'Mean ±1 sample SD across 20 matched scripted conversations. Last assistant-prefix token; no generation or steering.\nRaw projection scales are model-specific. Instructions target desperation; other emotions are exploratory readouts.',
                     fontsize=8.5, color='#61717d', linespacing=1.5)
            fk.save(fig, output / (e + '.png'), max_panels=2, max_series=3, words=100, min_pt=8)
            fig.savefig(output / (e + '.svg'))
            pdf_file.savefig(fig)
            plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--model', choices=SLUGS)
    args = parser.parse_args()
    build(args.root.resolve(), (args.model,) if args.model else SLUGS)
