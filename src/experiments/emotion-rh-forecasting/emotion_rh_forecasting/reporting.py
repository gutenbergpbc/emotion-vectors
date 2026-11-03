"""Figures use saved metrics; bands are subsampling variability, not confidence."""
import csv
import html

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from shared.io import atomic
from .analysis import EMOTIONS

FORMATS = ('png', 'svg', 'pdf')


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figure_names(spec):
    names = ['accuracy-trends', 'accuracy-heatmap', 'accuracy-gain-vs-position']
    if 0 in spec['horizons'] and len(spec['horizons']) > 1:
        names.append('accuracy-trends-future')
    return names


def report(root, spec, result):
    folder = root / 'figures'
    folder.mkdir(exist_ok=True)
    estimates = {(r['model'], r['horizon']): r for r in result['estimates'] if r['cohort'] == 'all_available'}
    counts = {r['horizon']: r['balanced_eval_pairs'] for r in result['populations'] if r['cohort'] == 'all_available'}
    horizons = spec['horizons']
    method = spec['evaluation']['method']
    groups = f"{result['counts']['traces']} traces · {result['counts']['folds']} folds · " + (
        'held-out task prompts' if method == 'held_out_task' else 'held-out source groups')
    subtitle = f"{spec['title']} · {spec['model']['name']}"
    caveat = ('Cutoff and polarity refit on training folds at every horizon; each held-out fold is balanced separately.\n'
              'h=0 classifies the current sentence. Future targets can reflect ongoing hacking; this is not a first-onset forecast.')
    if 'replay' in spec['protocol']:
        caveat += '\nNative-template replay of archived ATIF text; label provenance is in the report. These are reconstructed activations.'

    def save(fig, name):
        for ext in FORMATS:
            fig.savefig(folder / f'{name}.{ext}', dpi=180, facecolor='white')
        plt.close(fig)

    for name, hs in [('accuracy-trends', horizons), ('accuracy-trends-future', [h for h in horizons if h > 0])]:
        if name not in figure_names(spec):
            continue
        fig, axes = plt.subplots(3, 4, figsize=(17, 11), sharex=True, sharey=True)
        fig.subplots_adjust(left=.07, right=.98, top=.79, bottom=.17, wspace=.18, hspace=.37)
        for ax, emotion in zip(axes.flat, EMOTIONS):
            rows = [estimates[emotion, h] for h in hs]
            ax.plot(hs, [100*r['repeated_sampling_mean_accuracy'] for r in rows], color='#3169a8', marker='o', ms=3, label='Emotion probe')
            ax.fill_between(hs, [100*r['sampling_interval95'][0] for r in rows], [100*r['sampling_interval95'][1] for r in rows], color='#3169a8', alpha=.13)
            ax.plot(hs, [100*estimates['token_position', h]['repeated_sampling_mean_accuracy'] for h in hs], color='#67727d', ls=':', lw=1.6, label='Token position')
            ax.axhline(50, color='#a5652e', ls='--', lw=1, label='Always non-hacking')
            # Keep the reference's range when possible; never clip a new model.
            values = [100*estimates[m, h]['repeated_sampling_mean_accuracy'] for m in EMOTIONS+['token_position'] for h in hs]
            bounds = [100*v for m in EMOTIONS for h in hs for v in estimates[m, h]['sampling_interval95']]
            low, high = min(25, 5*np.floor(min(values+bounds)/5)), max(85, 5*np.ceil(max(values+bounds)/5))
            ax.set(title=emotion.capitalize(), ylim=(low, high), xticks=hs if len(hs) <= 11 else hs[::2])
            if len(hs) > 1:
                ax.set_xlim(min(hs), max(hs))
            ax.grid(axis='y', alpha=.12)
        for ax in axes[:, 0]:
            ax.set_ylabel('Balanced-evaluation accuracy (%)')
        for ax in axes[-1]:
            ax.set_xlabel('Sentence horizon (+h)')
        fig.suptitle('How each emotion probe forecasts future reward hacking', x=.07, y=.97, ha='left', fontsize=22)
        fig.text(.07, .93, subtitle+' · '+groups, fontsize=11)
        fig.text(.07, .89, f"Lines: mean across {spec['repeats']:,} balanced evaluations · bands: sampling variability only, not generalization confidence", fontsize=11)
        fig.text(.07, .85, f'Balanced pair counts (+{min(hs)} → +{max(hs)}): '+' / '.join(str(counts[h]) for h in hs), fontsize=11, color='#51565c')
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='lower left', bbox_to_anchor=(.065, .085), ncol=3, frameon=False)
        fig.text(.07, .025, caveat, fontsize=10, color='#51565c')
        save(fig, name)
    for delta in (False, True):
        models = EMOTIONS if delta else EMOTIONS+['token_position']
        matrix = np.array([[100*estimates[m, h]['repeated_sampling_mean_accuracy'] for h in horizons] for m in models])
        if delta:
            matrix -= np.array([100*estimates['token_position', h]['repeated_sampling_mean_accuracy'] for h in horizons])
        fig, ax = plt.subplots(figsize=(15.5, 10))
        fig.subplots_adjust(left=.15, right=.88, top=.79, bottom=.20)
        limit = max(10, float(np.ceil(np.abs(matrix).max()/5)*5))
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit) if delta else TwoSlopeNorm(vmin=0, vcenter=50, vmax=100)
        img = ax.imshow(matrix, aspect='auto', cmap='RdBu', norm=norm)
        for i in range(len(models)):
            for j in range(len(horizons)):
                ax.text(j, i, f'{matrix[i,j]:+.1f}' if delta else f'{matrix[i,j]:.1f}', ha='center', va='center', fontsize=9,
                        color='white' if abs(norm(matrix[i,j])-.5) > .30 else '#203047')
        ax.set(xticks=range(len(horizons)), xticklabels=[f'+{h}\nn={counts[h]}' for h in horizons],
               yticks=range(len(models)), yticklabels=[m.replace('_', ' ').capitalize() for m in models], xlabel='Sentence horizon · balanced evaluation pairs')
        fig.colorbar(img, ax=ax, fraction=.035, pad=.025).set_label('Accuracy gain (percentage points)' if delta else 'Balanced-evaluation accuracy (%)')
        fig.suptitle('Emotion forecast accuracy relative to token position' if delta else 'Predicting reward hacking 0–10 sentences ahead', x=.15, y=.96, ha='left', fontsize=21)
        fig.text(.15, .91, subtitle, fontsize=12)
        fig.text(.15, .86, groups+f" · mean of {spec['repeats']:,} balanced evaluations", fontsize=11)
        fig.text(.15, .04, caveat, fontsize=10, color='#51565c')
        save(fig, 'accuracy-gain-vs-position' if delta else 'accuracy-heatmap')
    atomic(root / 'chart-data.json', {'estimates': result['estimates'], 'populations': result['populations'],
                                    'horizons': horizons, 'emotions': EMOTIONS, 'interval': result['interval']})
    for name, rows in [('metrics', result['estimates']), ('fold-statistics', result['fold_statistics']), ('populations', result['populations'])]:
        write_csv(root / f'{name}.csv', rows)
    images = ''.join(f'<h2>{html.escape(name.replace("-", " "))}</h2><img src="figures/{name}.svg" alt="{name}">' for name in figure_names(spec))
    (root / 'report.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>Emotion probes predicting reward hacking</title><style>body{font:16px/1.5 system-ui;max-width:1350px;margin:32px auto;padding:0 20px;color:#233140}img{width:100%}a{color:#3169a8}</style>'
        f'<h1>{html.escape(subtitle)}</h1><p>{html.escape(groups)}. Frozen twelve emotion directions; training-only cutoff and polarity.</p>'
        + (f'<p>{html.escape(spec["protocol"]["replay"])}</p>' if 'replay' in spec['protocol'] else '') +
        '<p>Horizon 0 is current-sentence classification; horizon h predicts the label exactly h eligible prose sentences later within the same trace. '
        'No future-token features. The oracle current-label persistence baseline is an annotation-dependent diagnostic saved only in the metrics.</p>'
        f'<p>{html.escape(result["interval"])} This retrospective benchmark uses model-generated labels and fixed probes. '
        'It does not establish generalization to new tasks or advance warning before hacking begins. Longer horizons change the population. '
        'Common-source and certain-only sensitivity results are included in the metrics.</p>'
        '<p><a href="metrics.json">Metrics and protocol</a> · <a href="metrics.csv">CSV</a> · <a href="verification.json">Verification</a> · '
        '<a href="config.resolved.json">Configuration</a> · <a href="index.html">Progress</a></p>'+images)
