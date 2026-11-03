import html
import numpy as np

from shared.io import atomic, read, require, sha
from .data import EMOTIONS, RECIPES
from .fitting import normalize, winning_layer
from .statistics import verify_threshold


def report(root, spec, progress):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    data = read(root / 'metrics.json')['results']
    progress.stage('report', 0, 3, 'figures', 'Rendering held-out balanced accuracy')
    figure, ax = plt.subplots(figsize=(17, 6))
    x = np.arange(len(EMOTIONS))
    colors = ['#456fa4', '#df8845', '#3c9588']
    legends = ['Trained vs neutral', 'Trained vs other emotions + neutral', 'Trained vs other emotions']
    for j, recipe in enumerate(RECIPES):
        values = [data[f'{e}__{recipe}'] for e in EMOTIONS]
        ba = np.array([v['heldout']['balanced_accuracy'] for v in values])
        ci = np.array([v['heldout_bootstrap']['balanced_accuracy_95_interval'] for v in values])
        pos = x + (j-1)*.26
        ax.bar(pos, ba, .26, color=colors[j], label=legends[j])
        # Draw percentile intervals directly; a percentile CI need not contain
        # the point estimate, unlike matplotlib's nonnegative yerr convention.
        ax.vlines(pos, ci[:, 0], ci[:, 1], color='#526173', linewidth=1)
        ax.hlines(ci[:, 0], pos-.03, pos+.03, color='#526173', linewidth=1)
        ax.hlines(ci[:, 1], pos-.03, pos+.03, color='#526173', linewidth=1)
        for p, value, interval in zip(pos, ba, ci):
            ax.text(p, max(value, interval[1])+.015, f'{value:.0%}', ha='center', fontsize=8)
    ax.set_xticks(x, [e.title() for e in EMOTIONS], rotation=20, ha='right')
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel('Mixed-label balanced accuracy')
    ax.axhline(.5, color='#87939b', linestyle='--', linewidth=.8)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.15)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.13), ncol=3, frameon=False)
    family = spec.get('training_family', 'diversity')
    title = f"{'SYNTHETIC DEMO · ' if spec['backend'] == 'synthetic' else ''}{spec['model']} · {spec['quantization'] or spec['dtype']} · {family} probes"
    figure.suptitle(title, fontsize=14, y=1.02)
    figure.text(.5, -.03, f"95% source-conversation bootstrap intervals · {spec['bootstrap_draws']:,} resamples · fixed probes and cutoffs", ha='center', color='#617080')
    figure.tight_layout()
    (root / 'figures').mkdir(exist_ok=True)
    for i, ext in enumerate(('png', 'svg', 'pdf'), 1):
        figure.savefig(root / f'figures/balanced-accuracy.{ext}', bbox_inches='tight', dpi=160)
        progress.stage('report', i, 3, 'figures', f'Saved {ext.upper()}')
    plt.close(figure)
    rows = ''.join(f"<tr><td>{html.escape(v['emotion'])}</td><td>{html.escape(v['recipe'])}</td><td>{v['layer']}</td><td>{v['heldout']['balanced_accuracy']:.1%}</td></tr>" for v in data.values())
    training_note = ('240 emotion stories + 20 shared neutral dialogues; raw-text all-token pooling, including special tokens. '
                     'Five topic-grouped training folds. Evaluation pools marked assistant sentences. Story and neutral dialogue formats differ; '
                     'centering does not establish that format effects are removed.' if family == 'story' else 'Assistant-sentence training and evaluation.')
    (root / 'report.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>body{{font:16px/1.55 system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#243440}}img{{width:100%}}table{{border-collapse:collapse}}th,td{{padding:6px 18px;text-align:left;border-bottom:1px solid #ddd}}.note{{background:#eef3f6;padding:16px;border-radius:8px}}</style>
<h1>{html.escape(title)}</h1><p>{html.escape(training_note)} · {spec['centering']} centering · checkpoint {html.escape(spec['revision'])}</p>
<img src="figures/balanced-accuracy.svg" alt="Held-out balanced accuracy for twelve emotions and three training contrasts">
<p class="note">Each bar averages target recall and mean specificity across the other eleven emotions and neutral. Layers use training-only grouped cross-validation; thresholds and the selected twelve probes use calibration data. All bars use the same held-out rows. These intervals hold the fitted probes, layers and thresholds fixed. This reused benchmark is retrospective; emotion classification does not establish subjective experience. Precision identifies the model on which these probes were trained.</p>
<p><a href="index.html">Progress</a> · <a href="metrics.csv">Metrics CSV</a> · <a href="metrics.json">Metrics JSON</a> · <a href="probes/manifest.json">Probe bundle</a> · <a href="verification.json">Verification</a></p>
<table><thead><tr><th>Emotion</th><th>Contrast</th><th>Layer</th><th>Held-out BA</th></tr></thead><tbody>{rows}</tbody></table></html>''')


def verify(root, spec, progress):
    progress.stage('verify', 0, 36, 'probes', 'Recomputing results from portable bundle')
    manifest = read(root / 'probes/manifest.json')
    require(manifest['identity'] == spec['identity'], 'Bundle identity mismatch')
    for name, checksum in manifest['files'].items():
        require(sha(root / 'probes' / name) == checksum, f'Bundle checksum mismatch: {name}')
    models = read(root / 'calibrated_models.json')['models']
    cv = read(root / 'cv_results.json')
    metrics = read(root / 'metrics.json')['results']
    expected = {f'{e}__{r}' for e in EMOTIONS for r in RECIPES}
    require(set(models) == set(metrics) == expected, 'Expected exactly 36 probes')
    with np.load(root / 'probes/uncentered-vectors.npz', allow_pickle=False) as raw, \
         np.load(root / 'probes/all-layer-uncentered-vectors.npz', allow_pickle=False) as raw_layers, \
         np.load(root / 'probes/all-layer-vectors.npz', allow_pickle=False) as primary_layers:
        require(set(raw.files) == set(raw_layers.files) == set(primary_layers.files) == expected,
                'Expected both vector variants for all 36 probes')
        for key, model in models.items():
            values = raw_layers[key]
            require(values.shape == (spec['layers'], spec['width']), 'Uncentered vector shape mismatch')
            require(np.isfinite(values).all() and np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-6),
                    'Non-unit uncentered direction')
            require(np.array_equal(raw[key], values[model['layer']]), 'Uncentered selected-layer mismatch')
        for recipe in RECIPES:
            keys = [f'{e}__{recipe}' for e in EMOTIONS]
            values = np.stack([raw_layers[key] for key in keys]).astype(np.float64)
            transformed = normalize(values-values.mean(axis=0)) if spec['centering'] == 'within_contrast' else values
            require(np.allclose(transformed, np.stack([primary_layers[key] for key in keys]), atol=2e-6),
                    'Centered directions do not reproduce from uncentered export')
    with np.load(root / 'probes/vectors.npz', allow_pickle=False) as vectors, \
         np.load(root / 'probes/scores/calibration.npz', allow_pickle=False) as calibration, \
         np.load(root / 'probes/scores/heldout.npz', allow_pickle=False) as held:
        require(set(calibration['ids']).isdisjoint(held['ids']), 'Evaluation split overlap')
        for i, (key, model) in enumerate(models.items(), 1):
            require(np.isfinite(vectors[key]).all() and np.isclose(np.linalg.norm(vectors[key]), 1, atol=1e-6), 'Non-unit vector')
            require(np.array_equal(vectors[key], np.asarray(model['vector'], dtype=np.float32)), 'Bundle vector differs from fitted vector')
            require(winning_layer(np.mean(cv[key]['fold_aurocs'], axis=0)) == model['layer'], 'Layer selection mismatch')
            verify_threshold(calibration['labels'], calibration[key], model['emotion'], model['threshold'])
            rates = {lab: float(np.mean(held[key][held['labels'] == lab] >= model['threshold'])) for lab in set(held['labels'])}
            ba = .5 * (rates[model['emotion']] + 1 - np.mean([v for lab, v in rates.items() if lab != model['emotion']]))
            require(abs(ba-metrics[key]['heldout']['balanced_accuracy']) < 1e-12, f'BA does not reproduce: {key}')
            progress.stage('verify', i, 36, 'probes', key)
    for path in ('report.html', 'figures/balanced-accuracy.png', 'figures/balanced-accuracy.svg', 'figures/balanced-accuracy.pdf'):
        require((root / path).is_file(), f'Missing report artifact: {path}')
    atomic(root / 'verification.json', {'passed': True, 'identity': spec['identity'], 'probes': 36,
            'synthetic': spec['backend'] == 'synthetic', 'checks': ['bundle checksums', 'unit vectors',
            'uncentered directions at all layers and primary selected layers', 'centering reproduces from uncentered directions', 'training-only layer selection',
            'calibration optimum and tie rules', 'held-out BA from saved scores', 'report artifacts']})
