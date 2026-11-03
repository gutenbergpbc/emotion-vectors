"""Plot k=0, 5, 10 out-of-fold ROC curves for saved twelve-probe forecasts."""

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
import hashlib
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'figures/_kit'))
import figkit as fk

from render_combined import TITLES

HORIZONS = (0, 5, 10)
COHORTS = ('all_available', 'common_sources', 'certain_only')
COLORS = {0: '#182a38', 5: '#bb562a', 10: '#418268'}
STYLES = {0: '-', 5: '--', 10: ':'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(run):
    out = run / 'combined-all12'
    receipt = json.loads((out / 'verification.json').read_text())
    assert receipt['passed'] and all(sha(out / p) == h for p, h in receipt['output_hashes'].items())
    bundle = json.loads((out / 'result-bundle.json').read_text())
    populations = {(p['cohort'], p['k']): p for p in bundle['populations']}
    result = {'source_bundle_sha256': sha(out / 'result-bundle.json'), 'horizons': list(HORIZONS), 'cohorts': {}}
    fk.use()
    plt.rcParams['figure.constrained_layout.use'] = False
    for cohort in COHORTS:
        curves = {}
        for k in HORIZONS:
            if (cohort, k) not in populations:
                continue
            path = out / cohort / f'k{k:02d}' / 'oof-predictions.npz'
            with np.load(path, allow_pickle=False) as saved:
                y, folds, prob = saved['labels'], saved['folds'], saved['probabilities']
            pairs = json.loads((path.parent / 'pairs.json').read_text())
            assert len(pairs) == len(y) == populations[cohort, k]['pairs']
            assert np.array_equal(y, [p['hacking'] for p in pairs])
            assert np.array_equal(folds, [p['fold'] for p in pairs])
            assert y.sum() == populations[cohort, k]['rh']
            assert np.isfinite(prob).all() and np.logical_and(prob >= 0, prob <= 1).all()
            assert set(y) == {False, True}
            fpr, tpr, _ = roc_curve(y, prob, drop_intermediate=True)
            full_fpr, full_tpr, _ = roc_curve(y, prob, drop_intermediate=False)
            operating = prob >= .5
            curves[k] = {
                'eligible_pairs': len(y), 'rh': int(y.sum()), 'non_rh': int((~y).sum()),
                'traces': len({p['trace_id'] for p in pairs}), 'folds': len(set(folds.tolist())),
                'auroc': float(roc_auc_score(y, prob)),
                'tpr_at_fpr': {str(b): float(full_tpr[full_fpr <= b+1e-12].max()) for b in (.01, .05, .10)},
                'cutoff': .5, 'operating_tpr': float(operating[y].mean()),
                'operating_fpr': float(operating[~y].mean()),
                'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
            }
        result['cohorts'][cohort] = curves
        fig, ax = plt.subplots(figsize=(9, 5.5))
        fig.subplots_adjust(left=.12, right=.97, top=.93, bottom=.18)
        chance = np.r_[0, np.geomspace(1e-5, 1, 300)]
        ax.plot(chance, chance, color=fk.LIGHT, lw=1, label='_chance')
        for k in HORIZONS:
            if k not in curves:
                continue
            c = curves[k]
            ax.step(c['fpr'], c['tpr'], where='post', color=COLORS[k], ls=STYLES[k], lw=2,
                    label=f'k={k}')
        c0 = curves[0]
        ax.scatter(c0['operating_fpr'], c0['operating_tpr'], marker='*', s=115,
                   color='#d59a22', edgecolor='#735514', zorder=5)
        for i, k in enumerate(HORIZONS):
            label = f"k={k}  AUROC {curves[k]['auroc']:.2f}" if k in curves else f'k={k}  unavailable'
            ax.text(.03, .97-i*.07, label,
                    transform=ax.transAxes, va='top', color=COLORS[k] if k in curves else fk.GREY, fontsize=10)
        ax.set_xscale('symlog', linthresh=.001)
        ax.set(xlim=(0, 1), ylim=(0, 1.02),
               xticks=[0, .001, .01, .1, 1], xticklabels=['0', '0.1%', '1%', '10%', '100%'],
               yticks=[0, .25, .5, .75, 1], yticklabels=['0%', '25%', '50%', '75%', '100%'],
               xlabel='False-positive rate', ylabel='True-positive rate')
        ax.grid(alpha=.12)
        stem = f'roc-k0-5-10-{cohort}'
        fk.save(fig, out / f'{stem}.png', words=30)
        fig.savefig(out / f'{stem}.svg', facecolor='white')
        plt.close(fig)
    (out / 'roc-metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    sections = ''.join(
        f'<h2>{html.escape(cohort.replace("_", " ").title())}</h2>'
        f'<figure><img src="roc-k0-5-10-{cohort}.svg" alt="Out-of-fold ROC at k 0, 5, 10">'
        f'<figcaption>k=0: {result["cohorts"][cohort][0]["rh"]} RH and '
        f'{result["cohorts"][cohort][0]["non_rh"]:,} non-RH units. '
        + (f'Unavailable horizons: {", ".join(str(k) for k in HORIZONS if k not in result["cohorts"][cohort])}. '
           if any(k not in result['cohorts'][cohort] for k in HORIZONS) else '')
        +
        'Star: k=0 prediction cutoff 0.5. Zero FPR is retained on the near-log axis. '
        'Repeated units within traces are not independent.</figcaption></figure>' for cohort in COHORTS)
    (out / 'roc-report.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Combined ROC curves</title>'
        '<style>body{font:17px/1.5 system-ui;max-width:1050px;margin:32px auto;padding:0 20px;color:#263746}'
        'img{width:100%}figure{margin:24px 0}figcaption{font-size:14px;color:#54616b}</style>'
        f'<h1>{html.escape(TITLES[run.name])} · twelve-probe ROC</h1>'
        '<p>Out-of-fold logistic probabilities predict the RH label exactly k eligible units later. '
        'k=0 is current-unit classification; later positives can include ongoing RH. '
        'Cohorts and target populations match the accuracy curves.</p>'
        + sections + '<p><a href="report.html">Accuracy curves</a> · '
        '<a href="roc-metrics.json">ROC values and counts</a></p></html>')
    report_path = out / 'report.html'
    report = report_path.read_text()
    if 'roc-report.html' not in report:
        report = report.replace('</html>', '<p><a href="roc-report.html">ROC curves at k=0, 5, 10</a></p></html>')
        report_path.write_text(report)
    receipt['output_hashes'] = {str(p.relative_to(out)): sha(p) for p in out.rglob('*')
                                if p.is_file() and p.name != 'verification.json'}
    (out / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dirs', nargs='+', type=Path)
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        main(run_dir.resolve())
