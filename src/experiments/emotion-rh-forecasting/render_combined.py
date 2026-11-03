"""Render the saved twelve-emotion forecast bundles without refitting models."""

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

sys.path.insert(0, str(Path(__file__).resolve().parent / 'figures/_kit'))
import figkit as fk

TITLES = {
    'gpt-oss-120b-coding-v1': 'GPT-OSS 120B · Coding',
    'gpt-oss-120b-sales-v1': 'GPT-OSS 120B · Sales',
    'gpt-oss-120b-terminalwrench-deepseek-balanced50-v1': 'GPT-OSS 120B · TerminalWrench',
    'gemma-3-27b-terminalwrench-pilot50-v1': 'Gemma 27B · TerminalWrench',
    'gemma-3-27b-coding-verbose-haiku55-v1': 'Gemma 27B · Coding, verbose prompt',
    'gemma-3-27b-sales-haiku55-v1': 'Gemma 27B · Sales',
    'gemma3-4b-coding-lines-v1': 'Gemma 4B · Coding code lines',
    'gemma3-4b-sales-gpt55-v1': 'Gemma 4B · Sales',
    'glm-5.3-coding-1000-gpt55-v1': 'GLM 5.3 · Coding',
    'glm-5.3-balanced-source-best12-v1/sales': 'GLM 5.3 · Sales',
}
COHORTS = ('all_available', 'common_sources', 'certain_only')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(run):
    out = run / 'combined-all12'
    bundle = json.loads((out / 'result-bundle.json').read_text())
    estimates = bundle['estimates']
    populations = bundle['populations']
    run_key = f'{run.parent.name}/{run.name}' if run.name == 'sales' else run.name
    title = TITLES[run_key]
    unit = 'code line' if 'code lines' in title else 'sentence'
    fk.use()
    plt.rcParams['figure.constrained_layout.use'] = False
    for cohort in COHORTS:
        by_model = {name: sorted((r for r in estimates if r['cohort'] == cohort and r['model'] == name), key=lambda r: r['k'])
                    for name in ('all_12', 'token_position')}
        fig, ax = plt.subplots(figsize=(9, 5.2))
        fig.set_layout_engine(None)
        fig.subplots_adjust(left=.11, right=.78, top=.93, bottom=.15)
        for name, color, style in [('all_12', fk.ACCENT, '-o'), ('token_position', fk.GREY, ':')]:
            rows = by_model[name]
            ax.plot([r['k'] for r in rows], [100*r['accuracy'] for r in rows], style,
                    color=color, lw=2.3, ms=4, label=name)
            if name == 'all_12':
                ax.fill_between([r['k'] for r in rows],
                                [100*r['sampling_interval95'][0] for r in rows],
                                [100*r['sampling_interval95'][1] for r in rows],
                                color=color, alpha=.14)
        ax.axhline(50, color=fk.LIGHT, lw=1, ls='--')
        sign = 1 if by_model['all_12'][-1]['accuracy'] >= by_model['token_position'][-1]['accuracy'] else -1
        for name, label, offset in [('all_12', '12 probes', 2.0*sign), ('token_position', 'Token position', -2.0*sign)]:
            final = by_model[name][-1]
            ax.text(10.25, 100*final['accuracy']+offset, label, color=fk.ACCENT if name == 'all_12' else fk.GREY,
                    va='center', fontsize=10)
        all_values = [50] + [100*v for rows in by_model.values() for r in rows for v in (r['accuracy'], *r['sampling_interval95'])]
        ax.set(xlim=(-.15, 10.1), ylim=(max(20, min(all_values)-7), min(100, max(all_values)+7)),
               xticks=list(range(11)), xlabel=f'{unit.capitalize()} offset k', ylabel='Balanced accuracy (%)')
        ax.grid(axis='y', alpha=.12)
        fk.save(fig, out / f'{cohort}.png', words=25)
        fig.savefig(out / f'{cohort}.svg', facecolor='white')
        fig.savefig(out / f'{cohort}.pdf', facecolor='white')
        plt.close(fig)

    table = ''.join('<tr><td>'+str(k)+'</td>'+''.join(
        '<td>'+f"{next(r['accuracy'] for r in estimates if r['cohort']=='all_available' and r['k']==k and r['model']==model)*100:.1f}%"+'</td>'
        for model in ('all_12', 'token_position'))+'</tr>' for k in range(11))
    sections = []
    for cohort, heading in [('all_available', 'All available pairs'), ('common_sources', 'Same sources at every horizon'),
                            ('certain_only', 'Review-flagged endpoints excluded')]:
        counts = ', '.join(f"k={p['k']}: {p['balanced_pairs']}" for p in populations if p['cohort']==cohort)
        sections.append(f'<h2>{heading}</h2><figure><img src="{cohort}.svg" alt="{heading}: twelve-probe and token-position accuracy by horizon">'
                        f'<figcaption>Balanced evaluation pairs: {counts}. Band: majority-class subsampling variability; '
                        'it is not a confidence interval for new tasks. Dashed line: 50% chance.</figcaption></figure>')
    individual_link = '../figures/accuracy-trends.png' if run_key == 'glm-5.3-balanced-source-best12-v1/sales' else '../report.html'
    report = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Combined RH forecast</title>'
              '<style>body{font:17px/1.5 system-ui;max-width:1050px;margin:32px auto;padding:0 20px;color:#263746}'
              'img{width:100%}figure{margin:24px 0}figcaption{font-size:14px;color:#54616b}'
              'td,th{padding:5px 20px;text-align:right}a{color:#286aa4}</style>'
              f'<h1>{html.escape(title)} · twelve emotion probes</h1>'
              '<p>Training-fold standardization and fixed L2 logistic regression (C=1, balanced class weights, 0.5 cutoff). '
              f'The probes, labels, source/target pairs, and folds match the individual forecast. k=0 classifies the current {unit}; '
              f'k&gt;0 predicts the label exactly k later {unit}s. Later positives can reflect ongoing reward hacking.</p>'
              + ''.join(sections) + '<h2>Primary accuracy</h2><table><tr><th>k</th><th>12 probes</th><th>Token position</th></tr>'
              + table + f'</table><p><a href="{individual_link}">Individual forecast</a> · '
              '<a href="result-bundle.json">Result bundle</a> · <a href="metrics.csv">Metrics CSV</a></p></html>')
    (out / 'report.html').write_text(report)
    receipt = json.loads((out / 'verification.json').read_text())
    assert receipt['passed']
    receipt['output_hashes'] = {str(p.relative_to(out)): sha(p) for p in out.rglob('*')
                                if p.is_file() and p.name != 'verification.json'}
    (out / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dirs', nargs='+', type=Path)
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        main(run_dir.resolve())
