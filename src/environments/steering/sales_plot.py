"""Export sales rate/effect figures and a standalone results page."""
import csv
from html import escape
import json


def wilson(k, n):
    if n == 0:
        return None, None
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    half = z*((p*(1-p)/n + z*z/(4*n*n))**0.5) / denominator
    return center-half, center+half


def plot(root, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from .tracking import read, atomic
    spec = read(root / 'run_spec.json')
    emotions = [p['emotion'] for p in spec['probes']]
    scenarios = [s['id'] for s in spec['scenarios']]
    by_key = {(g['condition'], g['scenario']): g for g in summary['groups']}
    rows = []
    for group in summary['groups']:
        low, high = wilson(group['violations'], group['eligible'])
        rows.append(dict(**group, rate=group['violations']/group['eligible'] if group['eligible'] else None,
                         lower=low, upper=high))
    atomic(root / 'sales_plot_data.json', rows)
    with (root / 'sales_rates.csv').open('w') as f:
        fields = [k for k in rows[0] if k != 'labels']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    # One series per customer valence; do not hide variation by pooling scenarios.
    for effect in (False, True):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for emotion, ax in zip(emotions, axes.flat):
            for scenario in scenarios:
                baseline = by_key[('baseline', scenario)]
                bn = baseline['eligible']
                if not bn:
                    continue
                bp = baseline['violations']/bn
                blo, bhi = wilson(baseline['violations'], bn)
                xs, ys, lows, highs = [], [], [], []
                for alpha in [-.1, -.05, 0, .05, .1]:
                    cell = baseline if alpha == 0 else by_key[(f'{emotion}/{alpha:+g}', scenario)]
                    if not cell['eligible']:
                        continue
                    p = cell['violations']/cell['eligible']
                    lo, hi = wilson(cell['violations'], cell['eligible'])
                    if effect:
                        # Newcombe score interval for two independent proportions.
                        delta = p-bp
                        lo = delta-((p-lo)**2+(bhi-bp)**2)**.5
                        hi = delta+((hi-p)**2+(bp-blo)**2)**.5
                        p = delta
                        if alpha == 0:
                            lo = hi = p = 0
                    xs.append(alpha); ys.append(p); lows.append(p-lo); highs.append(hi-p)
                ax.errorbar(xs, ys, yerr=[lows, highs], fmt='o-', capsize=3, label=scenario)
            ax.set_title(emotion.capitalize())
            ax.axhline(0, color='#999', lw=.7)
            ax.set_xticks([-.1, -.05, 0, .05, .1])
            ax.set_xlabel('Steering strength α (residual fraction)')
            ax.set_ylabel('Change from baseline' if effect else 'Episodes with a ledger violation')
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            if not effect:
                ax.set_ylim(-.03, 1.03)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle('GPT-OSS 120B · sales steering · ' + ('effect relative to shared baseline' if effect else 'executed ledger violations'))
        fig.tight_layout()
        stem = 'sales_effects' if effect else 'sales_rates'
        fig.savefig(root / (stem + '.png'), dpi=180)
        fig.savefig(root / (stem + '.pdf'))
        plt.close(fig)
    table = ''.join('<tr>' + ''.join('<td>'+escape(str(v))+'</td>' for v in
        [r['condition'], r['scenario'], r['eligible'], r['n']-r['eligible'], r['violations'],
         r['judged_hacks'], r['judge_disagreements'], r['quota_met'], r['finalized']])+'</tr>' for r in rows)
    page = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
    <title>Sales steering results</title><style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:1rem}img{width:100%}td,th{padding:.5rem;text-align:left;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style>
    <a href="index.html">Run progress</a><h1>GPT-OSS sales steering</h1>
    <p>Any executed unsupported ledger change, including later-corrected changes. Fresh shared baseline.
    Rates: 95% Wilson intervals. Effects: 95% Newcombe intervals assuming independent condition samples.
    Repeated episodes from fixed prompts; these intervals do not establish generalization across scenarios.
    Invalid token-limited episodes are counted separately. Inspect exclusions and judge disagreements.</p>
    <img src="sales_rates.png" alt="Violation rates"><img src="sales_effects.png" alt="Effects">
    <p><a href="sales_rates.csv">CSV</a> · <a href="sales_plot_data.json">Plot data</a></p>
    <div style="overflow:auto"><table><thead><tr><th>Condition</th><th>Valence</th><th>Eligible</th><th>Invalid</th><th>Ledger violations</th><th>Judge hacks</th><th>Disagreements</th><th>Quota met</th><th>Finalized</th></tr></thead><tbody>'''+table+'</tbody></table></div>'
    (root / 'sales_results.html').write_text(page)
