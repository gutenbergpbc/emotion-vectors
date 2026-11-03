"""Correlation figure and a compact report; no display-scale fitting."""
import csv
import html

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def undefined_correlation_label(metric, rows, result):
    paired = [r for r in rows if r['emotion'] == metric['emotion']
              and r['label'] not in result['exclude_labels']
              and r['rating'] is not None and result['primary_score'] in r]
    if not paired:
        return 'No paired observations'
    if len(paired) < 2:
        return 'Insufficient data: only 1 paired observation'
    ratings = {r['rating'] for r in paired}
    projections = {r[result['primary_score']] for r in paired}
    reasons = []
    if len(ratings) == 1:
        reasons.append(f'Constant self-report: always {next(iter(ratings)):g}')
    if len(projections) == 1:
        reasons.append('Constant probe score')
    return '; '.join(reasons) or 'Correlation undefined'


def report(root, result, rows):
    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    write_csv(root / 'metrics.csv', result['results'])
    write_csv(root / 'paired-scores.csv', rows)
    synthetic = 'SYNTHETIC DEMO · ' if result['synthetic'] else ''
    fig, ax = plt.subplots(figsize=(10, max(5, len(result['results']) * .38 + 2.5)))
    fig.subplots_adjust(left=.17, right=.97, bottom=.22, top=.81)
    for name, color, offset in [('pearson', '#387c74', -.12), ('spearman', '#8a68a1', .12)]:
        for index, row in enumerate(result['results']):
            value = row[name]
            y = index + offset
            if value is not None:
                ax.plot(value, y, 'o', color=color, markersize=5)
            if row[name + '_ci_low'] is not None:
                # Percentile intervals need not bracket the point estimate.
                ax.hlines(y, row[name + '_ci_low'], row[name + '_ci_high'], color=color, linewidth=2)
        ax.plot([], [], 'o', color=color, label=name.capitalize())
    for i, row in enumerate(result['results']):
        if row['status'] != 'ok':
            ax.text(.98, i, undefined_correlation_label(row, rows, result), transform=ax.get_yaxis_transform(),
                    ha='right', va='center', fontsize=8, color='#69736f')
    ax.set_yticks(range(len(result['results'])), [r['emotion'] for r in result['results']])
    ax.invert_yaxis()
    ax.set_xlim(-1, 1)
    ax.axvline(0, color='#89958b', linewidth=1)
    ax.grid(axis='x', alpha=.15)
    ax.spines[['top', 'right']].set_visible(False)
    score_label = 'raw projection' if result['primary_score'] == 'raw_projection' else 'projection display score'
    ax.set_xlabel(f'Correlation between {score_label} and self-report (0–10)')
    ax.legend(loc='lower right', frameon=False)
    unit = result.get('unit_label', 'sentence')
    fig.suptitle(synthetic + f'Projection vs. self-report: {unit}-level correlation', fontsize=14, fontweight='bold', y=.97)
    excluded = ', '.join(result['exclude_labels']) or 'none'
    fig.text(.5, .89, f"{result['model']} · {result['environment']} · excluded labels: {excluded}\n"
             'Outcomes pooled · 95% trace-cluster bootstrap intervals', ha='center', fontsize=10)
    counts = [r['traces'] for r in result['results']]
    trace_note = str(min(counts)) if min(counts) == max(counts) else f'{min(counts)}–{max(counts)}'
    fig.text(.5, .06, f'Each available {unit} pair has equal weight; longer traces contribute more pairs.\n'
             f"{trace_note} paired traces per emotion; {result['bootstrap_replicates']:,} requested whole-trace resamples.\n"
             'Undefined correlations are omitted. Self-report is a comparison baseline, not ground truth.',
             ha='center', fontsize=9)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(figures / f'score-correlations.{ext}', dpi=180)
    plt.close(fig)
    # Show the baseline distribution even when no matched probe scores exist.
    fig, ax = plt.subplots(figsize=(10, max(4, .34 * len(result['results']))))
    counts = np.array([[sum(r['emotion'] == row['emotion'] and r['rating'] == score and
                            r['label'] not in result['exclude_labels'] for r in rows)
                        for score in range(11)] for row in result['results']])
    image = ax.imshow(counts, aspect='auto', cmap='Blues')
    ax.set_xticks(range(11))
    ax.set_yticks(range(len(result['results'])), [r['emotion'] for r in result['results']])
    ax.set_xlabel('Self-report rating (0–10); missing responses excluded')
    ax.set_title(synthetic + 'Self-report rating counts')
    fig.colorbar(image, ax=ax, label='Sentence checkpoints')
    fig.tight_layout()
    fig.savefig(figures / 'rating-distribution.png', dpi=160)
    plt.close(fig)
    esc = html.escape
    notes = ''.join(f'<li>{esc(note)}</li>' for note in result['limitations'])
    methods = ''.join(f'<li><b>{esc(key)}:</b> {esc(value)}</li>' for key, value in result['methods'].items())
    (root / 'report.html').write_text(f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Emotion self-report baseline</title>
<style>body{{font:16px/1.55 system-ui;max-width:1100px;margin:32px auto;padding:0 24px;color:#243440;background:#fafbf9}}img{{width:100%;height:auto}}.note{{background:#edf3f6;padding:16px;border-radius:8px}}a{{color:#286a94}}</style></head><body>
<h1>{esc(synthetic + result['model'])} · self-report baseline</h1>
<p>{esc(result['environment'])} · {result['traces']} traces · {result['checkpoint_count']} checkpoints · {result['total_ratings']} responses.
Prompt: {esc(result['prompt'])}. Scale: 0 = not at all, 5 = moderately, 10 = extremely.</p>
<p>Response status counts: {esc(str(result['rating_status_counts']))}. Actual providers: {esc(str(result['providers']))}.</p>
<p><a href="index.html">Progress</a> · <a href="metrics.json">Metrics and methods JSON</a> · <a href="metrics.csv">Metrics CSV</a> ·
<a href="config.resolved.json">Configuration and provenance</a> · <a href="verification.json">Verification</a></p>
<img src="figures/score-correlations.png" alt="Pearson and Spearman correlations with trace-cluster bootstrap intervals">
<p><a href="figures/score-correlations.svg">SVG</a> · <a href="figures/score-correlations.pdf">PDF</a></p>
<img src="figures/rating-distribution.png" alt="Self-report rating counts by emotion">
<h2>Methods</h2><ul>{methods}</ul><div class="note"><ul>{notes}</ul></div>
<p>Row-level ratings and paired scores are saved beside this report. Raw requests and conversation text are available on disk, outside the dashboard's public allowlist.</p>
</body></html>''')
