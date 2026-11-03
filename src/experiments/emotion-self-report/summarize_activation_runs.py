"""Build a local index of the four complete-trace activation runs."""

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

from pathlib import Path
import csv
import html
import json

from select_activation_traces import ROOT, read, sha

RUNS = [('GPT-OSS 120B', 'gptoss-complete-traces-v1'),
        ('Gemma 4B', 'gemma4b-complete-traces-v1'),
        ('Gemma 27B', 'gemma27b-complete-traces-v1'),
        ('GLM 5.3', 'glm53-complete-traces-v2')]


def main():
    base = ROOT / 'runs/emotion-self-report-activations'
    rows, table = [], []
    for model, name in RUNS:
        root = base / name
        selection = read(root / 'selection.json')
        verification_path = root / 'verification.json'
        complete = False
        verification = {}
        if verification_path.exists():
            verification = read(verification_path)
            complete = verification['status'] == 'complete' and all(
                sha(root / filename) == expected for filename, expected in verification['input_sha256'].items())
            if not complete:
                raise ValueError(f'Completed run files changed: {name}')
        row = {'model': model, 'status': 'complete' if complete else 'pending',
               'complete_traces': selection['selected_traces'] if complete else 0,
               'collected_checkpoints': selection['selected_checkpoints'] if complete else 0,
               'selected_traces': selection['selected_traces'],
               'selected_checkpoints': selection['selected_checkpoints'],
               'projection_rows': verification.get('projection_rows', 0), 'run': name}
        rows.append(row)
        links = ''
        if complete:
            links = f'<a href="{name}/trajectories.html">Trajectories</a> · '
            links += ' · '.join(f'<a href="{name}/{dataset}/report.html">{dataset.title()} correlations</a>'
                                for dataset in sorted(selection['dataset_traces']))
            links += f' · <a href="{name}/paired-scores.csv">Paired CSV</a> · <a href="{name}/provenance.json">Provenance</a>'
        table.append(f'<tr><td>{html.escape(model)}</td><td>{row["status"]}</td>'
                     f'<td>{row["complete_traces"]}</td><td>{row["collected_checkpoints"]}</td>'
                     f'<td>{row["projection_rows"]:,}</td><td>{links}</td></tr>')
    summary = {'status': 'complete' if all(r['status'] == 'complete' for r in rows) else 'in_progress',
               'runs': rows, 'total_complete_traces': sum(r['complete_traces'] for r in rows),
               'total_checkpoints': sum(r['collected_checkpoints'] for r in rows),
               'total_projection_rows': sum(r['projection_rows'] for r in rows)}
    (base / 'coverage.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (base / 'coverage.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    page = '''<!doctype html><meta charset="utf-8"><title>Emotion activation collection</title>
<style>body{font:16px/1.5 system-ui;max-width:1300px;margin:40px auto;padding:0 24px;color:#253b40;background:#fafbf9}table{border-collapse:collapse;width:100%}th,td{padding:12px;text-align:left;border-bottom:1px solid #ddd}a{color:#286a94}</style>
<h1>Emotion probes vs. self-report</h1><p>All 12 existing selected probes at every frozen checkpoint in each selected complete trace. GPT-OSS coding excluded. Reports separate sales and coding.</p>
<table><thead><tr><th>Model</th><th>Status</th><th>Complete traces</th><th>Checkpoints</th><th>Probe scores</th><th>Results</th></tr></thead><tbody>__TABLE__</tbody></table>
<p>The target is 2,000 checkpoints per model, allowing an overshoot to finish a trace; smaller fully rated populations are retained in full. Larger pools favor shorter traces, so these are not representative samples. Completeness refers to every existing self-report checkpoint, including command/code-block endpoints where applicable.</p>
<p>Primary readout: raw projection at the last causally completed assistant token at each checkpoint. Self-report remains 0–10. Correlations use 2,000 whole-trace bootstrap replicates. Single-checkpoint traces have no within-trace trajectory; Gemma 4B coding consists of 100 such code-block endpoints. Gemma 27B has only three sales traces and one coding trace; coding has no trace-bootstrap interval.</p>
<p>Model replay preserves the archived visible conversation before the self-report question. Provider weights, precision, or hidden reasoning may differ from local replay; exact extraction identities and normalization are recorded in each run's provenance.</p>
<p><a href="coverage.json">Coverage JSON</a> · <a href="coverage.csv">Coverage CSV</a></p>'''
    (base / 'index.html').write_text(page.replace('__TABLE__', ''.join(table)))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
