"""Pool Coding/Sales observations, validate source parity, and bootstrap traces."""

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
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = _REPO
sys.path[:0] = [str(ROOT / 'src/experiments'),
                str(ROOT / 'src/experiments/emotion-self-report')]
from emotion_self_report.analysis import analyze, correlations, pearson, ranks

ACT = ROOT / 'runs/emotion-self-report-activations'
RUNS = {
    'glm53': ('GLM 5.3', ACT / 'glm53-complete-traces-v2/coding', ACT / 'glm53-complete-traces-v2/sales'),
    'gptoss120b': ('GPT-OSS 120B', ROOT / 'runs/emotion-self-report/gpt-oss-120b-coding-self-report-20261022-rerun-v1', ACT / 'gptoss-complete-traces-v1/sales'),
    'gemma4b': ('Gemma 3 4B', ACT / 'gemma4b-complete-traces-v1/coding', ACT / 'gemma4b-complete-traces-v1/sales'),
    'gemma27b': ('Gemma 3 27B', ACT / 'gemma27b-complete-traces-v1/coding', ACT / 'gemma27b-complete-traces-v1/sales'),
}


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def file_identity(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def csv_out(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


# Coding and Sales used the same extraction provenance for GLM and both Gemmas.
# GPT-OSS used separate replays: verify the actual vectors and layers match.
gp = ROOT / 'experiments/gpt_oss_self_report_plan_20261016/probe_replay_12'
coding_probes = read(gp / 'probes.json')['models']
sales_probes_path = ROOT / 'experiments/gpt_oss_sales_sentence_rh_12probes_20261016/probes.json'
sales_probes = read(sales_probes_path)['models']
assert set(coding_probes) == set(sales_probes)
for key in coding_probes:
    assert all(coding_probes[key].get(f) == sales_probes[key].get(f)
               for f in ('vector', 'layer', 'intercept'))
gpt_sales = read(ACT / 'gptoss-complete-traces-v1/provenance.json')
assert read(gp / 'run-spec.json')['model']['revision'] == gpt_sales['extraction_model']['revision']

summary, verification, provenance = [], [], []
for slug, (name, coding, sales) in RUNS.items():
    output = HERE / slug
    output.mkdir(parents=True, exist_ok=True)
    rows, sources, task_counts = [], [], {}
    source_metrics = [read(p / 'metrics.json') for p in (coding, sales)]
    a, b = source_metrics
    for key in ('model', 'prompt', 'scale', 'primary_score', 'exclude_labels'):
        assert a[key] == b[key], (name, key)
    assert a['primary_score'] == 'raw_projection' and a['scale'] == [0, 10]
    if slug != 'gptoss120b':
        assert a['projection_provenance'] == b['projection_provenance']
    for task, run, metric in zip(('coding', 'sales'), (coding, sales), source_metrics):
        source_rows = []
        with (run / 'paired-scores.csv').open() as stream:
            for r in csv.DictReader(stream):
                row = {k: r[k] for k in ('trace_id', 'unit_id', 'emotion', 'label', 'rating_status', 'provider')}
                row.update(environment=task, original_trace_id=row['trace_id'],
                           trace_id=task + '::' + row['trace_id'],
                           rating=float(r['rating']) if r['rating'] else None)
                if r.get('raw_projection'):
                    row['raw_projection'] = float(r['raw_projection'])
                    assert math.isfinite(row['raw_projection'])
                source_rows.append(row)
        eligible = [r for r in source_rows if r['label'] not in a['exclude_labels']
                    and r['rating'] is not None and 'raw_projection' in r]
        for m in metric['results']:
            rr = [r for r in eligible if r['emotion'] == m['emotion']]
            assert len(rr) == m['pairs']
            for key, value in zip(('pearson', 'spearman'), correlations(
                    [r['raw_projection'] for r in rr], [r['rating'] for r in rr])):
                assert (value is None and m[key] is None) or np.isclose(value, m[key], atol=1e-12)
        task_counts[task] = {'traces': len({r['trace_id'] for r in eligible}),
                             'checkpoints': len({(r['trace_id'], r['unit_id']) for r in eligible})}
        rows.extend(source_rows)
        sources.extend(file_identity(run / f) for f in ('metrics.json', 'paired-scores.csv'))
    keys = [(r['trace_id'], r['unit_id'], r['emotion']) for r in rows]
    assert len(keys) == len(set(keys)), 'Duplicate observation'
    spec = dict(identity=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
                model=a['model'], environment='coding+sales', synthetic=False, mode='pooled_saved_pairs',
                prompt=a['prompt'], correlation_score='raw_projection', bootstrap_replicates=2000,
                seed=20261021, exclude_labels=a['exclude_labels'], emotions=[r['emotion'] for r in a['results']])
    print(f'{name}: pooling {task_counts}', flush=True)
    result = analyze(rows, spec)
    result.update(display_name=name, task_counts=task_counts, sources=sources,
                  paired_trace_count=sum(t['traces'] for t in task_counts.values()),
                  paired_checkpoint_count=sum(t['checkpoints'] for t in task_counts.values()),
                  source_limitations={task: m['limitations'] for task, m in zip(('coding', 'sales'), source_metrics)})
    result['methods']['point_estimate'] = 'Concatenate Coding and Sales eligible raw-projection/self-report pairs per emotion, then recompute Pearson and Spearman. Each checkpoint pair has equal weight; no task balancing or taskwise standardization.'
    result['methods']['interval'] = '95% percentile intervals from 2,000 unstratified resamples of all whole traces across the two tasks, with replacement. Trace IDs are namespaced by environment; ranks are recomputed in every draw.'
    result['methods']['task_adjusted_check'] = 'Descriptive Pearson correlation after subtracting each task mean from projection and rating separately; no interval. Also residualize global midranks by task to give a task-adjusted rank-correlation diagnostic.'
    result['limitations'] += ['Pooled correlations can be driven by differences between task means and by unequal task coverage; they do not establish within-task agreement.']
    if slug == 'gemma27b':
        result['limitations'].append('Gemma 27B has only four traces (one Coding, three Sales); its trace bootstrap intervals are highly uncertain.')
    if slug == 'gptoss120b':
        result['limitations'].append('GPT-OSS Coding uses sentence endpoints, Sales uses mixed saved checkpoints; sampling and conversation rendering differ. Source-run limitations are retained.')
    for record in result['results']:
        rr = [r for r in rows if r['emotion'] == record['emotion'] and r['label'] not in spec['exclude_labels']
              and r['rating'] is not None and 'raw_projection' in r]
        x, y = (np.array([r[k] for r in rr], dtype=float) for k in ('raw_projection', 'rating'))
        assert record['pairs'] == sum(next(v for v in m['results'] if v['emotion'] == record['emotion'])['pairs'] for m in source_metrics)
        assert np.isclose(record['pearson'], pearsonr(x, y).statistic, atol=1e-12)
        assert np.isclose(record['spearman'], spearmanr(x, y).statistic, atol=1e-12)
        task = np.array([r['environment'] for r in rr])
        for method, xx, yy in [('pearson', x.copy(), y.copy()), ('rank', ranks(x), ranks(y))]:
            for env in ('coding', 'sales'):
                mask = task == env
                xx[mask] -= xx[mask].mean()
                yy[mask] -= yy[mask].mean()
            record['task_adjusted_' + method] = pearson(xx, yy)
        summary.append({'model': name, **record})
    write(output / 'metrics.json', result)
    csv_out(output / 'metrics.csv', result['results'])
    csv_out(output / 'paired-scores.csv', rows)
    provenance.append({'model': name, 'task_counts': task_counts, 'sources': sources})
    verification.append({'model': name, 'source_coefficients_reproduced': True,
                         'pooled_coefficients_match_scipy': True, 'pair_counts_add_up': True,
                         'unique_observation_keys': True, 'paired_trace_count': sum(t['traces'] for t in task_counts.values())})
csv_out(HERE / 'summary.csv', summary)
write(HERE / 'sources.json', provenance)
write(HERE / 'verification.json', verification)
print('Verified source parity, observation uniqueness, counts, and SciPy agreement for all 48 model/emotion results (96 coefficients).', flush=True)
