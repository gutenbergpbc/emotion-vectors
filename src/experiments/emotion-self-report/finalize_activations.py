"""Validate complete replay coverage, join existing ratings, and report by dataset.

Usage: uv run python emotion-self-report/finalize_activations.py RUN_DIRECTORY
No self-report API calls or GPU execution. Input score rows must use original
trace/checkpoint IDs and character offsets in the source assistant message.
"""
from __future__ import annotations

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
from collections import Counter, defaultdict
import hashlib
import html
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from emotion_self_report.analysis import analyze, join
from emotion_self_report.reporting import report, write_csv
from select_activation_traces import read, sha, EMOTIONS


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def validate(points, ratings, scores):
    expected = {}
    for point in points:
        key = (point['trace_id'], point['unit_id'])
        if key in expected:
            raise ValueError(f'Duplicate checkpoint: {key}')
        expected[key] = point
    wanted = {(tid, uid, emotion) for tid, uid in expected for emotion in EMOTIONS}
    indexed = {}
    for score in scores:
        key = (score['trace_id'], score['unit_id'], score['emotion'])
        if key in indexed or key not in wanted:
            raise ValueError(f'Duplicate or unexpected projection: {key}')
        point = expected[key[:2]]
        value = score['raw_projection']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Nonfinite or nonnumeric projection: {key}')
        if (score['message_index'] != point['message_index']
                or score['sentence_char_end'] != point['char_end']
                or not 0 < score['probe_char_end'] <= point['char_end']):
            raise ValueError(f'Misaligned or future projection: {key}')
        indexed[key] = score
    if set(indexed) != wanted:
        raise ValueError(f'Incomplete projections: {len(wanted - set(indexed))} missing')
    rating_keys = [(r['trace_id'], r['unit_id'], r['emotion']) for r in ratings]
    if len(set(rating_keys)) != len(rating_keys) or set(rating_keys) != wanted:
        raise ValueError('Rating coverage differs from full selected checkpoints')
    for row in ratings:
        value = row['rating']
        if (row['rating_status'] != 'numeric' or isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 10):
            raise ValueError('Non-numeric selected self-report')
    return indexed


def verify_correlations(rows, result):
    from scipy.stats import pearsonr, spearmanr
    for metric in result['results']:
        paired = [r for r in rows if r['emotion'] == metric['emotion'] and
                  r['label'] not in result['exclude_labels']]
        x, y = [r['raw_projection'] for r in paired], [r['rating'] for r in paired]
        defined = len(x) >= 2 and len(set(x)) > 1 and len(set(y)) > 1
        for name, function in [('pearson', pearsonr), ('spearman', spearmanr)]:
            expected = float(function(x, y).statistic) if defined else None
            actual = metric[name]
            if (actual is None) != (expected is None) or (actual is not None and abs(actual - expected) > 1e-10):
                raise ValueError(f'Independent correlation mismatch: {metric["emotion"]}/{name}')


def trajectory_viewer(root, rows, selection):
    grouped = defaultdict(dict)
    for row in rows:
        key = row['unit_id']
        point = grouped[row['trace_id']].setdefault(key, {
            'unit_id': key, 'sentence_index': row['sentence_index'], 'message_index': row['message_index'],
            'char_start': row['char_start'], 'char_end': row['char_end'], 'text': row.get('text', ''),
            'dataset': row['dataset'], 'checkpoint_kind': row.get('checkpoint_kind', 'prose'), 'emotions': {}})
        point['emotions'][row['emotion']] = {'projection': row['raw_projection'], 'rating': row['rating']}
    traces = [{'id': tid, 'points': sorted(points.values(), key=lambda p: (p['message_index'], p['char_end']))}
              for tid, points in grouped.items()]
    write(root / 'trajectories.json', traces)
    payload = json.dumps({'traces': traces, 'emotions': EMOTIONS}, ensure_ascii=True).replace('<', '\\u003c')
    template = '''<!doctype html><meta charset="utf-8"><title>Emotion trajectories</title>
<style>body{font:15px system-ui;margin:24px;color:#253b40;background:#fafbf9}select{max-width:100%;padding:8px}#plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:16px}section{background:white;padding:12px;border:1px solid #ddd}svg{width:100%}p{max-width:1000px}pre{white-space:pre-wrap}h3{margin:0}</style>
<h1>__MODEL__ · emotion trajectories</h1><p>All selected checkpoints in each complete trace, in source order.
Teal: raw probe projection (left axis). Purple: self-report 0–10 (right axis).
Projection axes vary by emotion and trace; they are not calibrated emotion intensity. Points include code/command endpoints where collected.</p>
<select id="trace"></select><p id="summary"></p><div id="plots"></div><h2>Checkpoint text</h2><div id="text"></div>
<script>const data=__DATA__;const selector=document.querySelector('#trace');
data.traces.forEach((t,i)=>{let o=document.createElement('option');o.value=i;o.textContent=t.id;selector.append(o)});
function draw(){const t=data.traces[Number(selector.value)],pts=t.points;document.querySelector('#summary').textContent=pts[0].dataset+' · '+pts.length+' checkpoints';
const area=document.querySelector('#plots');area.replaceChildren();
data.emotions.forEach(e=>{let section=document.createElement('section'),title=document.createElement('h3');title.textContent=e;section.append(title);
let vs=pts.map(p=>p.emotions[e].projection),lo=Math.min(...vs),hi=Math.max(...vs);if(lo===hi){lo-=.5;hi+=.5}let pad=(hi-lo)*.08;lo-=pad;hi+=pad;
const x=i=>48+i*264/Math.max(1,pts.length-1),yp=v=>150-(v-lo)/(hi-lo)*120,yr=v=>150-v/10*120;
let svg='<svg viewBox="0 0 370 185"><path d="M48 30 V150 H312 V30" fill="none" stroke="#aaa"/>';
[[vs,yp,'#387c74'],[pts.map(p=>p.emotions[e].rating),yr,'#8a68a1']].forEach(([values,y,color])=>{svg+='<polyline fill="none" stroke="'+color+'" points="'+values.map((v,i)=>x(i)+','+y(v)).join(' ')+'"/>';values.forEach((v,i)=>svg+='<circle r="2.2" fill="'+color+'" cx="'+x(i)+'" cy="'+y(v)+'"><title>Checkpoint '+(i+1)+': '+v+'</title></circle>')});
svg+='<g font-size="10" fill="#47555a"><text x="2" y="34">'+hi.toPrecision(3)+'</text><text x="2" y="151">'+lo.toPrecision(3)+'</text><text x="320" y="34">10</text><text x="320" y="151">0</text><text x="48" y="170">1</text><text x="300" y="170">'+pts.length+'</text></g></svg>';section.insertAdjacentHTML('beforeend',svg);area.append(section)});
let texts=document.querySelector('#text');texts.replaceChildren();pts.forEach((p,i)=>{let d=document.createElement('details'),s=document.createElement('summary'),b=document.createElement('pre');s.textContent='Checkpoint '+(i+1)+' · '+p.checkpoint_kind+' · message '+p.message_index+' · characters '+p.char_start+'–'+p.char_end;b.textContent=p.text;d.append(s,b);texts.append(d)})}selector.onchange=draw;draw();</script>'''
    (root / 'trajectories.html').write_text(template.replace('__MODEL__', html.escape(selection['model'])).replace('__DATA__', payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--bootstrap-replicates', type=int, default=2000)
    args = parser.parse_args()
    root = args.root.resolve()
    selection = read(root / 'selection.json')
    points, ratings, scores = [read(root / name) for name in ('checkpoints.json', 'ratings.json', 'probe-scores.json')]
    provenance = read(root / 'provenance.json')
    if provenance['model'] != selection['model']:
        raise ValueError('Provenance source model differs from selected self-report model')
    indexed = validate(points, ratings, scores)
    points_by_key = {(p['trace_id'], p['unit_id']): p for p in points}
    datasets = {t['trace_id']: t['dataset'] for t in selection['traces']}
    ratings = [{**points_by_key[(r['trace_id'], r['unit_id'])], **r,
                'dataset': datasets[r['trace_id']], 'provider': r.get('provider')} for r in ratings]
    rows = join(ratings, indexed)
    if len(points) != selection['selected_checkpoints'] or len(datasets) != selection['selected_traces']:
        raise ValueError('Selected counts differ')
    actual_counts = Counter(p['trace_id'] for p in points)
    if any(actual_counts[t['trace_id']] != t['checkpoints'] for t in selection['traces']):
        raise ValueError('Per-trace completeness differs from selection')
    identity_files = ['selection.json', 'checkpoints.json', 'ratings.json', 'probe-scores.json', 'provenance.json']
    hashes = {name: sha(root / name) for name in identity_files}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    write(root / 'paired-scores.json', rows)
    write_csv(root / 'paired-scores.csv', rows)
    trajectory_viewer(root, rows, selection)
    for dataset in sorted(set(datasets.values())):
        subset = [r for r in rows if r['dataset'] == dataset]
        spec = {'identity': identity, 'model': selection['model'], 'environment': dataset,
                'synthetic': False, 'mode': 'activation_replay_existing_ratings', 'prompt': 'task_stance',
                'correlation_score': 'raw_projection', 'bootstrap_replicates': args.bootstrap_replicates,
                'seed': 20261021, 'exclude_labels': ['ambiguous'], 'emotions': EMOTIONS}
        result = analyze(subset, spec)
        result['unit_label'] = 'checkpoint'
        result['limitations'][-1] = selection['completeness']
        result['limitations'].extend([selection['selection_policy'],
            'Replay uses the archived visible conversation. API provider weights/precision and hidden reasoning may differ from the local extraction model; see provenance.json.',
            'A code-block endpoint is one checkpoint, not multiple sentence observations. Single-checkpoint traces cannot show within-trace trends.'])
        result['projection_provenance'] = provenance
        verify_correlations(subset, result)
        output = root / dataset
        output.mkdir(exist_ok=True)
        write(output / 'metrics.json', result)
        write(output / 'config.resolved.json', {**spec, 'selection': selection, 'provenance': provenance})
        report(output, result, subset)
        # Shared report links expect a progress page; link to the actual trajectories.
        report_path = output / 'report.html'
        report_path.write_text(report_path.read_text().replace('href="index.html">Progress', 'href="../trajectories.html">Trajectories'))
    verification = {'status': 'complete', 'identity': identity, 'input_sha256': hashes,
                    'analysis_code_sha256': {str(path.relative_to(HERE)): sha(path) for path in
                        [Path(__file__), HERE / 'emotion_self_report/analysis.py',
                         HERE / 'emotion_self_report/reporting.py']},
                    'traces': len(datasets), 'checkpoints': len(points), 'projection_rows': len(scores),
                    'all_12_numeric_ratings_and_finite_projections': True,
                    'complete_trace_checkpoint_coverage': True, 'causal_endpoint_checks': True,
                    'independent_scipy_correlations': True,
                    'lagged_checkpoint_emotion_rows': sum(s['probe_char_end'] < s['sentence_char_end'] for s in scores),
                    'dataset_traces': selection['dataset_traces'], 'dataset_checkpoints': selection['dataset_checkpoints']}
    for dataset in sorted(set(datasets.values())):
        write(root / dataset / 'verification.json', verification)
    write(root / 'verification.json', verification)
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
