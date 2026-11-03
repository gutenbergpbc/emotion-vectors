"""Paired sentence correlations with whole-trace bootstrap intervals.

Promoted from gpt_oss_self_report_plan_20261016/analyze_score_correlations.py.
Ranks are recomputed in every bootstrap replicate; constants remain undefined.
"""
from collections import Counter, defaultdict

import numpy as np

from shared.io import require


def ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def pearson(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    require(a.shape == b.shape and a.ndim == 1 and np.isfinite(a).all() and np.isfinite(b).all(),
            'Correlation requires finite, aligned vectors')
    if len(a) < 2:
        return None
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(np.dot(a, b) / denominator, -1, 1)) if denominator > 0 else None


def correlations(a, b):
    return pearson(a, b), pearson(ranks(a), ranks(b))


def join(ratings, scores):
    joined, seen = [], set()
    for rating in ratings:
        key = (rating['trace_id'], rating['unit_id'], rating['emotion'])
        require(key not in seen, 'Duplicate rating')
        seen.add(key)
        joined.append({**rating, **scores.get(key, {})})
    return sorted(joined, key=lambda r: (r['trace_id'], r['sentence_index'], r['emotion']))


def analyze(rows, spec, progress=None):
    rng = np.random.default_rng(spec['seed'])
    result = dict(identity=spec['identity'], model=spec['model'], environment=spec['environment'],
                  synthetic=spec['synthetic'], mode=spec['mode'], prompt=spec['prompt'], scale=[0, 10],
                  primary_score=spec['correlation_score'], bootstrap_replicates=spec['bootstrap_replicates'],
                  seed=spec['seed'], exclude_labels=spec['exclude_labels'],
                  total_ratings=len(rows), rating_status_counts=dict(Counter(r['rating_status'] for r in rows)),
                  providers=dict(Counter(r['provider'] or 'unreported' for r in rows)),
                  traces=len({r['trace_id'] for r in rows}),
                  checkpoint_count=len({(r['trace_id'], r['unit_id']) for r in rows}),
                  methods={
                      'point_estimate': 'Pooled sentence pairs per emotion; each pair has equal weight, so longer traces contribute more observations.',
                      'pearson': 'Centered dot product divided by product of Euclidean norms.',
                      'spearman': 'Pearson correlation of average ranks; ties receive midranks.',
                      'interval': '95% percentile intervals from resampling whole traces with replacement; recompute ranks in every draw. At least two paired traces required.',
                      'within_trace': 'Arithmetic mean of separate within-trace coefficients, using traces with at least three pairs and nonconstant scores. Descriptive only.',
                      'missing': 'Null, nonnumeric and missing probe scores are excluded pairwise, never replaced with zero.',
                  },
                  limitations=[
                      'Self-report is an elicited baseline, not ground truth for emotion or a claim of subjective experience. Correlation does not establish probe accuracy.',
                      'API replay uses visible assistant prefixes. Native model revision, template and hidden reasoning are not guaranteed to match probe extraction.',
                      'Bootstrap intervals account for dependence within traces, not shared prompts across different traces. They are exploratory, without multiple-comparison correction.',
                      'Positive affine scaling does not change correlation; discretized 1–10 projection bands can change it and are not calibrated emotional intensity.',
                      'Any retained final fragment is a checkpoint; code blocks and headings are context, not automatic sentence targets.',
                  ], results=[])
    for index, emotion in enumerate(spec['emotions']):
        all_rows = [r for r in rows if r['emotion'] == emotion]
        eligible = [r for r in all_rows if r['label'] not in spec['exclude_labels']]
        numeric = [r for r in eligible if r['rating'] is not None]
        paired = [r for r in numeric if spec['correlation_score'] in r]
        x = np.array([r[spec['correlation_score']] for r in paired])
        y = np.array([r['rating'] for r in paired])
        groups = defaultdict(list)
        for i, row in enumerate(paired):
            groups[row['trace_id']].append(i)
        indices = [np.asarray(v) for v in groups.values()]
        point = correlations(x, y)
        raw = correlations([r['raw_projection'] for r in paired], y)
        bootstrap, within = [[], []], [[], []]
        attempted = 0
        if len(indices) >= 2 and point != (None, None):
            for draw in range(spec['bootstrap_replicates']):
                attempted += 1
                selected = rng.integers(len(indices), size=len(indices))
                ix = np.concatenate([indices[i] for i in selected])
                for k, value in enumerate(correlations(x[ix], y[ix])):
                    if value is not None:
                        bootstrap[k].append(value)
                if progress and (draw + 1) % 200 == 0:
                    progress.stage('analyze', index, len(spec['emotions']), 'emotions',
                                   f"{emotion}: bootstrap {draw + 1} / {spec['bootstrap_replicates']}")
        for ix in indices:
            if len(ix) >= 3:
                for k, value in enumerate(correlations(x[ix], y[ix])):
                    if value is not None:
                        within[k].append(value)
        record = dict(emotion=emotion, total=len(all_rows), excluded_label=len(all_rows) - len(eligible),
                      missing_rating=len(eligible) - len(numeric), missing_probe_score=len(numeric) - len(paired),
                      pairs=len(paired), traces=len(indices),
                      unique_projection_scores=len(np.unique(x)), unique_self_report_scores=len(np.unique(y)))
        for k, name in enumerate(('pearson', 'spearman')):
            interval = np.quantile(bootstrap[k], [.025, .975]).tolist() if bootstrap[k] else [None, None]
            record.update({name: point[k], name + '_ci_low': interval[0], name + '_ci_high': interval[1],
                           name + '_valid_bootstraps': len(bootstrap[k]),
                           name + '_undefined_bootstraps': attempted - len(bootstrap[k]),
                           name + '_skipped_bootstraps': spec['bootstrap_replicates'] - attempted,
                           'raw_' + name: raw[k],
                           'mean_within_trace_' + name: float(np.mean(within[k])) if within[k] else None,
                           'eligible_within_trace_' + name: len(within[k])})
        record['status'] = 'ok' if point != (None, None) else ('no_pairs' if not paired else 'constant_or_insufficient')
        result['results'].append(record)
        if progress:
            progress.stage('analyze', index + 1, len(spec['emotions']), 'emotions', emotion)
    return result
