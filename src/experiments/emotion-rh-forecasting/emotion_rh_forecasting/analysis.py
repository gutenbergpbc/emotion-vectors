"""Frozen directions; training-only scalar cutoffs; group-disjoint evaluation.

Threshold and sampling rules preserve the historical GPT-OSS coding experiment.
Horizon zero uses the same strictly causal source pooling as future horizons.
"""
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from shared.io import require

EMOTIONS = sorted(['hostile', 'frustrated', 'excited', 'appreciative', 'sad', 'afraid',
                   'surprised', 'anxious', 'guilty', 'ashamed', 'desperate', 'calm'])
BASELINES = ['token_position', 'always_not_hacking', 'oracle_label_persistence']


def fit_threshold(x, y):
    """Historical midpoint/constant candidates and deterministic polarity ties."""
    x, y = np.asarray(x, float), np.asarray(y, int)
    require(x.ndim == y.ndim == 1 and len(x) == len(y) and np.isfinite(x).all()
            and set(y) == {0, 1}, 'Threshold training needs finite scores and both classes')
    values, inverse = np.unique(x, return_inverse=True)
    pos = np.bincount(inverse, weights=y, minlength=len(values))
    neg = np.bincount(inverse, weights=1-y, minlength=len(values))
    cp, cn = np.r_[0., np.cumsum(pos)], np.r_[0., np.cumsum(neg)]
    high_ba = .5 * ((pos.sum()-cp)/pos.sum() + cn/neg.sum())
    cutoffs = np.r_[np.nextafter(values[0], -np.inf), values[:-1]/2+values[1:]/2,
                    np.nextafter(values[-1], np.inf)]
    require(np.isfinite(cutoffs).all(), 'Threshold candidates overflowed')
    candidates = []
    for polarity, scores in [(1, high_ba), (-1, 1-high_ba)]:
        for i in np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-12, rtol=0)):
            candidates.append((float(scores[i]), -abs(i-(len(cutoffs)-1)/2), polarity, int(i)))
    ba, _, polarity, i = max(candidates)
    return dict(cutoff=float(cutoffs[i]), polarity=polarity, training_balanced_accuracy=ba,
                scale=max(float(x.std()), 1e-12), training_n=len(y))


def predict_threshold(x, rule):
    x = np.asarray(x)
    return x >= rule['cutoff'] if rule['polarity'] == 1 else x < rule['cutoff']


def assign_folds(rows, evaluation, seed):
    """Freeze once at h=0, including unscorable rows; never split sentences randomly."""
    method = evaluation['method']
    if method == 'held_out_task':
        tasks = sorted({r['task'] for r in rows})
        require(len(tasks) >= 2 and all(tasks), 'Task-held-out evaluation needs at least two named tasks')
        folds = [tasks.index(r['task']) for r in rows]
    elif method == 'provided':
        require(all(type(r.get('fold')) is int and r['fold'] >= 0 for r in rows), 'Missing/invalid frozen fold')
        folds = [r['fold'] for r in rows]
    else:
        n = evaluation['folds']
        require(len({r['group_id'] for r in rows}) >= n, 'Not enough source groups for requested folds')
        groups = np.array([r['group_id'] for r in rows])
        y = np.array([r['hacking'] for r in rows])
        folds = np.full(len(rows), -1, dtype=int)
        for fold, (_, test) in enumerate(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed).split(
                np.zeros(len(rows)), y, groups)):
            folds[test] = fold
    require(len(set(folds)) >= 2, 'Evaluation needs at least two folds')
    trace_folds, group_folds = {}, {}
    for row, fold in zip(rows, folds):
        row['fold'] = int(fold)
        for mapping, key in [(trace_folds, row['trace_id']), (group_folds, row['group_id'])]:
            require(mapping.setdefault(key, int(fold)) == fold, f'Trace/source group crosses folds: {key}')
    return {'method': method, 'trace_folds': trace_folds, 'group_folds': group_folds}


def make_pairs(rows, horizon, allowed=None, certain_only=False):
    require(type(horizon) is int and horizon >= 0, 'Horizon must be a nonnegative integer')
    traces = defaultdict(list)
    for row in rows:
        traces[row['trace_id']].append(row)
    result = []
    for trace in sorted(traces):
        sentences = sorted(traces[trace], key=lambda r: r['sentence_index'])
        require([r['sentence_index'] for r in sentences] == list(range(len(sentences))),
                f'Incomplete original sentence order: {trace}')
        # Pair before exclusions. Missing features/uncertain endpoints must not
        # compress the sequence or change the meaning of i+h.
        for i in range(max(0, len(sentences)-horizon)):
            source, target = sentences[i], sentences[i+horizon]
            if source['projections'] is None or (allowed is not None and (trace, source['unit_id']) not in allowed):
                continue
            if certain_only and (source['needs_review'] or target['needs_review']):
                continue
            require(source['last_feature_token_char_end'] <= source['char_end'], 'Future token in source features')
            if horizon and source['turn_id'] == target['turn_id']:
                require(source['char_end'] <= target['char_start'], 'Overlapping source/target sentence spans')
            result.append(dict(source, horizon=horizon, hacking=target['hacking'],
                               source_hacking=source['hacking'], target_unit_id=target['unit_id'],
                               target_sentence_index=target['sentence_index'], target_turn_id=target['turn_id'],
                               source_needs_review=source['needs_review'], target_needs_review=target['needs_review']))
    return result


def balanced_groups(y, folds):
    groups = []
    for fold in sorted(set(folds)):
        a, b = np.flatnonzero((folds == fold) & y), np.flatnonzero((folds == fold) & ~y)
        require(len(a) and len(b), f'Fold {fold} has only one target class; balanced evaluation is undefined')
        groups.append((a, b, min(len(a), len(b))))
    return groups


def draw_balanced(groups, rng):
    selected = []
    for a, b, n in groups:
        selected.extend(a if len(a) == n else rng.choice(a, n, replace=False))
        selected.extend(b if len(b) == n else rng.choice(b, n, replace=False))
    return np.array(sorted(selected), dtype=int)


def score(y, predicted):
    tp, fn = int((y & predicted).sum()), int((y & ~predicted).sum())
    tn, fp = int((~y & ~predicted).sum()), int((~y & predicted).sum())
    require(tp+fn > 0 and tn+fp > 0, 'Metrics need both target classes')
    return dict(n=len(y), rh_n=int(y.sum()), non_rh_n=int((~y).sum()),
                accuracy=(tp+tn)/len(y), balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)),
                rh_recall=tp/(tp+fn), non_rh_specificity=tn/(tn+fp), tp=tp, fn=fn, tn=tn, fp=fp)


def evaluate(rows, cohort, horizon, seed, repeats, expected_folds):
    require(rows, f'No eligible pairs: {cohort}/h{horizon}')
    models = EMOTIONS + BASELINES
    y = np.array([r['hacking'] for r in rows], dtype=bool)
    folds = np.array([r['fold'] for r in rows])
    require(set(folds) == set(expected_folds), f'A fold disappeared at {cohort}/h{horizon}')
    groups = balanced_groups(y, folds)
    predictions = np.zeros((len(models), len(rows)), dtype=bool)
    rules, stats = [], []
    for j, model in enumerate(models):
        if model == 'always_not_hacking':
            continue
        if model == 'oracle_label_persistence':
            predictions[j] = [r['source_hacking'] for r in rows]
            continue
        x = np.array([r['mean_token_position'] if model == 'token_position' else r['projections'][model] for r in rows])
        for fold in sorted(set(folds)):
            train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            for key in ('trace_id', 'group_id'):
                require(not {rows[i][key] for i in train} & {rows[i][key] for i in test}, f'{key} leakage')
            rule = fit_threshold(x[train], y[train])
            predictions[j, test] = predict_threshold(x[test], rule)
            rules.append(dict(model=model, fold=int(fold), **rule))
    chosen = draw_balanced(groups, np.random.default_rng(seed))
    rng = np.random.default_rng(seed+1)
    samples = np.empty((repeats, len(models)))
    for repeat in range(repeats):
        ix = draw_balanced(groups, rng)
        samples[repeat] = (predictions[:, ix] == y[ix]).mean(axis=1)
    estimates = [dict(cohort=cohort, horizon=horizon, model=m, **score(y[chosen], predictions[j, chosen]),
                      repeated_sampling_mean_accuracy=float(samples[:, j].mean()),
                      sampling_interval95=np.percentile(samples[:, j], [2.5, 97.5]).tolist())
                 for j, m in enumerate(models)]
    chosen_set = set(chosen.tolist())
    fields = ['trace_id', 'group_id', 'task', 'unit_id', 'target_unit_id', 'sentence_index',
              'target_sentence_index', 'turn_id', 'target_turn_id', 'source_hacking', 'hacking', 'fold',
              'source_needs_review', 'target_needs_review']
    index = [{**{k: r[k] for k in fields}, 'selected': i in chosen_set} for i, r in enumerate(rows)]
    for fold in sorted(set(folds)):
        train, test, sample = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold), chosen[folds[chosen] == fold]
        stats.append(dict(cohort=cohort, horizon=horizon, fold=int(fold),
                          train_pairs=len(train), train_rh=int(y[train].sum()), train_non_rh=int((~y[train]).sum()),
                          train_groups=len({rows[i]['group_id'] for i in train}),
                          train_trace_ids=sorted({rows[i]['trace_id'] for i in train}),
                          test_trace_ids=sorted({rows[i]['trace_id'] for i in test}),
                          eval_pairs_before_balancing=len(test), eval_rh_before_balancing=int(y[test].sum()),
                          eval_non_rh_before_balancing=int((~y[test]).sum()), balanced_eval_pairs=len(sample)))
    population = dict(cohort=cohort, horizon=horizon, eligible_pairs=len(rows),
                      traces=len({r['trace_id'] for r in rows}), groups=len({r['group_id'] for r in rows}),
                      balanced_eval_pairs=len(chosen), balanced_rh=int(y[chosen].sum()),
                      balanced_non_rh=int((~y[chosen]).sum()))
    return (dict(estimates=estimates, fold_statistics=stats, population=population, pair_index=index, rules=rules),
            dict(models=np.array(models), predictions=predictions, labels=y, folds=folds,
                 selected_indices=chosen, sampling_accuracies=samples))
