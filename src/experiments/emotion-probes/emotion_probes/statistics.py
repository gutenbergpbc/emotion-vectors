"""Condition-weighted metrics, deterministic cutoffs and grouped intervals.

Reused from src/experiments/emotion-probes/adapters/gemma27b/probe_harness/statistics.py.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from .data import LABELS
from shared.io import require


def condition_weights(labels, target):
    labels = np.asarray(labels)
    require(set(labels) == set(LABELS), "Each mixed split must contain all 13 labels")
    counts = {e: np.sum(labels == e) for e in LABELS}
    return np.array(
        [0.5 / counts[e] if e == target else 0.5 / 12 / counts[e] for e in labels]
    )


def measure(labels, scores, cutoff, target):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    y = labels == target
    p = scores >= cutoff
    require(
        np.isfinite(scores).all() and np.isfinite(cutoff), "Nonfinite scores or cutoff"
    )
    weights = condition_weights(labels, target)
    rates = {e: float(p[labels == e].mean()) for e in LABELS}
    rec = rates[target]
    fpr = float(np.mean([v for e, v in rates.items() if e != target]))
    tp, tn, fp, fn = [int(a.sum()) for a in (p & y, ~p & ~y, p & ~y, ~p & y)]
    return {
        "n": len(y),
        "n_positive": int(y.sum()),
        "n_negative": int((~y).sum()),
        "balanced_accuracy": (rec + 1 - fpr) / 2,
        "recall": rec,
        "mean_negative_condition_fpr": fpr,
        "condition_weighted_auroc": float(
            roc_auc_score(y, scores, sample_weight=weights)
        ),
        "condition_weighted_average_precision": float(
            average_precision_score(y, scores, sample_weight=weights)
        ),
        "raw_accuracy": (tp + tn) / len(y),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "firing_rate_by_label": rates,
        "count_by_label": {e: int(np.sum(labels == e)) for e in LABELS},
    }


def candidates(scores):
    u = np.unique(np.asarray(scores, dtype=np.float64))
    require(len(u) > 0 and np.isfinite(u).all(), "Invalid calibration scores")
    result = np.r_[
        np.nextafter(u[0], -np.inf),
        u[:-1] + np.diff(u) / 2,
        np.nextafter(u[-1], np.inf),
    ]
    require(np.isfinite(result).all(), "Calibration cutoffs overflow")
    return result


def choose_threshold(labels, scores, target):
    labels = np.asarray(labels)
    weights = condition_weights(labels, target)
    y = labels == target
    best = None
    for t in candidates(scores):
        p = scores >= t
        recall = float(weights[y & p].sum() / weights[y].sum())
        fpr = float(weights[~y & p].sum() / weights[~y].sum())
        ba = (recall + 1 - fpr) / 2
        if (
            best is None
            or ba > best[0] + 1e-12
            or (abs(ba - best[0]) <= 1e-12 and (-fpr, float(t)) > (best[1], best[2]))
        ):
            best = (ba, -fpr, float(t))
    return best[2]


def verify_threshold(labels, scores, target, cutoff):
    # Independent per-label calculation rather than reusing weighted sums.
    labels = np.asarray(labels)
    best = None
    for t in candidates(scores):
        rates = {e: float(np.mean(scores[labels == e] >= t)) for e in LABELS}
        fpr = float(np.mean([v for e, v in rates.items() if e != target]))
        ba = (rates[target] + 1 - fpr) / 2
        if (
            best is None
            or ba > best[0] + 1e-12
            or (abs(ba - best[0]) <= 1e-12 and (-fpr, float(t)) > (best[1], best[2]))
        ):
            best = (ba, -fpr, float(t))
    require(best[2] == cutoff, "Threshold fails independent optimum/tie check")


def bootstrap_setup(rows, draws=2000, seed=20261009):
    groups, gi = np.unique([r["group_id"] for r in rows], return_inverse=True)
    li = np.array([LABELS.index(r["label"]) for r in rows])
    counts = np.zeros((len(groups), len(LABELS)))
    np.add.at(counts, (gi, li), 1)
    samples = (
        np.random.default_rng(seed)
        .multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=draws)
        .astype(float)
    )
    return gi, li, samples, samples @ counts, groups


def bootstrap(scores, threshold, target, setup):
    gi, li, draws, den, groups = setup
    num = np.zeros((len(groups), len(LABELS)))
    np.add.at(num, (gi, li), (scores >= threshold).astype(float))
    valid = (den > 0).all(axis=1)
    require(valid.any(), "All bootstrap draws lack a label")
    rates = (draws @ num)[valid] / den[valid]
    ti = LABELS.index(target)
    ba = (rates[:, ti] + 1 - np.delete(rates, ti, axis=1).mean(axis=1)) / 2
    return {
        "method": "native-source-group percentile bootstrap; fixed vector/layer/cutoff; shared draws",
        "groups": len(groups),
        "draws": len(draws),
        "valid_replicates": int(valid.sum()),
        "missing_label_replicates": int((~valid).sum()),
        "balanced_accuracy_95_interval": np.percentile(ba, [2.5, 97.5]).tolist(),
    }
