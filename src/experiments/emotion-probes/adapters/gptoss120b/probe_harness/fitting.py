import numpy as np
from sklearn.metrics import roc_auc_score
from .common import EMOTIONS, RECIPES, atomic, digest, read, require, sha
from .extraction import feature_manifest, layer_features


def negative_conditions(emotion, recipe):
    return (
        [(e, e) for e in EMOTIONS if e != emotion] if recipe != "neutral" else []
    ) + ([(emotion, "neutral")] if recipe != "other11" else [])


def direction(means, emotion, recipe):
    v = means[(emotion, emotion)] - np.mean(
        [means[k] for k in negative_conditions(emotion, recipe)], axis=0
    )
    norm = np.linalg.norm(v)
    require(norm > 0 and np.isfinite(norm), "Zero/nonfinite mean difference")
    return v / norm


def winning_layer(mean):
    return int(np.flatnonzero(mean >= np.max(mean) - 1e-6)[0])


def fit(run):
    run.need("extract")
    spec = run.spec()
    cases, manifest = feature_manifest(run, "training")
    conditions = {
        k: np.array([i for i, c in enumerate(cases) if (c["emotion"], c["label"]) == k])
        for e in EMOTIONS
        for k in [(e, e), (e, "neutral")]
    }
    folds = np.array([c["fold"] for c in cases])
    keys = [(e, r) for e in EMOTIONS for r in RECIPES]
    binding = digest({"spec": spec, "features": manifest, "fitter": sha(__file__)})
    checkpoint = run.root / "fit_layers"
    checkpoint.mkdir(exist_ok=True)
    aurocs = {e + "__" + r: np.empty((5, spec["layers"])) for e, r in keys}
    for layer in range(spec["layers"]):
        path = checkpoint / f"{layer:03d}.json"
        if path.exists():
            saved = read(path)
            require(
                saved["binding"] == binding, "Stale CV checkpoint; create a new run"
            )
            for key in aurocs:
                aurocs[key][:, layer] = saved["aurocs"][key]
            run.progress(
                layer + 1,
                spec["layers"],
                phase="cross_validation",
                current_layer=layer,
                cache_reused=True,
            )
            continue
        x = layer_features(run, "training", layer, cases, manifest)
        for fold in range(5):
            train_ix = {k: ix[folds[ix] != fold] for k, ix in conditions.items()}
            require(
                all(len(ix) for ix in train_ix.values()), "Empty training condition"
            )
            means = {
                k: x[ix].mean(axis=0, dtype=np.float64) for k, ix in train_ix.items()
            }
            for emotion, recipe in keys:
                neg = negative_conditions(emotion, recipe)
                cond = [(emotion, emotion)] + neg
                ixs = [conditions[k][folds[conditions[k]] == fold] for k in cond]
                require(all(len(q) for q in ixs), "Empty validation condition")
                ix = np.concatenate(ixs)
                y = np.concatenate([np.full(len(q), j == 0) for j, q in enumerate(ixs)])
                w = np.concatenate(
                    [
                        np.full(len(q), (1 if j == 0 else 1 / len(neg)) / len(q))
                        for j, q in enumerate(ixs)
                    ]
                )
                scores = x[ix] @ direction(means, emotion, recipe)
                aurocs[emotion + "__" + recipe][fold, layer] = roc_auc_score(
                    y, scores, sample_weight=w
                )
            run.progress(
                layer * 5 + fold + 1,
                spec["layers"] * 5,
                phase="cross_validation",
                current_layer=layer,
                current_fold=fold,
            )
        atomic(
            path,
            {
                "binding": binding,
                "aurocs": {k: a[:, layer].tolist() for k, a in aurocs.items()},
            },
        )
    cv = {}
    models = {}
    for key, a in aurocs.items():
        mean = a.mean(axis=0)
        layer = winning_layer(mean)
        cv[key] = {
            "fold_aurocs": a.tolist(),
            "mean_auroc": mean.tolist(),
            "selected_layer": layer,
        }
    atomic(run.root / "cv_results.json", cv)
    for layer in sorted({v["selected_layer"] for v in cv.values()}):
        x = layer_features(run, "training", layer, cases, manifest)
        means = {
            k: x[ix].mean(axis=0, dtype=np.float64) for k, ix in conditions.items()
        }
        for emotion, recipe in keys:
            key = emotion + "__" + recipe
            if cv[key]["selected_layer"] != layer:
                continue
            neg = negative_conditions(emotion, recipe)
            models[key] = {
                "emotion": emotion,
                "recipe": recipe,
                "layer": layer,
                "intercept": 0.0,
                "vector": direction(means, emotion, recipe).tolist(),
                "negative_conditions": neg,
                "negative_condition_weights": [1 / len(neg)] * len(neg),
                "training_cv_auroc": cv[key]["mean_auroc"][layer],
            }
            run.progress(len(models), 36, phase="refit", current_probe=key)
    atomic(
        run.root / "selected_models.json",
        {
            "binding": binding,
            "models": models,
            "fitter_sha256": sha(__file__),
            "training_cases_sha256": spec["input_hashes"]["training_cases.json"],
            "feature_manifest_sha256": sha(
                run.root / "training_features/manifest.json"
            ),
        },
    )
    with (run.root / "probe_vectors.npz").open("wb") as f:
        np.savez(f, **{k: np.array(m["vector"]) for k, m in models.items()})
