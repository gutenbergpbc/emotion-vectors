import csv
import numpy as np
from .common import atomic, digest, read, require, sha
from .extraction import feature_manifest, layer_features
from .statistics import (
    bootstrap,
    bootstrap_setup,
    choose_threshold,
    measure,
    verify_threshold,
)


def evaluate(run):
    run.need("fit")
    spec = run.spec()
    split = read(run.root / "inputs/split_manifest.json")
    models = read(run.root / "selected_models.json")["models"]
    cases, manifest = feature_manifest(run, "eval")
    index = {c["id"]: i for i, c in enumerate(cases)}
    rows = split["rows"]
    retained = [index[r["id"]] for r in rows]
    labels = np.array([r["label"] for r in rows])
    cal = np.array([r["split"] == "calibration" for r in rows])
    held = ~cal
    scores = {}
    # Each selected layer is read once; the feature matrix never includes all layers in RAM.
    for layer in sorted({m["layer"] for m in models.values()}):
        x = layer_features(run, "eval", layer, cases, manifest)
        for key, model in models.items():
            if model["layer"] != layer:
                continue
            values = x @ np.asarray(model["vector"])
            require(np.isfinite(values).all(), "Invalid probe score")
            for cid, aliases in split["aliases"].items():
                require(
                    np.allclose(
                        values[[index[a] for a in aliases]],
                        values[index[cid]],
                        rtol=1e-6,
                        atol=1e-6,
                    ),
                    "Saved aliases produce different scores",
                )
            scores[key] = values[retained]
        run.progress(len(scores), 36, phase="scoring", current_layer=layer)
    with (run.root / "all_eval_probe_scores.npz").open("wb") as f:
        np.savez(f, ids=np.array([r["id"] for r in rows]), **scores)
    binding = digest(
        {
            "spec": spec,
            "models": sha(run.root / "selected_models.json"),
            "features": manifest,
            "scores": digest({k: s.tolist() for k, s in scores.items()}),
            "evaluator": sha(__file__),
        }
    )
    # Freeze ALL thresholds before calculating any held-out metrics.
    calibrated = {}
    for key, model in models.items():
        s = scores[key]
        cut = choose_threshold(labels[cal], s[cal], model["emotion"])
        verify_threshold(labels[cal], s[cal], model["emotion"], cut)
        calibrated[key] = {**model, "threshold": cut}
        run.progress(len(calibrated), 36, phase="calibration", current_probe=key)
    atomic(
        run.root / "calibrated_models.json",
        {
            "binding": binding,
            "model": spec["model"],
            "revision": spec["revision"],
            "width": spec["width"],
            "models": calibrated,
        },
    )
    atomic(
        run.root / "calibration_results.json",
        {
            "results": {
                key: measure(
                    labels[cal], scores[key][cal], m["threshold"], m["emotion"]
                )
                for key, m in calibrated.items()
            }
        },
    )
    setup = bootstrap_setup(
        [r for r in rows if r["split"] == "heldout"],
        spec["bootstrap_draws"],
        spec["bootstrap_seed"],
    )
    results = {}
    checkpoints = run.root / "evaluation_probes"
    checkpoints.mkdir(exist_ok=True)
    for key, model in calibrated.items():
        path = checkpoints / (key + ".json")
        s = scores[key]
        target = model["emotion"]
        cut = model["threshold"]
        if path.exists():
            saved = read(path)
            require(
                saved["binding"] == binding,
                "Stale evaluation checkpoint; create a new run",
            )
            result = saved["result"]
        else:
            result = {
                "emotion": target,
                "recipe": model["recipe"],
                "layer": model["layer"],
                "threshold": cut,
                "calibration": measure(labels[cal], s[cal], cut, target),
                "heldout": measure(labels[held], s[held], cut, target),
                "heldout_bootstrap": bootstrap(s[held], cut, target, setup),
            }
            atomic(path, {"binding": binding, "result": result})
        results[key] = result
        atomic(
            run.root / "evaluation_results.json",
            {
                "binding": binding,
                "state": "partial" if len(results) < 36 else "complete",
                "evaluator_sha256": sha(__file__),
                "results": results,
            },
        )
        run.progress(len(results), 36, phase="heldout", current_probe=key)
    # Streaming CSV avoids holding 81,288 row dictionaries in memory.
    path = run.root / "sentence_scores.csv"
    temp = path.with_suffix(".tmp")
    with temp.open("w") as f:
        fields = [
            "probe",
            "id",
            "group_id",
            "label",
            "split",
            "score",
            "threshold",
            "prediction",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, model in calibrated.items():
            for row, score in zip(rows, scores[key]):
                writer.writerow(
                    {
                        **{k: row[k] for k in fields[1:5]},
                        "probe": key,
                        "score": float(score),
                        "threshold": model["threshold"],
                        "prediction": int(score >= model["threshold"]),
                    }
                )
    temp.replace(path)
