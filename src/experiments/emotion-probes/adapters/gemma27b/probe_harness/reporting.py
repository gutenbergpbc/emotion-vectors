import base64
import csv
import html
import io
from pathlib import Path
import re

import numpy as np

from .common import (
    EMOTIONS,
    RECIPES,
    LABELS,
    atomic,
    digest,
    read,
    require,
    sha,
    validate_inputs,
)
from .extraction import feature_manifest
from .fitting import negative_conditions, winning_layer
from .statistics import measure, verify_threshold, bootstrap, bootstrap_setup

LIMITATIONS = (
    "Retrospective development evaluation on one shared sales fixture. Five surprised evaluation pairs retain review disagreements. "
    "95% intervals use 2,000 native-source-group bootstrap draws, conditional on fixed vectors, selected layers and cutoffs. "
    "Supplied reasoning is replayed text, not newly generated Gemma reasoning. Each probe makes an independent binary decision."
)
ASSETS = [
    "report.html",
    "all_emotions_unified.png",
    "all_emotions_unified.svg",
    "all_emotions_unified.pdf",
    "plot_data.json",
    "selected_models.json",
    "calibrated_models.json",
    "cv_results.json",
    "probe_vectors.npz",
    "evaluation_results.json",
    "calibration_results.json",
    "sentence_scores.csv",
    "run_spec.json",
]


def verify(run, assets=False):
    spec = run.spec()
    _, _, split = validate_inputs(run.root / "inputs")
    for part in ("training", "eval"):
        feature_manifest(run, part)
    if spec["mode"] == "gemma":
        require(
            read(run.root / "smoke.json")["status"] == "passed",
            "GPU smoke checks missing",
        )
    fitted = read(run.root / "selected_models.json")
    require(
        fitted["feature_manifest_sha256"]
        == sha(run.root / "training_features/manifest.json"),
        "Training features changed since fit",
    )
    models = read(run.root / "calibrated_models.json")["models"]
    cv = read(run.root / "cv_results.json")
    results = read(run.root / "evaluation_results.json")["results"]
    keys = {e + "__" + r for e in EMOTIONS for r in RECIPES}
    require(
        set(models) == set(cv) == set(results) == keys, "Expected exactly 36 probes"
    )
    byid = {r["id"]: r for r in split["rows"]}
    exports = {k: [] for k in keys}
    with (run.root / "sentence_scores.csv").open() as f:
        for row in csv.DictReader(f):
            require(row["probe"] in exports, "Unexpected exported probe")
            exports[row["probe"]].append(row)
    setup = bootstrap_setup(
        [r for r in split["rows"] if r["split"] == "heldout"],
        spec["bootstrap_draws"],
        spec["bootstrap_seed"],
    )
    vector_arrays = np.load(run.root / "probe_vectors.npz", allow_pickle=False)
    for key, model in models.items():
        require(
            {k: v for k, v in model.items() if k != "threshold"}
            == fitted["models"][key],
            "Calibrated direction changed since fitting",
        )
        require(
            np.array_equal(vector_arrays[key], np.asarray(model["vector"])),
            "Vector export mismatch",
        )
        require(
            len(model["vector"]) == spec["width"]
            and abs(np.linalg.norm(model["vector"]) - 1) < 1e-10,
            "Invalid unit vector",
        )
        require(model["intercept"] == 0, "Nonzero intercept")
        a = np.asarray(cv[key]["fold_aurocs"])
        require(
            a.shape == (5, spec["layers"]) and np.isfinite(a).all(), "Invalid CV array"
        )
        require(
            np.allclose(a.mean(axis=0), cv[key]["mean_auroc"], atol=1e-12, rtol=0),
            "CV mean mismatch",
        )
        require(
            model["layer"] == winning_layer(a.mean(axis=0)),
            "Layer selection rule mismatch",
        )
        expected = negative_conditions(model["emotion"], model["recipe"])
        require(
            model["negative_conditions"] == [list(c) for c in expected],
            "Wrong negative conditions",
        )
        require(
            np.allclose(
                model["negative_condition_weights"], [1 / len(expected)] * len(expected)
            ),
            "Wrong condition weights",
        )
        rows = exports[key]
        require(
            len(rows) == 2258 and {r["id"] for r in rows} == set(byid),
            "Export ID mismatch",
        )
        for r in rows:
            require(
                all(r[k] == byid[r["id"]][k] for k in ("label", "group_id", "split")),
                "Export split metadata mismatch",
            )
            require(
                float(r["threshold"]) == model["threshold"], "Export cutoff mismatch"
            )
            require(
                int(r["prediction"]) == int(float(r["score"]) >= model["threshold"]),
                "Export prediction mismatch",
            )
        for part in ("calibration", "heldout"):
            selected = [r for r in rows if r["split"] == part]
            labels = np.array([r["label"] for r in selected])
            scores = np.array([float(r["score"]) for r in selected])
            actual = measure(labels, scores, model["threshold"], model["emotion"])
            saved = results[key][part]
            for field in (
                "balanced_accuracy",
                "raw_accuracy",
                "tp",
                "tn",
                "fp",
                "fn",
                "condition_weighted_auroc",
                "condition_weighted_average_precision",
            ):
                require(
                    abs(actual[field] - saved[field]) < 1e-10,
                    f"Exported metric mismatch: {key}/{part}/{field}",
                )
            require(
                actual["count_by_label"] == saved["count_by_label"],
                "Denominator mismatch",
            )
            if part == "calibration":
                verify_threshold(labels, scores, model["emotion"], model["threshold"])
            else:
                # CSV preserves split-manifest order, so shared group draws align exactly.
                require(
                    [r["id"] for r in selected]
                    == [r["id"] for r in split["rows"] if r["split"] == "heldout"],
                    "Prediction order changed",
                )
                interval = bootstrap(
                    scores, model["threshold"], model["emotion"], setup
                )
                require(
                    interval == results[key]["heldout_bootstrap"],
                    "Bootstrap interval or missing-label counts mismatch",
                )
        require(
            results[key]["layer"] == model["layer"]
            and results[key]["threshold"] == model["threshold"],
            "Result/model mismatch",
        )
    vector_arrays.close()
    checks = [
        "Frozen hashes, source groups and aliases",
        "Complete finite activation shards with matching identities",
        "36 unit directions, five-fold all-layer CV and shallowest tie rule",
        "All calibration cutoffs independently optimal",
        "81,288 exported predictions reproduce metrics and counts",
    ]
    if assets:
        for name in ASSETS:
            require((run.root / name).is_file(), f"Missing report asset: {name}")
        plot = read(run.root / "plot_data.json")
        require(len(plot["rows"]) == 36, "Wrong chart count")
        for row in plot["rows"]:
            r = results[row["key"]]
            require(
                row["balanced_accuracy"] == r["heldout"]["balanced_accuracy"],
                "Chart metric mismatch",
            )
            require(
                row["ci95"] == r["heldout_bootstrap"]["balanced_accuracy_95_interval"],
                "Chart interval mismatch",
            )
        for link in re.findall(
            r'href="([^"]+)"', (run.root / "report.html").read_text()
        ):
            require((run.root / link).is_file(), f"Broken report link: {link}")
        checks.append("Report assets, links and all 36 chart values")
    record = {
        "status": "passed",
        "mode": spec["mode"],
        "checks": checks,
        "artifacts": {
            name: sha(run.root / name) for name in ASSETS if (run.root / name).exists()
        },
        "spec_sha256": digest(spec),
    }
    atomic(run.root / "verification.json", record)
    return record


def report(run):
    run.need("evaluate")
    spec = run.spec()
    run.progress(0, 3, phase="verify_predictions")
    verify(run)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    results = read(run.root / "evaluation_results.json")["results"]
    fig, ax = plt.subplots(figsize=(17, 7))
    fig.subplots_adjust(top=0.77, bottom=0.23, left=0.06, right=0.98)
    x = np.arange(12)
    width = 0.25
    rows = []
    for j, (recipe, color) in enumerate(
        zip(RECIPES, ["#416FA4", "#D88443", "#3C9284"])
    ):
        items = [results[e + "__" + recipe] for e in EMOTIONS]
        values = np.array([r["heldout"]["balanced_accuracy"] for r in items])
        ci = np.array(
            [r["heldout_bootstrap"]["balanced_accuracy_95_interval"] for r in items]
        )
        centers = x + (j - 1) * width
        ax.bar(centers, values, width, color=color, label=recipe)
        ax.vlines(centers, ci[:, 0], ci[:, 1], color="#334155", lw=1)
        for side in (0, 1):
            ax.hlines(
                ci[:, side], centers - 0.025, centers + 0.025, color="#334155", lw=1
            )
        for e, value, interval in zip(EMOTIONS, values, ci):
            rows.append(
                {
                    "key": e + "__" + recipe,
                    "emotion": e,
                    "recipe": recipe,
                    "balanced_accuracy": float(value),
                    "ci95": interval.tolist(),
                }
            )
    ax.set_xticks(x, [e.capitalize() for e in EMOTIONS], rotation=20)
    ax.set_ylim(0, 1.09)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.axhline(0.5, ls="--", color="#8a97a4")
    ax.set_ylabel("Held-out mixed-emotion balanced accuracy")
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    title = (
        ("SYNTHETIC DEMO — " if spec["mode"] == "synthetic" else "")
        + spec["model"]
        + " · 36 emotion probes"
    )
    fig.suptitle(title, fontsize=18)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
    )
    fig.text(0.06, 0.13, "Model revision: " + spec["revision"], fontsize=8)
    fig.text(
        0.06,
        0.05,
        "Equal weighting across negative labels · 2,000 source-group bootstrap draws\nRetrospective sales fixture · five surprised review disagreements · fixed probes and thresholds",
        fontsize=9,
    )
    for ext in ("svg", "png", "pdf"):
        fig.savefig(run.root / f"all_emotions_unified.{ext}", dpi=160)
    plt.close(fig)
    atomic(
        run.root / "plot_data.json",
        {"rows": rows, "results_sha256": sha(run.root / "evaluation_results.json")},
    )
    run.progress(1, 3, phase="render_report")
    image = base64.b64encode(
        (run.root / "all_emotions_unified.png").read_bytes()
    ).decode()
    table = []
    for row in rows:
        r = results[row["key"]]
        h = r["heldout"]
        ci = row["ci95"]
        table.append(
            "<tr>"
            + "".join(
                "<td>" + html.escape(str(v)) + "</td>"
                for v in [
                    r["emotion"],
                    r["recipe"],
                    r["layer"],
                    f"{h['balanced_accuracy']:.2%}",
                    f"{ci[0]:.2%}–{ci[1]:.2%}",
                    f"{h['recall']:.2%}",
                    f"{h['mean_negative_condition_fpr']:.2%}",
                    r["threshold"],
                ]
            )
            + "</tr>"
        )
    links = " · ".join(
        f'<a href="{name}">{name}</a>' for name in ASSETS if name != "report.html"
    )
    (run.root / "report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'
        + html.escape(title)
        + "</title><style>body{font:15px/1.6 system-ui;max-width:1400px;margin:40px auto;padding:24px;color:#233140}img{width:100%}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}.scroll{overflow:auto}</style><h1>"
        + html.escape(title)
        + "</h1><p>2,700 training cases · 1,129 calibration replays · 1,129 held-out replays. Zero-based layer indices.</p><p>"
        + html.escape(LIMITATIONS)
        + '</p><img alt="All 36 held-out probe balanced accuracies" src="data:image/png;base64,'
        + image
        + '"><div class="scroll"><table><thead><tr>'
        + "".join(
            "<th>" + h + "</th>"
            for h in [
                "Emotion",
                "Contrast",
                "Layer",
                "Held-out BA",
                "95% interval",
                "Recall",
                "Mean negative FPR",
                "Cutoff",
            ]
        )
        + "</tr></thead><tbody>"
        + "".join(table)
        + "</tbody></table></div><p>"
        + links
        + "</p></html>"
    )
    run.progress(2, 3, phase="verify_report")
    verify(run, assets=True)
    run.progress(3, 3, phase="verified", verification="passed")
