"""Plot task-disjoint GLM 5.2 DeepSWE sentence forecasts from frozen replay."""
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
import csv
import html
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CANONICAL = HERE.parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(CANONICAL), str(HERE)]
from shared.io import atomic, read, require, sha  # noqa: E402
from emotion_rh_forecasting.analysis import EMOTIONS, evaluate, make_pairs  # noqa: E402
from plot_glm_hack_forecast import frozen_folds  # noqa: E402
from prepare_goodfire_glm52_replay import OUT as REPLAY, PROBES, labels as REPLAY_LABELS  # noqa: E402
from materialize_goodfire_glm52_manual_labels import OUT as LABEL_ROOT  # noqa: E402

OUT = ROOT / "runs/emotion-rh-forecasting/goodfire-glm52-deepswe-manual-pilot-v1"


def load_rows(label_root=LABEL_ROOT):
    labels_path = label_root / "labels.json"
    labels = read(labels_path)
    require(labels["summary"]["status"] == "complete" and len(labels["traces"]) == 12,
            "Sentence labels incomplete")
    spec = read(REPLAY / "replay-spec.json")
    require(spec["source_probe_manifest_sha256"] == sha(PROBES / "manifest.json") and
            spec["selection_sha256"] == labels["summary"]["selection_sha256"] and
            spec["label_request_manifest_sha256"] == sha(REPLAY_LABELS.OUT / "manifest.json"),
            "Replay source identity changed")
    if "base_labels_sha256" in labels["summary"]:
        require(labels["summary"]["base_labels_sha256"] == sha(LABEL_ROOT / "labels.json") and
                labels["summary"]["judge_manifest_sha256"] == sha(label_root / "manifest.json"),
                "GPT judge inputs changed")
    else:
        require(labels["summary"]["adjudications_sha256"] == sha(label_root / "adjudications.json"),
                "Manual decisions changed")
    by_trace = {t["id"]: t for t in labels["traces"]}
    emotions = [p["emotion"] for p in read(REPLAY / "probes.json")["probes"]]
    rows = []
    output_offsets = {}
    for entry in spec["jobs"]:
        trace = by_trace[entry["trace_id"]]
        job_path = REPLAY / "inputs" / f"{entry['id']}.json"
        require(sha(job_path) == entry["sha256"], "Replay job changed")
        job = read(job_path)
        score_path = REPLAY / "scores" / f"{entry['id']}.npz"
        receipt = read(score_path.with_suffix(".receipt.json"))
        require(receipt["replay_identity"] == spec["identity"] and
                receipt["job_sha256"] == entry["sha256"] and
                receipt["sha256"] == sha(score_path), "Replay score changed")
        with np.load(score_path, allow_pickle=False) as archive:
            scores = archive["scores"]
        require(scores.shape == (job["tokens"], 12) and np.isfinite(scores).all(),
                "Invalid score matrix")
        for replay in job["units"]:
            index = replay["unit_id"]
            unit = trace["units"][index]
            require(all(unit[k] == replay[k] for k in
                        ("unit_id", "node_index", "channel", "char_start", "char_end", "text")),
                    "Labeled sentence differs from replay")
            indices = np.asarray(replay["token_indices"], dtype=int)
            weights = np.asarray(replay["token_weights"], dtype=float)
            require(len(indices) and np.all(indices < len(scores)) and np.all(weights > 0),
                    "Invalid causal pooling")
            values = np.average(scores[indices], axis=0, weights=weights)
            rows.append({"trace_id": trace["id"], "group_id": trace["task"], "task": trace["task"],
                         "sentence_index": index, "unit_id": unit["unit_id"],
                         "turn_id": f"{unit['node_index']}-{unit['channel']}",
                         "char_start": unit["char_start"], "char_end": unit["char_end"],
                         "text": unit["text"], "hacking": unit["hacking"],
                         "needs_review": unit["needs_review"],
                         "projections": dict(zip(emotions, map(float, values))),
                         "mean_token_position": output_offsets.get(trace["id"], 0) +
                                                float(np.average(replay["output_positions"], weights=weights)),
                         "completed_feature_tokens": len(indices),
                         "last_feature_token_char_end": unit["char_end"]})
        output_offsets[trace["id"]] = output_offsets.get(trace["id"], 0) + job["output_tokens"]
    require(len(rows) == spec["total_sentences"] == labels["summary"]["labeled_sentences"],
            "Total sentences changed")
    return rows, spec, labels_path


def plot(estimates, summary, out=OUT, position_label="Token position"):
    by_key = {(e["model"], e["horizon"]): e for e in estimates}
    fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharex=True, sharey=True)
    for ax, emotion in zip(axes.flat, EMOTIONS):
        for model, color, style, label in ((emotion, "#256f9d", "-o", "Emotion probe"),
                                            ("token_position", "#687d8c", ":", position_label),
                                            ("always_not_hacking", "#bb6438", "--", "Always non-RH")):
            points = [by_key[(model, h)] for h in range(11)]
            x = np.arange(11)
            y = [p["repeated_sampling_mean_accuracy"] * 100 for p in points]
            ax.plot(x, y, style, color=color, linewidth=1.6, markersize=3,
                    label=label if emotion == EMOTIONS[0] else None)
            if model == emotion:
                low = [p["sampling_interval95"][0] * 100 for p in points]
                high = [p["sampling_interval95"][1] * 100 for p in points]
                ax.fill_between(x, low, high, color=color, alpha=.13)
        ax.set_title(emotion.capitalize())
        ax.set_xlim(0, 10)
        ax.set_xticks(range(11))
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=.18)
    for ax in axes[:, 0]:
        ax.set_ylabel("Balanced-evaluation accuracy (%)")
    for ax in axes[-1]:
        ax.set_xlabel("Sentence horizon (+k)")
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower left", bbox_to_anchor=(.04, .025), ncol=3)
    fig.suptitle("GLM 5.2 · DeepSWE · emotion probes forecasting reward-hacking sentences",
                 fontsize=16, y=.99)
    fig.text(.06, .944,
             f"{summary['traces']} traces / 3 task-disjoint folds · {summary['confirmed_rh_sentences']} RH labels used · "
             f"{summary['label_method']} · NVFP4 replay", fontsize=10)
    fig.text(.06, .012,
             f"Balanced pairs: {summary['populations'][0]['balanced_eval_pairs']} at k=0, "
             f"{summary['populations'][-1]['balanced_eval_pairs']} at k=10. "
             "Unresolved endpoints excluded; bands show subsample variation, not generalization uncertainty.",
             fontsize=9)
    fig.tight_layout(rect=[0, .09, 1, .91])
    dest = out / "figures" / "accuracy-trends"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(dest.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-root", type=Path, default=LABEL_ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out
    rows, spec, labels_path = load_rows(args.label_root)
    folds = frozen_folds(rows, "coding")
    expected_folds = set(folds["trace_folds"].values())
    estimates, populations = [], []
    for horizon in range(11):
        pairs = make_pairs(rows, horizon, certain_only=True)
        result, _ = evaluate(pairs, "certain_only", horizon, 20261023, 2000, expected_folds)
        estimates.extend(result["estimates"])
        populations.append(result["population"])
        print(f"k={horizon}: {result['population']['balanced_eval_pairs']} balanced pairs", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"status": "complete", "traces": len({r["trace_id"] for r in rows}),
               "sentences": len(rows), "rh_sentences": sum(r["hacking"] for r in rows),
               "confirmed_rh_sentences": sum(r["hacking"] and not r["needs_review"] for r in rows),
               "needs_review": sum(r["needs_review"] for r in rows),
               "source": "Goodfire/reward-hack-data/deepswe-glm-5.2-fp8",
               "labels_sha256": sha(labels_path), "replay_identity": spec["identity"],
               "probe_model": spec["model"], "folds": folds,
               "interpretation": spec["interpretation"], "populations": populations,
               "label_method": labels_path.parent.name,
               "caveat": "Short outcome-stratified pilot; source-vote, manual, and GPT labels where applicable; unresolved endpoints excluded. NVFP4 activations are an exploratory transfer from FP8 traces."}
    atomic(out / "metrics.json", {"summary": summary, "estimates": estimates})
    with (out / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(estimates[0]))
        writer.writeheader()
        writer.writerows(estimates)
    plot(estimates, summary, out)
    (out / "report.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>GLM 5.2 DeepSWE sentence forecast pilot</title>'
        '<style>body{font:16px/1.5 system-ui;max-width:1500px;margin:2rem auto;color:#172d2b}'
        'img{max-width:100%}</style><h1>GLM 5.2 DeepSWE sentence forecast pilot</h1>'
        f'<p>{summary["traces"]} distinct tasks, {summary["sentences"]:,} indexed sentences, '
        f'{summary["confirmed_rh_sentences"]} RH labels used, '
        f'{summary["needs_review"]} unresolved or flagged. {html.escape(summary["interpretation"])}</p>'
        '<p>Each point is out-of-fold accuracy after balancing each test fold by target label. '
        'k=0 classifies the current sentence; k&gt;0 predicts exactly the sentence k positions later. '
        'The primary analysis excludes pairs with an unresolved source or target label without shifting sentence positions. '
        f'Certain labels come from {html.escape(summary["label_method"])}; see the label dataset for per-sentence provenance. '
        'Bands are repeated balanced-subsample variability, not generalization intervals. '
        'The pilot selection is outcome- and length-stratified, so performance does not estimate the full archive.</p>'
        '<img src="figures/accuracy-trends.svg" alt="Twelve emotion-probe forecast panels">'
        '<p><a href="metrics.csv">Metrics CSV</a> · <a href="metrics.json">Metrics and folds</a></p></html>')
    atomic(out / "verification.json", {"passed": True, "labels_sha256": sha(labels_path),
          "replay_identity": spec["identity"], "horizons": list(range(11)),
          "metrics_sha256": sha(out / "metrics.json"),
          "figure_sha256": sha(out / "figures/accuracy-trends.svg")})
    print(json.dumps({k: summary[k] for k in ("traces", "sentences", "rh_sentences", "needs_review")}, indent=2))


if __name__ == "__main__":
    main()
