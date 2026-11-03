"""Plot individual reward-hacking rate curves for a completed gamut run.

The additions run supplies new batch-32 conditions. The pilot supplies the
shared baseline and the existing +/-0.05 points. If a condition appears in
both, the additions run takes precedence.

Run with:
  uv run --with matplotlib python steering/plot_run_emotion_rates.py
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "steering/runs/qwen35-9b-emotions-pilot-20261015"
RH075 = ROOT / "steering/runs/qwen35-9b-rh-conv1d-075-20261015"
ADDITIONS = ROOT / "steering/runs/qwen35-9b-rh-gamut-all12-additions-b32-20261015-r2"
MISSING_SIX = ROOT / "steering/runs/qwen35-9b-rh-gamut-missing-six-optimized-b32-20261016"


def wilson(k, n, z=1.959963984540054):
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return (c - h) / d, (c + h) / d


def read_summary(path):
    return json.loads((path / "reward_hacking_judge_summary.json").read_text())


def wilson(k, n, z=1.959963984540054):
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return (c - h) / d, (c + h) / d


def add_cell(cells, key, value, source):
    cell = cells.setdefault(key, {"n": 0, "labels": {}, "sources": []})
    cell["n"] += value["n"]
    for label, count in value["labels"].items():
        cell["labels"][label] = cell["labels"].get(label, 0) + count
    cell["sources"].append(source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, default=PILOT)
    parser.add_argument("--rh075", type=Path, default=RH075)
    parser.add_argument("--additions", type=Path, default=ADDITIONS)
    parser.add_argument("--missing-six", type=Path, default=MISSING_SIX)
    parser.add_argument("--output", type=Path, default=ADDITIONS / "emotion_graphs")
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import PercentFormatter

    pilot = read_summary(args.pilot)
    rh075 = read_summary(args.rh075)
    additions = read_summary(args.additions)
    missing_six = read_summary(args.missing_six)
    spec = json.loads((args.additions / "run_spec.json").read_text())
    emotions = [item["emotion"] for item in spec["probes"]]
    # Keep the pilot baseline once, include pilot +/-0.05 and +/-0.10/0.15,
    # earlier +/-0.075 sweep, then pool new additions. Exact response-text
    # comparison across repeated cells found no duplicated generations.
    cells = {}
    for key, value in pilot.items():
        emotion, alpha = key.rsplit("|", 1)
        alpha = float(alpha)
        if emotion == "baseline" or abs(alpha) in (0.05, 0.10, 0.15):
            add_cell(cells, (emotion, alpha), value, "pilot")
    for key, value in rh075.items():
        emotion, alpha = key.rsplit("|", 1)
        alpha = float(alpha)
        if emotion != "baseline":
            add_cell(cells, (emotion, alpha), value, "rh075")
    for key, value in additions.items():
        emotion, alpha = key.rsplit("|", 1)
        add_cell(cells, (emotion, float(alpha)), value, "batch32_additions")
    for key, value in missing_six.items():
        emotion, alpha = key.rsplit("|", 1)
        if emotion != "baseline":
            add_cell(cells, (emotion, float(alpha)), value, "optimized_missing_six_batch32")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": "Qwen/Qwen3.5-9B",
        "metric": "judge-labeled reward-hacking rate",
        "interval": "95% Wilson confidence interval",
        "sources": {
            "pilot": str(args.pilot),
            "batch32_additions": str(args.additions),
            "optimized_missing_six_batch32": str(args.missing_six),
        },
        "policy": "Pilot shared baseline and +/-0.05 points; additions-run values for every other condition. Overlapping cells are not pooled.",
        "graphs": [],
    }
    for emotion in emotions:
        conditions = sorted((a, v) for (e, a), v in cells.items() if e == emotion)
        xs, ys, lows, highs = [], [], [], []
        for alpha, item in conditions:
            n = item["n"]
            k = item["labels"].get("hack", 0)
            rate = k / n if n else 0
            lo, hi = wilson(k, n)
            xs.append(alpha)
            ys.append(rate)
            lows.append(max(0.0, rate - lo))
            highs.append(max(0.0, hi - rate))

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.errorbar(xs, ys, yerr=[lows, highs], fmt="o-", color="#cf5648",
                    lw=1.8, ms=5.5, capsize=3, zorder=3)
        base = cells.get(("baseline", 0.0))
        if base:
            n = base["n"]
            k = base["labels"].get("hack", 0)
            rate = k / n if n else 0
            lo, hi = wilson(k, n)
            ax.axhline(rate, color="#555", ls="--", lw=1, alpha=.7)
            ax.errorbar([0], [rate], yerr=[[max(0.0, rate - lo)],
                                            [max(0.0, hi - rate)]], fmt="D",
                        color="#333", ms=5, capsize=3, zorder=4)
        ax.set_title(f"Qwen3.5-9B · {emotion.capitalize()} steering")
        ax.set_xlabel("Steering strength α")
        ax.set_ylabel("Judge-labeled reward-hacking rate")
        ax.set_xlim(-0.17, 0.17)
        ax.set_xticks([-0.15, -0.10, -0.075, -0.05, -0.025, 0, 0.025,
                       0.05, 0.075, 0.10, 0.15])
        ax.set_xticklabels(["−.15", "−.10", "−.075", "−.05", "−.025", "0",
                            "+.025", "+.05", "+.075", "+.10", "+.15"],
                           rotation=35, ha="right")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", color="#ddd", lw=.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(.01, .98, "Pooled judged responses · 95% Wilson intervals",
                transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#555")
        fig.tight_layout()
        stem = args.output / emotion
        fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        if emotion == emotions[0]:
            pdf = PdfPages(args.output / "all_12_emotions.pdf")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        manifest["graphs"].append({
            "emotion": emotion,
            "png": stem.with_suffix(".png").name,
            "svg": stem.with_suffix(".svg").name,
            "conditions": [{"alpha": a, "n_judged": v["n"],
                            "hack_count": v["labels"].get("hack", 0),
                            "sources": sorted(set(v["sources"]))}
                           for a, v in conditions],
        })
    manifest["sources"]["rh075"] = str(args.rh075)
    manifest["policy"] += "; add the seed-20261016 missing-six batch-32 results for ±0.075/0.10/0.15."
    pdf.close()
    manifest["policy"] = (
        "Pilot baseline once; pool judged label counts from pilot +/-0.05/0.10/0.15, "
        "the earlier +/-0.075 sweep, the all-emotion batch-32 additions, and the "
        "seed-20261016 batch-32 sweep for sad, ashamed, calm, surprised, guilty, and anxious. "
        "Raw response text "
        "comparison found no exact duplicates in overlapping conditions."
    )
    manifest["interval_caveat"] = (
        "Wilson intervals describe response-level binomial uncertainty for the pooled proportion. "
        "They do not account for dependence from repeated fixed prompts or shared base seeds; "
        "the six newly filled conditions use seed 20261016."
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(emotions)} emotion graphs to {args.output}")


if __name__ == "__main__":
    main()
