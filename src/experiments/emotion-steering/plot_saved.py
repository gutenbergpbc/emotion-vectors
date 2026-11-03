#!/usr/bin/env python3
"""Plot a verified steering run without generating responses or changing labels."""

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
import sys
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from emotion_steering.pipeline import joined_rows
from emotion_steering.reporting import summarize, wilson

from shared.io import atomic, read, require, sha

LABELS = {
    "reward_hack": ("Reward hacking", "#2376a9"),
    "ordinary_failure": ("Ordinary failure", "#a05c91"),
    "honest_limit": ("Honest limitation", "#4c9b83"),
    "uncertain": ("Uncertain", "#b3bcc6"),
    "benign": ("Benign", "#b2cfa2"),
    "incoherent": ("Incoherent", "#cc9949"),
}
LEGACY = "#a67642"
TITLES = {
    "impossible_sort": "Sorting",
    "impossible_prime": "Primality",
    "impossible_search": "Searching",
}


def cells(rows):
    result = []
    for strength in sorted({r["strength"] for r in rows}):
        group = [r for r in rows if r["strength"] == strength]
        counts = Counter(r["label"] for r in group)
        old = [r for r in group if r["legacy_label"] is not None]
        n = len(group)
        result.append(
            {
                "strength": strength,
                "n": n,
                "rh_count": counts["reward_hack"],
                "rh_rate": counts["reward_hack"] / n,
                "ci95": wilson(counts["reward_hack"], n),
                "legacy_n": len(old),
                "legacy_rh_rate": sum(r["legacy_label"] == "hack" for r in old)
                / len(old)
                if old
                else None,
                "truncated": sum(r["truncated"] for r in group),
                **{label: counts[label] for label in LABELS},
            }
        )
    return result


def setup_axis(ax, points, ylabel):
    x = [c["strength"] for c in points]
    ax.set(
        xticks=x,
        xticklabels=[f"{v:+.2f}" if v else "0\nbaseline" for v in x],
        ylim=(0, 1.05),
        yticks=np.linspace(0, 1, 5),
        ylabel=ylabel,
        xlabel="Steering strength (fraction of residual norm)",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0, color="#d4dae0", linewidth=1, zorder=0)


def rh_curve(ax, points, annotations=False, audited_label="Revised RH labels"):
    x = np.array([c["strength"] for c in points])
    y = np.array([c["rh_rate"] for c in points])
    intervals = np.array([c["ci95"] for c in points])
    if all(c["legacy_n"] for c in points):
        require(
            all(c["legacy_n"] == c["n"] for c in points),
            "Legacy/new populations differ",
        )
        ax.plot(
            x,
            [c["legacy_rh_rate"] for c in points],
            "--s",
            color=LEGACY,
            linewidth=1.8,
            markersize=5,
            label="Legacy RH labels",
            zorder=3,
        )
    ax.errorbar(
        x,
        y,
        # Boundary rates can produce a negative roundoff error near 1e-17.
        yerr=np.maximum(0, np.array([y - intervals[:, 0], intervals[:, 1] - y])),
        fmt="-o",
        color=LABELS["reward_hack"][1],
        linewidth=2.4,
        markersize=6,
        capsize=4,
        elinewidth=1.2,
        label=audited_label,
        zorder=4,
    )
    if annotations:
        for c in points:
            ax.annotate(
                f"{c['rh_count']}/{c['n']}",
                (c["strength"], c["rh_rate"]),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                color=LABELS["reward_hack"][1],
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "pad": 1,
                    "alpha": 0.85,
                },
            )
    setup_axis(ax, points, "Judge-labeled RH rate")


def save(fig, stem):
    paths = []
    for ext in ("png", "svg", "pdf"):
        path = stem.with_suffix("." + ext)
        fig.savefig(path, dpi=180, facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def presentation(spec):
    historical = spec["mode"] == "audit-existing"
    return {
        "origin": "Rejudged historical subset" if historical else "Fresh generations",
        "label": "Revised RH labels" if historical else "Audited RH labels",
        "stem": "all-emotions-rejudged" if historical else "all-emotions-steering",
        "scope": (
            "Judge-label comparison only; this small, truncated subset does not establish causal effects or preserved capability."
            if historical
            else "Results describe these fixed scenarios; RH labels alone do not establish preserved capability."
        ),
    }


def protocol_disclosure(root, observations):
    path = root / "protocol.json"
    if not path.exists():
        return "", None
    protocol = read(path)
    require(
        protocol.get("type") == "mixed_per_turn_token_budget"
        and protocol["preserved_samples"] + protocol["remaining_samples"]
        == observations,
        "Invalid mixed token-budget protocol",
    )
    note = (
        "Mixed per-turn token limits: "
        f"{protocol['preserved_samples']} saved traces at "
        f"{protocol['preserved_max_new_tokens']:,}; "
        f"{protocol['remaining_samples']} continuation traces at "
        f"{protocol['new_max_new_tokens']:,}."
    )
    return note, protocol


def overview(spec, rows, output, protocol_note=""):
    wording = presentation(spec)
    emotions = [p["emotion"] for p in spec["probes"]]
    columns = min(3, len(emotions))
    height = (len(emotions) + columns - 1) // columns
    fig, axes = plt.subplots(
        height,
        columns,
        figsize=(4.7 * columns, 3 * height + 2.1),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.85,
        bottom=0.15 if protocol_note else 0.12,
        hspace=0.35,
        wspace=0.15,
    )
    fig.text(
        0.07,
        0.975,
        f"Emotion steering · {spec['model_key']} · {spec['environment'].title()}",
        fontsize=20,
        weight="bold",
        va="top",
    )
    group_sizes = sorted(
        {
            c["n"]
            for e in emotions
            for c in cells([r for r in rows if r["emotion"] in (None, e)])
        }
    )
    fig.text(
        0.07,
        0.94,
        f"{wording['origin']} · {len(spec['scenarios'])} fixed tasks · "
        f"{spec['samples']} samples/task/strength · n={','.join(map(str, group_sizes))} per point",
        fontsize=12,
        color="#5a6b78",
    )
    for index, (emotion, ax) in enumerate(zip(emotions, axes.flat)):
        rh_curve(
            ax,
            cells([r for r in rows if r["emotion"] in (None, emotion)]),
            audited_label=wording["label"],
        )
        ax.set_title(emotion.title(), loc="left", fontsize=14, weight="bold")
        if index % columns:
            ax.set_ylabel("")
        if index // columns != height - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Steering strength", fontsize=10)
        ax.tick_params(labelsize=10)
    for ax in list(axes.flat)[len(emotions) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.917),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.07,
        0.025,
        f"{len(rows)} unique saved traces; one shared baseline reused in every panel. "
        f"{sum(r['truncated'] for r in rows)}/{len(rows)} traces truncated.\n"
        "Whiskers: descriptive 95% Wilson intervals, conditional on these fixed tasks. All labels remain in the denominator.\n"
        f"{wording['scope']}" + (f"\n{protocol_note}" if protocol_note else ""),
        fontsize=10,
        color="#5a6b78",
        linespacing=1.6,
    )
    return save(fig, output / wording["stem"])


def gallery(spec, rows, output, protocol_note=""):
    wording = presentation(spec)
    emotions = [p["emotion"] for p in spec["probes"]]
    cards = "".join(
        f'<section id="{html.escape(e)}"><h2>{html.escape(e.title())}</h2>'
        f'<a href="{e}-revised-labels.png"><img src="{e}-revised-labels.svg" alt="{html.escape(e)} steering plot" loading="lazy"></a>'
        f'<p><a href="{e}-revised-labels.png">PNG</a> · <a href="{e}-revised-labels.pdf">PDF</a> · <a href="{e}-revised-labels.svg">SVG</a></p></section>'
        for e in emotions
    )
    path = output / "index.html"
    path.write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Emotion steering plots</title><style>body{{font:16px/1.5 system-ui;max-width:1300px;margin:30px auto;padding:20px;color:#20303d}}img{{max-width:100%;height:auto}}section{{border-top:1px solid #dde3e8;margin-top:35px}}a{{color:#2376a9}}</style>
<h1>{html.escape(spec["model_key"])} · {html.escape(spec["environment"].title())}: {len(emotions)} emotion steering plots</h1>
<p>{wording["origin"]} · {wording["label"]}.</p>
<p>{len(rows)} unique saved traces, {spec["samples"]} samples per fixed task and strength. One shared baseline. All labels retained in the denominator.</p>
<p>{sum(r["truncated"] for r in rows)}/{len(rows)} traces truncated. {wording["scope"]}</p>
{f'<p><strong>{html.escape(protocol_note)}</strong> All saved traces remain in the denominator.</p>' if protocol_note else ''}
<p><a href="{wording["stem"]}.pdf">Overview PDF</a> · <a href="rates.csv">Rates CSV</a> · <a href="emotion-plots.zip">Download plots</a></p>
<img src="{wording["stem"]}.svg" alt="One audited RH curve per emotion">{cards}</html>""")
    return path


def draw(root, spec, rows, output):
    wording = presentation(spec)
    protocol_note, protocol = protocol_disclosure(root, len(rows))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "text.color": "#20303d",
            "axes.labelcolor": "#20303d",
            "axes.edgecolor": "#81909b",
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    artifacts, records = [], []
    origin = (
        "Rejudged saved traces" if spec["mode"] == "audit-existing" else "Fresh samples"
    )
    for emotion in sorted({r["emotion"] for r in rows if r["emotion"]}):
        selected = [r for r in rows if r["emotion"] in (None, emotion)]
        points = cells(selected)
        names = sorted({r["scenario"] for r in selected})
        total, truncated = len(selected), sum(r["truncated"] for r in selected)
        n = ", ".join(str(v) for v in sorted({c["n"] for c in points}))
        title = (
            f"{emotion.title()} steering · {spec['model_key']} · {spec['environment']}"
        )
        has_legacy = all(c["legacy_n"] == c["n"] for c in points)
        subtitle = f"{origin} · {len(names)} fixed task(s) · n = {n} per strength"
        if has_legacy:
            subtitle += " · same traces for both rubrics"
        footer = (
            f"{truncated}/{total} traces hit the output limit; incomplete and uncertain traces remain in the denominator.\n"
            "Whiskers: descriptive 95% Wilson intervals for repeated generations on fixed prompts, not uncertainty across tasks.\n"
            f"{wording['scope']}" + (f"\n{protocol_note}" if protocol_note else "")
        )
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4))
        fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.27, wspace=0.23)
        fig.text(0.07, 0.94, title, fontsize=19, weight="bold")
        fig.text(0.07, 0.89, subtitle, fontsize=11, color="#5a6b78")
        rh_curve(axes[0], points, annotations=True, audited_label=wording["label"])
        axes[0].set_title(
            "Reward hacking: audited vs. legacy labels"
            if has_legacy
            else "Reward hacking: audited labels",
            loc="left",
            pad=14,
        )
        axes[0].legend(loc="upper left", frameon=False, fontsize=10)
        ax = axes[1]
        x = np.array([c["strength"] for c in points])
        bottom = np.zeros(len(points))
        width = min(np.diff(x)) * 0.64
        for label, (text, color) in LABELS.items():
            y = np.array([c[label] / c["n"] for c in points])
            if not y.any():
                continue
            ax.bar(
                x,
                y,
                width=width,
                bottom=bottom,
                color=color,
                label=text,
                edgecolor="white",
                linewidth=0.7,
            )
            for xx, yy, bb in zip(x, y, bottom):
                if yy >= 0.16:
                    ax.text(
                        xx,
                        bb + yy / 2,
                        f"{yy:.0%}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="#fff" if label != "uncertain" else "#20303d",
                    )
            bottom += y
        require(np.allclose(bottom, 1), "Label counts do not sum to denominator")
        setup_axis(ax, points, "Fraction of all traces")
        ax.set_title("Breakdown of revised labels", loc="left", pad=14)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=2,
            frameon=False,
            fontsize=9,
            columnspacing=1.5,
        )
        fig.text(0.07, 0.025, footer, fontsize=9, color="#5a6b78", linespacing=1.5)
        artifacts += save(fig, output / f"{emotion}-revised-labels")
        records += [{"emotion": emotion, "scenario": "all", **c} for c in points]

        fig, axes = plt.subplots(
            1, len(names), figsize=(5 * len(names), 5.8), squeeze=False
        )
        fig.subplots_adjust(left=0.06, right=0.98, top=0.71, bottom=0.28, wspace=0.24)
        fig.text(0.06, 0.94, title + " · by task", fontsize=18, weight="bold")
        fig.text(
            0.06,
            0.885,
            "Same fixed prompts and replicates at every strength; baseline reused within each task.",
            fontsize=11,
            color="#5a6b78",
        )
        for ax, name in zip(axes.flat, names):
            per_task = cells([r for r in selected if r["scenario"] == name])
            rh_curve(ax, per_task, audited_label=wording["label"])
            counts = ", ".join(str(v) for v in sorted({c["n"] for c in per_task}))
            ax.set_title(
                f"{TITLES.get(name, name)} · n={counts}/strength", loc="left", pad=12
            )
            records += [{"emotion": emotion, "scenario": name, **c} for c in per_task]
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.52, 0.84),
            ncol=2,
            frameon=False,
        )
        fig.text(0.06, 0.04, footer, fontsize=9, color="#5a6b78", linespacing=1.5)
        artifacts += save(fig, output / f"{emotion}-by-task")
    path = output / "rates.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    artifacts.append(path)
    if len(spec["probes"]) > 1:
        artifacts += overview(spec, rows, output, protocol_note)
        artifacts.append(gallery(spec, rows, output, protocol_note))
        archive = output / "emotion-plots.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for artifact in artifacts:
                if "-by-task." not in artifact.name:
                    z.write(artifact, artifact.name)
        artifacts.append(archive)
    atomic(
        output / "provenance.json",
        {
            "source_run": str(root),
            "identity": spec["identity"],
            "source_verification_sha256": sha(root / "verification.json"),
            "script_sha256": sha(Path(__file__)),
            "observations": len(rows),
            "new_generation_calls": 0,
            "new_judge_calls": 0,
            "protocol": protocol,
            "protocol_sha256": sha(root / "protocol.json") if protocol else None,
            "artifacts": {str(p.relative_to(output)): sha(p) for p in artifacts},
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    verification = read(root / "verification.json")
    spec = read(root / "config.resolved.json")
    require(
        verification["passed"] and verification["identity"] == spec["identity"],
        "Run not verified",
    )
    for name, checksum in verification["artifacts"].items():
        path = (root / name).resolve()
        require(
            path.is_relative_to(root) and sha(path) == checksum,
            "Verified artifact changed: " + name,
        )
    rows = joined_rows(root, spec)
    require(rows == read(root / "audit-rows.json"), "Saved rows do not reproduce")
    require(
        summarize(rows, spec) == read(root / "metrics.json"),
        "Saved metrics do not reproduce",
    )
    output = root / "figures" / "revised-labels"
    draw(root, spec, rows, output)
    print(json.dumps({"output": str(output), "observations": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
