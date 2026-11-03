"""Audit neutral v7 sentence labels and task-held-out forecast feasibility."""
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


from collections import defaultdict
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from shared.io import atomic, read, require, sha
from emotion_rh_forecasting.inputs import Sources, load_probes

LABEL_ROOT = ROOT / "runs/emotion-rh-sentence-labels"
OUT = ROOT / "runs/emotion-rh-forecasting/terminalwrench-neutral-v7-20261026-preflight"
SOURCES = [
    LABEL_ROOT / "terminalwrench-v7-steering-gpt61sol-v2",
    LABEL_ROOT / "terminalwrench-v7-neutral-extension-gpt61sol-v2",
    LABEL_ROOT / "terminalwrench-neutral20-20261026-astra-review-v1",
]
PROBES = {
    "GPT-OSS 120B": ROOT / "data/probes/gptoss120b/probes.json",
    "Gemma 4B": ROOT / "data/probes/gemma4b/probes.json",
    "GLM 5.3": ROOT / "data/probes/glm53/probes.json",
}


def run():
    by_model = defaultdict(list)
    seen = set()
    provenance = []
    for folder in SOURCES:
        labels_path = folder / "labels.json"
        labels = read(labels_path)
        require(labels["summary"]["status"] == "complete", f"Incomplete labels: {folder}")
        provenance.append({"path": str(labels_path.relative_to(ROOT)), "sha256": sha(labels_path)})
        for trace in labels["traces"]:
            task = trace["task"]
            if task["strength"] != 0:
                continue
            require(task["condition"] == "baseline" and task["emotion"] is None,
                    "Unexpected neutral task")
            identity = (trace["model"], task["id"])
            require(identity not in seen, f"Duplicate neutral episode: {identity}")
            seen.add(identity)
            by_model[trace["model"]].append(trace)
    result = {"status": "complete", "label_sources": provenance, "models": {}}
    for model, path in PROBES.items():
        bundle = load_probes(path, Sources())
        traces = by_model[model]
        rows = []
        # Freeze two task-disjoint folds using the rare +10 positive task groups.
        # The other tasks are assigned by sentence load, without consulting probes.
        h10_by_task = defaultdict(int)
        sentence_load = defaultdict(int)
        for trace in traces:
            task_name = trace["task"]["scenario"]
            sentence_load[task_name] += len(trace["units"])
            h10_by_task[task_name] += sum(u["hacking"] and not u["needs_review"]
                                          for u in trace["units"][10:])
        positive_tasks = sorted((t for t, n in h10_by_task.items() if n),
                                key=lambda t: (-h10_by_task[t], t))
        fold_tasks = [[], []]
        fold_positive = [0, 0]
        fold_sentences = [0, 0]
        for task_name in positive_tasks:
            fold = min(range(2), key=lambda i: (fold_positive[i], fold_sentences[i], i))
            fold_tasks[fold].append(task_name)
            fold_positive[fold] += h10_by_task[task_name]
            fold_sentences[fold] += sentence_load[task_name]
        for task_name in sorted(set(sentence_load) - set(positive_tasks),
                                key=lambda t: (-sentence_load[t], t)):
            fold = min(range(2), key=lambda i: (fold_sentences[i], i))
            fold_tasks[fold].append(task_name)
            fold_sentences[fold] += sentence_load[task_name]
        task_folds = {task_name: fold for fold, tasks_in_fold in enumerate(fold_tasks)
                      for task_name in tasks_in_fold}
        for h in range(0, 11):
            positives = []
            candidates = 0
            fold_counts = [{"positive": 0, "negative": 0} for _ in range(2)]
            for trace in traces:
                units = trace["units"]
                fold = task_folds[trace["task"]["scenario"]]
                for target in units[h:]:
                    candidates += 1
                    if target["hacking"] and not target["needs_review"]:
                        positives.append(trace["task"]["scenario"])
                        fold_counts[fold]["positive"] += 1
                    else:
                        fold_counts[fold]["negative"] += 1
            groups = sorted(set(positives))
            rows.append({"horizon": h, "candidate_pairs_before_activation_checks": candidates,
                         "positive_pairs": len(positives), "positive_task_groups": groups,
                         "task_held_out_possible": all(c["positive"] and c["negative"] for c in fold_counts),
                         "fold_counts_before_activation_checks": fold_counts,
                         "balanced_pairs_before_activation_checks": 2 * sum(
                             min(c["positive"], c["negative"]) for c in fold_counts)})
        result["models"][model] = {
            "labelable_traces": len(traces),
            "probe_bundle": str(path.relative_to(ROOT)), "probe_bundle_sha256": sha(path),
            "checkpoint": bundle["model"], "probe_count": len(bundle["probes"]),
            "task_folds": task_folds,
            "horizons": rows,
            "readiness": "needs_activation_replay" if all(r["task_held_out_possible"] for r in rows)
                         else "insufficient_positive_task_groups",
        }
    result["interpretation"] = (
        "Neutral v7 labels only. Positive means hacking=true and needs_review=false. "
        "Counts precede token-scoring exclusions. A task-held-out fold needs positives in at least two task groups "
        "so both training and evaluation can contain positives; this is a necessary, not sufficient, condition. "
        "No v7 sentence-level emotion probe activations are included in these label sources.")
    OUT.mkdir(parents=True, exist_ok=True)
    atomic(OUT / "readiness.json", result)
    print(json.dumps({model: {"traces": data["labelable_traces"],
                               "positive_groups_h10": len(data["horizons"][10]["positive_task_groups"]),
                               "readiness": data["readiness"]}
                      for model, data in result["models"].items()}, indent=2))


if __name__ == "__main__":
    run()
