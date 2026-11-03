"""Freeze exact neutral ATIF sentence inputs for the canonical forecast runner.

This prepares labels and native-visible histories; it does not create probe
activations or imply that a model-specific causal replay has been validated.
"""
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
from shared.io import atomic, read, require, sha  # noqa: E402
from preflight_terminalwrench_neutral_v7_20261026 import OUT, SOURCES  # noqa: E402

DEST = ROOT / "runs/emotion-rh-forecasting/terminalwrench-neutral-v7-20261026-inputs"


def prepare():
    audit = read(OUT / "readiness.json")
    by_model = defaultdict(list)
    for folder in SOURCES:
        labels_path = folder / "labels.json"
        require(any(entry["path"] == str(labels_path.relative_to(ROOT)) and
                    entry["sha256"] == sha(labels_path) for entry in audit["label_sources"]),
                f"Label source changed: {labels_path}")
        for trace in read(labels_path)["traces"]:
            task = trace["task"]
            if task["condition"] == "baseline" and task["emotion"] is None and task["strength"] == 0:
                by_model[trace["model"]].append(trace)
    for model, labels in by_model.items():
        key = {"GPT-OSS 120B": "gptoss120b", "Gemma 4B": "gemma4b", "GLM 5.3": "glm53"}[model]
        task_folds = audit["models"][model]["task_folds"]
        traces = []
        for labeled in sorted(labels, key=lambda t: t["id"]):
            source = ROOT / labeled["source_path"]
            require(sha(source) == labeled["source_sha256"], f"ATIF episode changed: {source}")
            episode = read(source)
            require(episode["task"] == labeled["task"], "Episode task changed")
            trajectory = json.loads(episode["trajectory_text"])
            units_by_step = defaultdict(list)
            for unit in labeled["units"]:
                require(unit["kind"] == "sentence_or_fragment" and unit["field"] == "message",
                        "Unexpected label unit")
                units_by_step[str(unit["step_id"])].append(unit)
            messages, turns = [], []
            for step in trajectory["steps"]:
                if step["source"] == "user":
                    messages.append({"role": "user", "content": step["message"]})
                    continue
                if step["source"] != "agent":
                    continue
                step_id = str(step["step_id"])
                body = step.get("message", "")
                units = []
                for unit in units_by_step.pop(step_id, []):
                    lo, hi = unit["char_start"], unit["char_end"]
                    require(body[lo:hi] == unit["text"], "Sentence differs from ATIF")
                    units.append({k: unit[k] for k in ("unit_id", "kind", "char_start", "char_end",
                                                       "text", "hacking", "needs_review")})
                if units:
                    require(messages and messages[0]["role"] == "user", "Missing user prompt")
                    turns.append({"id": step_id, "messages": list(messages),
                                  "completion": body, "units": units})
                history_body = body
                calls = step.get("tool_calls") or []
                if calls:
                    history_body += "\n\n# Submitted commands\n```json\n" + json.dumps(
                        calls, ensure_ascii=False, indent=2) + "\n```"
                messages.append({"role": "assistant", "content": history_body})
                outputs = [r.get("content", "") for r in (step.get("observation") or {}).get("results", [])]
                if outputs:
                    messages.append({"role": "user", "content": "\n\n".join(outputs)})
            require(not units_by_step and turns, "Unmatched label step or prose-free trace")
            traces.append({"id": labeled["id"], "task": labeled["task"]["scenario"],
                           "group_id": labeled["task"]["scenario"],
                           "fold": task_folds[labeled["task"]["scenario"]], "turns": turns})
        dest = DEST / f"{key}.json"
        atomic(dest, {"schema_version": 1, "traces": traces})
        print(json.dumps({"model": model, "traces": len(traces),
                          "turns": sum(len(t["turns"]) for t in traces),
                          "sentences": sum(len(u["units"]) for t in traces for u in t["turns"]),
                          "path": str(dest)}))


if __name__ == "__main__":
    prepare()
