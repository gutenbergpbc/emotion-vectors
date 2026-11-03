"""Freeze neutral TerminalWrench ATIF histories for sentence self-report."""
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


import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CANONICAL = HERE.parent
PROJECT = CANONICAL.parents[1]
sys.path.insert(0, str(HERE))
from emotion_self_report.segmentation import segment

SOURCE = PROJECT / "runs/emotion-steering"
DEST = HERE / "inputs/terminalwrench-neutral-v7-20261026"
MODELS = {
    "gptoss120b": "openai/gpt-oss-120b",
    "gemma4b": "google/gemma-3-4b-it",
    "glm53": "z-ai/glm-5.3",
}
RUN_STEM = "-terminalwrench-interactive-steering-active-time-v2-rh-top4-continuous-v7"
EXTENSION = "-neutral-extension-8-20261025"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_frozen(path: Path, value):
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != content:
        raise ValueError(f"Frozen input changed: {path}")
    if not path.exists():
        path.write_text(content)


def export(model_key: str, model_id: str):
    traces, points, provenance = [], [], []
    seen = set()
    for suffix in ("", EXTENSION):
        root = SOURCE / (model_key + RUN_STEM + suffix)
        if not root.exists():
            continue
        for path in sorted((root / "episodes").glob("*.json")):
            episode = json.loads(path.read_text())
            task = episode.get("task") or {}
            if not (task.get("condition") == "baseline" and task.get("emotion") is None
                    and task.get("strength") == 0 and not episode.get("generation_error")
                    and episode.get("termination") == "complete"):
                continue
            trajectory = json.loads(episode.get("trajectory_text") or "null")
            steps = trajectory.get("steps", [])
            agents = [step for step in steps if step.get("source") == "agent"]
            counts = [sum(unit["kind"] == "sentence_or_fragment" for unit in segment(step.get("message", "")))
                      for step in agents]
            if len(agents) < 2 or sum(counts) < 10 or sum(n > 0 for n in counts) < 2:
                continue
            key = (task["scenario"], task["sample"])
            if key in seen:
                raise ValueError(f"Duplicate task/sample: {model_key} {key}")
            seen.add(key)
            for artifact in episode.get("native_artifacts") or []:
                native = root / artifact["path"]
                if not native.is_file() or sha(native) != artifact["sha256"]:
                    raise ValueError(f"Invalid native artifact: {native}")

            trace_id = f"{model_key}/{task['scenario']}/{task['sample']:04d}"
            messages = []
            trace_points = []
            uid = 0
            for step in steps:
                if step.get("source") == "user":
                    messages.append({"role": "user", "content": step["message"]})
                elif step.get("source") == "agent":
                    body = step.get("message", "")
                    calls = step.get("tool_calls") or []
                    if calls:
                        body += "\n\n# Submitted commands\n```json\n" + json.dumps(calls, ensure_ascii=False, indent=2) + "\n```"
                    index = len(messages)
                    messages.append({"role": "assistant", "content": body})
                    for unit in segment(step.get("message", "")):
                        if unit["kind"] == "sentence_or_fragment":
                            trace_points.append({"trace_id": trace_id, "unit_id": uid,
                                                 "message_index": index,
                                                 "char_start": unit["char_start"],
                                                 "char_end": unit["char_end"],
                                                 "text": unit["text"]})
                        uid += 1
                    observation = step.get("observation") or {}
                    outputs = [r.get("content", "") for r in observation.get("results", [])]
                    if outputs:
                        messages.append({"role": "user", "content": "\n\n".join(outputs)})
            if not messages or messages[0]["role"] != "user" or not trace_points:
                raise ValueError(f"Invalid trace: {trace_id}")
            traces.append({"id": trace_id, "model": model_id, "scenario": task["scenario"],
                           "label": "neutral", "messages": messages})
            points.extend(trace_points)
            provenance.append({"trace_id": trace_id, "task_id": task["id"],
                               "source": str(path), "source_sha256": sha(path),
                               "native_artifacts": episode.get("native_artifacts"),
                               "agent_turns": len(agents), "covered_turns": sum(n > 0 for n in counts),
                               "sentence_checkpoints": len(trace_points),
                               "source_model": json.loads((root / "config.resolved.json").read_text()).get("model")})
    traces.sort(key=lambda row: row["id"])
    points.sort(key=lambda row: (row["trace_id"], row["message_index"], row["char_end"]))
    provenance.sort(key=lambda row: row["trace_id"])
    root = DEST / model_key
    write_frozen(root / "traces.json", traces)
    write_frozen(root / "checkpoints.json", points)
    write_frozen(root / "provenance.json", {"api_model": model_id,
        "normalization": "ATIF visible agent message; submitted commands appended as fenced JSON; terminal observations replayed as user text; reasoning excluded.",
        "traces": provenance})
    return {"model": model_id, "traces": len(traces), "checkpoints": len(points),
            "requests": len(points) * 12, "traces_path": str(root / "traces.json")}


if __name__ == "__main__":
    print(json.dumps([export(k, v) for k, v in MODELS.items()], indent=2))
