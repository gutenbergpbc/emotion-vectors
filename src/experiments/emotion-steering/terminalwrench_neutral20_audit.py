"""Audit the saved neutral20 TerminalWrench traces, including partial timeouts."""
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


from collections import Counter
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from shared.io import atomic, read, sha
from emotion_steering.terminalwrench import alias

from terminalwrench_neutral20_20261026 import SOURCES, RUN_SUFFIX


def audit(root: Path) -> dict:
    plan = read(root / "planned-episodes.json")
    selection = read(root / "selection.json")
    rows = []
    for task in plan:
        path = root / "episodes" / f"{alias(task)}.json"
        row = {"id": task["id"], "path": str(path), "present": path.is_file()}
        if not path.is_file():
            rows.append(row)
            continue
        episode = read(path)
        try:
            trajectory = json.loads(episode["trajectory_text"])
            steps = trajectory["steps"]
            agent_steps = sum(step.get("source") == "agent" for step in steps)
        except (KeyError, ValueError, TypeError, AttributeError):
            agent_steps = 0
        artifacts = episode.get("native_artifacts") or []
        receipts_valid = bool(artifacts) and all(
            (root / receipt["path"]).is_file()
            and sha(root / receipt["path"]) == receipt["sha256"]
            for receipt in artifacts
        )
        neutral = (episode.get("task") == task and task["condition"] == "baseline"
                   and task["emotion"] is None and task["strength"] == 0
                   and task["sample"] == 2)
        termination = episode.get("termination")
        error = episode.get("generation_error")
        usable = bool(neutral and agent_steps and receipts_valid
                      and termination in ("complete", "active_budget_timeout")
                      and error in (None, "active_budget_timeout"))
        row.update(termination=termination, generation_error=error,
                   agent_steps=agent_steps, native_artifacts=len(artifacts),
                   receipts_valid=receipts_valid, neutral=neutral,
                   usable=usable, sha256=sha(path))
        rows.append(row)
    ids = [task["scenario"] for task in plan]
    summary = {
        "status": "complete" if len(rows) == 20 and all(r.get("usable") for r in rows)
                  and len(set(ids)) == 20 and sorted(ids) == selection["scenario_ids"]
                  else "incomplete",
        "planned": len(plan),
        "saved": sum(r["present"] for r in rows),
        "usable": sum(r.get("usable", False) for r in rows),
        "normal_completions": sum(r.get("termination") == "complete" and r.get("usable")
                                  for r in rows),
        "partial_timeouts": sum(r.get("termination") == "active_budget_timeout"
                                and r.get("usable") for r in rows),
        "terminations": dict(Counter(r.get("termination", "missing") for r in rows)),
        "episodes": rows,
    }
    atomic(root / "trace-quality.json", summary)
    return summary


if __name__ == "__main__":
    base = HERE.parents[2] / "runs/emotion-steering"
    for name in ("gpt_oss", "gemma", "glm"):
        root = base / f"{SOURCES[name]}-{RUN_SUFFIX}"
        if (root / "validation.json").is_file():
            result = audit(root)
            print(f"{name}: {result['status']}; {result['usable']}/20 usable, "
                  f"{result['normal_completions']} complete, "
                  f"{result['partial_timeouts']} partial timeouts")
