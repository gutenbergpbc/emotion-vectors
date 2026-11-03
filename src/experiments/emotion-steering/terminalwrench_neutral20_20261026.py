"""Collect twenty new, shared-task neutral TerminalWrench episodes per model."""
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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(HERE.parent)]

from shared.io import atomic, digest, read, sha
from emotion_steering.config import tasks
from emotion_steering.terminalwrench import alias, generate_one, preflight, serve
from emotion_steering.active_budget import ActiveClock
from emotion_steering.timed_proxy import TimedProxy


SOURCES = {
    "gpt_oss": "gptoss120b-terminalwrench-interactive-steering-active-time-v2-rh-top4-continuous-v7",
    "gemma": "gemma4b-terminalwrench-interactive-steering-active-time-v2-rh-top4-continuous-v7",
    "glm": "glm53-terminalwrench-interactive-steering-active-time-v2-rh-top4-continuous-v7",
}
RUN_SUFFIX = "neutral20-sample2-20261026"


def paths(model):
    source = PROJECT / "runs/emotion-steering" / SOURCES[model]
    root = source.parent / f"{SOURCES[model]}-{RUN_SUFFIX}"
    return source, root


def shared_scenarios(spec):
    by_stratum = {}
    for scenario in spec["scenarios"]:
        by_stratum.setdefault(scenario["stratum"], []).append(scenario["id"])
    if len(by_stratum) != 6 or any(len(v) != 4 for v in by_stratum.values()):
        raise ValueError("Expected the frozen six-stratum, 24-task TerminalWrench suite")
    chosen = []
    for index, (stratum, ids) in enumerate(sorted(by_stratum.items())):
        # Four strata contribute three tasks and two contribute four: 20 total.
        count = 3 if index < 4 else 4
        chosen.extend(sorted(ids, key=lambda value: digest([stratum, value]))[:count])
    return sorted(chosen)


def prepare(model):
    source, root = paths(model)
    if root.exists():
        raise ValueError(f"Run already exists: {root}")
    root.mkdir(parents=True)
    spec = read(source / "config.resolved.json")
    selected = shared_scenarios(spec)
    spec.update(run_id=root.name, samples=3)
    spec.pop("identity", None)
    spec["identity"] = digest(spec)
    cfg = read(source / "config.json")
    cfg.update(run_id=root.name, samples=3)
    atomic(root / "config.json", cfg)
    atomic(root / "config.resolved.json", spec)
    atomic(root / "dataset.json", read(source / "dataset.json"))
    plan = [t for t in tasks(spec) if t["condition"] == "baseline"
            and t["sample"] == 2 and t["scenario"] in selected]
    if len(plan) != 20 or len({t["scenario"] for t in plan}) != 20:
        raise ValueError("Expected twenty distinct neutral task scenarios")
    for task in plan:
        if task["emotion"] is not None or task["strength"] != 0:
            raise ValueError("Non-neutral task in plan")
    atomic(root / "planned-episodes.json", plan)
    atomic(root / "selection.json", {
        "source_run": str(source), "model": model, "sample": 2,
        "scenario_ids": selected, "runner_sha256": sha(__file__),
        "source_config_sha256": sha(source / "config.resolved.json"),
        "source_dataset_sha256": sha(source / "dataset.json"),
        "method": "Twenty shared scenarios, 3-4 per task stratum; deterministic hash selection before outcomes; fresh replicate sample 2.",
    })
    atomic(root / "progress.json", {"status": "prepared", "planned": 20, "completed": 0})
    return root


def validate(root):
    plan = read(root / "planned-episodes.json")
    rows = []
    for task in plan:
        path = root / "episodes" / f"{alias(task)}.json"
        if not path.exists():
            rows.append({"id": task["id"], "status": "missing"})
            continue
        episode = read(path)
        try:
            trajectory = json.loads(episode.get("trajectory_text") or "null")
            agent_steps = sum(step.get("source") == "agent" for step in trajectory.get("steps", []))
        except (ValueError, AttributeError, TypeError):
            agent_steps = 0
        artifacts = episode.get("native_artifacts") or []
        artifacts_valid = all((root / a["path"]).is_file()
                              and sha(root / a["path"]) == a["sha256"] for a in artifacts)
        valid = (episode.get("task") == task and not episode.get("generation_error")
                 and agent_steps > 0 and bool(artifacts) and artifacts_valid)
        rows.append({"id": task["id"], "status": "valid" if valid else "invalid",
                     "termination": episode.get("termination"), "agent_steps": agent_steps,
                     "path": str(path), "sha256": sha(path)})
    result = {"status": "complete" if all(r["status"] == "valid" for r in rows) else "incomplete",
              "planned": len(plan), "saved": sum(r["status"] != "missing" for r in rows),
              "valid": sum(r["status"] == "valid" for r in rows),
              "terminations": dict(Counter(r.get("termination", "missing") for r in rows)),
              "episodes": rows}
    atomic(root / "validation.json", result)
    return result


def run(model, port, concurrency, app_name=None):
    _, root = paths(model)
    plan = read(root / "planned-episodes.json")
    spec = read(root / "config.resolved.json")
    if read(root / "selection.json")["runner_sha256"] != sha(__file__):
        raise ValueError("Frozen runner changed")
    app = app_name or f"terminalwrench-{model.replace('_', '-')}-neutral20-20261026"
    env = dict(os.environ, EMOTION_STEERING_RUN=str(root),
               TERMINALWRENCH_WORKER_APP=app, TERMINALWRENCH_INFERENCE_LIMIT="4")
    with (root / "deploy.log").open("a") as log:
        deployed = subprocess.run([sys.executable, "-m", "modal", "deploy",
                                   str(HERE / "terminalwrench_async_worker.py")],
                                  env=env, stdout=log, stderr=subprocess.STDOUT)
    if deployed.returncode:
        raise RuntimeError(f"Modal deploy failed; inspect {root / 'deploy.log'}")

    try:
        import modal
        worker = modal.Cls.from_name(app, "AsyncInteractive")()
        info = worker.info.spawn().get(timeout=1800)
        if info.get("identity") != spec["identity"] or info.get("max_context") != 65536:
            raise RuntimeError("Worker identity or context differs from frozen plan")
        atomic(root / "worker-info.json", info)
        spec["terminalwrench"].update(api_base=f"http://127.0.0.1:{port}/v1",
                                       budget_base=f"http://127.0.0.1:{port}",
                                       budget_file_base=str(root / "clock-state"))
        clock = ActiveClock(state_dir=root / "clock-state")
        proxy = TimedProxy(worker, clock, root, "lens")
        threading.Thread(target=serve, args=(spec, root, proxy, "127.0.0.1", port),
                         kwargs={"clock": clock}, daemon=True).start()
        time.sleep(0.5)
        preflight(spec)
        episodes = root / "episodes"
        episodes.mkdir(exist_ok=True)
        scenarios = {s["id"]: s for s in spec["scenarios"]}
        finished = {t["id"] for t in plan if (episodes / f"{alias(t)}.json").exists()}
        lock = threading.Lock()
        running = set()

        def status():
            with lock:
                atomic(root / "progress.json", {"status": "running", "planned": len(plan),
                       "completed": len(finished), "running": sorted(running),
                       "updated_unix": time.time()})

        def execute(task):
            with lock:
                running.add(task["id"])
            status()
            try:
                output = generate_one(scenarios[task["scenario"]], task, spec, root)
            except Exception as exc:
                output = {"generation_error": repr(exc), "termination": "execution_failure"}
            if alias(task) in clock.episodes:
                output["active_budget"] = clock.stop(alias(task))
            atomic(episodes / f"{alias(task)}.json", {"task": task, **output})
            with lock:
                running.remove(task["id"])
                finished.add(task["id"])
            status()
            print(f"{model}: {len(finished)}/{len(plan)} {task['id']} {output.get('termination')}", flush=True)

        status()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(execute, [t for t in plan if t["id"] not in finished]))
        result = validate(root)
        atomic(root / "progress.json", {k: v for k, v in result.items() if k != "episodes"})
        print(f"{model}: {result['valid']}/{len(plan)} valid neutral episodes", flush=True)
        if result["status"] != "complete":
            raise RuntimeError(f"Incomplete episode set; inspect {root / 'validation.json'}")
    finally:
        stopped = subprocess.run([sys.executable, "-m", "modal", "app", "stop", "--yes", app],
                                 capture_output=True, text=True)
        atomic(root / "compute-stopped.json", {"app": app, "returncode": stopped.returncode,
                                                 "stdout": stopped.stdout, "stderr": stopped.stderr})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=SOURCES, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--port", type=int, default=8180)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if args.prepare:
        print(prepare(args.model))
    else:
        run(args.model, args.port, args.concurrency)
