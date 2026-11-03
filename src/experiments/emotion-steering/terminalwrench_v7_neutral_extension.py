"""Run a small, provenance-preserving neutral-only extension to a v7 sweep."""
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
import concurrent.futures
import datetime as dt
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(HERE.parent)]
from shared.io import atomic, digest
from emotion_steering.config import tasks
from emotion_steering.terminalwrench import alias, generate_one, preflight, serve
from emotion_steering.active_budget import ActiveClock
from emotion_steering.timed_proxy import TimedProxy

SOURCES = {
    "glm": "glm53-terminalwrench-interactive-steering-active-time-v2-rh-top4-continuous-v7",
    "gemma": "gemma4b-terminalwrench-interactive-steering-active-time-v2-rh-top4-continuous-v7",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=SOURCES, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--concurrency", type=int, default=4)
    a = p.parse_args()
    source = PROJECT / "runs/emotion-steering" / SOURCES[a.model]
    suffix = "neutral-extension-8-20261025"
    run_id = f"{SOURCES[a.model]}-{suffix}"
    root = PROJECT / "runs/emotion-steering" / run_id
    if root.exists():
        raise SystemExit(f"Output already exists; refusing to overwrite: {root}")
    root.mkdir(parents=True)

    spec = json.loads((source / "config.resolved.json").read_text())
    spec["run_id"] = run_id
    # Sample 1 is a fresh stochastic replicate; all tasks remain unsteered.
    spec["samples"] = 2
    spec.pop("identity", None)
    spec["identity"] = digest(spec)
    atomic(root / "config.resolved.json", spec)
    cfg = json.loads((source / "config.json").read_text())
    cfg.update(run_id=run_id, samples=2)
    atomic(root / "config.json", cfg)
    atomic(root / "dataset.json", json.loads((source / "dataset.json").read_text()))

    source_episodes = source / "episodes"
    available = [
        t for t in json.loads((source / "planned-episodes.json").read_text())
        if t["condition"] == "baseline"
        and not (source_episodes / (alias(t) + ".json")).exists()
    ]
    # Choose distinct, unrun task scenarios with a deterministic balanced shuffle.
    by_id = {s["id"]: s for s in spec["scenarios"]}
    rng = random.Random(20261025 + (0 if a.model == "glm" else 1))
    strata = {}
    for t in available:
        strata.setdefault(by_id[t["scenario"]]["stratum"], []).append(t)
    for group in strata.values():
        rng.shuffle(group)
    chosen = []
    while strata and len(chosen) < 8:
        for name in sorted(list(strata)):
            if strata[name]:
                chosen.append(strata[name].pop())
                if len(chosen) == 8:
                    break
            if not strata[name]:
                del strata[name]
    if len(chosen) != 8:
        raise SystemExit(f"Only {len(chosen)} unrun baseline scenarios available")
    selected_ids = {t["scenario"] for t in chosen}
    extension = [t for t in tasks(spec) if t["condition"] == "baseline" and t["sample"] == 1 and t["scenario"] in selected_ids]
    if len(extension) != 8:
        raise RuntimeError(f"Expected 8 extension tasks, got {len(extension)}")
    atomic(root / "planned-episodes.json", extension)
    atomic(root / "extension-provenance.json", {
        "source_run": str(source), "model_key": a.model,
        "selection": "Eight distinct baseline scenarios not saved in v7, balanced across task strata; sample=1 fresh seed.",
        "planned_episodes": extension,
    })

    app = f"terminalwrench-{a.model}-neutral-extension-v1"
    env = dict(os.environ, EMOTION_STEERING_RUN=str(root), TERMINALWRENCH_WORKER_APP=app,
               TERMINALWRENCH_INFERENCE_LIMIT="4")
    deploy_log = (root / "deploy.log").open("a")
    deployed = subprocess.run([sys.executable, "-m", "modal", "deploy", str(HERE / "terminalwrench_async_worker.py")], env=env, stdout=deploy_log, stderr=subprocess.STDOUT)
    deploy_log.close()
    if deployed.returncode:
        raise RuntimeError(f"Modal deployment failed; see {root/'deploy.log'}")

    import modal
    worker = modal.Cls.from_name(app, "AsyncInteractive")()
    info = worker.info.spawn().get(timeout=1800)
    if info.get("identity") != spec["identity"] or info.get("max_context") != 65536:
        raise RuntimeError(f"Deployed worker identity/context mismatch: {info}")
    atomic(root / "worker-info.json", info)

    spec["terminalwrench"]["api_base"] = f"http://127.0.0.1:{a.port}/v1"
    clock = ActiveClock(state_dir=root / "clock-state")
    spec["terminalwrench"]["budget_base"] = f"http://127.0.0.1:{a.port}"
    spec["terminalwrench"]["budget_file_base"] = str(root / "clock-state")
    proxy = TimedProxy(worker, clock, root, "lens")
    threading.Thread(target=serve, args=(spec, root, proxy, "127.0.0.1", a.port), kwargs={"clock": clock}, daemon=True).start()
    time.sleep(0.5)
    preflight(spec)

    results = root / "episodes"
    results.mkdir()
    status_path = PROJECT / "dashboards/terminalwrench-three-models/status" / f"{a.model}.json"
    started = time.time()
    lock = threading.Lock()
    done, running = {}, set()

    def write_status(status="running", message="Generating eight neutral TerminalWrench episodes."):
        with lock:
            atomic(status_path, {
                "model": spec["model"]["name"], "status": status,
                "stage": "Neutral-only v7 extension", "message": message,
                "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "planned_episodes": 8, "completed_episodes": len(done),
                "failed_episodes": sum(bool(x.get("generation_error")) for x in done.values()),
                "running_episodes": len(running), "run_id": run_id,
                "raw_artifacts": str(root),
            })

    write_status()

    def episode(t):
        with lock:
            running.add(t["id"])
        write_status()
        scenario = by_id[t["scenario"]]
        try:
            out = generate_one(scenario, t, spec, root)
        except Exception as exc:
            out = {"generation_error": repr(exc), "termination": "execution_failure"}
        if alias(t) in clock.episodes:
            out["active_budget"] = clock.stop(alias(t))
        out = {"task": t, **out}
        atomic(results / (alias(t) + ".json"), out)
        with lock:
            running.remove(t["id"])
            done[t["id"]] = out
        write_status()
        return out

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            list(pool.map(episode, extension))
        atomic(root / "runtime.json", {
            "wall_seconds": time.time() - started, "episodes": len(done),
            "cost_usd": None, "cost_note": "Retrieve actual billed usage from Modal; not inferred from wall time.",
        })
        write_status("generated", "Eight neutral episodes saved; RH adjudication remains pending.")
    except BaseException as exc:
        write_status("blocked", repr(exc))
        raise
    finally:
        subprocess.run([sys.executable, "-m", "modal", "app", "stop", "--yes", app], check=False)


if __name__ == "__main__":
    main()
