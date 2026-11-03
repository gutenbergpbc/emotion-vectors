"""Prepare, generate, judge, report, verify. Historical imports are explicit."""

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from shared.io import atomic, read, require, sha
from shared.progress import Progress, lock

from .config import HERE, PROJECT, STAGES, resolve, run_root, tasks

sys.path.insert(0, str(PROJECT))


def progress(root, spec):
    p = Progress(root, f"{spec['model_key']} · {spec['environment']} emotion steering")
    if [s["id"] for s in p.data["stages"]] != STAGES:
        p.data["stages"] = [
            {"id": s, "label": s.title(), "state": "pending"} for s in STAGES
        ]
    return p


def commit(root, stage, task_id, value, spec):
    path = root / stage / (task_id + ".json")
    atomic(path, {"identity": spec["identity"], **value})
    receipts_path = root / "receipts.json"
    receipts = read(receipts_path) if receipts_path.exists() else {}
    receipts[stage + "/" + task_id] = {
        "path": str(path.relative_to(root)),
        "sha256": sha(path),
    }
    atomic(receipts_path, receipts)
    return path


def verify_receipts(root, spec):
    receipts = read(root / "receipts.json")
    for receipt in receipts.values():
        path = (root / receipt["path"]).resolve()
        require(
            path.is_relative_to(root.resolve()) and sha(path) == receipt["sha256"],
            "Committed artifact changed",
        )
        require(
            read(path)["identity"] == spec["identity"],
            "Committed artifact belongs to another run",
        )
    return receipts


def initialize(root, spec, resume):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "config.resolved.json"
    if path.exists():
        require(resume, "Run exists; pass --resume or choose a new run_id")
        require(
            read(path) == spec,
            "Configuration, code or input changed; choose a new run_id",
        )
        verify_receipts(root, spec)
    else:
        atomic(path, spec)
        atomic(root / "receipts.json", {})
        (root / "logs").mkdir(exist_ok=True)
        for source, dest in [
            ("dashboard.html", "index.html"),
            ("dashboard.js", "dashboard.js"),
            ("dashboard.css", "dashboard.css"),
        ]:
            shutil.copyfile(PROJECT / "dashboards" / source, root / dest)
    atomic(root / "tasks.json", tasks(spec))


def generate(root, spec, p, backend=None):
    from .environments import generate_batch, generate_one

    planned = tasks(spec)
    receipts = verify_receipts(root, spec)
    missing = [t for t in planned if "responses/" + t["id"] not in receipts]
    p.stage(
        "generate",
        len(planned) - len(missing),
        len(planned),
        "traces",
        "Loading model" if missing else "Verified cached generation",
    )
    if missing and backend is None and spec["environment"] != "terminalwrench":
        from .backend import make_backend

        backend = make_backend(spec)
    if missing and spec["environment"] == "terminalwrench":
        from .terminalwrench import preflight

        preflight(spec)
    probes = {x["emotion"]: x for x in spec["probes"]}
    scenarios = {s["id"]: s for s in spec["scenarios"]}
    batch_size = spec.get("batch_size", 1)
    for offset in range(0, len(missing), batch_size):
        batch = missing[offset : offset + batch_size]
        p.stage(
            "generate",
            len(planned) - len(missing) + offset,
            len(planned),
            "traces",
            f"{batch[0]['id']} ({len(batch)} independent episodes)",
        )
        started = time.monotonic()
        if batch_size > 1:
            results = generate_batch(backend, scenarios, batch, probes, spec)
        else:
            task = batch[0]
            results = [
                generate_one(
                    backend,
                    scenarios[task["scenario"]],
                    task,
                    probes.get(task["emotion"]),
                    spec,
                    root,
                )
            ]
        require(
            len(results) == len(batch)
            and all(
                isinstance(r, dict) and isinstance(r.get("response"), str)
                for r in results
            ),
            "Invalid generation result or batch population",
        )
        elapsed = time.monotonic() - started
        for i, (task, result) in enumerate(zip(batch, results), 1):
            commit(
                root,
                "responses",
                task["id"],
                {
                    **task,
                    **result,
                    "elapsed_seconds": elapsed,
                    "generation_batch_size": len(batch),
                    "elapsed_scope": "whole batch"
                    if len(batch) > 1
                    else "single episode",
                },
                spec,
            )
            p.event(
                "task.completed", task_id=task["id"], stage_id="generate", attempt=1
            )
            p.stage(
                "generate",
                len(planned) - len(missing) + offset + i,
                len(planned),
                "traces",
                task["id"],
            )
    p.complete_stage("generate")


def import_existing(root, spec, p):
    """Audit saved generations, never pass off historical rows as new samples."""
    historical = Path(spec["historical_run"])
    canonical = (historical / "config.resolved.json").exists()
    old_spec = read(
        historical / ("config.resolved.json" if canonical else "run_spec.json")
    )
    require(
        old_spec["model"]["name"] == spec["model"]["name"]
        and old_spec["model"]["revision"] == spec["model"]["revision"],
        "Historical model identity mismatch",
    )
    old_probes = {p["emotion"]: p for p in old_spec["probes"]}
    for probe in spec["probes"]:
        old = old_probes[probe["emotion"]]
        require(
            (old["probe"], old["layer_zero_based"], old["source_sha256"])
            == (probe["probe"], probe["layer_zero_based"], probe["source_sha256"]),
            "Historical intervention mismatch",
        )
    receipts = read(historical / "receipts.json")
    scenarios = {s["id"]: s for s in spec["scenarios"]}
    if canonical:
        require(
            old_spec["environment"] == spec["environment"]
            and old_spec["dataset_sha256"] == spec["dataset_sha256"],
            "Canonical source dataset differs",
        )
        verify_receipts(historical, old_spec)
    else:
        require(spec["environment"] == "coding", "Legacy import supports coding traces")
        old_scenarios = {s["id"]: s for s in old_spec["scenarios"]}
        for scenario in scenarios.values():
            old = old_scenarios.get(scenario["id"], {})
            require(
                scenario["messages"]
                == [{"role": "user", "content": old.get("prompt")}],
                "Legacy source prompt differs from the requested dataset",
            )
    current = verify_receipts(root, spec)
    for i, task in enumerate(tasks(spec), 1):
        if "responses/" + task["id"] not in current:
            receipt = receipts[
                ("responses/" if canonical else "generate/") + task["id"]
            ]
            source = historical / receipt["path"]
            require(
                sha(source) == receipt["sha256"],
                "Historical response checksum mismatch",
            )
            record = read(source)
            require(
                record["id"] == task["id"] and record["strength"] == task["strength"],
                "Historical task mismatch",
            )
            if not canonical:
                require(
                    record["prompt"]
                    == scenarios[task["scenario"]]["messages"][0]["content"],
                    "Historical response prompt differs from the requested dataset",
                )
            jr = receipts.get(("judgments/" if canonical else "judge/") + task["id"])
            judgment = None
            if jr:
                require(
                    sha(historical / jr["path"]) == jr["sha256"],
                    "Historical judgment changed",
                )
                judgment = read(historical / jr["path"])
                require(
                    judgment["response_sha256"] == receipt["sha256"],
                    "Historical judge/response mismatch",
                )
                if canonical:
                    judgment = judgment.get("legacy")
            record.update(
                identity=spec["identity"],
                historical_seed=record["seed"],
                historical_judgment=judgment,
                source={"path": str(source), "sha256": sha(source)},
                import_only=True,
            )
            commit(root, "responses", task["id"], record, spec)
        p.stage(
            "generate",
            i,
            len(tasks(spec)),
            "traces",
            "Import verified historical response",
        )
    p.complete_stage("generate")


def reuse_judgments(root, spec):
    """Reuse only verified labels from the identical rubric and identical saved traces."""
    source = spec.get("reuse_judgments_from")
    if not source:
        return
    source = Path(source)
    old_spec = read(source / "config.resolved.json")
    verification = read(source / "verification.json")
    require(
        verification["passed"] and verification["identity"] == old_spec["identity"],
        "Cached judge source is not verified",
    )
    for name, checksum in verification["artifacts"].items():
        path = (source / name).resolve()
        require(
            path.is_relative_to(source.resolve()) and sha(path) == checksum,
            "Cached judge artifact changed",
        )
    for key in (
        "judge_model",
        "environment",
        "dataset_sha256",
        "legacy_rubric",
        "compare_legacy_judge",
    ):
        require(
            old_spec[key] == spec[key], "Cached judge configuration differs: " + key
        )
    judge_source = str((HERE / "emotion_steering/judging.py").relative_to(PROJECT))
    require(
        old_spec["source_hashes"][judge_source] == spec["source_hashes"][judge_source],
        "Cached judge rubric/code differs",
    )
    prior = verify_receipts(source, old_spec)
    current = verify_receipts(root, spec)
    for task in tasks(spec):
        key = "judgments/" + task["id"]
        if key in current or key not in prior:
            continue
        response_key = "responses/" + task["id"]
        old_response = read(source / prior[response_key]["path"])
        new_response = read(root / current[response_key]["path"])
        require(
            {k: v for k, v in old_response.items() if k != "identity"}
            == {k: v for k, v in new_response.items() if k != "identity"},
            "Cached judgment response differs",
        )
        judgment = read(source / prior[key]["path"])
        require(
            judgment["response_sha256"] == prior[response_key]["sha256"],
            "Cached judgment response binding differs",
        )
        commit(
            root,
            "judgments",
            task["id"],
            {
                "task_id": task["id"],
                "audit": judgment["audit"],
                "legacy": judgment["legacy"],
                "response_sha256": current[response_key]["sha256"],
                "reused_from": {
                    "path": str(source / prior[key]["path"]),
                    "sha256": prior[key]["sha256"],
                },
            },
            spec,
        )


def judge(root, spec, p, send=None):
    from .judging import audit, check_fixtures, legacy, request

    send = send or request
    reuse_judgments(root, spec)
    receipts = verify_receipts(root, spec)
    planned = tasks(spec)
    require(
        all("responses/" + t["id"] in receipts for t in planned),
        "Generate every planned trace before judging",
    )
    p.stage("judge", 0, len(planned), "traces", "Checking judge regression fixtures")
    fixture_path = root / "judge-fixtures.json"
    if not fixture_path.exists():
        atomic(fixture_path, check_fixtures(spec, send))
    fixture = read(fixture_path)
    require(
        fixture["passed"],
        "Judge failed authored regression fixtures; inspect judge-fixtures.json",
    )
    scenarios = {s["id"]: s for s in spec["scenarios"]}
    missing = [t for t in planned if "judgments/" + t["id"] not in receipts]
    completed = len(planned) - len(missing)
    p.stage(
        "judge",
        completed,
        len(planned),
        "traces",
        "Verified cached labels; judging remaining traces",
    )

    def classify(task):
        record = read(root / receipts["responses/" + task["id"]]["path"])
        scenario = scenarios[task["scenario"]]
        result = audit(record, scenario, spec, send)
        old = (
            legacy(record, scenario, spec, send)
            if spec["compare_legacy_judge"]
            else None
        )
        return {
            "task_id": task["id"],
            "audit": result,
            "legacy": old,
            "response_sha256": receipts["responses/" + task["id"]]["sha256"],
        }

    failures = []
    with ThreadPoolExecutor(max_workers=spec.get("judge_workers", 1)) as pool:
        pending = {pool.submit(classify, task): task for task in missing}
        for future in as_completed(pending):
            task = pending[future]
            try:
                value = future.result()
            except Exception as exc:  # noqa: BLE001 -- retain successful judgments and report every failed task
                failures.append({"task_id": task["id"], "error": repr(exc)})
                atomic(root / "judge-errors.json", failures)
                p.event("judge.failed", task_id=task["id"], error=str(exc))
                continue
            # All receipt/progress writes stay on the coordinator thread.
            commit(root, "judgments", task["id"], value, spec)
            completed += 1
            p.stage("judge", completed, len(planned), "traces", task["id"])
    atomic(root / "judge-errors.json", failures)
    require(
        not failures,
        f"{len(failures)} judgments failed; inspect judge-errors.json and resume",
    )
    p.complete_stage("judge")


def joined_rows(root, spec):
    from .judging import evidence, validate_label

    receipts = verify_receipts(root, spec)
    rows = []
    for t in tasks(spec):
        r = read(root / receipts["responses/" + t["id"]]["path"])
        j = read(root / receipts["judgments/" + t["id"]]["path"])
        require(
            j["response_sha256"] == receipts["responses/" + t["id"]]["sha256"],
            "Judge/response binding mismatch",
        )
        a = validate_label(j["audit"], evidence(r))
        rows.append(
            dict(
                **{**t, "seed": r["seed"]},
                label=a["label"],
                quote=a["quote"],
                explanation=a["explanation"],
                mechanism=a["mechanism"],
                legacy_label=j["legacy"]["label"] if j["legacy"] else None,
                legacy_quote=j["legacy"].get("quote", "") if j["legacy"] else "",
                truncated=r.get("finish_reason")
                in ("length", "turn_limit", "episode_timeout"),
                generation_error=r.get("generation_error"),
                objective=r.get("objective"),
                transcript=evidence(r),
            )
        )
    return rows


def verify(root, spec):
    from .reporting import summarize

    rows = joined_rows(root, spec)
    require(
        read(root / "metrics.json") == summarize(rows, spec),
        "Metrics do not reproduce from saved judgments",
    )
    expected = tasks(spec)
    require(
        len(rows) == len(expected) and len({r["id"] for r in rows}) == len(expected),
        "Missing/duplicate observations",
    )
    require(read(root / "judge-fixtures.json")["passed"], "Judge fixtures failed")
    artifacts = [
        "config.resolved.json",
        "tasks.json",
        "receipts.json",
        "metrics.json",
        "metrics.csv",
        "audit-rows.json",
        "judge-fixtures.json",
        "report.html",
        "figures/rh-rates.png",
        "figures/rh-rates.svg",
        "figures/rh-rates.pdf",
    ]
    checksums = {name: sha(root / name) for name in artifacts}
    result = {
        "passed": True,
        "identity": spec["identity"],
        "observations": len(rows),
        "mode": spec["mode"],
        "artifacts": checksums,
        "checks": [
            "complete condition grid",
            "single shared baseline",
            "probe/checkpoint identity",
            "exact paired task IDs",
            "response/judge checksums",
            "quoted evidence present",
            "authored judge regression cases",
            "RH/failure rates recomputed from observations",
            "report artifacts",
        ],
    }
    atomic(root / "verification.json", result)
    return result


def main(config_path, dry_run=False, resume=False, serve_port=None, through="verify"):
    spec = resolve(config_path)
    root = run_root(spec)
    if serve_port:
        from shared.server import serve

        serve(root, serve_port)
        return
    require(through in STAGES, "Steering supports prepare/generate/judge/report/verify")
    if dry_run:
        print(
            json.dumps(
                {
                    "run_dir": str(root),
                    "model": spec["model"]["name"],
                    "environment": spec["environment"],
                    "tasks": len(tasks(spec)),
                    "probes": len(spec["probes"]),
                    "strengths": spec["strengths"],
                    "samples": spec["samples"],
                    "batch_size": spec.get("batch_size", 1),
                    "gpu": spec["model"]["gpu"],
                    "executor": spec["executor"],
                    "mode": spec["mode"],
                },
                indent=2,
            )
        )
        return
    with lock(root):
        initialize(root, spec, resume)
        with progress(root, spec) as p:
            p.stage(
                "prepare",
                1,
                1,
                "configurations",
                "Frozen configuration and probe vectors",
            )
            p.complete_stage("prepare")
            if through == "prepare":
                p.finish(False)
                return
            if spec["mode"] == "audit-existing":
                import_existing(root, spec, p)
            elif spec["executor"] == "modal":
                receipts = verify_receipts(root, spec)
                if not all("responses/" + t["id"] in receipts for t in tasks(spec)):
                    p.stage(
                        "generate",
                        0,
                        len(tasks(spec)),
                        "traces",
                        "Starting Modal GPU worker",
                    )
                    # The worker owns generation snapshots; no local heartbeat should mask its liveness.
                    p.stop.set()
                    p.thread.join()
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "modal",
                            "run",
                            "--detach",
                            str(HERE / "modal_worker.py") + "::main",
                            "--run-dir",
                            str(root),
                        ],
                        check=True,
                        env={**os.environ, "EMOTION_STEERING_RUN": str(root)},
                    )
                    p.data = read(root / "progress.json")
                    # Remote events use their own file; restart a local heartbeat for judging.
                    import threading

                    p.stop.clear()
                    p.thread = threading.Thread(target=p._heartbeat, daemon=True)
                    p.thread.start()
                p.complete_stage("generate")
            else:
                generate(root, spec, p)
            if through == "generate":
                p.finish(False)
                return
            judge(root, spec, p)
            if through == "judge":
                p.finish(False)
                return
            from .reporting import report

            p.stage("report", 0, 1, "reports", "Building curves and trace audit")
            report(root, spec, joined_rows(root, spec))
            p.stage("report", 1, 1, "reports")
            p.complete_stage("report")
            if through == "report":
                p.finish(False)
                return
            p.stage(
                "verify", 0, 1, "checks", "Recomputing statistics and checking bindings"
            )
            verify(root, spec)
            p.stage("verify", 1, 1, "checks")
            p.complete_stage("verify")
            p.finish(True)
    print(f"Verified results: {root}", flush=True)
