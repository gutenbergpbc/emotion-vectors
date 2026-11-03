"""GLM batch32 continuation with recorded, checkpoint-preserving batch16 fallback.

The canonical scientific code stays frozen. This adapter is included in the new
run's source hashes and keeps the requested batch size separate from runtime
fallback decisions.
"""

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
import json
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from emotion_steering.config import PROJECT, resolve, run_root, tasks
from emotion_steering.pipeline import (
    commit,
    generate,
    initialize,
    joined_rows,
    judge,
    progress,
    verify,
    verify_receipts,
)

from shared.io import atomic, digest, read, require, sha
from shared.progress import lock

EXTRA_SOURCES = [HERE / "glm_adaptive.py", HERE / "glm_modal_worker.py"]


def resolve_adaptive(config):
    spec = resolve(config)
    require(
        spec["model_key"] == "glm-5.3-nvfp4" and spec["batch_size"] == 32,
        "This adapter requires the pinned GLM model and requested batch32",
    )
    for path in EXTRA_SOURCES:
        spec["source_hashes"][str(path.relative_to(PROJECT))] = sha(path)
    spec.pop("identity")
    spec["identity"] = digest(spec)
    return spec


def check_sources(spec):
    for name, expected in spec["source_hashes"].items():
        require(sha(PROJECT / name) == expected, f"Frozen source changed: {name}")


def import_checkpoint(source, config, queued=False):
    import modal

    source = Path(source).resolve()
    spec = resolve_adaptive(config)
    root = run_root(spec)
    require(root != source, "Continuation needs a new run ID")
    volume = modal.Volume.from_name("canonical-emotion-steering-v1", create_if_missing=True)

    def download(relative):
        if relative.startswith("responses/") and (source / relative).exists():
            # Every cached byte is checked against the original receipt below.
            return
        data = b"".join(volume.read_file(f"/{source.name}/{relative}"))
        path = source / relative
        if path.exists() and relative == "config.resolved.json":
            require(read(path) == json.loads(data), "Source specification changed")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    if queued:
        try:
            remote = json.loads(
                b"".join(volume.read_file(f"/{source.name}/receipts.json"))
            )
        except (FileNotFoundError, modal.exception.NotFoundError):
            remote = {}
        require(
            not remote and not read(source / "receipts.json"),
            "Queued source unexpectedly has completed samples",
        )
        boundary = {
            "completed_responses": 0,
            "source_never_started": True,
            "receipts_sha256": sha(source / "receipts.json"),
        }
    else:
        for relative in (
            "batch32-boundary.json",
            "config.resolved.json",
            "receipts.json",
        ):
            download(relative)
        boundary = read(source / "batch32-boundary.json")
    old = read(source / "config.resolved.json")
    receipts = read(source / "receipts.json")
    require(
        sha(source / "receipts.json") == boundary["receipts_sha256"],
        "Source advanced after the stopped boundary",
    )
    require(
        len(receipts) == boundary["completed_responses"]
        and all(k.startswith("responses/") for k in receipts),
        "Invalid source checkpoint population",
    )
    allowed = {"identity", "run_id", "batch_size", "source_hashes"}
    require(
        {k: v for k, v in old.items() if k not in allowed}
        == {k: v for k, v in spec.items() if k not in allowed},
        "Scientific configuration changed",
    )
    require(
        old["batch_size"] == 16 and old["model_key"] == "glm-5.3-nvfp4",
        "Expected the original GLM batch16 run",
    )
    require(
        all(spec["source_hashes"].get(k) == v for k, v in old["source_hashes"].items())
        and set(spec["source_hashes"]) - set(old["source_hashes"])
        == {str(p.relative_to(PROJECT)) for p in EXTRA_SOURCES},
        "Only the explicit adaptive runtime sources may be added",
    )
    require(tasks(old) == tasks(spec), "Task IDs or seeds changed")
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(download, [r["path"] for r in receipts.values()]))
    verify_receipts(source, old)
    require(not (root / "config.resolved.json").exists(), "Continuation already exists")
    initialize(root, spec, resume=False)
    (root / "responses").mkdir(exist_ok=True)
    planned = {t["id"]: t for t in tasks(spec)}
    bindings = {}
    for key, receipt in receipts.items():
        original = source / receipt["path"]
        row = read(original)
        require(all(row[k] == v for k, v in planned[row["id"]].items()), "Task differs")
        provenance = {
            "run_id": old["run_id"],
            "identity": old["identity"],
            "path": str(original),
            "sha256": receipt["sha256"],
            "generation_batch_size": row["generation_batch_size"],
        }
        row["identity"] = spec["identity"]
        row["continuation_source"] = provenance
        output = commit(root, "responses", row["id"], row, spec)
        restored = read(output)
        restored.pop("continuation_source")
        restored["identity"] = old["identity"]
        require(restored == read(original), "Imported episode content changed")
        bindings[key] = {"source": provenance, "continued_sha256": sha(output)}
    current = verify_receipts(root, spec)
    require(set(current) == set(receipts), "Completed sample was added or dropped")
    atomic(
        root / "continuation.json",
        {
            "identity": spec["identity"],
            "source_run": str(source),
            "source_identity": old["identity"],
            "source_boundary": boundary,
            "previous_batch_size": 16,
            "new_batch_size": 32,
            "fallback_batch_size": 16,
            "preserved_samples": len(receipts),
            "remaining_samples": len(planned) - len(receipts),
            "completed_samples_rerun": 0,
            "bindings": bindings,
            "adapter_source_hashes": {
                str(p.relative_to(PROJECT)): sha(p) for p in EXTRA_SOURCES
            },
        },
    )
    for name, expected in spec["source_hashes"].items():
        require(sha(PROJECT / name) == expected, "Frozen source changed")
        target = root / "frozen-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / name, target)
    with progress(root, spec) as p:
        p.stage("prepare", 1, 1, "configurations", "Preserved GLM batch16 checkpoint")
        p.complete_stage("prepare")
        p.stage(
            "generate",
            len(receipts),
            len(planned),
            "traces",
            "Ready for batch32; automatic batch16 memory fallback",
        )
        p.finish(False)
    with volume.batch_upload() as upload:
        for relative in (
            "config.resolved.json",
            "receipts.json",
            "continuation.json",
            "progress.json",
        ):
            upload.put_file(root / relative, f"/{root.name}/{relative}")
        upload.put_directory(root / "responses", f"/{root.name}/responses")
    require(
        json.loads(b"".join(volume.read_file(f"/{root.name}/receipts.json")))
        == current,
        "Uploaded checkpoint differs",
    )
    atomic(
        source / "continuation-target.json",
        {
            "run_id": root.name,
            "identity": spec["identity"],
            "preserved_samples": len(receipts),
            "reason": "User requested batch32 with batch16 fallback; committed episodes retained",
        },
    )
    print(
        json.dumps(
            {
                "run_dir": str(root),
                "preserved": len(receipts),
                "remaining": len(planned) - len(receipts),
            }
        ),
        flush=True,
    )


def memory_or_engine_failure(exc):
    if (
        isinstance(exc, (ValueError, AssertionError))
        or type(exc).__name__ == "BackendValidationError"
    ):
        return False
    name = type(exc).__name__
    text = str(exc).lower()
    return name in ("OutOfMemoryError", "EngineDeadError") or (
        isinstance(exc, RuntimeError)
        and any(
            term in text
            for term in (
                "out of memory",
                "cuda error: memory allocation",
                "engine core initialization failed",
                "enginecore encountered an issue",
            )
        )
    )


def run_attempts(run_child, cleanup):
    """A child owns the complete CUDA process group; never reuse a failed engine."""
    statuses = []
    for size in (32, 16):
        try:
            result = run_child(size)
        finally:
            cleanup()
        statuses.append(result)
        if result["ok"] or not result.get("fallback_eligible") or size == 16:
            break
    return statuses


def child(specs_path, effective_batch_size, status_path):
    import modal
    from emotion_steering.backend import make_backend

    specs = read(Path(specs_path))
    status_path = Path(status_path)
    backend = None
    status = {
        "ok": False,
        "effective_batch_size": effective_batch_size,
        "started_at": time.time(),
        "fallback_eligible": False,
    }
    try:
        for spec in specs:
            check_sources(spec)
            root = Path("/data") / spec["run_id"]
            with lock(root):
                require(
                    read(root / "config.resolved.json") == spec,
                    "Remote specification changed",
                )
                receipts = verify_receipts(root, spec)
                missing = [
                    t for t in tasks(spec) if "responses/" + t["id"] not in receipts
                ]
                with progress(root, spec) as p:
                    p.stage(
                        "prepare",
                        1,
                        1,
                        "configurations",
                        "Verified preserved checkpoint",
                    )
                    p.complete_stage("prepare")
                    if not missing:
                        p.stage(
                            "generate",
                            len(tasks(spec)),
                            len(tasks(spec)),
                            "traces",
                            "All saved samples retained",
                        )
                        p.complete_stage("generate")
                        p.finish(False)
                        continue
                    runtime_spec = {**spec, "batch_size": effective_batch_size}
                    if backend is None:
                        p.stage(
                            "generate",
                            len(tasks(spec)) - len(missing),
                            len(tasks(spec)),
                            "traces",
                            f"Loading cached GLM weights for batch{effective_batch_size}",
                        )
                        started = time.monotonic()
                        backend = make_backend(runtime_spec)
                        load_seconds = time.monotonic() - started
                    else:
                        backend.spec, backend.cfg = runtime_spec, spec["model"]
                    previous = (
                        read(root / "runtime.json")
                        if (root / "runtime.json").exists()
                        else {}
                    )
                    atomic(
                        root / "runtime.json",
                        {
                            "identity": spec["identity"],
                            "model": spec["model"]["name"],
                            "revision": spec["model"]["revision"],
                            "gpu": spec["model"]["gpu"],
                            "requested_batch_size": 32,
                            "effective_batch_size": effective_batch_size,
                            "load_seconds": load_seconds,
                            "suite_run_ids": [s["run_id"] for s in specs],
                            "previous_runtime": previous or None,
                            "adapter_source_hashes": {
                                str(p.relative_to(PROJECT)): sha(p)
                                for p in EXTRA_SOURCES
                            },
                        },
                    )
                    generate(root, runtime_spec, p, backend)
                    p.finish(False)
        status["ok"] = True
    except Exception as exc:  # noqa: BLE001 -- persist failure before the parent destroys the CUDA process group
        status.update(
            error=traceback.format_exc(),
            error_type=type(exc).__name__,
            fallback_eligible=memory_or_engine_failure(exc),
        )
        if getattr(exc, "diagnostics", None) is not None:
            atomic(
                root / "generation-error.json",
                {
                    "identity": spec["identity"],
                    "error": str(exc),
                    "diagnostics": exc.diagnostics,
                },
            )
    finally:
        status["ended_at"] = time.time()
        atomic(status_path, status)
        modal.Volume.from_name("canonical-emotion-steering-v1", create_if_missing=True).commit()
    print(json.dumps(status), flush=True)
    return 0 if status["ok"] else 1


def finish(config):
    from emotion_steering.reporting import report

    spec = resolve_adaptive(config)
    root = run_root(spec)
    check_sources(spec)
    with lock(root):
        require(
            read(root / "config.resolved.json") == spec,
            "Continuation specification changed",
        )
        receipts = verify_receipts(root, spec)
        require(
            all("responses/" + t["id"] in receipts for t in tasks(spec)),
            "Generation is incomplete",
        )
        with progress(root, spec) as p:
            p.stage(
                "generate",
                len(tasks(spec)),
                len(tasks(spec)),
                "traces",
                "All generation receipts verified",
            )
            p.complete_stage("generate")
            judge(root, spec, p)
            p.stage("report", 0, 1, "reports", "Building steering curves")
            report(root, spec, joined_rows(root, spec))
            p.stage("report", 1, 1, "reports")
            p.complete_stage("report")
            p.stage(
                "verify", 0, 1, "checks", "Recomputing statistics and checking bindings"
            )
            verify(root, spec)
            p.stage("verify", 1, 1, "checks")
            p.complete_stage("verify")
            p.finish(True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--import-source", type=Path)
    parser.add_argument("--queued", action="store_true")
    parser.add_argument("--finish", action="store_true")
    parser.add_argument("--child-specs", type=Path)
    parser.add_argument("--effective-batch-size", type=int, choices=(16, 32))
    parser.add_argument("--status-path", type=Path)
    args = parser.parse_args()
    if args.child_specs:
        sys.exit(child(args.child_specs, args.effective_batch_size, args.status_path))
    elif args.import_source:
        import_checkpoint(args.import_source, args.config, args.queued)
    elif args.finish:
        finish(args.config)
    else:
        parser.error("Choose --import-source, --finish, or --child-specs")
