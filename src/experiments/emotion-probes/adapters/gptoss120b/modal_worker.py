"""Run an existing prepared directory on one H200 and mirror progress locally.

Example:
  uv run --extra modal modal run modal_worker.py --run-dir /absolute/run --through report
HF_TOKEN uses an ephemeral Modal secret when provided locally; otherwise the named huggingface secret.
The persistent volume holds run data; model weights use a separate cache volume.
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


from pathlib import Path
import json
import copy
import contextlib
import traceback
import subprocess
import time
import tarfile

import modal
import os

from probe_harness.common import Run, atomic, digest, read, require, sha, now

HERE = Path(__file__).resolve().parent
app = modal.App("gpt-oss120-twelve-emotion-probes-v1")
cache = modal.Volume.from_name("gpt-oss120-probe-model-cache-v1", create_if_missing=True)
data = modal.Volume.from_name("gpt-oss120-probe-runs-v1", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(HERE / "requirements-extract.txt"))
    .env(
        {
            "HF_HOME": "/models",
            "PYTHONUNBUFFERED": "1",
            "MPLCONFIGDIR": "/tmp/matplotlib",
        }
    )
    .add_local_dir(HERE / "probe_harness", remote_path="/root/probe_harness", copy=True)
)


@app.function(
    image=image,
    gpu="H200",
    cpu=16,
    memory=131072,
    timeout=86400,
    volumes={"/models": cache, "/data": data},
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})] if os.environ.get("HF_TOKEN") else [],
)
def worker(remote_dir: str, through: str, smoke_only: bool):
    directory = Path("/data") / remote_dir
    if not (directory / "status.json").exists():
        archive = Path("/data") / (remote_dir + ".tar.gz")
        directory.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(directory, filter="data")
        data.commit()
    status = read(directory / "status.json")
    call = modal.FunctionCall.from_id(modal.current_function_call_id())
    status["provider"] = {
        "name": "Modal",
        "job_url": call.get_dashboard_url(),
        "call_id": call.object_id,
    }
    atomic(directory / "status.json", status)
    stages = ["extract"] if smoke_only else ["extract", "fit", "evaluate", "report"]
    if not smoke_only:
        stages = stages[: stages.index(through) + 1]
    # The payload contains real status snapshots, not progress inferred from log text.
    last_commit = 0.0
    versions = {}

    def artifacts():
        payload = {}
        for name in (
            "smoke.json",
            "preflight.json",
            "runtime.json",
            "calibrated_models.json",
            "cv_results.json",
            "calibration_results.json",
            "evaluation_results.json",
            "verification.json",
        ):
            path = directory / name
            if path.exists() and path.stat().st_mtime_ns != versions.get(name):
                payload[name] = read(path)
                versions[name] = path.stat().st_mtime_ns
        return payload

    for stage in stages:
        command = ["python", "-m", "probe_harness.cli", "--run", str(directory), stage]
        if smoke_only:
            command.append("--smoke-only")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            status_path = directory / "status.json"
            if time.monotonic() - last_commit > 60:
                data.commit()
                last_commit = time.monotonic()
            yield {
                "status": read(status_path) if status_path.exists() else None,
                "log": line.rstrip(),
                "artifacts": artifacts(),
            }
        code = process.wait()
        data.commit()
        cache.commit()
        yield {
            "status": read(directory / "status.json"),
            "log": f"{stage} exited {code}",
            "stage_exit": code,
            "artifacts": artifacts(),
        }
        if code:
            return
    # No activation bytes in logs. The caller downloads committed artifacts separately.
    atomic(directory / "transfer_manifest.json", {str(p.relative_to(directory)): sha(p) for p in directory.rglob("*") if p.is_file() and p.name not in (".run.lock", "status.json", "transfer_manifest.json")})
    data.commit()
    yield {"finished": True, "status": read(directory / "status.json")}


@contextlib.contextmanager
def transport_guard(root):
    try:
        yield
    except BaseException as exc:
        run = Run(root)
        if run.status["stage"] == "prepare":
            run.status["stage"] = "extract"
        error = f"{type(exc).__name__}: {exc}"
        run.status.update(state="failed", error=error)
        run.status["stages"][run.status["stage"]].update(state="failed", error=error)
        with (root / "run.log").open("a") as log:
            log.write(traceback.format_exc())
        run.event(
            "provider_failed",
            error=error,
            resume="Resolve the provider/transfer error and rerun the same Modal command.",
        )
        raise


@app.local_entrypoint()
def main(run_dir: str, through: str = "report", smoke_only: bool = False):
    require(
        through in ("extract", "fit", "evaluate", "report"), "Unknown --through stage"
    )
    root = Path(run_dir).resolve()
    spec = read(root / "run_spec.json")
    require(spec["mode"] == "gpt_oss", "Modal extraction requires a real GPT-OSS run")
    from probe_harness.extraction import identity
    preflight = read(root / "preflight.json")
    require(preflight["status"] == "passed" and preflight["identity"] == identity(spec), "Preflight required before allocating GPU")
    with Run(root).locked(), transport_guard(root):
        remote_dir = root.name + "-" + digest(spec)[:16]
        # Fixed remote name means retries resume verified shards instead of starting over.
        # Do not overwrite existing remote status on a retry.
        existing = {e.path.lstrip("/") for e in data.iterdir("/", recursive=False)}
        if remote_dir not in existing:
            archive = root / "prepared_payload.tar.gz"
            with tarfile.open(archive, "w:gz", compresslevel=3) as tf:
                for name in ("run_spec.json", "status.json", "preflight.json", "inputs", "tokens"):
                    tf.add(root / name, arcname=name)
            print(f"Uploading checked preparation archive: {archive.stat().st_size} bytes", flush=True)
            with data.batch_upload(force=True) as upload:
                upload.put_file(archive, "/" + remote_dir + ".tar.gz")
            print("Preparation archive uploaded; starting GPU smoke/extraction", flush=True)
        success = False
        remote_status = None

        def publish_status(snapshot):
            local = copy.deepcopy(snapshot)
            if local["state"] == "complete":
                local["state"] = "running"
                local["stages"]["report"].update(
                    state="running", phase="downloading_artifacts"
                )
            local["updated_at"] = now()
            atomic(root / "status.json", local)

        with (root / "provider.log").open("a") as log:
            for item in worker.remote_gen(remote_dir, through, smoke_only):
                if item.get("status"):
                    remote_status = item["status"]
                    publish_status(remote_status)
                for name, value in item.get("artifacts", {}).items():
                    atomic(root / name, value)
                if item.get("log"):
                    log.write(item["log"] + "\n")
                    log.flush()
                    print(item["log"], flush=True)
                    try:
                        event = json.loads(item["log"])
                        if "event" in event:
                            with (root / "events.jsonl").open("a") as f:
                                f.write(json.dumps(event) + "\n")
                    except json.JSONDecodeError:
                        pass
                success = item.get("finished", False) or success
        # Includes logs on failure, so the local dashboard remains useful for debugging.
        try:
            transfer_checksums = json.loads(b"".join(data.read_file("/" + remote_dir + "/transfer_manifest.json")))
        except Exception:
            transfer_checksums = {}
        entries = list(data.iterdir("/" + remote_dir, recursive=True))
        entries.sort(key=lambda e: ("_features/" in e.path or "/tokens/" in e.path, e.path))
        transferred = 0
        for entry in entries:
            if entry.type != modal.volume.FileEntryType.FILE:
                continue
            rel = Path(entry.path.lstrip("/")).relative_to(remote_dir)
            if str(rel) in (".run.lock", "status.json"):
                continue
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            expected = transfer_checksums.get(str(rel))
            if expected and target.exists() and sha(target) == expected:
                continue
            temp = target.with_name(target.name + ".download")
            with temp.open("wb") as f:
                for chunk in data.read_file(entry.path):
                    f.write(chunk)
            if expected:
                require(sha(temp) == expected, f"Transfer checksum mismatch: {rel}")
            temp.replace(target)
            transferred += target.stat().st_size
            if remote_status:
                remote_status["transfer"] = {"last_file": str(rel), "bytes": transferred, "state": "running"}
                publish_status(remote_status)
        if remote_status and remote_status["state"] == "complete":
            verification = read(root / "verification.json")
            for name, expected in verification["artifacts"].items():
                require(
                    sha(root / name) == expected,
                    f"Downloaded artifact checksum mismatch: {name}",
                )
        if remote_status:
            remote_status["transfer"] = {"state": "complete"}
            remote_status["updated_at"] = now()
            atomic(root / "status.json", remote_status)
        if not success:
            raise RuntimeError(
                "Remote stage failed; see run.log and status.json. Rerun to resume verified artifacts."
            )
        print("Mirrored run:", root)
