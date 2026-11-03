"""Pinned model-specific GPU image; canonical coordinator owns local judging."""

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
import io
import json
import os
import tarfile
import threading
import time
import traceback
from pathlib import Path

import modal

HERE = (
    Path(__file__).resolve().parent
    if modal.is_local()
    else Path("/study/src/experiments/emotion-steering")
)
CANONICAL = HERE.parent
PROJECT = CANONICAL.parents[1]
LOCAL_RUN = Path(os.environ["EMOTION_STEERING_RUN"]).resolve()
SPEC = json.loads((LOCAL_RUN / "config.resolved.json").read_text())
MODEL = SPEC["model"]
models = modal.Volume.from_name(MODEL["cache_volume"], create_if_missing=True)
runs = modal.Volume.from_name("canonical-emotion-steering-v1", create_if_missing=True)
uses_lens = MODEL.get("backend") == "vllm_lens" or MODEL["family"] == "glm"
volumes = {"/models": models, "/data": runs}
if uses_lens:
    volumes["/kernel-cache"] = modal.Volume.from_name(
        "canonical-glm53-kernel-cache-v1", create_if_missing=True
    )
image = modal.Image.debian_slim(python_version="3.12")
if MODEL["family"] == "gpt_oss" and not uses_lens:
    image = image.pip_install_from_requirements(
        str(PROJECT / "src/experiments/emotion-probes/adapters/gptoss120b/requirements-extract.txt")
    )
elif uses_lens:
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6",
            add_python="3.12",
        )
        .entrypoint([])
        .apt_install("build-essential", "ninja-build")
        .uv_pip_install(
            "vllm==0.30.0", "vllm-lens==1.2.1", "transformers==5.17.0", "numpy==2.3.5"
        )
        .uv_pip_install(
            "flashinfer-jit-cache @ https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.18.post1/flashinfer_jit_cache-0.6.18.post1%2Bcu130-cp39-abi3-manylinux_2_28_x86_64.whl#sha256=d1729490636a98f22f158518e10c257eec5a188d3dadf30b4c9de298675d8d4b",
            "flashinfer-cubin @ https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.18.post1/flashinfer_cubin-0.6.18.post1-py3-none-any.whl#sha256=bbacb5b8bbf429e43bf2740bb45551d981f41d0462842e36ee6fb3e3452763ab",
        )
    )
else:
    image = image.uv_pip_install(
        *[f"{k}=={v}" for k, v in MODEL["packages"].items()],
        "accelerate==1.15.0",
        "numpy==2.3.5",
        "sentencepiece",
        "protobuf",
    )
    if MODEL["family"] == "qwen":
        image = image.uv_pip_install("flash-linear-attention==0.5.2")
image = (
    image.uv_pip_install("pandas==2.3.3")
    .env(
        {
            "TERMINALWRENCH_INFERENCE_LIMIT": os.environ.get("TERMINALWRENCH_INFERENCE_LIMIT", str(SPEC.get("batch_size", 1))),
            "HF_HOME": "/models",
            "HF_HUB_CACHE": MODEL["hub_cache"],
            "PYTHONUNBUFFERED": "1",
            "EMOTION_STEERING_RUN": "/study/run",
            "PYTHONPATH": "/study:/study/src/experiments:/study/src/experiments/emotion-steering",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "CUDA_HOME": "/usr/local/cuda",
            "FLASHINFER_WORKSPACE_BASE": "/kernel-cache/cuda13-gcc13-v1",
            "MAX_JOBS": "2",
            "OMP_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
        }
    )
    .add_local_file(
        LOCAL_RUN / "config.resolved.json", "/study/run/config.resolved.json"
    )
    .add_local_dir(
        HERE / "emotion_steering",
        "/study/src/experiments/emotion-steering/emotion_steering",
        ignore=["__pycache__"],
    )
    .add_local_dir(
        CANONICAL / "shared",
        "/study/src/experiments/shared",
        ignore=["__pycache__"],
    )
    .add_local_file(
        HERE / "models.json",
        "/study/src/experiments/emotion-steering/models.json",
    )
    .add_local_file(
        HERE / "modal_worker.py",
        "/study/src/experiments/emotion-steering/modal_worker.py",
    )
    .add_local_dir(PROJECT / "src/environments/sales_env", "/study/sales_env", ignore=["__pycache__"])
)
secret_values = {
    k: os.environ[k]
    for k in ("HF_TOKEN", "TERMINALWRENCH_API_KEY")
    if os.environ.get(k)
}
secret = (
    modal.Secret.from_dict(secret_values)
    if secret_values
    else modal.Secret.from_name("huggingface")
)
app = modal.App("canonical-emotion-steering")


@app.function(
    image=image,
    gpu=MODEL["gpu"],
    cpu=16,
    memory=131072,
    timeout=86400,
    volumes=volumes,
    secrets=[secret],
    max_containers=1,
)
def worker(spec, following_specs=None):
    """Generate one or more same-model runs using one loaded GPU allocation."""
    from emotion_steering.backend import make_backend
    from emotion_steering.pipeline import generate, progress

    from shared.io import atomic, read, require, sha
    from shared.progress import lock

    specs = [spec, *(following_specs or [])]
    for current in specs:
        require(
            {k: v for k, v in current["model"].items() if k != "answer_prefix"}
            == {k: v for k, v in spec["model"].items() if k != "answer_prefix"},
            "Suite requires one model runtime",
        )
        require(
            current.get("batch_size", 1) <= spec.get("batch_size", 1),
            "First suite run must set largest batch size",
        )
        for name, expected in current["source_hashes"].items():
            require(
                sha(Path("/study") / name) == expected,
                f"Remote source differs from frozen run: {name}",
            )
    error = None
    backend = None
    runtime = None
    stop = threading.Event()

    def persist():
        while not stop.wait(10):
            runs.commit()

    thread = threading.Thread(target=persist, daemon=True)
    thread.start()
    try:
        for current in specs:
            root = Path("/data") / current["run_id"]
            with lock(root):
                if (root / "config.resolved.json").exists():
                    require(
                        read(root / "config.resolved.json") == current,
                        "Remote specification changed",
                    )
                else:
                    atomic(root / "config.resolved.json", current)
                    atomic(root / "receipts.json", {})
                with progress(root, current) as p:
                    p.stage(
                        "prepare",
                        1,
                        1,
                        "configurations",
                        "Frozen model and probe vectors",
                    )
                    p.complete_stage("prepare")
                    if backend is None:
                        p.stage(
                            "generate",
                            0,
                            0,
                            "traces",
                            "Loading shared model for sequential suite",
                        )
                        started = time.monotonic()
                        backend = make_backend(current)
                        runtime = {
                            "loaded_for_run_id": current["run_id"],
                            "load_seconds": time.monotonic() - started,
                            "model": current["model"]["name"],
                            "revision": current["model"]["revision"],
                            "gpu": current["model"]["gpu"],
                            "suite_run_ids": [s["run_id"] for s in specs],
                        }
                    else:
                        backend.spec = current
                        backend.cfg = current["model"]
                    atomic(
                        root / "runtime.json",
                        {"identity": current["identity"], **runtime},
                    )
                    generate(root, current, p, backend)
                    p.finish(False)
    except Exception as exc:  # noqa: BLE001 -- preserve committed outputs even if a later run fails
        error = traceback.format_exc()
        if getattr(exc, "diagnostics", None) is not None:
            atomic(
                root / "generation-error.json",
                {
                    "identity": current["identity"],
                    "error": str(exc),
                    "diagnostics": exc.diagnostics,
                },
            )
    finally:
        stop.set()
        thread.join()
        runs.commit()
    archives = {
        current["run_id"]: archive_run(Path("/data") / current["run_id"])
        for current in specs
    }
    return {"archive": archives[spec["run_id"]], "archives": archives, "error": error}


def archive_run(root):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name not in (
                ".run.lock",
                "config.resolved.json",
            ):
                tar.add(path, arcname=str(path.relative_to(root)))
    return archive.getvalue()


@app.local_entrypoint()
def main(run_dir: str):
    root = Path(run_dir).resolve()
    if root != LOCAL_RUN:
        raise ValueError("EMOTION_STEERING_RUN differs from --run-dir")
    run_local([root])


@app.local_entrypoint()
def suite(run_dirs_json: str):
    """JSON array of prepared run directories; one model load and one GPU worker."""
    roots = [
        Path(value).resolve() for value in json.loads(Path(run_dirs_json).read_text())
    ]
    if not roots or roots[0] != LOCAL_RUN or len(set(roots)) != len(roots):
        raise ValueError(
            "Suite must start with EMOTION_STEERING_RUN and have distinct roots"
        )
    run_local(roots)


def run_local(roots):
    import sys

    sys.path.insert(0, str(HERE))
    from emotion_steering.remote import result_archives, saved_call_id, suite_members

    specs = [json.loads((root / "config.resolved.json").read_text()) for root in roots]
    for spec in specs:
        for name, expected in spec["source_hashes"].items():
            if hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Local source changed during deployment: {name}")
    connection = roots[0] / "connection.json"
    saved = json.loads(connection.read_text()) if connection.exists() else {}
    existing = saved_call_id(saved, specs)
    if existing:
        call = modal.FunctionCall.from_id(existing)
    else:
        call = worker.spawn(specs[0], specs[1:])
    for root, spec in zip(roots, specs, strict=True):
        (root / "connection.json").write_text(
            json.dumps(
                {
                    "function_call_id": call.object_id,
                    "volume": "canonical-emotion-steering-v1",
                    "identity": spec["identity"],
                    "suite_run_ids": [s["run_id"] for s in specs],
                    "suite_members": suite_members(specs),
                    "state": "running",
                }
            )
        )
    try:
        while True:
            try:
                result = call.get(timeout=10)
                break
            except TimeoutError:
                for root, spec in zip(roots, specs, strict=True):
                    try:
                        snapshot = b"".join(
                            runs.read_file(f"/{spec['run_id']}/progress.json")
                        )
                        temp = root / "progress.remote.tmp"
                        temp.write_bytes(snapshot)
                        temp.replace(root / "progress.json")
                    except (FileNotFoundError, modal.exception.NotFoundError):
                        pass
    except (modal.exception.FunctionTimeoutError, modal.exception.RemoteError):
        for root in roots:
            saved = json.loads((root / "connection.json").read_text())
            saved["state"] = "failed"
            (root / "connection.json").write_text(json.dumps(saved))
        raise
    archives = result_archives(result, specs)
    for root, spec in zip(roots, specs, strict=True):
        download_archive(root, archives[spec["run_id"]])
        saved = json.loads((root / "connection.json").read_text())
        saved["state"] = "failed" if result["error"] else "complete"
        (root / "connection.json").write_text(json.dumps(saved))
    if result["error"]:
        raise RuntimeError(result["error"])
    print(f"Downloaded generation artifacts to {', '.join(map(str, roots))}")


def download_archive(root, archive):
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            relative = Path(member.name)
            if not member.isfile() or relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid remote artifact path")
            # Keep local coordinator events independent from remote worker events.
            if relative == Path("events.jsonl"):
                relative = Path("events/gpu.jsonl")
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(".download")
            temp.write_bytes(tar.extractfile(member).read())
            temp.replace(target)


if SPEC["environment"] == "terminalwrench":

    @app.function(
        image=image,
        gpu=MODEL["gpu"],
        cpu=16,
        memory=131072,
        timeout=14400,
        volumes=volumes,
        secrets=[secret],
        max_containers=1,
        scaledown_window=120,
    )
    @modal.concurrent(max_inputs=8)
    @modal.web_server(8000, startup_timeout=1800)
    def native_endpoint():
        """Explicit `modal serve` endpoint for the separate local Harbor coordinator."""
        import subprocess

        from shared.io import atomic

        if SPEC["environment"] != "terminalwrench":
            raise ValueError(
                "Native endpoint requires a prepared TerminalWrench config"
            )
        if not os.environ.get("TERMINALWRENCH_API_KEY"):
            raise ValueError("Set TERMINALWRENCH_API_KEY before serving")
        root = Path("/data") / (SPEC["run_id"] + "-endpoint")
        atomic(root / "config.resolved.json", SPEC)
        subprocess.Popen(
            [
                "python",
                "-m",
                "emotion_steering.terminalwrench",
                "--run-dir",
                str(root),
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ]
        )
