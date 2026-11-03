"""One GLM GPU allocation; isolated CUDA subprocesses make batch fallback safe."""

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


import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal
import modal_worker as base

HERE = (
    Path(__file__).resolve().parent
    if modal.is_local()
    else Path("/study/src/experiments/emotion-steering")
)
sys.path[:0] = [str(HERE), str(HERE.parent)]
from glm_sales_capped import check_sources, run_attempts

from shared.io import atomic, read, require

app = base.app
image = base.image
for name in (
    "glm_adaptive.py",
    "glm_modal_worker.py",
    "glm_sales_capped.py",
    "glm_sales_capped_worker.py",
):
    image = image.add_local_file(
        HERE / name, f"/study/src/experiments/emotion-steering/{name}"
    )


@app.function(
    image=image,
    gpu=base.MODEL["gpu"],
    cpu=16,
    memory=131072,
    timeout=86400,
    volumes=base.volumes,
    secrets=[base.secret],
    max_containers=1,
)
def adaptive_worker(spec, following_specs=None):
    specs = [spec, *(following_specs or [])]
    for current in specs:
        check_sources(current)
        require(
            {k: v for k, v in current["model"].items() if k != "answer_prefix"}
            == {k: v for k, v in spec["model"].items() if k != "answer_prefix"},
            "Suite requires one model runtime",
        )
    spec_path = Path("/tmp/glm-adaptive-specs.json")
    atomic(spec_path, specs)
    stop = threading.Event()

    def persist():
        while not stop.wait(10):
            base.runs.commit()

    thread = threading.Thread(target=persist, daemon=True)
    thread.start()
    process = None

    def cleanup():
        if process is None:
            return
        # Every engine/TP process inherits this subprocess's process group.
        # Kill the entire group before attempting a new CUDA engine.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        time.sleep(2)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def run_child(size):
        nonlocal process
        status_path = Path("/data") / spec["run_id"] / f"adaptive-attempt-{size}.json"
        if status_path.exists():
            status_path.unlink()
        process = subprocess.Popen(
            [
                sys.executable,
                str(HERE / "glm_sales_capped.py"),
                "--child-specs",
                str(spec_path),
                "--effective-batch-size",
                str(size),
                "--status-path",
                str(status_path),
            ],
            start_new_session=True,
        )
        # A failed CUDA engine can hang during Python interpreter shutdown.
        # The status is written before shutdown, so release the whole process
        # group as soon as failure is recorded instead of waiting indefinitely.
        while process.poll() is None:
            if status_path.exists() and not read(status_path)["ok"]:
                cleanup()
                break
            time.sleep(1)
        code = process.wait()
        if status_path.exists():
            result = read(status_path)
        else:
            result = {
                "ok": False,
                "effective_batch_size": size,
                "fallback_eligible": size == 32 and code in (-9, 137),
                "error": f"Generation subprocess exited {code} without a status artifact",
            }
        result["exit_code"] = code
        for current in specs:
            root = Path("/data") / current["run_id"]
            previous = (
                read(root / "adaptive-runtime.json")
                if (root / "adaptive-runtime.json").exists()
                else {"attempts": []}
            )
            atomic(
                root / "adaptive-runtime.json",
                {
                    "identity": current["identity"],
                    "requested_batch_size": 32,
                    "fallback_batch_size": 16,
                    "attempts": previous["attempts"] + [result],
                },
            )
        base.runs.commit()
        return result

    try:
        statuses = run_attempts(run_child, cleanup)
    finally:
        cleanup()
        stop.set()
        thread.join()
        base.runs.commit()
    archives = {
        s["run_id"]: base.archive_run(Path("/data") / s["run_id"]) for s in specs
    }
    return {
        "archives": archives,
        "archive": archives[spec["run_id"]],
        "error": None
        if statuses[-1]["ok"]
        else statuses[-1].get("error", "Generation failed"),
    }


@app.local_entrypoint()
def continue_suite(run_dirs_json: str):
    roots = [Path(p).resolve() for p in json.loads(Path(run_dirs_json).read_text())]
    require(
        roots and roots[0] == base.LOCAL_RUN,
        "Suite must start with EMOTION_STEERING_RUN",
    )
    # Reuse the canonical, identity-bound reconnect/download implementation.
    base.worker = adaptive_worker
    base.run_local(roots)
