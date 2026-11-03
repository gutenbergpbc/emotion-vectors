#!/usr/bin/env python3
"""Run through Modal container exec to persist a whole batch before a handoff.

Only use for a worker whose reload was requested. This pauses the designated
coordinator and its VLLM processes; after the final volume commit, its app may be
stopped and the saved responses imported into a continuation run.
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
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import modal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--target", type=int)
    parser.add_argument("--coordinator-pid", type=int, default=2)
    args = parser.parse_args()
    root = Path("/data") / args.run_id
    receipts_path = root / "receipts.json"
    initial = len(json.loads(receipts_path.read_text()))
    target = args.target or min(588, (initial // args.batch_size + 1) * args.batch_size)
    print(json.dumps({"initial": initial, "pause_after": target}), flush=True)
    while len(json.loads(receipts_path.read_text())) < target:
        time.sleep(0.01)
    os.kill(args.coordinator_pid, signal.SIGSTOP)
    paused = [args.coordinator_pid]
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in (1, args.coordinator_pid, os.getpid()):
            continue
        try:
            if b"VLLM::" in (proc / "cmdline").read_bytes():
                os.kill(int(proc.name), signal.SIGSTOP)
                paused.append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    receipts = json.loads(receipts_path.read_text())
    marker = {
        "reason": "User requested batch32 continuation after completed batch",
        "previous_batch_size": args.batch_size,
        "completed_responses": len(receipts),
        "initial_responses": initial,
        "paused_pids": paused,
        "receipts_sha256": hashlib.sha256(receipts_path.read_bytes()).hexdigest(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (root / "batch32-boundary.json").write_text(json.dumps(marker, indent=2) + "\n")
    modal.Volume.from_name("canonical-emotion-steering-v1", create_if_missing=True).commit()
    print(json.dumps(marker), flush=True)


if __name__ == "__main__":
    main()
