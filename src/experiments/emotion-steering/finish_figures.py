"""Render each campaign's figures when its saved run passes verification."""

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
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared.io import atomic, read, sha


def render_ready(campaign_path):
    campaign = read(campaign_path)
    results = campaign_path.parent.parent
    completed, failures = [], {}
    for run in campaign["runs"]:
        root = results / run["id"]
        verification_path = root / "verification.json"
        if not verification_path.exists():
            continue
        try:
            verification = read(verification_path)
            if not verification.get("passed"):
                continue
            output = root / "figures/revised-labels"
            provenance_path = output / "provenance.json"
            provenance = read(provenance_path) if provenance_path.exists() else {}
            current = (
                provenance.get("identity") == verification["identity"]
                and provenance.get("source_verification_sha256")
                == sha(verification_path)
                and provenance.get("script_sha256") == sha(HERE / "plot_saved.py")
                and all(
                    (output / name).is_file() and sha(output / name) == checksum
                    for name, checksum in provenance.get("artifacts", {}).items()
                )
            )
            if not current:
                subprocess.run(
                    [
                        sys.executable,
                        str(HERE / "plot_saved.py"),
                        "--run-dir",
                        str(root),
                    ],
                    check=True,
                )
            completed.append(run["id"])
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            failures[run["id"]] = str(error)
    state = {"completed": completed, "errors": failures, "total": len(campaign["runs"])}
    atomic(campaign_path.with_name("figure-progress.json"), state)
    if completed:
        campaign["status"] = (
            "All runs are verified and their 12-emotion figure galleries are ready."
            if len(completed) == len(campaign["runs"])
            else f"{len(completed)} of {len(campaign['runs'])} verified figure galleries are ready. Remaining runs continue below."
        )
        atomic(campaign_path, campaign)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    previous = None
    while True:
        state = render_ready(args.campaign.resolve())
        if state != previous:
            print(json.dumps(state), flush=True)
            previous = state
        if args.once or len(state["completed"]) == state["total"]:
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
