#!/usr/bin/env python3
"""Preserve a stopped checkpoint when changing only steering batch size.

The source run stays immutable. Imported payloads retain their generation batch
size and bind to their original bytes through a per-record provenance receipt.
This helper is deliberately outside the frozen generation package.
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
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from emotion_steering.config import PROJECT, resolve, run_root, tasks
from emotion_steering.pipeline import commit, initialize, progress, verify_receipts
from shared.io import atomic, read, require, sha


def verify_continuation(root):
    root = Path(root).resolve()
    spec = read(root / "config.resolved.json")
    manifest = read(root / "continuation.json")
    require(manifest["identity"] == spec["identity"], "Continuation identity mismatch")
    source = Path(manifest["source_run"])
    require(sha(source / "config.resolved.json") == manifest["source_config_sha256"],
            "Source configuration changed")
    require(sha(source / "receipts.json") == manifest["source_receipts_sha256"],
            "Source checkpoint changed")
    receipts = verify_receipts(root, spec)
    for key, binding in manifest["bindings"].items():
        receipt = receipts[key]
        require(receipt["sha256"] == binding["continued_sha256"],
                "Previously completed sample was overwritten")
        original = Path(binding["source"]["path"])
        require(sha(original) == binding["source"]["sha256"], "Original sample changed")
        continued = read(root / receipt["path"])
        require(continued.pop("continuation_source") == binding["source"],
                "Sample provenance changed")
        continued["identity"] = manifest["source_identity"]
        require(continued == read(original), "Continued episode differs from original")
    result = {"identity": spec["identity"], "preserved_samples": len(manifest["bindings"]),
              "completed_samples_overwritten_or_rerun": 0,
              "checks": ["source configuration and checkpoint hashes", "original response hashes",
                         "imported response hashes", "complete episode payload equality"]}
    atomic(root / "continuation-verification.json", result)
    print(json.dumps(result), flush=True)


def import_checkpoint(source, config):
    import modal

    source = Path(source).resolve()
    spec = resolve(config)
    root = run_root(spec)
    require(root != source, "Continuation requires a distinct run ID")
    volume = modal.Volume.from_name("canonical-emotion-steering-v1", create_if_missing=True)
    run_id = source.name

    def download(relative):
        data = b"".join(volume.read_file(f"/{run_id}/{relative}"))
        destination = source / relative
        if destination.exists() and relative in ("config.resolved.json",):
            require(json.loads(destination.read_bytes()) == json.loads(data),
                    "Source configuration changed remotely")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

    # The boundary watcher pauses the producer and commits the whole batch first.
    for name in ("batch32-boundary.json", "config.resolved.json", "receipts.json"):
        download(name)
    old = read(source / "config.resolved.json")
    boundary = read(source / "batch32-boundary.json")
    receipts = read(source / "receipts.json")
    require(sha(source / "receipts.json") == boundary["receipts_sha256"],
            "Source advanced after the stopped batch boundary")
    require(all(k.startswith("responses/") for k in receipts),
            "Continuation expects generation-only source receipts")
    require(len(receipts) == boundary["completed_responses"], "Boundary count differs")
    allowed = {"identity", "run_id", "batch_size"}
    require({k: v for k, v in old.items() if k not in allowed}
            == {k: v for k, v in spec.items() if k not in allowed},
            "Only run ID and batch size may change; model/data/code/seeds must match")
    require(old["batch_size"] == 12 and spec["batch_size"] == 32,
            "This continuation was requested for batch12 to batch32")
    require(old["model_key"] == "gemma-3-4b", "Gemma-only continuation")
    planned = {t["id"]: t for t in tasks(spec)}
    require(tasks(old) == tasks(spec), "Task IDs or seeds changed")
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(download, [r["path"] for r in receipts.values()]))
    verify_receipts(source, old)
    require(not (root / "config.resolved.json").exists(),
            "Continuation already exists; do not overwrite an active run")
    initialize(root, spec, resume=False)
    bindings = {}
    for key, receipt in receipts.items():
        original = source / receipt["path"]
        row = read(original)
        task = planned[row["id"]]
        require(all(row[k] == v for k, v in task.items()), "Task payload differs")
        provenance = {
            "run_id": old["run_id"], "identity": old["identity"],
            "path": str(original), "sha256": receipt["sha256"],
            "generation_batch_size": row["generation_batch_size"],
        }
        row["identity"] = spec["identity"]
        row["continuation_source"] = provenance
        output = commit(root, "responses", row["id"], row, spec)
        # Prove the entire episode is unchanged apart from bookkeeping fields.
        restored = read(output)
        restored.pop("continuation_source")
        restored["identity"] = old["identity"]
        require(restored == read(original), "Imported episode content changed")
        bindings[key] = {"source": provenance, "continued_sha256": sha(output)}
    current = verify_receipts(root, spec)
    require(set(current) == set(receipts), "A completed sample was dropped or added")
    manifest = {
        "identity": spec["identity"], "source_run": str(source),
        "source_identity": old["identity"], "source_config_sha256": sha(source / "config.resolved.json"),
        "source_receipts_sha256": sha(source / "receipts.json"),
        "source_boundary": boundary, "previous_batch_size": 12, "new_batch_size": 32,
        "preserved_samples": len(receipts), "remaining_samples": len(planned) - len(receipts),
        "allowed_changes": ["run_id", "batch_size", "identity"],
        "completed_samples_rerun": 0, "bindings": bindings,
    }
    atomic(root / "continuation.json", manifest)
    snapshot = root / "frozen-source"
    for name, expected in spec["source_hashes"].items():
        require(sha(PROJECT / name) == expected, "Frozen source changed")
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / name, target)
    atomic(snapshot / "manifest.json", {"identity": spec["identity"], "files": spec["source_hashes"]})
    with progress(root, spec) as p:
        p.stage("prepare", 1, 1, "configurations", "Preserved batch12 checkpoint")
        p.complete_stage("prepare")
        p.stage("generate", len(receipts), len(planned), "traces", "Ready to continue remaining samples at batch32")
        p.finish(False)
    # Worker generation only checks its mounted volume, so seed both locations.
    with volume.batch_upload() as upload:
        for relative in ["config.resolved.json", "receipts.json", "continuation.json", "progress.json"]:
            upload.put_file(root / relative, f"/{root.name}/{relative}")
        upload.put_directory(root / "responses", f"/{root.name}/responses")
    remote = json.loads(b"".join(volume.read_file(f"/{root.name}/receipts.json")))
    require(remote == current, "Remote continuation receipts differ")
    atomic(source / "continuation-target.json", {
        "run_id": root.name, "identity": spec["identity"], "preserved_samples": len(receipts),
        "reason": "User requested batch32; all committed episodes retained in continuation",
    })
    print(json.dumps({"run_dir": str(root), "preserved": len(receipts),
                      "remaining": len(planned) - len(receipts), "batch_size": 32}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source-run", type=Path)
    mode.add_argument("--verify-run", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.verify_run:
        verify_continuation(args.verify_run)
    else:
        if args.config is None:
            parser.error("--source-run requires --config")
        import_checkpoint(args.source_run, args.config)
