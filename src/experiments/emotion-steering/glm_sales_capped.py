"""Explicit mixed-budget Sales continuation; preserve completed episodes verbatim."""

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
import html
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import glm_adaptive as adaptive
from emotion_steering import pipeline
from emotion_steering.config import PROJECT, run_root, tasks

from shared.io import atomic, digest, read, require, sha

EXTRA_SOURCES = [HERE / "glm_sales_capped.py", HERE / "glm_sales_capped_worker.py"]
original_resolve = adaptive.resolve_adaptive
check_sources = adaptive.check_sources
run_attempts = adaptive.run_attempts


def resolve_capped(config):
    spec = original_resolve(config)
    require(
        spec["environment"] == "sales" and spec["max_new_tokens"] == 2048,
        "This continuation is Sales with 2048 tokens per assistant turn",
    )
    for path in EXTRA_SOURCES:
        spec["source_hashes"][str(path.relative_to(PROJECT))] = sha(path)
    spec.pop("identity")
    spec["identity"] = digest(spec)
    return spec


def import_checkpoint(source, config):
    import modal

    source = Path(source).resolve()
    spec = resolve_capped(config)
    root = run_root(spec)
    volume = modal.Volume.from_name("canonical-emotion-steering-v1", create_if_missing=True)
    require(
        root != source and not (root / "config.resolved.json").exists(),
        "Use a new continuation ID",
    )

    def download(relative):
        data = b"".join(volume.read_file(f"/{source.name}/{relative}"))
        path = source / relative
        if relative == "config.resolved.json" and path.exists():
            require(
                read(path) == json.loads(data), "Remote source specification differs"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    for relative in ("batch32-boundary.json", "config.resolved.json", "receipts.json"):
        download(relative)
    old, receipts, boundary = (
        read(source / name)
        for name in ("config.resolved.json", "receipts.json", "batch32-boundary.json")
    )
    require(
        sha(source / "receipts.json") == boundary["receipts_sha256"],
        "Source advanced after boundary",
    )
    require(
        len(receipts) == boundary["completed_responses"]
        and all(k.startswith("responses/") for k in receipts),
        "Invalid boundary population",
    )
    allowed = {"run_id", "identity", "max_new_tokens", "source_hashes"}
    require(
        {k: v for k, v in old.items() if k not in allowed}
        == {k: v for k, v in spec.items() if k not in allowed},
        "Unexpected scientific change",
    )
    require(
        old["max_new_tokens"] == 8192 and old["batch_size"] == 32,
        "Expected the original 8192-token batch32 Sales run",
    )
    require(
        all(spec["source_hashes"].get(k) == v for k, v in old["source_hashes"].items())
        and set(spec["source_hashes"]) - set(old["source_hashes"])
        == {str(p.relative_to(PROJECT)) for p in EXTRA_SOURCES},
        "Unexpected source change",
    )
    require(tasks(old) == tasks(spec), "Task IDs or seeds changed")
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(download, [v["path"] for v in receipts.values()]))
    pipeline.verify_receipts(source, old)
    pipeline.initialize(root, spec, resume=False)
    bindings = {}
    for key, receipt in receipts.items():
        original = read(source / receipt["path"])
        row = dict(original)
        provenance = {
            "run_id": old["run_id"],
            "identity": old["identity"],
            "path": str(source / receipt["path"]),
            "sha256": receipt["sha256"],
            "generation_batch_size": row["generation_batch_size"],
            "generation_max_new_tokens": old["max_new_tokens"],
        }
        row.update(
            identity=spec["identity"],
            continuation_source=provenance,
            generation_max_new_tokens=old["max_new_tokens"],
        )
        output = pipeline.commit(root, "responses", row["id"], row, spec)
        restored = read(output)
        for field in ("identity", "continuation_source", "generation_max_new_tokens"):
            if field in original:
                restored[field] = original[field]
            else:
                restored.pop(field, None)
        require(restored == original, "Completed episode changed")
        bindings[key] = {"source": provenance, "continued_sha256": sha(output)}
    pipeline.verify_receipts(root, spec)
    require(
        set(read(root / "receipts.json")) == set(receipts), "Sample added or dropped"
    )
    protocol = {
        "type": "mixed_per_turn_token_budget",
        "identity": spec["identity"],
        "source_run_id": old["run_id"],
        "preserved_samples": len(receipts),
        "preserved_max_new_tokens": 8192,
        "remaining_samples": len(tasks(spec)) - len(receipts),
        "new_max_new_tokens": 2048,
        "unit": "tokens per assistant turn",
        "completed_samples_rerun": 0,
        "note": "User-authorized continuation: preserved completed responses used 8192 tokens per turn; remaining responses use 2048. All samples remain in the denominator.",
    }
    atomic(root / "protocol.json", protocol)
    atomic(
        root / "continuation.json",
        {
            "identity": spec["identity"],
            "source_run": str(source),
            "source_identity": old["identity"],
            "source_boundary": boundary,
            "bindings": bindings,
            "preserved_samples": len(receipts),
            "remaining_samples": len(tasks(spec)) - len(receipts),
            "completed_samples_rerun": 0,
            "protocol": protocol,
        },
    )
    for name, expected in spec["source_hashes"].items():
        require(sha(PROJECT / name) == expected, "Frozen source changed")
        target = root / "frozen-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / name, target)
    with pipeline.progress(root, spec) as p:
        p.stage("prepare", 1, 1, "configurations", "Preserved completed Sales episodes")
        p.complete_stage("prepare")
        p.stage(
            "generate",
            len(receipts),
            len(tasks(spec)),
            "traces",
            "Ready to resume missing episodes at 2048 tokens per turn",
        )
        p.finish(False)
    with volume.batch_upload() as upload:
        for name in (
            "config.resolved.json",
            "receipts.json",
            "protocol.json",
            "continuation.json",
            "progress.json",
        ):
            upload.put_file(root / name, f"/{root.name}/{name}")
        upload.put_directory(root / "responses", f"/{root.name}/responses")
    require(
        json.loads(b"".join(volume.read_file(f"/{root.name}/receipts.json")))
        == read(root / "receipts.json"),
        "Uploaded checkpoint differs",
    )
    atomic(
        source / "continuation-target.json",
        {
            "run_id": root.name,
            "identity": spec["identity"],
            "preserved_samples": len(receipts),
            "reason": "User requested 2048 tokens per turn; completed responses preserved",
        },
    )
    print(
        json.dumps(
            {
                "run_dir": str(root),
                "preserved": len(receipts),
                "remaining": len(tasks(spec)) - len(receipts),
            }
        ),
        flush=True,
    )


def child(specs_path, effective_batch_size, status_path):
    original_commit = pipeline.commit

    def commit_with_budget(root, stage, task_id, value, spec):
        if stage == "responses":
            value = {**value, "generation_max_new_tokens": spec["max_new_tokens"]}
        return original_commit(root, stage, task_id, value, spec)

    pipeline.commit = commit_with_budget
    return adaptive.child(specs_path, effective_batch_size, status_path)


def finish(config):
    adaptive.resolve_adaptive = resolve_capped
    adaptive.finish(config)
    spec = resolve_capped(config)
    root = run_root(spec)
    protocol = read(root / "protocol.json")
    note = (
        '<aside data-token-budget-protocol="mixed"><strong>Token-budget protocol:</strong> '
        + html.escape(protocol["note"])
        + "</aside>"
    )
    report = root / "report.html"
    content = report.read_text()
    if 'data-token-budget-protocol="mixed"' not in content:
        require("<h1>" in content, "Report heading missing")
        report.write_text(content.replace("<h1>", note + "<h1>", 1))
    result = pipeline.verify(root, spec)
    result["artifacts"]["protocol.json"] = sha(root / "protocol.json")
    atomic(root / "verification.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--import-source", type=Path)
    parser.add_argument("--finish", action="store_true")
    parser.add_argument("--child-specs", type=Path)
    parser.add_argument("--effective-batch-size", type=int, choices=(16, 32))
    parser.add_argument("--status-path", type=Path)
    args = parser.parse_args()
    if args.child_specs:
        sys.exit(child(args.child_specs, args.effective_batch_size, args.status_path))
    elif args.import_source:
        import_checkpoint(args.import_source, args.config)
    elif args.finish:
        finish(args.config)
    else:
        parser.error("Choose --import-source or --finish")
