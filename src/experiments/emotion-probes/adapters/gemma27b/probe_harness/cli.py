import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from urllib.request import urlopen

from .common import (
    FROZEN,
    INPUTS,
    Run,
    atomic,
    digest,
    read,
    require,
    sha,
    source_hashes,
    validate_inputs,
)
from .rendering import RENDERER


def resolve_revision(model, revision):
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    from urllib.parse import quote

    url = (
        f"https://huggingface.co/api/models/{model}/revision/{quote(revision, safe='')}"
    )
    with urlopen(url, timeout=30) as response:
        resolved = json.load(response)["sha"]
    require(
        re.fullmatch(r"[0-9a-f]{40}", resolved),
        "Could not resolve immutable model revision",
    )
    return resolved


def prepare(run, args, synthetic=False):
    bundle = Path(args.bundle).resolve()
    manifest = read(bundle / "manifest.json")
    for name, expected in manifest["files"].items():
        require(
            sha(bundle / name) == expected, f"Bundle file checksum mismatch: {name}"
        )
    train, ev, split = validate_inputs(bundle / "inputs")
    revision = (
        "synthetic-fixture"
        if synthetic
        else resolve_revision(args.model, args.revision)
    )
    spec = {
        "schema_version": 1,
        "mode": "synthetic" if synthetic else "gemma",
        "model": args.model,
        "revision": revision,
        "tokenizer_revision": revision,
        "weight_format": "safetensors",
        "dtype": "bfloat16",
        "layers": 3 if synthetic else 62,
        "width": 16 if synthetic else 5376,
        "layer_indexing": "zero_based",
        "renderer": RENDERER,
        "readout": "full post-block residual; marked target-token mean; causal prefix",
        "attention": args.attention,
        "batch_size": 1,
        "tensor_parallel": 1,
        "max_tokens": args.max_tokens,
        "shard_size": args.shard_size,
        "input_hashes": FROZEN,
        "review_overlay_sha256": read(bundle / "inputs/protocol.json")[
            "review_overlay_sha256"
        ],
        "source_hashes": source_hashes(),
        "dependency_lock_sha256": sha(Path(__file__).resolve().parents[1] / "uv.lock"),
        "training_folds": 5,
        "bootstrap_draws": 2000,
        "bootstrap_seed": 20261009,
        "selection": "greatest mean weighted training-fold AUROC; shallowest within 1e-6",
        "calibration": "max equal-negative-label BA; lower mean negative FPR, then higher finite cutoff",
        "expected": {
            "training": 2700,
            "eval_raw": 2300,
            "eval_unique": 2258,
            "calibration": 1129,
            "heldout": 1129,
            "probes": 36,
        },
    }
    require(
        args.shard_size > 0 and args.max_tokens > 0,
        "Shard size and token limit must be positive",
    )
    require(
        args.model == "google/gemma-3-27b-it",
        "This adapter is validated for the Gemma 3 27B instruction checkpoint",
    )
    path = run.root / "run_spec.json"
    if path.exists():
        require(
            read(path) == spec, "Run identity changed; choose a new --run directory"
        )
    else:
        atomic(path, spec)
    inputs = run.root / "inputs"
    inputs.mkdir(exist_ok=True)
    for name in INPUTS:
        dest = inputs / name
        if dest.exists():
            require(sha(dest) == FROZEN[name], "Existing input copy changed")
        else:
            shutil.copyfile(bundle / "inputs" / name, dest)
    run.status.update(
        model=spec["model"],
        revision=revision,
        mode=spec["mode"],
        spec_sha256=digest(spec),
    )
    run.progress(
        25,
        25,
        phase="verified_inputs",
        expected=spec["expected"],
        feature_storage_bytes=5000 * spec["layers"] * spec["width"] * 4,
        message="Ready for CUDA extraction; tokenizer and longest-prefix checks run before model load",
    )


def synthetic_features(run):
    """Deterministic non-model fixtures for end-to-end harness/UI testing only."""
    import numpy as np
    from .common import EMOTIONS
    from .extraction import identity

    spec = run.spec()
    require(
        spec["mode"] == "synthetic", "Synthetic features only permitted in a demo run"
    )
    rng = np.random.default_rng(83)
    split = read(run.root / "inputs/split_manifest.json")
    alias_key = {a: k for k, aliases in split["aliases"].items() for a in aliases}
    generated = {}
    for part in ("training", "eval"):
        cases = read(run.root / "inputs" / f"{part}_cases.json")
        folder = run.root / f"{part}_features"
        folder.mkdir(exist_ok=True)
        entries = []
        for start in range(0, len(cases), 64):
            batch = cases[start : start + 64]
            acts = []
            for c in batch:
                key = alias_key.get(c["id"], c["id"])
                if key not in generated:
                    a = rng.normal(0, 1, (spec["layers"], spec["width"])).astype(
                        np.float32
                    )
                    if c["label"] in EMOTIONS:
                        a[:, EMOTIONS.index(c["label"])] += np.array([0.6, 1.4, 1.0])
                    generated[key] = a
                acts.append(generated[key])
            path = folder / f"shard_{start:05d}.npy"
            np.save(path, np.stack(acts), allow_pickle=False)
            entry = {
                "file": path.name,
                "sha256": sha(path),
                "identity": identity(spec),
                "ids": [c["id"] for c in batch],
                "case_hashes": [digest(c) for c in batch],
                "synthetic": True,
            }
            atomic(path.with_suffix(".json"), entry)
            entries.append(entry)
            run.progress(
                min(start + 64, len(cases)),
                len(cases),
                partition=part,
                phase="SYNTHETIC_FIXTURE",
            )
        atomic(
            folder / "manifest.json",
            {"identity": identity(spec), "rows": len(cases), "files": entries},
        )


def parser():
    p = argparse.ArgumentParser(description="Frozen twelve-emotion Gemma probe harness")
    p.add_argument(
        "--run",
        required=True,
        help="Versioned run directory (never reuse across identities)",
    )
    sub = p.add_subparsers(dest="command", required=True)
    for cmd in ("prepare", "demo"):
        s = sub.add_parser(cmd)
        s.add_argument("--bundle", required=True)
        s.add_argument("--model", default="google/gemma-3-27b-it")
        s.add_argument(
            "--revision",
            default="main",
            help="Resolved once to immutable commit at prepare",
        )
        s.add_argument(
            "--attention", default="sdpa", choices=["sdpa", "flash_attention_2"]
        )
        s.add_argument("--max-tokens", type=int, default=131072)
        s.add_argument("--shard-size", type=int, default=16)
    s = sub.add_parser("extract")
    s.add_argument("--smoke-only", action="store_true")
    for cmd in ("fit", "evaluate", "report", "verify"):
        sub.add_parser(cmd)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s = sub.add_parser("status")
    s.add_argument("--watch", action="store_true")
    s.add_argument("--interval", type=float, default=2)
    return p


def main():
    args = parser().parse_args()
    run = Run(args.run)
    if args.command == "serve":
        from .server import serve

        serve(run.root, args.host, args.port)
        return
    if args.command == "status":
        last = None
        try:
            while True:
                status = read(run.root / "status.json")
                if status != last:
                    print(json.dumps(status, indent=2), flush=True)
                    last = status
                if not args.watch:
                    return
                time.sleep(max(0.2, args.interval))
        except KeyboardInterrupt:
            return
    with run.locked():
        if args.command in ("prepare", "demo"):
            require(
                run.status["stages"]["prepare"]["state"] != "complete",
                "Run already prepared; resume a later stage or choose a new run directory",
            )
        if args.command == "demo":
            with run.stage("prepare"):
                prepare(run, args, synthetic=True)
            with run.stage("extract"):
                synthetic_features(run)
            from .fitting import fit
            from .evaluation import evaluate
            from .reporting import report

            for name, fn in [("fit", fit), ("evaluate", evaluate), ("report", report)]:
                with run.stage(name):
                    fn(run)
            return
        if args.command == "verify":
            from .reporting import verify

            verify(run, assets=True)
            print("Verification passed")
            return
        with run.stage(args.command):
            if args.command == "prepare":
                prepare(run, args)
            elif args.command == "extract":
                from .extraction import extract

                extract(run, smoke_only=args.smoke_only)
            elif args.command == "fit":
                from .fitting import fit

                fit(run)
            elif args.command == "evaluate":
                from .evaluation import evaluate

                evaluate(run)
            elif args.command == "report":
                from .reporting import report

                report(run)


if __name__ == "__main__":
    main()
