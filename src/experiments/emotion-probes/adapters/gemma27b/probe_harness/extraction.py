"""Correctness-first CUDA extraction, batch size one, resumable checked shards."""

import importlib.metadata
import os
from pathlib import Path
import time

import numpy as np

from .common import atomic, digest, read, require, sha, source_hashes, validate_inputs
from .rendering import RENDERER, tokenize_case


def identity(spec):
    return digest(
        {
            k: spec[k]
            for k in (
                "model",
                "revision",
                "tokenizer_revision",
                "dtype",
                "layers",
                "width",
                "renderer",
                "readout",
                "input_hashes",
                "source_hashes",
                "attention",
            )
        }
    )


def validate_shard(root, entry, spec, cases):
    path = root / entry["file"]
    require(sha(path) == entry["sha256"], f"Shard checksum mismatch: {path}")
    require(entry["identity"] == identity(spec), "Cache identity mismatch")
    a = np.load(path, mmap_mode="r", allow_pickle=False)
    require(
        a.shape == (len(entry["ids"]), spec["layers"], spec["width"]),
        "Wrong activation shape",
    )
    require(a.dtype == np.float32 and np.isfinite(a).all(), "Invalid activation values")
    require(
        entry["case_hashes"] == [digest(cases[cid]) for cid in entry["ids"]],
        "Shard case identities changed",
    )
    if spec["mode"] == "gemma":
        require(
            len(entry.get("evidence", [])) == len(entry["ids"]),
            "Missing per-case extraction evidence",
        )
        for cid, record in zip(entry["ids"], entry["evidence"]):
            require(
                record["id"] == cid and record["case_sha256"] == digest(cases[cid]),
                "Wrong per-case evidence",
            )
            lo, hi = record["target_token_span"]
            require(
                0 <= lo < hi == record["prefix_token_count"]
                and hi - lo == record["target_token_count"],
                "Wrong target-token span",
            )
            require(
                sha(root.parent / record["token_file"]) == record["tokens_sha256"],
                "Token file checksum mismatch",
            )
    return a


def feature_manifest(run, partition):
    spec = run.spec()
    cases = read(run.root / "inputs" / f"{partition}_cases.json")
    byid = {c["id"]: c for c in cases}
    folder = run.root / f"{partition}_features"
    manifest = read(folder / "manifest.json")
    require(
        manifest["identity"] == identity(spec), "Feature manifest identity mismatch"
    )
    seen = []
    for entry in manifest["files"]:
        require(set(entry["ids"]) <= set(byid), "Unexpected feature ID")
        validate_shard(folder, entry, spec, byid)
        seen.extend(entry["ids"])
    require(
        len(seen) == len(set(seen)) == len(cases) and set(seen) == set(byid),
        "Missing/duplicate features",
    )
    require(manifest["rows"] == len(cases), "Manifest count mismatch")
    return cases, manifest


def layer_features(run, partition, layer, cases, manifest):
    byid = {c["id"]: i for i, c in enumerate(cases)}
    x = np.empty((len(cases), run.spec()["width"]), dtype=np.float32)
    for entry in manifest["files"]:
        a = np.load(
            run.root / f"{partition}_features" / entry["file"],
            mmap_mode="r",
            allow_pickle=False,
        )
        x[[byid[cid] for cid in entry["ids"]]] = a[:, layer, :]
    return x


def prepare_tokens(run, tokenizer, spec, cases, partition):
    folder = run.root / "tokens" / partition
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.monotonic()
    for i, c in enumerate(cases):
        path = folder / f"{i:05d}.npy"
        meta = path.with_suffix(".json")
        if meta.exists() and path.exists():
            record = read(meta)
            require(
                record["identity"] == identity(spec)
                and record["case_sha256"] == digest(c),
                "Token cache identity mismatch",
            )
            require(
                sha(path) == record["tokens_sha256"], "Token cache checksum mismatch"
            )
        else:
            record = tokenize_case(c, tokenizer, spec["max_tokens"])
            ids = record.pop("input_ids")
            record.pop("positions")
            temp = path.with_suffix(".tmp")
            with temp.open("wb") as f:
                np.save(f, np.asarray(ids, dtype=np.int64), allow_pickle=False)
            temp.replace(path)
            record.update(
                identity=identity(spec),
                tokens_sha256=sha(path),
                token_file=str(path.relative_to(run.root)),
            )
            atomic(meta, record)
        records.append(record)
        if i % 25 == 0 or i == len(cases) - 1:
            run.progress(
                i + 1,
                len(cases),
                phase="tokenize",
                partition=partition,
                current_case=c["id"],
                elapsed_tokenization_seconds=round(time.monotonic() - started, 2),
            )
    atomic(folder / "manifest.json", {"identity": identity(spec), "records": records})
    return records


def extract(run, smoke_only=False, device="cuda"):
    run.need("prepare")
    spec = run.spec()
    require(spec["renderer"] == RENDERER, "Unknown rendering adapter")
    require(
        spec["source_hashes"] == source_hashes(),
        "Harness source changed after prepare; create a new run",
    )
    train, ev, _ = validate_inputs(run.root / "inputs")
    require(
        spec["mode"] == "gemma",
        "Use demo for synthetic fixtures; real extraction requires a Gemma run",
    )
    if not smoke_only and all(
        (run.root / f"{part}_features/manifest.json").exists()
        for part in ("training", "eval")
    ):
        complete = all(
            read(run.root / f"{part}_features/manifest.json")["rows"] == count
            for part, count in [("training", 2700), ("eval", 2300)]
        )
        if complete:
            for part in ("training", "eval"):
                feature_manifest(run, part)
            smoke = read(run.root / "smoke.json")
            require(
                smoke["status"] == "passed" and smoke["identity"] == identity(spec),
                "Missing matching GPU smoke evidence",
            )
            run.progress(5000, 5000, phase="verified_cache", verified_cache_reuse=5000)
            return True
    import torch
    from transformers import AutoConfig, AutoTokenizer, Gemma3ForConditionalGeneration

    require(
        torch.cuda.is_available() and device == "cuda",
        "CUDA GPU required. Use the Modal worker or a CUDA host; no GPU found here.",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        spec["model"], revision=spec["tokenizer_revision"], use_fast=True
    )
    config = AutoConfig.from_pretrained(spec["model"], revision=spec["revision"])
    require(
        config._commit_hash == spec["revision"],
        "Loaded configuration revision mismatch",
    )
    tc = config.text_config
    require(
        (tc.num_hidden_layers, tc.hidden_size) == (spec["layers"], spec["width"]),
        "Loaded model shape differs from run spec",
    )
    require(
        spec["max_tokens"] <= tc.max_position_embeddings,
        "Run token limit exceeds checkpoint limit",
    )
    prepared = {}
    for part, cases in [("training", train), ("eval", ev)]:
        prepared[part] = prepare_tokens(run, tokenizer, spec, cases, part)
    lengths = [r["prefix_token_count"] for rows in prepared.values() for r in rows]
    runtime = {
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "batch_size": 1,
        "attention": spec["attention"],
        "dtype": spec["dtype"],
        "accumulator": "float32",
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "identity": identity(spec),
        "tensor_parallel": 1,
        "hook": "decoder output[0], after both residual additions, before final model norm",
    }
    runtime_path = run.root / "runtime.json"
    if runtime_path.exists():
        require(read(runtime_path) == runtime, "Runtime changed; create a new run")
    atomic(runtime_path, runtime)
    run.progress(0, 5000, phase="loading_model", max_prefix_tokens=max(lengths))
    model = Gemma3ForConditionalGeneration.from_pretrained(
        spec["model"],
        revision=spec["revision"],
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation=spec["attention"],
    ).eval()
    language = model.model.language_model
    layers = language.layers
    require(len(layers) == spec["layers"], "Wrong decoder module path or layer count")

    def capture(record, full_ids=None):
        return capture_case(model, record, run.root, spec, device, full_ids)

    # Short matched pair + longest causal prefix + explicit reasoning example.
    pair = train[0]["pair_id"]
    samples = [("training", i) for i, c in enumerate(train) if c["pair_id"] == pair]
    samples += [
        ("eval", int(np.argmax([r["prefix_token_count"] for r in prepared["eval"]]))),
        ("eval", 0),
    ]
    checks = []
    for part, i in dict.fromkeys(samples):
        record = prepared[part][i]
        run.progress(0, 5000, phase="smoke", current_case=record["id"])
        started = time.monotonic()
        a = capture(record)
        b = capture(record)
        relative = float(
            np.max(
                np.linalg.norm(a - b, axis=1)
                / np.maximum(np.linalg.norm(a, axis=1), 1e-12)
            )
        )
        require(relative <= 1e-5, "Repeated batch-one activations disagree")
        checks.append(
            {
                "id": record["id"],
                "tokens": record["prefix_token_count"],
                "relative_l2": relative,
                "seconds_two_passes": time.monotonic() - started,
            }
        )
    # Verify suffix truncation against a complete short training rendering.
    from .rendering import render_case

    rendered, _, _ = render_case(train[0], tokenizer)
    full_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    short = capture(prepared["training"][0])
    full = capture(prepared["training"][0], full_ids)
    relative = float(
        np.max(
            np.linalg.norm(full - short, axis=1)
            / np.maximum(np.linalg.norm(full, axis=1), 1e-12)
        )
    )
    require(relative <= 0.02, "Suffix truncation changed pooled activations")
    atomic(
        run.root / "smoke.json",
        {
            "status": "passed",
            "identity": identity(spec),
            "checks": checks,
            "suffix_truncation_max_relative_l2": relative,
            "batch_size": 1,
            "runtime": runtime,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        },
    )
    if smoke_only:
        run.progress(
            0,
            5000,
            phase="smoke_passed",
            message="Smoke only; full extraction remains pending",
        )
        return False
    completed = reused = 0
    for part, cases in [("training", train), ("eval", ev)]:
        folder = run.root / f"{part}_features"
        folder.mkdir(exist_ok=True)
        byid = {c["id"]: c for c in cases}
        entries = []
        for start in range(0, len(cases), spec["shard_size"]):
            records = prepared[part][start : start + spec["shard_size"]]
            path = folder / f"shard_{start:05d}.npy"
            meta = path.with_suffix(".json")
            if meta.exists() and path.exists():
                entry = read(meta)
                require(
                    entry["ids"] == [r["id"] for r in records], "Shard ordering changed"
                )
                validate_shard(folder, entry, spec, byid)
                reused += len(records)
            else:
                arrays = []
                for record in records:
                    begin = time.monotonic()
                    run.progress(
                        completed + len(arrays),
                        5000,
                        phase="capture",
                        partition=part,
                        current_case=record["id"],
                        current_shard=path.name,
                        prefix_tokens=record["prefix_token_count"],
                        verified_cache_reuse=reused,
                    )
                    arrays.append(capture(record))
                    run.event(
                        "case_extracted",
                        case_id=record["id"],
                        seconds=round(time.monotonic() - begin, 3),
                        target_tokens=record["target_token_count"],
                        prefix_tokens=record["prefix_token_count"],
                    )
                temp = path.with_suffix(".tmp")
                with temp.open("wb") as f:
                    np.save(f, np.stack(arrays), allow_pickle=False)
                temp.replace(path)
                entry = {
                    "file": path.name,
                    "sha256": sha(path),
                    "identity": identity(spec),
                    "ids": [r["id"] for r in records],
                    "case_hashes": [r["case_sha256"] for r in records],
                    "evidence": records,
                }
                atomic(meta, entry)
            entries.append(entry)
            completed += len(records)
            atomic(
                folder / "manifest.json",
                {
                    "identity": identity(spec),
                    "rows": sum(len(e["ids"]) for e in entries),
                    "files": entries,
                },
            )
            run.progress(
                completed,
                5000,
                phase="extract",
                partition=part,
                current_shard=path.name,
                verified_cache_reuse=reused,
            )
        feature_manifest(run, part)
    return True


def capture_case(model, record, root, spec, device, full_ids=None):
    import torch

    layers = model.model.language_model.layers
    ids = (
        np.load(Path(root) / record["token_file"], allow_pickle=False)
        if full_ids is None
        else full_ids
    )
    tensor = torch.as_tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    a, b = record["target_token_span"]
    found = {}
    hooks = []

    def hook(index):
        def collect(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            require(
                hidden.shape == (1, len(ids), spec["width"]),
                "Unexpected decoder output shape",
            )
            pooled = hidden[0, a:b].float().mean(dim=0)
            require(b - a == record["target_token_count"], "Wrong target token count")
            found[index] = pooled.cpu().numpy()

        return collect

    try:
        for i, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(hook(i)))
        with torch.inference_mode():
            model(
                input_ids=tensor,
                use_cache=False,
                output_hidden_states=False,
                logits_to_keep=1,
            )
    finally:
        for handle in hooks:
            handle.remove()
    require(set(found) == set(range(spec["layers"])), "Some decoder hooks did not fire")
    x = np.stack([found[i] for i in range(spec["layers"])])
    require(np.isfinite(x).all(), "Nonfinite activations")
    return x
