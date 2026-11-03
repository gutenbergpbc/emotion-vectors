import re
from copy import deepcopy
from pathlib import Path

import numpy as np

from shared.io import digest, read, require, sha

HERE = Path(__file__).resolve().parents[1]
CANONICAL = HERE.parent
PROJECT = CANONICAL.parents[1]
STAGES = ["prepare", "generate", "judge", "report", "verify"]
DEFAULTS = {
    "schema_version": 1,
    "experiment": "emotion-steering",
    "executor": "modal",
    "strengths": [-0.1, -0.05, 0.05, 0.1],
    "samples": 32,
    "batch_size": 1,
    "seed": 20261016,
    "max_new_tokens": 400,
    "temperature": 0.8,
    "top_p": 0.95,
    "max_turns": 8,
    "judge_model": "anthropic/claude-haiku-5.5",
    "judge_workers": 1,
    "compare_legacy_judge": True,
    "mode": "generate",
    "bootstrap_draws": 2000,
}


def safe_id(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value)


def load_vector(p, base):
    path = (base / p["source"]).resolve()
    require(sha(path) == p["source_sha256"], f"Probe source changed: {path}")
    key = p["probe"].split("::")[-1]
    if path.suffix == ".npy":
        v = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            v = z[key]
            if v.ndim == 2:
                v = v[p["layer_zero_based"]]
    elif path.suffix == ".json":
        v = np.asarray(read(path)["models"][key]["vector"])
    else:
        raise ValueError("Use JSON, NPY or NPZ probe exports")
    v = np.asarray(v, dtype=np.float32)
    require(
        v.ndim == 1 and np.isfinite(v).all() and np.linalg.norm(v) > 1e-8,
        "Invalid probe vector",
    )
    return {
        **p,
        "source": str(path),
        "vector": v.tolist(),
        "vector_sha256": digest(v.tolist()),
    }


def load_probes(cfg, base, model):
    if "probe_catalog" in cfg:
        path = (base / cfg["probe_catalog"]).resolve()
        catalog = read(path)
        require(
            (catalog["model"], catalog["revision"])
            == (model["name"], model["revision"]),
            "Probe/model checkpoint mismatch",
        )
        probes = [load_vector(p, PROJECT) for p in catalog["probes"]]
        provenance = {
            "path": str(path),
            "sha256": sha(path),
            "selection": catalog.get("selection"),
        }
    else:
        path = (base / cfg["probe_bundle"]).resolve()
        manifest = read(path / "manifest.json")
        require(
            not manifest.get("synthetic"), "Synthetic probes cannot steer real models"
        )
        require(
            (manifest["model"], manifest["revision"])
            == (model["name"], model["revision"]),
            "Probe/model checkpoint mismatch",
        )
        for name, checksum in manifest["files"].items():
            target = (path / name).resolve()
            require(
                target.is_relative_to(path) and sha(target) == checksum,
                "Bundle checksum/path mismatch",
            )
        models = manifest["models"]
        selected = read(path / "selected-12.json")["probes"]
        probes = []
        for emotion, key in selected.items():
            m = models[key]
            probes.append(
                load_vector(
                    {
                        "emotion": emotion,
                        "probe": key,
                        "layer_zero_based": m["layer"],
                        "source": str(path / "vectors.npz"),
                        "source_sha256": sha(path / "vectors.npz"),
                        "centering": manifest["centering"],
                    },
                    PROJECT,
                )
            )
        provenance = {
            "path": str(path),
            "sha256": sha(path / "manifest.json"),
            "selection": read(path / "selected-12.json")["method"],
        }
    requested = cfg.get("emotions", [p["emotion"] for p in probes])
    require(
        requested and len(requested) == len(set(requested)),
        "Emotions must be unique and nonempty",
    )
    by_emotion = {p["emotion"]: p for p in probes}
    require(
        len(by_emotion) == len(probes) and set(requested) <= set(by_emotion),
        "Missing/duplicate probes",
    )
    selected = [by_emotion[e] for e in requested]
    for p in selected:
        require(
            safe_id(p["emotion"])
            and len(p["vector"]) == model["width"]
            and type(p["layer_zero_based"]) is int
            and 0 <= p["layer_zero_based"] < model["layers"],
            "Probe dimension/layer mismatch",
        )
    return selected, provenance


def resolve(path):
    path = Path(path).resolve()
    supplied = read(path)
    allowed = set(DEFAULTS) | {
        "run_id",
        "model",
        "dataset",
        "probe_catalog",
        "probe_bundle",
        "emotions",
        "historical_run",
        "reuse_judgments_from",
        "terminalwrench",
    }
    require(
        not (set(supplied) - allowed),
        f"Unknown configuration fields: {set(supplied) - allowed}",
    )
    cfg = {**DEFAULTS, **supplied}
    require(
        cfg["experiment"] == "emotion-steering" and cfg["schema_version"] == 1,
        "Unsupported experiment",
    )
    require(safe_id(cfg["run_id"]), "Unsafe run ID")
    require(cfg["executor"] in ("local", "modal"), "Unknown executor")
    require(cfg["mode"] in ("generate", "audit-existing"), "Unknown run mode")
    require(
        ("probe_catalog" in cfg) != ("probe_bundle" in cfg),
        "Supply exactly one probe_catalog or probe_bundle",
    )
    for k in ("samples", "max_new_tokens", "max_turns", "bootstrap_draws"):
        require(type(cfg[k]) is int and cfg[k] > 0, f"{k} must be positive")
    require(
        type(cfg["judge_workers"]) is int and 1 <= cfg["judge_workers"] <= 16,
        "judge_workers must be between 1 and 16",
    )
    require(type(cfg["seed"]) is int and cfg["seed"] >= 0, "Invalid seed")
    require(
        0 <= cfg["temperature"] <= 2 and 0 < cfg["top_p"] <= 1,
        "Invalid sampling settings",
    )
    strengths = cfg["strengths"]
    require(
        strengths
        and len(strengths) == len(set(strengths))
        and all(
            type(s) in (int, float) and np.isfinite(s) and 0 < abs(s) <= 1
            for s in strengths
        ),
        "Strengths must be distinct finite nonzero residual fractions; baseline is added once",
    )
    models = read(HERE / "models.json")
    require(
        cfg["model"] in models,
        "Unknown model preset; add a pinned runtime to models.json",
    )
    model = deepcopy(models[cfg["model"]])
    require(
        type(cfg["batch_size"]) is int and 1 <= cfg["batch_size"] <= 32,
        "batch_size must be between 1 and 32",
    )
    require(
        cfg["batch_size"] == 1
        or model.get("backend") == "vllm_lens"
        or model["family"] == "glm",
        "Batched steering requires the vLLM-Lens backend",
    )
    probes, provenance = load_probes(cfg, path.parent, model)
    dataset_path = (path.parent / cfg["dataset"]).resolve()
    dataset = read(dataset_path)
    require(
        dataset["environment"] in ("coding", "sales", "terminalwrench"),
        "Unknown environment",
    )
    scenarios = dataset["scenarios"]
    require(
        scenarios and len({s["id"] for s in scenarios}) == len(scenarios),
        "Empty/duplicate scenarios",
    )
    for s in scenarios:
        require(safe_id(s["id"]), "Unsafe scenario ID")
        if dataset["environment"] != "terminalwrench":
            require(
                s.get("messages")
                and all(
                    m["role"] in ("system", "user", "assistant", "tool")
                    and isinstance(m["content"], str)
                    for m in s["messages"]
                ),
                "Invalid scenario messages",
            )
        if dataset["environment"] == "sales":
            require(
                "sales_config" in s, "Sales scenario needs exact environment settings"
            )
    if dataset["environment"] != "coding":
        model["answer_prefix"] = ""
    if dataset["environment"] == "terminalwrench":
        require(cfg["batch_size"] == 1 or cfg.get("terminalwrench", {}).get("environment") == "modal", "Concurrent TerminalWrench requires Modal")
        require(
            cfg["executor"] == "local",
            "TerminalWrench uses a local Harbor coordinator; its task sandboxes and GPU may be remote",
        )
        require(
            "terminalwrench" in cfg,
            "TerminalWrench requires api_base and pinned Harbor executable",
        )
        for s in scenarios:
            folder = (dataset_path.parent / s["task_path"]).resolve()
            s["task_path"] = str(folder)
            require(
                all(
                    (folder / n).is_file()
                    for n in (
                        "instruction.md",
                        "task.toml",
                        "environment/Dockerfile",
                        "tests/test.sh",
                    )
                ),
                f"Missing native Harbor task: {folder}",
            )
            s["task_hashes"] = {
                str(p.relative_to(folder)): sha(p)
                for p in sorted(folder.rglob("*"))
                if p.is_file()
            }
            s["instruction"] = (folder / "instruction.md").read_text()
    cfg["dataset"] = str(dataset_path)
    if cfg["mode"] == "audit-existing":
        cfg["historical_run"] = str((path.parent / cfg["historical_run"]).resolve())
    if cfg.get("reuse_judgments_from"):
        cfg["reuse_judgments_from"] = str(
            (path.parent / cfg["reuse_judgments_from"]).resolve()
        )
    source_paths = list((HERE / "emotion_steering").glob("*.py")) + [
        HERE / "models.json",
        HERE / "modal_worker.py",
    ]
    source_paths += [CANONICAL / "shared/io.py", CANONICAL / "shared/progress.py"]
    if dataset["environment"] == "sales":
        source_paths += [PROJECT / "src/environments/sales_env/sales_environment.py"]
    spec = {
        **cfg,
        "model_key": cfg["model"],
        "model": model,
        "probes": probes,
        "probe_provenance": provenance,
        "dataset_sha256": sha(dataset_path),
        "environment": dataset["environment"],
        "scenarios": scenarios,
        "legacy_rubric": dataset.get("legacy_rubric"),
        "intervention": "h + strength * norm(h) * unit(vector); all prompt/decode tokens; zero has no hook",
        "source_hashes": {str(p.relative_to(PROJECT)): sha(p) for p in source_paths},
    }
    spec["identity"] = digest(spec)
    return spec


def tasks(spec):
    result = []
    conditions = [(None, 0.0)] + [
        (p, s) for p in spec["probes"] for s in spec["strengths"]
    ]
    for p, strength in conditions:
        condition = f"{p['emotion']}/{strength:+g}" if p else "baseline"
        for si, scenario in enumerate(spec["scenarios"]):
            for sample in range(spec["samples"]):
                result.append(
                    {
                        "id": f"{condition}/{scenario['id']}/{sample:04d}",
                        "condition": condition,
                        "emotion": p["emotion"] if p else None,
                        "strength": strength,
                        "scenario": scenario["id"],
                        "sample": sample,
                        "seed": spec["seed"] + si * 10000 + sample,
                        "layer": p["layer_zero_based"] if p else None,
                    }
                )
    # Stable pair blocks; preserve the seed of every sample regardless of resume/batching.
    import random

    random.Random(spec["seed"]).shuffle(result)
    return result


def run_root(spec):
    return PROJECT / "runs/emotion-steering" / spec["run_id"]
