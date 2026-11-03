"""Validate trace collection and freeze equal-size finalized sales cohorts."""

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


from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "runs/emotion-rh-traces"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory):
    spec = json.loads((directory / "run-spec.json").read_text())
    files = sorted((directory / "traces").glob("*/*.json"))
    rows = [(path, json.loads(path.read_text())) for path in files]
    assert len(rows) == spec["target_count"], (directory.name, len(rows), spec["target_count"])
    assert len({row["id"] for _, row in rows}) == len(rows)
    assert all(row["model_requested"] == spec["model_requested"] for _, row in rows)
    assert all(row["dataset"] == spec["dataset"] for _, row in rows)
    by_scenario = defaultdict(Counter)
    providers = Counter()
    finish = Counter()
    cost = 0.0
    request_count = 0
    reasoning_traces = 0
    for path, row in rows:
        assert row["id"] == f"{row['dataset']}/{row['scenario']}/{row['sample']:04d}"
        completions = [row["completion"]] if spec["dataset"] == "coding" else [turn["completion"] for turn in row["turns"]]
        assert completions
        assert all(x["model_returned"] == spec["model_requested"] for x in completions), row["id"]
        by_scenario[row["scenario"]][row["status"]] += 1
        request_count += len(completions)
        reasoning_traces += bool(row.get("reasoning_text") or any(turn.get("reasoning_text") for turn in row.get("turns", [])))
        for completion in completions:
            providers[completion.get("provider") or "unknown"] += 1
            finish[completion.get("finish_reason") or "unknown"] += 1
            cost += (completion.get("usage") or {}).get("cost") or 0
    report = {"audited_at": datetime.now(timezone.utc).isoformat(),
              "model": spec["model_requested"], "dataset": spec["dataset"],
              "trace_count": len(rows), "by_scenario": {k: dict(v) for k, v in sorted(by_scenario.items())},
              "completion_requests_saved": request_count, "providers": dict(providers),
              "finish_reasons": dict(finish), "traces_with_reasoning": reasoning_traces,
              "saved_request_cost_usd": round(cost, 6),
              "cost_note": "Successful saved requests only; retries and failed attempts may add cost."}
    if spec["dataset"] == "sales":
        chosen = []
        for scenario in sorted(by_scenario):
            candidates = [(p, x) for p, x in rows if x["scenario"] == scenario and x["status"] == "complete"]
            candidates.sort(key=lambda pair: pair[1]["sample"])
            if len(candidates) < 50:
                report["cohort_error"] = f"{scenario}: only {len(candidates)} finalized episodes; need 50"
                continue
            chosen.extend({"id": x["id"], "path": str(p.relative_to(ROOT)),
                           "sha256": digest(p)} for p, x in candidates[:50])
        if len(chosen) == 150:
            cohort = {"model": spec["model_requested"], "dataset": "sales",
                      "selection": "first 50 finalized episodes by sample index per original tone; selected without labels or probe scores",
                      "source_run": directory.name, "traces": chosen}
            (directory / "cohort-finalized-150.json").write_text(json.dumps(cohort, indent=2) + "\n")
            report["finalized_cohort"] = "cohort-finalized-150.json"
    (directory / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(directory.name, json.dumps(report, sort_keys=True))
    return "cohort_error" not in report


if __name__ == "__main__":
    success = all(audit(path) for path in sorted(BASE.glob("*-openrouter-v1")))
    raise SystemExit(0 if success else 1)
