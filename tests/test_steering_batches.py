import sys
from copy import deepcopy
from pathlib import Path

CANONICAL = (Path(__file__).resolve().parents[1] / "src/experiments")
sys.path[:0] = [str(CANONICAL), str(CANONICAL / "emotion-steering")]
from emotion_steering.config import resolve, tasks
from emotion_steering.environments import generate_batch, sales
from emotion_steering.pipeline import generate, initialize, progress, verify_receipts

from shared.io import read


class EpisodeBackend:
    def __init__(self):
        self.requests = []
        self.batch_sizes = []

    def complete(self, messages, probe, strength, seed, max_tokens):
        self.requests.append((seed, strength))
        base = seed % 1000003
        if base == 103:
            text = "```COMMAND\nADD_SALE Alex Chen, Fake Customer, TechFlow Pro, 999999\n```"
            finish = "length"
        elif base == 102 and seed < 1000003:
            text = "```COMMAND\nADD_SALE Alex Chen, MedTech Innovations, TechFlow Pro, 30024\n```"
            finish = "eos"
        else:
            text = "```COMMAND\nFINALIZE_AND_END_DAY\n```"
            finish = "eos"
        return {"response": text, "finish_reason": finish, "sampling_seed": seed}

    def complete_batch(self, requests):
        self.batch_sizes.append(len(requests))
        return [self.complete(**request) for request in requests]


def test_sales_batches_match_independent_episodes_and_never_execute_truncation():
    scenario = read(CANONICAL / "emotion-steering/data/sales-smoke.json")["scenarios"][
        0
    ]
    spec = {"environment": "sales", "max_turns": 3, "max_new_tokens": 4096}
    plan = [
        {"scenario": scenario["id"], "seed": seed, "emotion": None, "strength": 0}
        for seed in (101, 102, 103)
    ]
    original = deepcopy(scenario)
    backend = EpisodeBackend()
    batch = generate_batch(backend, {scenario["id"]: scenario}, plan, {}, spec)
    single = [sales(EpisodeBackend(), scenario, task, None, spec) for task in plan]
    assert batch == single
    assert scenario == original
    assert backend.batch_sizes == [3, 1]
    assert backend.requests == [(101, 0), (102, 0), (103, 0), (1000105, 0)]
    assert batch[0]["finalized"] and batch[1]["finalized"]
    assert batch[1]["objective"]["final"]["valid_sales"] == 1
    assert batch[2]["finish_reason"] == "length" and not batch[2]["turns"]
    assert not batch[2]["objective"]["final"]["violation"]
    assert all(
        row["customer"] != "Fake Customer" for row in batch[2]["final_sales_recorded"]
    )


def test_pipeline_batch_resume_preserves_population_and_seed_pairing(tmp_path):
    spec = resolve(CANONICAL / "../../configs/gpt-oss-120b-steering-smoke.json")
    spec["batch_size"] = 2
    initialize(tmp_path, spec, False)
    backend = EpisodeBackend()
    with progress(tmp_path, spec) as p:
        generate(tmp_path, spec, p, backend)
        generate(tmp_path, spec, p, backend)
    assert backend.batch_sizes == [2, 1]
    receipts = verify_receipts(tmp_path, spec)
    assert len(receipts) == 3
    for task in tasks(spec):
        row = read(tmp_path / receipts["responses/" + task["id"]]["path"])
        assert row["sampling_seed"] == task["seed"]
        assert row["strength"] == task["strength"]


def test_complete_grid_has_twelve_paired_samples_per_point():
    spec = resolve(CANONICAL / "../../configs/gemma-3-4b-coding-steering-12pp-v1.json")
    plan = tasks(spec)
    assert len(plan) == 588
    baseline = {
        (t["scenario"], t["sample"], t["seed"]) for t in plan if t["emotion"] is None
    }
    assert len(baseline) == 12
    for condition in {t["condition"] for t in plan}:
        assert {
            (t["scenario"], t["sample"], t["seed"])
            for t in plan
            if t["condition"] == condition
        } == baseline
