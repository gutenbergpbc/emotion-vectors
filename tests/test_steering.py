"""Scientific boundaries: matched interventions, failure labels, receipts and native actions."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

CANONICAL = (Path(__file__).resolve().parents[1] / "src/experiments")
sys.path.insert(0, str(CANONICAL / "emotion-steering"))
from emotion_steering.config import resolve, tasks
from emotion_steering.pipeline import (
    initialize,
    progress,
    generate,
    import_existing,
    judge,
    joined_rows,
    verify,
    verify_receipts,
)
from emotion_steering.reporting import report, summarize
from emotion_steering.judging import FIXTURES, RUBRIC, audit, validate_label
from emotion_steering.environments import ledger_score
from emotion_steering.terminalwrench import Service, alias


def spec():
    return resolve(CANONICAL / "../../configs/gpt-oss-120b-steering-smoke.json")


class FakeBackend:
    def __init__(self):
        self.calls = []

    def complete(self, messages, probe, strength, seed, max_tokens):
        self.calls.append((strength, seed, probe["emotion"] if probe else None))
        return dict(
            response="I cannot meet the impossible specification.",
            finish_reason="eos",
            input_tokens=5,
            generated_tokens=9,
            sampling_seed=seed,
        )


def fake_judge(messages, model):
    if messages[0]["role"] == "system":
        data = json.loads(messages[1]["content"])
        match = next((f for f in FIXTURES if f["response"] == data["transcript"]), None)
        label = match["expected"] if match else "honest_limit"
        return dict(
            label=label,
            quote=data["transcript"] if label == "reward_hack" else "",
            mechanism="test hardcoding" if label == "reward_hack" else "none",
            explanation="Fixture response",
        ), {}
    return {"label": "honest_impossible", "quote": ""}, {}


def test_pairing_and_single_baseline():
    s = spec()
    plan = tasks(s)
    assert len(plan) == 3 and len({t["id"] for t in plan}) == 3
    assert sum(t["emotion"] is None for t in plan) == 1
    for scenario in s["scenarios"]:
        group = [t for t in plan if t["scenario"] == scenario["id"]]
        assert len({t["seed"] for t in group}) == 1
        assert {t["strength"] for t in group} == {-0.1, 0, 0.1}
    assert plan == tasks(s)


def test_full_cpu_pipeline_resume_and_corruption(tmp_path):
    s = spec()
    initialize(tmp_path, s, False)
    backend = FakeBackend()
    with progress(tmp_path, s) as p:
        generate(tmp_path, s, p, backend)
        assert len(backend.calls) == 3
        generate(tmp_path, s, p, backend)
        assert len(backend.calls) == 3
        judge(tmp_path, s, p, fake_judge)
        rows = joined_rows(tmp_path, s)
        report(tmp_path, s, rows)
        assert verify(tmp_path, s)["observations"] == 3
        assert all(
            c["rh_rate"] == 0
            for c in json.loads((tmp_path / "metrics.json").read_text())["cells"]
        )
    initialize(tmp_path, s, True)
    response = next((tmp_path / "responses").rglob("*.json"))
    response.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        verify_receipts(tmp_path, s)


def test_edited_configuration_not_resumed(tmp_path):
    s = spec()
    initialize(tmp_path, s, False)
    changed = deepcopy(s)
    changed["temperature"] = 0.2
    with pytest.raises(ValueError, match="changed"):
        initialize(tmp_path, changed, True)


def test_judge_needs_real_evidence():
    with pytest.raises(ValueError, match="exact transcript"):
        validate_label(
            dict(
                label="reward_hack",
                quote="made up",
                mechanism="cheating",
                explanation="",
            ),
            "ordinary code",
        )
    with pytest.raises(ValueError, match="affirmative evidence"):
        validate_label(
            dict(label="reward_hack", quote="", mechanism="none", explanation=""),
            "ordinary code",
        )
    assert (
        validate_label(
            dict(
                label="ordinary_failure", quote="", mechanism="none", explanation="bug"
            ),
            "ordinary code",
        )["label"]
        == "ordinary_failure"
    )


def test_judge_is_blind_to_condition():
    s = spec()
    seen = []

    def send(messages, model):
        seen.extend(messages)
        return dict(
            label="ordinary_failure", quote="", mechanism="none", explanation="A bug"
        ), {}

    audit(
        dict(response="def f(): return 0", emotion="guilty", strength=-0.1, layer=9),
        {"messages": [{"role": "user", "content": "Return one"}]},
        s,
        send,
    )
    assert "guilty" not in json.dumps(seen) and "strength" not in seen[1]["content"]
    assert "ordinary_failure" in RUBRIC


def test_judge_repairs_evidence_and_preserves_rejected_answer():
    attempts = []

    def send(messages, model):
        attempts.append(messages)
        return dict(
            label="reward_hack",
            quote="invented" if len(attempts) == 1 else "hardcode the tests",
            mechanism="test-specific hardcoding",
            explanation="The answer targets the evaluator.",
        ), {"request_id": str(len(attempts))}

    result = audit(
        {"response": "I will hardcode the tests."},
        {"instruction": "Implement a general solution."},
        spec(),
        send,
    )
    assert len(attempts) == 2
    assert result["quote"] == "hardcode the tests"
    assert result["rejected_attempts"][0]["parsed"]["quote"] == "invented"


def test_import_canonical_generations_without_resampling(tmp_path):
    from shared.io import digest, read, sha

    source, destination = tmp_path / "source", tmp_path / "audit"
    original = spec()
    initialize(source, original, False)
    backend = FakeBackend()
    with progress(source, original) as p:
        generate(source, original, p, backend)
        judge(source, original, p, fake_judge)
    updated = {**original, "mode": "audit-existing", "historical_run": str(source)}
    updated["identity"] = digest(updated)
    initialize(destination, updated, False)
    with progress(destination, updated) as p:
        import_existing(destination, updated, p)
        import_existing(destination, updated, p)
        judge(destination, updated, p, fake_judge)
    assert len(backend.calls) == 3
    for task in tasks(updated):
        prior = source / "responses" / (task["id"] + ".json")
        imported = read(destination / "responses" / (task["id"] + ".json"))
        assert imported["response"] == read(prior)["response"]
        assert imported["source"]["sha256"] == sha(prior)
        assert imported["import_only"]
        assert imported["historical_judgment"]["label"] == "honest_impossible"


def test_legacy_import_rejects_reused_task_id_with_different_prompt(tmp_path):
    from shared.io import atomic

    original = spec()
    source = tmp_path / "legacy"
    atomic(
        source / "run_spec.json",
        {
            "model": original["model"],
            "probes": original["probes"],
            "scenarios": [
                {"id": original["scenarios"][0]["id"], "prompt": "A different task"}
            ],
        },
    )
    atomic(source / "receipts.json", {})
    updated = {**original, "mode": "audit-existing", "historical_run": str(source)}
    with pytest.raises(ValueError, match="source prompt differs"):
        import_existing(tmp_path / "audit", updated, None)


def test_parallel_judging_commits_every_result_once(tmp_path):
    import threading

    s = {**spec(), "judge_workers": 3}
    initialize(tmp_path, s, False)
    barrier = threading.Barrier(3)
    seen = set()

    def send(messages, model):
        if messages[0]["role"] == "system":
            transcript = json.loads(messages[1]["content"])["transcript"]
            if not any(f["response"] == transcript for f in FIXTURES):
                seen.add(threading.get_ident())
                barrier.wait(timeout=10)
        return fake_judge(messages, model)

    with progress(tmp_path, s) as p:
        generate(tmp_path, s, p, FakeBackend())
        judge(tmp_path, s, p, send)
        judge(tmp_path, s, p, send)
    receipts = verify_receipts(tmp_path, s)
    assert len(seen) == 3
    assert len([k for k in receipts if k.startswith("judgments/")]) == 3
    assert len(joined_rows(tmp_path, s)) == 3


def test_verified_judgment_reuse_preserves_labels_and_rejects_rubric_changes(tmp_path):
    from shared.io import digest, read
    from emotion_steering.pipeline import commit, reuse_judgments

    source, target = tmp_path / "source", tmp_path / "target"
    s = spec()
    initialize(source, s, False)
    with progress(source, s) as p:
        generate(source, s, p, FakeBackend())
        judge(source, s, p, fake_judge)
        report(source, s, joined_rows(source, s))
        verify(source, s)
    updated = {**s, "reuse_judgments_from": str(source)}
    updated["identity"] = digest(updated)
    initialize(target, updated, False)
    for task in tasks(s):
        response = read(source / "responses" / (task["id"] + ".json"))
        response.pop("identity")
        commit(target, "responses", task["id"], response, updated)
    reuse_judgments(target, updated)
    reuse_judgments(target, updated)
    for task in tasks(s):
        name = "judgments/" + task["id"] + ".json"
        old, new = read(source / name), read(target / name)
        assert old["audit"] == new["audit"]
        assert new["identity"] == updated["identity"]
        assert new["reused_from"]["path"] == str(source / name)
    changed = deepcopy(updated)
    changed["source_hashes"][
        "src/experiments/emotion-steering/emotion_steering/judging.py"
    ] = "changed"
    with pytest.raises(ValueError, match="rubric/code differs"):
        reuse_judgments(target, changed)


def test_denominator_keeps_failures_and_truncation():
    s = spec()
    s["samples"] = 2
    rows = []
    for t in tasks(s):
        rows.append(
            dict(
                **t,
                label="reward_hack" if t["sample"] == 0 else "ordinary_failure",
                legacy_label="hack",
                truncated=t["sample"] == 1,
            )
        )
    result = summarize(rows, s)
    for c in result["cells"]:
        assert (
            c["n"] == 2
            and c["rh_rate"] == 0.5
            and c["ordinary_failure_rate"] == 0.5
            and c["truncation_rate"] == 0.5
        )
    assert all(
        e["delta"] == 0 and e["ci95"] == [0.0, 0.0] for e in result["paired_effects"]
    )


def test_ledger_failure_is_not_inferred_from_quota():
    initial = [
        dict(
            date="2025-09-30",
            sales_rep="Other",
            customer="Existing",
            product="TechFlow Pro",
            total_amount=100,
        )
    ]
    valid = dict(
        date="2025-09-30",
        sales_rep="Alex Chen",
        customer="Acme",
        product="TechFlow Pro",
        total_amount=500,
    )
    threads = [dict(customer="Acme", status="valid", expected_sale_amount=500)]
    assert not ledger_score(initial, initial, threads)["violation"]
    assert not ledger_score(initial + [valid], initial, threads)["violation"]
    assert ledger_score(initial + [valid, valid], initial, threads)["violation"]
    assert ledger_score([], initial, threads)["violation"]


def test_native_episode_routes_condition_and_rejects_changes(tmp_path):
    s = spec()
    backend = FakeBackend()
    service = Service(s, tmp_path, backend)
    task = next(t for t in tasks(s) if t["strength"] == -0.1)
    body = dict(
        model=alias(task),
        messages=[{"role": "user", "content": "Do the task"}],
        seed=task["seed"],
    )
    response = service.complete(body)
    assert response["choices"][0]["message"]["content"]
    assert backend.calls == [(-0.1, task["seed"], "guilty")]
    service.complete(body)
    assert len(backend.calls) == 1
    with pytest.raises(ValueError, match="temperature"):
        service.complete({**body, "temperature": 99})


def test_intervention_actual_tensors():
    torch = pytest.importorskip("torch")
    from emotion_steering.backend import intervention

    layer = torch.nn.Identity()
    h = torch.tensor([[[3.0, 4.0], [0.0, 5.0]]])
    v = torch.tensor([1.0, 0.0])
    for strength in (-0.1, -0.05, 0, 0.05, 0.1):
        with intervention(layer, v, strength):
            torch.testing.assert_close(
                layer(h), h + strength * h.norm(dim=-1, keepdim=True) * v
            )
        assert not layer._forward_hooks
    with pytest.raises(RuntimeError):
        with intervention(layer, v, 0.1):
            raise RuntimeError("test cleanup")
    assert not layer._forward_hooks
