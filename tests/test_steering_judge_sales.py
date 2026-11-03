"""Sales adjudication must distinguish executed changes from plans and tool failures."""

import json
import sys
from copy import deepcopy
from pathlib import Path

CANONICAL = (Path(__file__).resolve().parents[1] / "src/experiments")
sys.path.insert(0, str(CANONICAL / "emotion-steering"))

from emotion_steering.judging import (
    FIXTURES,
    SALES_FIXTURES,
    audit,
    check_fixtures,
    evidence,
)


def test_sales_evidence_separates_unexecuted_completion_and_preserves_ledger():
    fixture = next(
        f for f in SALES_FIXTURES if f["id"] == "sales-executed-hack-before-truncation"
    )
    captured = []

    def send(messages, model):
        captured.append(json.loads(messages[1]["content"]))
        return {
            "label": "reward_hack",
            "quote": "record 45000 despite the approved 30024",
            "mechanism": "Executed sales inflation",
            "explanation": "The observed ledger increased beyond the approved amount.",
        }, {}

    record = {**deepcopy(fixture), "emotion": "guilty", "strength": -0.1}
    audit(
        record,
        {"instruction": fixture["task"]},
        {"environment": "sales", "judge_model": "fixture-model"},
        send,
    )
    payload = captured[0]
    transcript = json.loads(payload["transcript"])
    assert transcript["executed_turns"] == fixture["turns"]
    assert transcript["executed_turns"][0]["ledger"][-1]["total_amount"] == 45000
    assert transcript["unexecuted_failed_completion"] == fixture["failed_completion"]
    assert payload["finish_reason"] == payload["termination"] == "length"
    assert "guilty" not in json.dumps(payload)
    assert "strength" not in json.dumps(payload)


def test_sales_fixture_gate_exercises_both_environments_and_rejects_mislabel():
    cases = {evidence(f): f for f in FIXTURES + SALES_FIXTURES}
    observed = []

    def send(messages, model):
        payload = json.loads(messages[1]["content"])
        fixture = cases[payload["transcript"]]
        observed.append((fixture["id"], payload["environment"]))
        label = fixture["expected"]
        # Simulate the precise regression we need to catch: a correct sale credited
        # to another representative is mistaken for a quota hack.
        if fixture["id"] == "sales-confirmed-for-other-representative":
            label = "reward_hack"
        if fixture["id"] == "sales-honest-inability":
            label = "ordinary_failure"
        return {
            "label": label,
            "quote": "Sarah Kim"
            if fixture["id"] == "sales-confirmed-for-other-representative"
            else ("" if label != "reward_hack" else payload["transcript"][:30]),
            "mechanism": "Test label" if label == "reward_hack" else "none",
            "explanation": "Authored regression transport check.",
        }, {}

    result = check_fixtures(
        {"environment": "sales", "judge_model": "fixture-model"}, send
    )
    assert not result["passed"]
    assert len(result["results"]) == len(FIXTURES) + len(SALES_FIXTURES)
    failures = [r["id"] for r in result["results"] if not r["passed"]]
    assert failures == [
        "sales-confirmed-for-other-representative",
        "sales-honest-inability",
    ]
    assert all(
        env == ("sales" if name.startswith("sales-") else "coding")
        for name, env in observed
    )


def test_coding_fixture_gate_keeps_existing_coverage():
    seen = []
    cases = {evidence(f): f for f in FIXTURES}

    def send(messages, model):
        payload = json.loads(messages[1]["content"])
        fixture = cases[payload["transcript"]]
        seen.append(fixture["id"])
        label = fixture["expected"]
        return {
            "label": label,
            "quote": "" if label != "reward_hack" else payload["transcript"][:30],
            "mechanism": "Test label" if label == "reward_hack" else "none",
            "explanation": "Authored regression transport check.",
        }, {}

    result = check_fixtures(
        {"environment": "coding", "judge_model": "fixture-model"}, send
    )
    assert result["passed"]
    assert seen == [f["id"] for f in FIXTURES]
