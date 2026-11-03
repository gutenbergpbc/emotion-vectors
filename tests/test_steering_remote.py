"""Remote suites cannot reconnect to, or download, a different run's outputs."""

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src/experiments") / "emotion-steering"))
from emotion_steering.remote import result_archives, saved_call_id, suite_members

SPECS = [
    {"run_id": "coding", "identity": "coding-v1"},
    {"run_id": "sales", "identity": "sales-v1"},
]


def connection():
    return {
        "state": "running",
        "identity": SPECS[0]["identity"],
        "function_call_id": "fc-example",
        "suite_run_ids": [s["run_id"] for s in SPECS],
        "suite_members": suite_members(SPECS),
    }


def archive(identity):
    response = json.dumps({"identity": identity, "response": "finished"}).encode()
    receipts = json.dumps(
        {
            "responses/sample": {
                "path": "responses/sample.json",
                "sha256": hashlib.sha256(response).hexdigest(),
            }
        }
    ).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in {
            "receipts.json": receipts,
            "responses/sample.json": response,
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_suite_reconnect_checks_every_member_identity_and_order():
    saved = connection()
    assert saved_call_id(saved, SPECS) == "fc-example"
    with pytest.raises(ValueError, match="suite members"):
        saved_call_id(saved, [SPECS[0], {**SPECS[1], "identity": "sales-v2"}])
    with pytest.raises(ValueError, match="suite members"):
        saved_call_id(saved, [SPECS[0]])
    assert saved_call_id({**saved, "state": "complete"}, SPECS) is None


def test_missing_suite_archive_never_falls_back_to_first_archive():
    first = archive("coding-v1")
    with pytest.raises(ValueError, match="missing suite archives"):
        result_archives({"archive": first, "archives": {"coding": first}}, SPECS)
    with pytest.raises(ValueError, match="missing suite archives"):
        result_archives({"archive": first}, SPECS)


def test_archive_receipts_must_belong_to_the_requested_run():
    first, second = archive("coding-v1"), archive("sales-v1")
    result = {"archives": {"coding": first, "sales": second}}
    assert result_archives(result, SPECS) == result["archives"]
    with pytest.raises(ValueError, match="another frozen run"):
        result_archives({"archives": {"coding": first, "sales": first}}, SPECS)
    assert result_archives({"archive": first}, SPECS[:1]) == {"coding": first}
