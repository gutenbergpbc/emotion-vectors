"""Bind remote resumptions and downloads to the complete frozen run suite."""

import hashlib
import io
import json
import tarfile
from pathlib import PurePosixPath


def suite_members(specs):
    return [{"run_id": spec["run_id"], "identity": spec["identity"]} for spec in specs]


def saved_call_id(saved, specs):
    if saved.get("state") != "running":
        return None
    expected = suite_members(specs)
    if saved.get("identity") != expected[0]["identity"]:
        raise ValueError("Running remote call belongs to another frozen configuration")
    members = saved.get("suite_members")
    if (
        members is None
        and len(specs) == 1
        and saved.get("suite_run_ids", [specs[0]["run_id"]]) == [specs[0]["run_id"]]
    ):
        # Before suites existed, a single run was bound by its one identity.
        members = expected
    if members != expected or saved.get(
        "suite_run_ids", [item["run_id"] for item in expected]
    ) != [item["run_id"] for item in expected]:
        raise ValueError(
            "Running remote call has different suite members or identities"
        )
    return saved["function_call_id"]


def result_archives(result, specs):
    archives = result.get("archives")
    if archives is None:
        if len(specs) != 1 or "archive" not in result:
            raise ValueError("Remote response is missing suite archives")
        archives = {specs[0]["run_id"]: result["archive"]}
    missing = [spec["run_id"] for spec in specs if spec["run_id"] not in archives]
    if missing:
        raise ValueError(f"Remote response is missing suite archives: {missing}")
    for spec in specs:
        validate_archive(archives[spec["run_id"]], spec)
    return archives


def validate_archive(archive, spec):
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        members = {}
        for member in tar.getmembers():
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or member.name in members
            ):
                raise ValueError("Invalid or duplicate remote artifact path")
            members[member.name] = member
        if not members:
            # A later suite member may not have started when an earlier run failed.
            return
        if "receipts.json" not in members:
            raise ValueError("Remote archive is missing its receipts")
        receipts = json.load(tar.extractfile(members["receipts.json"]))
        for receipt in receipts.values():
            member = members.get(receipt["path"])
            if member is None:
                raise ValueError("Remote archive is missing a committed response")
            content = tar.extractfile(member).read()
            if hashlib.sha256(content).hexdigest() != receipt["sha256"]:
                raise ValueError("Remote committed artifact checksum differs")
            if json.loads(content).get("identity") != spec["identity"]:
                raise ValueError("Remote archive belongs to another frozen run")
