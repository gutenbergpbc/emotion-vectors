from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import traceback

EMOTIONS = [
    "hostile",
    "frustrated",
    "excited",
    "appreciative",
    "sad",
    "afraid",
    "surprised",
    "anxious",
    "guilty",
    "ashamed",
    "desperate",
    "calm",
]
RECIPES = ["neutral", "other11_neutral", "other11"]
LABELS = EMOTIONS + ["neutral"]
STAGES = ["prepare", "extract", "fit", "evaluate", "report"]
INPUTS = [
    "training_cases.json",
    "eval_cases.json",
    "split_manifest.json",
    "protocol.json",
]
FROZEN = {
    "training_cases.json": "fca9cc2407965efa7a925ee392bb3ef588c8847c9cb320aac37403b8b014174e",
    "eval_cases.json": "a4175f24f31868a34281b233c9f35a740452ca0ba3f85f8aee9eb0ce9de08d04",
    "split_manifest.json": "5aaed7460b54dcf81088957d37d7adaaeb3b1117404a81656476fb1414188779",
    "protocol.json": "b90e2e476fe3ad6f136d0be0f803360659579918e453da714f6ea192e120d68b",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        temp.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def source_hashes():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def validate_inputs(root):
    root = Path(root)
    for name, expected in FROZEN.items():
        require(sha(root / name) == expected, f"Frozen input changed: {name}")
    train, ev, split = (read(root / n) for n in INPUTS[:3])
    require(
        (len(train), len(ev), len(split["rows"])) == (2700, 2300, 2258),
        "Unexpected case counts",
    )
    for cases in (train, ev):
        require(len({c["id"] for c in cases}) == len(cases), "Duplicate case ID")
        for c in cases:
            t = c["target"]
            body = c["messages"][t["message_index"]][t["stream"]]
            if c["render_mode"] == "sales_with_reasoning":
                body = body.strip()
            require(
                body[t["char_start"] : t["char_end"]] == t["text"],
                f"Target span mismatch: {c['id']}",
            )
    folds = {}
    for c in train:
        require(c["fold"] in range(5), "Invalid fold")
        folds.setdefault(c["group_id"], set()).add(c["fold"])
    require(all(len(f) == 1 for f in folds.values()), "Training group crosses folds")
    byid = {c["id"]: c for c in ev}
    used = set()
    groups = {}
    replays = {}
    require(len({r["id"] for r in split["rows"]}) == 2258, "Duplicate retained ID")
    for row in split["rows"]:
        cid = row["id"]
        part = row["split"]
        require(part in ("calibration", "heldout"), "Invalid split")
        groups.setdefault(row["group_id"], set()).add(part)
        require(row["label"] == byid[cid]["label"], "Split label mismatch")
        for alias in split["aliases"][cid]:
            require(alias not in used and alias in byid, "Duplicate or unknown alias")
            used.add(alias)
            require(byid[alias]["label"] == row["label"], "Alias label mismatch")
            identity = digest(
                {k: byid[alias][k] for k in ("messages", "target", "render_mode")}
            )
            replays.setdefault(identity, set()).add(part)
    require(used == set(byid), "Alias coverage does not match evaluation cases")
    require(all(len(s) == 1 for s in groups.values()), "Evaluation group crosses split")
    require(all(len(s) == 1 for s in replays.values()), "Exact replay crosses split")
    for part in ("calibration", "heldout"):
        rows = [r for r in split["rows"] if r["split"] == part]
        require(len(rows) == 1129, "Invalid split denominator")
        require(
            set(r["label"] for r in rows) == set(LABELS), "Missing evaluation label"
        )
        counts = {label: sum(r["label"] == label for r in rows) for label in LABELS}
        require(
            counts
            == {
                label: (554 if label == "neutral" else 25 if label == "hostile" else 50)
                for label in LABELS
            },
            "Frozen label denominators changed",
        )
    return train, ev, split


class Run:
    """Single-writer lock, atomic snapshots, append-only events and heartbeat."""

    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.mutex = threading.RLock()
        path = self.root / "status.json"
        self.status = (
            read(path)
            if path.exists()
            else {
                "schema_version": 1,
                "run_id": self.root.name,
                "state": "pending",
                "stage": "prepare",
                "started_at": now(),
                "updated_at": now(),
                "sequence": 0,
                "stages": {
                    s: {"state": "pending", "completed": 0, "total": None}
                    for s in STAGES
                },
                "links": {
                    "events": "events.jsonl",
                    "log": "run.log",
                    "spec": "run_spec.json",
                    "verification": "verification.json",
                    "report": "report.html",
                },
            }
        )

    @contextlib.contextmanager
    def locked(self):
        with (self.root / ".run.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(
                    "Another stage is running in this directory"
                ) from None
            if (self.root / "status.json").exists():
                self.status = read(self.root / "status.json")
            yield

    def event(self, event, **fields):
        with self.mutex:
            self.status["sequence"] += 1
            self.status["updated_at"] = now()
            payload = {
                "time": now(),
                "sequence": self.status["sequence"],
                "stage": self.status["stage"],
                "event": event,
                **fields,
            }
            with (self.root / "events.jsonl").open("a") as f:
                f.write(json.dumps(payload, allow_nan=False) + "\n")
            with (self.root / "run.log").open("a") as f:
                f.write(
                    f"{payload['time']} {payload['stage']} {event} {json.dumps(fields)}\n"
                )
            atomic(self.root / "status.json", self.status)
            print(json.dumps(payload), flush=True)

    def progress(self, completed=None, total=None, **fields):
        with self.mutex:
            record = self.status["stages"][self.status["stage"]]
            record.update(fields)
            if completed is not None:
                record["completed"] = completed
            if total is not None:
                record["total"] = total
            self.event("progress", **record)

    @contextlib.contextmanager
    def stage(self, stage):
        start = time.monotonic()
        stop = threading.Event()
        self.status.update(stage=stage, state="running", error=None)
        # A rerun invalidates downstream completion until those stages revalidate.
        for s in STAGES[STAGES.index(stage) + 1 :]:
            self.status["stages"][s] = {
                "state": "pending",
                "completed": 0,
                "total": None,
            }
        self.status["stages"][stage] = {
            "state": "running",
            "started_at": now(),
            "error": None,
            "completed": 0,
            "total": None,
        }
        self.event("stage_started", pid=os.getpid())

        def heartbeat():
            while not stop.wait(10):
                with self.mutex:
                    self.status["stages"][stage]["elapsed_seconds"] = round(
                        time.monotonic() - start, 2
                    )
                    self.event("heartbeat")

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            yield
        except BaseException as exc:
            with self.mutex:
                error = f"{type(exc).__name__}: {exc}"
                self.status.update(state="failed", error=error)
                self.status["stages"][stage].update(state="failed", error=error)
                with (self.root / "run.log").open("a") as f:
                    f.write(traceback.format_exc())
                self.event(
                    "stage_failed",
                    error=error,
                    resume=f"Rerun the {stage} command after resolving this error.",
                )
            raise
        else:
            with self.mutex:
                smoke = (
                    stage == "extract"
                    and self.status["stages"][stage].get("phase") == "smoke_passed"
                )
                self.status["stages"][stage].update(
                    state="pending" if smoke else "complete",
                    finished_at=now(),
                    elapsed_seconds=round(time.monotonic() - start, 2),
                )
                self.status["state"] = "complete" if stage == "report" else "pending"
                self.event("smoke_only_finished" if smoke else "stage_complete")
        finally:
            stop.set()
            thread.join()

    def spec(self):
        spec = read(self.root / "run_spec.json")
        require(
            digest(spec) == self.status.get("spec_sha256"),
            "Run specification changed; create a new run",
        )
        require(
            spec["source_hashes"] == source_hashes(),
            "Harness source changed since prepare; use a new run directory",
        )
        return spec

    def need(self, stage):
        require(
            self.status["stages"][stage]["state"] == "complete",
            f"Complete {stage} first",
        )
