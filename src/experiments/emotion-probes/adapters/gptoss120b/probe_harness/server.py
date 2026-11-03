"""Read-only, loopback-default dashboard server; only allowlisted artifacts served."""

from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .common import read
from .reporting import ASSETS

STATIC = Path(__file__).parent / "static"
DOWNLOADS = set(
    ASSETS
    + [
        "status.json",
        "events.jsonl",
        "run.log",
        "verification.json",
        "runtime.json",
        "smoke.json",
    ]
)


def handler_for(root):
    root = Path(root).resolve()

    @lru_cache(maxsize=1)
    def cases():
        return {c["id"]: c for c in read(root / "inputs/eval_cases.json")}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send(self, body, kind="application/json", status=200):
            if not isinstance(body, bytes):
                body = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            path = url.path
            q = parse_qs(url.query)
            try:
                if path in ("/", "/status"):
                    self.send(
                        (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8"
                    )
                    return
                if path in ("/app.js", "/style.css"):
                    self.send(
                        (STATIC / path[1:]).read_bytes(),
                        "application/javascript"
                        if path.endswith(".js")
                        else "text/css",
                    )
                    return
                if path == "/api/status":
                    self.send(read(root / "status.json"))
                    return
                if path == "/api/spec":
                    self.send(read(root / "run_spec.json"))
                    return
                if path == "/api/events":
                    p = root / "events.jsonl"
                    if not p.exists():
                        self.send([])
                        return
                    # Read a bounded tail even after multi-day runs.
                    with p.open("rb") as f:
                        f.seek(0, 2)
                        size = f.tell()
                        f.seek(max(0, size - 150000))
                        data = f.read().decode(errors="replace")
                    lines = data.splitlines()
                    if size > 150000:
                        lines = lines[1:]
                    self.send(
                        [json.loads(line) for line in lines[-150:] if line.strip()]
                    )
                    return
                status = read(root / "status.json")
                if path == "/api/cv":
                    valid = status["stages"]["fit"]["state"] == "complete"
                    self.send(read(root / "cv_results.json") if valid else {})
                    return
                if path == "/api/results":
                    e = status["stages"]["evaluate"]
                    valid = e["state"] == "complete" or (
                        e["state"] == "running" and e.get("phase") == "heldout"
                    )
                    self.send(
                        read(root / "evaluation_results.json")
                        if valid and (root / "evaluation_results.json").exists()
                        else {"results": {}}
                    )
                    return
                if path == "/api/calibration":
                    e = status["stages"]["evaluate"]
                    valid = e["state"] == "complete" or (
                        e["state"] == "running" and e.get("phase") == "heldout"
                    )
                    self.send(
                        read(root / "calibration_results.json")
                        if valid and (root / "calibration_results.json").exists()
                        else {"results": {}}
                    )
                    return
                if path == "/api/verification":
                    valid = status["stages"]["report"]["state"] == "complete"
                    self.send(
                        read(root / "verification.json")
                        if valid
                        else {"status": "pending"}
                    )
                    return
                if path == "/api/examples":
                    import numpy as np

                    probe = q.get("probe", [""])[0]
                    ready = status["stages"]["evaluate"]["state"] == "complete" and all((root / n).exists() for n in ("calibrated_models.json", "all_eval_probe_scores.npz"))
                    if not ready:
                        split = read(root / "inputs/split_manifest.json")
                        rows = [r.copy() for r in split["rows"] if q.get("split", ["heldout"])[0] in ("all", r["split"]) and q.get("label", ["all"])[0] in ("all", r["label"])]
                        total = len(rows)
                        offset = max(0, int(q.get("offset", ["0"])[0]))
                        rows = rows[offset:offset + 20]
                        for row in rows:
                            row.update(target=cases()[row["id"]]["target"]["text"], score=None, threshold=None, outcome="pending", prediction=None)
                        self.send({"ready": False, "rows": rows, "total": total, "message": "Frozen text available; scores pending compute or transfer"})
                        return
                    models = read(root / "calibrated_models.json")["models"]
                    if probe not in models:
                        self.send({"error": "Unknown probe"}, status=400)
                        return
                    model = models[probe]
                    split = read(root / "inputs/split_manifest.json")
                    answer = []
                    with np.load(
                        root / "all_eval_probe_scores.npz", allow_pickle=False
                    ) as arrays:
                        scores = arrays[probe]
                    calibration = np.asarray([float(v) for r, v in zip(split["rows"], scores) if r["split"] == "calibration"])
                    scale = max(float(np.quantile(np.abs(calibration - model["threshold"]), .95)), 1e-12)
                    for row, score in zip(split["rows"], scores):
                        pred = bool(score >= model["threshold"])
                        positive = row["label"] == model["emotion"]
                        outcome = (
                            ("TP" if positive else "FP")
                            if pred
                            else ("FN" if positive else "TN")
                        )
                        if q.get("split", ["heldout"])[0] not in ("all", row["split"]):
                            continue
                        if q.get("outcome", ["all"])[0] not in ("all", outcome):
                            continue
                        if q.get("label", ["all"])[0] not in ("all", row["label"]):
                            continue
                        answer.append(
                            {
                                **row,
                                "score": float(score),
                                "color_distance": min(1.0, abs(float(score) - model["threshold"]) / scale),
                                "threshold": model["threshold"],
                                "prediction": pred,
                                "outcome": outcome,
                            }
                        )
                    offset = max(0, int(q.get("offset", ["0"])[0]))
                    total = len(answer)
                    answer = answer[offset : offset + 20]
                    for row in answer:
                        row["target"] = cases()[row["id"]]["target"]["text"]
                    self.send({"ready": True, "rows": answer, "total": total, "color_scale": scale})
                    return
                if path == "/api/example":
                    cid = q.get("id", [""])[0]
                    if cid not in cases():
                        self.send({"error": "Unknown ID"}, status=404)
                        return
                    c = cases()[cid]
                    self.send(
                        {
                            k: c[k]
                            for k in (
                                "id",
                                "label",
                                "messages",
                                "target",
                                "render_mode",
                            )
                        }
                    )
                    return
                if path.startswith("/files/") and path[7:] in DOWNLOADS:
                    file = root / path[7:]
                    types = {
                        ".html": "text/html",
                        ".svg": "image/svg+xml",
                        ".png": "image/png",
                        ".pdf": "application/pdf",
                        ".json": "application/json",
                        ".csv": "text/csv",
                    }
                    self.send(file.read_bytes(), types.get(file.suffix, "text/plain"))
                    return
                self.send({"error": "Not found"}, status=404)
            except FileNotFoundError:
                self.send({"error": "Artifact not available yet"}, status=404)
            except (ValueError, KeyError) as exc:
                self.send({"error": str(exc)}, status=400)

    return Handler


def serve(root, host="127.0.0.1", port=8765):
    server = ThreadingHTTPServer((host, port), handler_for(root))
    print(
        f"Dashboard: http://{host}:{server.server_port}/\nStatus viewer: http://{host}:{server.server_port}/status",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
