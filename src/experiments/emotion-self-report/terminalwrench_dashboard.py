"""Read-only live dashboard for the three neutral TerminalWrench rating runs."""
from __future__ import annotations

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


from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[3] / "runs/emotion-self-report"
RUNS = (
    ("GPT-OSS 120B", "gpt-oss-120b-terminalwrench-neutral-self-report-v2", 850),
    ("Gemma 3 4B", "gemma-3-4b-terminalwrench-neutral-self-report-v1", 787),
    ("GLM 5.3", "glm-5.3-terminalwrench-neutral-self-report-v1", 360),
)
HTML = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TerminalWrench self-report scoring</title><style>
body{font:16px system-ui,sans-serif;background:#101b1d;color:#edf3ee;max-width:1150px;margin:35px auto;padding:0 20px}
h1{font-size:30px;margin-bottom:6px}p{color:#afc4bf}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px;margin-top:28px}
article{background:#1b2b2d;border:1px solid #385153;border-radius:14px;padding:20px}h2{font-size:21px;margin:0 0 4px}
.state{font-size:13px;text-transform:uppercase;letter-spacing:.1em;color:#80dbc0}.big{font-size:33px;font-weight:700;margin:14px 0 2px}
progress{width:100%;height:16px;accent-color:#70d0aa}.details{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:18px;font-size:14px}
.details b{display:block;font-size:18px;color:#fff}.muted{color:#a2bbb3}.error{color:#ffad9e}a{color:#9ce1c6}footer{margin-top:20px;font-size:13px}
</style><h1>TerminalWrench · self-report scoring</h1><p>Independent 0–10 ratings for 12 emotions at 1,997 frozen sentence checkpoints. Updates every 10 seconds.</p><p>GPT-OSS v2 uses Crusoe after Groq returned an empty visible answer. Its v1 responses remain archived separately; v2 starts at zero under a new routing identity.</p>
<div id="summary"></div><main id="cards"></main><footer id="updated"></footer><script>
const fmt=n=>Number(n||0).toLocaleString();
function card(r){let pct=r.total?Math.round(100*r.done/r.total):0;return `<article><h2>${r.label}</h2><div class="state">${r.state}</div><div class="big">${fmt(r.done)} / ${fmt(r.total)}</div><div class="muted">${pct}% of rating requests · ${fmt(r.checkpoints)} sentence checkpoints</div><progress value="${r.done}" max="${r.total}"></progress><div class="details"><div>Numeric<b>${fmt(r.numeric)}</b></div><div>Null / other<b>${fmt(r.explicit_null)} / ${fmt(r.non_numeric_response)}</b></div><div>Failures<b>${fmt(r.failed)}</b></div><div>Provider cost<b>$${Number(r.cost).toFixed(3)}</b></div><div>Rate<b>${Number(r.rate||0).toFixed(2)}/s</b></div><div>ETA<b>${r.eta==null?'—':(r.eta/3600).toFixed(1)+' h'}</b></div></div>${r.error?`<p class="error">${r.error}</p>`:''}</article>`}
async function refresh(){try{let response=await fetch('status',{cache:'no-store'});if(!response.ok)throw new Error(`HTTP ${response.status} from ${response.url}`);let d=await response.json();document.getElementById('cards').innerHTML=d.runs.map(card).join('');let done=d.runs.reduce((a,r)=>a+r.done,0),total=d.runs.reduce((a,r)=>a+r.total,0),cost=d.runs.reduce((a,r)=>a+r.cost,0);document.getElementById('summary').textContent=`${fmt(done)} / ${fmt(total)} requests · $${cost.toFixed(3)} reported cost`;document.getElementById('updated').textContent=`Updated ${new Date().toLocaleString()}`;}catch(e){document.getElementById('updated').textContent='Dashboard could not read status: '+e}}
refresh();setInterval(refresh,10000);
</script></html>'''

cache: dict[str, tuple[int, dict]] = {}


def status():
    runs = []
    for label, run_id, checkpoints in RUNS:
        root = ROOT / run_id
        progress = json.loads((root / "progress.json").read_text()) if (root / "progress.json").exists() else {}
        collect = next((s for s in progress.get("stages", []) if s.get("id") == "collect"), {})
        tally = Counter()
        cost = 0.0
        for file in (root / "responses").glob("*.json"):
            stamp = file.stat().st_mtime_ns
            saved = cache.get(str(file))
            if saved is None or saved[0] != stamp:
                try:
                    response = json.loads(file.read_text())
                    attempts = response.get("attempts") or []
                    amount = sum(float(a.get("response", {}).get("usage", {}).get("cost") or 0)
                                 for a in attempts)
                    value = {"status": response.get("rating_status"), "ok": response.get("ok"), "cost": amount}
                except (OSError, ValueError, TypeError):
                    continue
                cache[str(file)] = (stamp, value)
            else:
                value = saved[1]
            tally[value["status"]] += 1
            cost += value["cost"]
        p = collect.get("progress") or {}
        runs.append({"label": label, "run_id": run_id, "checkpoints": checkpoints,
                     "total": checkpoints * 12, "done": sum(tally[s] for s in ("numeric", "explicit_null", "non_numeric_response")),
                     "numeric": tally["numeric"], "explicit_null": tally["explicit_null"],
                     "non_numeric_response": tally["non_numeric_response"], "failed": tally["transport_error"],
                     "cost": cost, "rate": p.get("rate_per_second"), "eta": p.get("eta_seconds"),
                     "state": progress.get("state", "not started"), "error": collect.get("error")})
    return {"updated_epoch": time.time(), "runs": runs}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/status":
            body = json.dumps(status()).encode()
            content_type = "application/json"
        elif path == "/":
            body = HTML.encode()
            content_type = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8797)
    args = parser.parse_args()
    print(f"http://127.0.0.1:{args.port}/", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
