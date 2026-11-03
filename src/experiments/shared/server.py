"""Read-only loopback dashboard. Do not expose raw datasets or model tokens."""

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

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = unquote(urlsplit(self.path).path).lstrip('/') or 'index.html'
        allowed = {'index.html', 'dashboard.js', 'dashboard.css', 'progress.json', 'events.jsonl',
                   'report.html', 'config.resolved.json', 'metrics.json', 'metrics.csv', 'verification.json',
                   'smoke.json', 'tokenization.json', 'connection.json', 'probes/manifest.json', 'probes/selected-12.json'}
        parts = Path(path).parts
        if '..' in parts or (path not in allowed and not (parts[0] == 'figures' and Path(path).suffix in ('.svg', '.png', '.pdf'))):
            self.send_error(404)
            return
        super().do_GET()

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()


def serve(root, port):
    root = Path(root).resolve()
    print(f'Dashboard: http://127.0.0.1:{port}/ (serving {root})', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), partial(Handler, directory=str(root))).serve_forever()
