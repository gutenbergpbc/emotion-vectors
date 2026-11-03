"""Single-writer progress.json producer for model-forensics/dashboards."""
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


import contextlib
import datetime as dt
import fcntl
import json
from pathlib import Path
import threading
import traceback

from .io import atomic, read

STAGES = ['prepare', 'tokenize', 'extract', 'fit', 'calibrate', 'evaluate', 'report', 'verify']


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


@contextlib.contextmanager
def lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.run.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another runner owns {root}') from None
        yield


class Progress:
    def __init__(self, root, title):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.mutex = threading.RLock()
        self.stop = threading.Event()
        self.data = read(self.root / 'progress.json') if (self.root / 'progress.json').exists() else {
            'schema_version': 1, 'run_id': self.root.name, 'title': title,
            'state': 'pending', 'updated_at': now(),
            'stages': [{'id': s, 'label': s.title(), 'state': 'pending'} for s in STAGES],
            'links': [{'label': label, 'href': path} for label, path in [
                ('Results', 'report.html'), ('Configuration', 'config.resolved.json'),
                ('Events', 'events.jsonl'), ('Verification', 'verification.json')]],
        }
        self.sequence = 0
        if (self.root / 'events.jsonl').exists():
            for line in (self.root / 'events.jsonl').read_text().splitlines():
                if line.strip():
                    self.sequence = max(self.sequence, json.loads(line)['sequence'])

    def publish(self):
        with self.mutex:
            self.data['updated_at'] = now()
            atomic(self.root / 'progress.json', self.data)

    def event(self, event, **fields):
        with self.mutex:
            self.sequence += 1
            record = dict(schema_version=1, run_id=self.root.name, worker_id='runner',
                          sequence=self.sequence, time=now(), event=event, **fields)
            with (self.root / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')

    def __enter__(self):
        self.event('run.started', resumed=(self.data['state'] != 'pending'))
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        return self

    def _heartbeat(self):
        while not self.stop.wait(10):
            self.publish()

    def __exit__(self, kind, exc, tb):
        if exc is not None:
            with self.mutex:
                self.data.update(state='failed', error=str(exc), message='See logs/run.log')
                for stage in self.data['stages']:
                    if stage['state'] == 'running':
                        stage.update(state='failed', error=str(exc))
            (self.root / 'logs').mkdir(exist_ok=True)
            with (self.root / 'logs/run.log').open('a') as stream:
                stream.write(''.join(traceback.format_exception(kind, exc, tb)))
            self.event('run.failed', error=str(exc))
        self.stop.set()
        self.thread.join()
        self.publish()

    def stage(self, name, completed=0, total=None, unit='items', message=''):
        with self.mutex:
            stage = next(s for s in self.data['stages'] if s['id'] == name)
            previous = stage.get('progress', {})
            moved = completed != previous.get('completed')
            stage.update(state='running', message=message, progress={
                'completed': completed, 'total': total, 'unit': unit,
                'last_progress_at': now() if moved else previous.get('last_progress_at', now()),
            })
            self.data.update(state='running', active_stage=name, message=message)
            self.data.pop('error', None)
            stage.pop('error', None)
            self.publish()

    def complete_stage(self, name):
        with self.mutex:
            stage = next(s for s in self.data['stages'] if s['id'] == name)
            stage['state'] = 'complete'
            self.event('task.completed', task_id=name, stage_id=name, attempt=1)
            self.publish()

    def finish(self, complete):
        with self.mutex:
            self.data.pop('active_stage', None)
            self.data.update(state='complete' if complete else 'waiting',
                             message='Verified results ready' if complete else 'Requested stages finished; ready to continue')
            self.event('run.completed' if complete else 'task.completed', task_id='requested-stages', attempt=1)
            self.publish()
