"""Single-coordinator, artifact-backed dashboard producer (stdlib only)."""
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


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    try:
        temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


class Tracker:
    """Hold one writer lock for the entire coordinator lifetime.

    receipts.json is the authoritative committed-work manifest. progress.json is
    only a disposable view. Workers must send results to this coordinator, never
    write its files. Every resumed artifact is verified before it is counted.
    """
    def __init__(self, root, heartbeat_seconds=10, on_checkpoint=None, verify_existing=True):
        self.root = Path(root)
        self.interval = heartbeat_seconds
        self.on_checkpoint = on_checkpoint or (lambda force=False: None)
        self.verify_existing = verify_existing
        self.mutex = threading.RLock()
        self.stop = threading.Event()
        self.last_publish = 0

    def __enter__(self):
        self.lock = (self.root / '.writer.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError('Another coordinator owns this run') from None
        try:
            self.spec = read(self.root / 'run_spec.json')
            self.snapshot = read(self.root / 'progress.json')
            self.receipts = read(self.root / 'receipts.json')
            self.sequence = 0
            self.attempts = {}
            for line in (self.root / 'events.jsonl').read_text().splitlines():
                event = json.loads(line)
                self.sequence = max(self.sequence, event['sequence'])
                if 'task_id' in event:
                    self.attempts[event['task_id']] = max(self.attempts.get(event['task_id'], 0), event['attempt'])
            if self.verify_existing:
                for receipt in self.receipts.values():
                    if sha(self.root / receipt['path']) != receipt['sha256']:
                        raise ValueError(f"Committed artifact changed: {receipt['path']}")
            self.event('run.started', resumed=self.sequence > 0)
            self.thread = threading.Thread(target=self._heartbeat, daemon=True)
            self.thread.start()
            return self
        except BaseException:
            self.lock.close()
            raise

    def event(self, name, **fields):
        with self.mutex:
            self.sequence += 1
            event = dict(schema_version=1, run_id=self.spec['run_id'], worker_id='coordinator',
                         sequence=self.sequence, time=now(), event=name, **fields)
            with (self.root / 'events.jsonl').open('a') as f:
                f.write(json.dumps(event, allow_nan=False) + '\n')

    def publish(self, force=False):
        with self.mutex:
            if not force and time.monotonic() - self.last_publish < 1:
                return
            for st in self.snapshot['stages']:
                receipts = [r for r in self.receipts.values() if r['stage_id'] == st['id']]
                st['progress']['completed'] = len(receipts)
                if receipts:
                    st['progress']['last_progress_at'] = max(r['time'] for r in receipts)
            self.snapshot['updated_at'] = now()
            atomic(self.root / 'progress.json', self.snapshot)
            self.last_publish = time.monotonic()
            self.on_checkpoint(force=force)

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            try:
                self.publish()
            except Exception:
                # Surface publishing errors to the coordinator at the next boundary.
                self.heartbeat_error = traceback.format_exc()
                return

    def stage(self, stage_id, message):
        with self.mutex:
            transition = self.snapshot.get('active_stage') != stage_id
            self.snapshot['state'] = 'running'
            self.snapshot.pop('error', None)
            self.snapshot['active_stage'] = stage_id
            stage = next(s for s in self.snapshot['stages'] if s['id'] == stage_id)
            stage.update(state='running', message=message)
            # Batch status updates obey the normal checkpoint throttle.
            self.publish(force=transition)

    def start(self, stage_id, task_id, context=None):
        with self.mutex:
            if hasattr(self, 'heartbeat_error'):
                raise RuntimeError(self.heartbeat_error)
            key = f'{stage_id}/{task_id}'
            if key in self.receipts:
                return False
            attempt = self.attempts.get(key, 0) + 1
            self.attempts[key] = attempt
            self.event('task.retrying' if attempt > 1 else 'task.started', task_id=key,
                       stage_id=stage_id, attempt=attempt, context=context or {})
            self.current = (key, stage_id, attempt)
            self.current_tasks=getattr(self,'current_tasks',[])
            self.current_tasks.append((key,stage_id,attempt))
            st = next(s for s in self.snapshot['stages'] if s['id'] == stage_id)
            st['message'] = task_id
            self.publish()
            return True

    def commit(self, stage_id, task_id, path, context=None):
        self.commit_batch(stage_id, [(task_id, path, context)])

    def commit_batch(self, stage_id, artifacts):
        """Validate all artifacts, then atomically publish one receipt manifest.

        Uncommitted files are safe to regenerate after interruption. The manifest
        remains authoritative even if completion-event logging is interrupted.
        """
        with self.mutex:
            pending = {}
            for task_id, path, context in artifacts:
                key = f'{stage_id}/{task_id}'
                if key in self.receipts or key in pending:
                    continue
                if key not in self.attempts:
                    raise ValueError('Call start before committing a task')
                path = Path(path).resolve()
                relative = str(path.relative_to(self.root.resolve()))
                receipt = dict(path=relative, sha256=sha(path), stage_id=stage_id,
                               time=now(), context=context or {})
                read(path)
                pending[key] = receipt
            if not pending:
                return
            updated = {**self.receipts, **pending}
            atomic(self.root / 'receipts.json', updated)
            self.receipts = updated
            self.current = None
            self.current_tasks=[item for item in getattr(self,'current_tasks',[]) if item[0] not in pending]
            for key, receipt in pending.items():
                self.event('task.completed', task_id=key, stage_id=stage_id,
                           attempt=self.attempts[key], artifact=receipt, context=receipt['context'])
            self.publish()

    def finish_stage(self, stage_id):
        with self.mutex:
            self.publish(force=True)
            st = next(s for s in self.snapshot['stages'] if s['id'] == stage_id)
            if st['progress']['completed'] != st['progress']['total']:
                raise ValueError(f'{stage_id}: unfinished artifacts')
            st.update(state='complete', message='Artifacts saved and verified')
            self.snapshot.pop('active_stage', None)
            self.publish(force=True)

    def __exit__(self, kind, exc, tb):
        self.stop.set()
        self.thread.join()
        try:
            if exc:
                self.snapshot.update(state='cancelled' if kind is KeyboardInterrupt else 'failed', error=str(exc))
                for key, stage, attempt in getattr(self,'current_tasks',[]):
                    self.event('task.failed', task_id=key, stage_id=stage, attempt=attempt, message=str(exc))
                for st in self.snapshot['stages']:
                    if st['state'] == 'running':
                        st['state'] = self.snapshot['state']
                self.event('run.cancelled' if kind is KeyboardInterrupt else 'run.failed', message=str(exc), traceback=''.join(traceback.format_exception(kind, exc, tb)))
            elif all(s['state'] == 'complete' for s in self.snapshot['stages']):
                self.snapshot.update(state='complete', message='All stages verified')
                self.event('run.completed')
            else:
                self.snapshot.update(state='waiting', message='Coordinator stopped; resume the next stage with the CLI.')
                for st in self.snapshot['stages']:
                    if st['state'] == 'running':
                        st.update(state='waiting', message='Paused at the requested limit; rerun to resume.')
            self.publish(force=True)
        finally:
            self.lock.close()
