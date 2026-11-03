"""One emotion per fresh replay; durable attempts and strict 0–10 parsing."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

from shared.io import atomic, digest, read, require
from .config import QUESTIONS
from .inputs import checkpoint_key, job_key, prefix_messages

API = 'https://openrouter.ai/api/v1/chat/completions'


def payload(cfg, case, point):
    api = cfg['api']
    body = {k: api[k] for k in ('temperature', 'max_tokens')}
    for key in ('reasoning', 'provider'):
        if api[key]:
            body[key] = api[key]
    body.update(model=cfg['model'], messages=prefix_messages(case, point))
    return body


def request_body(cfg, case, point, emotion):
    body = payload(cfg, case, point)
    body['messages'].append({'role': 'user', 'content': QUESTIONS[cfg['prompt']].format(emotion=emotion)})
    return body


def parse_rating(content):
    try:
        value = json.loads(content)
    except (ValueError, TypeError):
        return None, 'non_numeric_response'
    if value is None:
        return None, 'explicit_null'
    if type(value) is int and 0 <= value <= 10:
        return value, 'numeric'
    return None, 'non_numeric_response'


def response_content(raw, model):
    require(isinstance(raw, dict) and raw.get('model') == model, 'API returned unexpected model')
    choices = raw.get('choices')
    require(isinstance(choices, list) and len(choices) == 1, 'Expected one API completion')
    require(choices[0].get('finish_reason') == 'stop', 'Truncated/unfinished API completion')
    content = choices[0].get('message', {}).get('content')
    require(isinstance(content, str) and content.strip(), 'Empty API completion')
    require(isinstance(raw.get('usage'), dict), 'Missing API usage')
    return content


class Client:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mutex = threading.Lock()
        self.next_start = 0.0

    def __call__(self, body):
        while True:
            with self.mutex:
                delay = self.next_start - time.monotonic()
                if delay <= 0:
                    self.next_start = time.monotonic() + self.cfg['api']['request_interval_seconds']
                    break
            time.sleep(min(delay, 1))
        token = os.environ.get('OPENROUTER_API_KEY')
        require(bool(token), 'Add OPENROUTER_API_KEY to the devbox Secrets settings')
        request = urllib.request.Request(API, data=json.dumps(body, ensure_ascii=False).encode(),
                                         headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=self.cfg['api']['timeout_seconds']) as response:
            return json.load(response)


def cached_record(path, identity, body, point, emotion):
    if not path.exists():
        return None
    record = read(path)
    require(record['identity'] == identity and record['request'] == body, 'Cached request binding changed')
    require(record['checkpoint'] == point and record['emotion'] == emotion and record['id'] == job_key(point, emotion),
            'Cached checkpoint/emotion changed')
    if record['ok']:
        raw = record['attempts'][-1]['response']
        value, status = parse_rating(response_content(raw, body['model']))
        require(record['value'] == value and record['rating_status'] == status, 'Cached rating does not reproduce')
        return record
    return None


def call(root, spec, case, point, emotion, send, sleep=time.sleep):
    body = request_body(spec, case, point, emotion)
    key = job_key(point, emotion)
    path = root / 'responses' / f'{key}.json'
    old = cached_record(path, spec['identity'], body, point, emotion)
    if old is not None:
        return old
    attempts = read(path)['attempts'] if path.exists() else []
    record = dict(id=key, identity=spec['identity'], checkpoint=point, emotion=emotion, request=body,
                  attempts=attempts, ok=False, value=None, rating_status='transport_error')
    for retry in range(spec['api']['max_attempts']):
        started = time.time()
        attempt = dict(attempt=len(attempts) + 1, started_epoch=started)
        delay = min(2 ** (retry + 1), 30)
        try:
            raw = send(body)
            attempt['response'] = raw
            content = response_content(raw, spec['model'])
            value, status = parse_rating(content)
            attempt['ok'] = True
            record.update(ok=True, value=value, rating_status=status, provider=raw.get('provider'),
                          served_model=raw['model'])
        except urllib.error.HTTPError as exc:
            attempt.update(ok=False, http_status=exc.code, error=exc.read().decode(errors='replace')[:3000])
            try:
                delay = max(delay, min(float(exc.headers.get('Retry-After', 0)), 30))
            except (TypeError, ValueError):
                pass
        except Exception as exc:
            attempt.update(ok=False, error=str(exc))
        attempt['elapsed_seconds'] = time.time() - started
        attempts.append(attempt)
        # Atomic response files preserve every returned attempt before retrying.
        atomic(path, record)
        if record['ok'] or attempt.get('http_status') in (400, 401, 402, 403, 404):
            break
        if retry + 1 < spec['api']['max_attempts']:
            sleep(delay)
    return record


def compact(record):
    point = record['checkpoint']
    return {k: point[k] for k in ('trace_id', 'unit_id', 'message_index', 'sentence_index', 'char_start',
                                  'char_end', 'prefix_sha256', 'task', 'label')} | {
        'emotion': record['emotion'], 'rating': record['value'], 'rating_status': record['rating_status'],
        'provider': record.get('provider'), 'served_model': record.get('served_model'),
    }


def collect(root, spec, cases, points, progress, send=None):
    jobs = [(point, emotion) for point in points for emotion in spec['emotions']]
    done, pending = [], []
    for point, emotion in jobs:
        body = request_body(spec, cases[point['trace_id']], point, emotion)
        record = cached_record(root / 'responses' / f'{job_key(point, emotion)}.json', spec['identity'], body, point, emotion)
        (done if record is not None else pending).append(record if record is not None else (point, emotion))
    progress.stage('collect', len(done), len(jobs), 'requests', 'Reusing verified responses; collecting missing requests')
    if pending and send is None:
        require(bool(os.environ.get('OPENROUTER_API_KEY')), 'Add OPENROUTER_API_KEY to devbox Secrets settings')
        require(not spec['synthetic'], 'Synthetic runs cannot send live API requests')
        send = Client(spec)
    start, initial = time.monotonic(), len(done)
    workers = spec['api']['workers']
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, len(pending), workers):
            futures = [pool.submit(call, root, spec, cases[p['trace_id']], p, e, send)
                       for p, e in pending[offset:offset + workers]]
            batch = [f.result() for f in futures]
            for record in batch:
                progress.event('task.completed' if record['ok'] else 'task.failed', stage_id='collect',
                               task_id=record['id'], attempt=len(record['attempts']),
                               context={'trace_id': record['checkpoint']['trace_id'], 'unit_id': record['checkpoint']['unit_id'],
                                        'emotion': record['emotion'], 'rating_status': record['rating_status']})
            done.extend(r for r in batch if r['ok'])
            progress.stage('collect', len(done), len(jobs), 'requests', f'{len(done)} / {len(jobs)} requests committed')
            with progress.mutex:
                counter = next(s for s in progress.data['stages'] if s['id'] == 'collect')['progress']
                rate = (len(done) - initial) / max(time.monotonic() - start, 1e-9)
                if rate > 0:
                    counter.update(rate_per_second=rate, eta_seconds=(len(jobs) - len(done)) / rate)
            progress.publish()
            if any(not r['ok'] for r in batch):
                raise RuntimeError('Request failed after retries. Responses retained; use --resume after resolving the error.')
    return [compact(r) for r in done]


def import_legacy(spec, cases, points, path, manifest, progress):
    """Validate saved request text and reparse completions; never trust CSV ratings."""
    point_map = {checkpoint_key(p): p for p in points}
    records = {}
    count = len(points) * len(spec['emotions'])
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            p, emotion = row['checkpoint'], row['emotion']
            if checkpoint_key(p) not in point_map or emotion not in spec['emotions'] or not row['ok']:
                continue
            point = point_map[checkpoint_key(p)]
            require(p['char_start'] == point['char_start'] and p['char_end'] == point['char_end'],
                    'Legacy checkpoint changed')
            expected_messages = prefix_messages(cases[point['trace_id']], point) + [
                {'role': 'user', 'content': manifest['question'].format(emotion=emotion)}]
            require(row['request'] == manifest['settings'] | {'messages': expected_messages},
                    'Legacy request/prefix/settings changed')
            successful = [a for a in row['attempts'] if a.get('ok')]
            require(successful, 'No successful legacy response')
            attempt = successful[-1]
            require(attempt['request']['messages'] == expected_messages and
                    attempt['request']['model'] == spec['model'], 'Legacy retry changed replay/model')
            value, status = parse_rating(response_content(attempt['response'], spec['model']))
            require(row['rating_status'] == status and row.get('value') == value, 'Legacy rating does not reproduce')
            normalized = compact(dict(checkpoint=point, emotion=emotion, value=value, rating_status=status,
                                      provider=attempt['response'].get('provider'), served_model=spec['model']))
            key = (*checkpoint_key(point), emotion)
            require(key not in records, 'Duplicate successful legacy request')
            records[key] = normalized
            if len(records) % 200 == 0:
                progress.stage('collect', len(records), count, 'requests', 'Validating existing self-reports')
    require(len(records) == count, f'Incomplete legacy collection: {len(records)} / {count}')
    progress.stage('collect', count, count, 'requests', 'Existing collection validated; no API requests')
    return list(records.values())
