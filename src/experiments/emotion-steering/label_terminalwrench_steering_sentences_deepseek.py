"""Label saved steering-episode assistant sentences with DeepSeek V4.1 Flash 1017.

Prepare: uv run --with pysbd python src/experiments/emotion-steering/label_terminalwrench_steering_sentences_deepseek.py --prepare-only
Pilot:   uv run --with pysbd python src/experiments/emotion-steering/label_terminalwrench_steering_sentences_deepseek.py --limit 3
Run:     uv run --with pysbd python src/experiments/emotion-steering/label_terminalwrench_steering_sentences_deepseek.py --workers 4
"""
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


import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import urllib.error
import urllib.request

CANONICAL = Path(__file__).resolve().parent.parent
ROOT = CANONICAL.parents[1]
sys.path[:0] = [str(CANONICAL), str(CANONICAL / 'emotion-self-report'), str(CANONICAL / 'emotion-rh-forecasting')]
import label_terminalwrench as judge  # noqa: E402
from emotion_self_report.segmentation import segment, validate_spans  # noqa: E402
from shared.io import atomic, digest, read, require, sha  # noqa: E402

MODEL = 'deepseek/deepseek-v4.1-flash-1017'
OUT = ROOT / 'runs/emotion-rh-sentence-labels/terminalwrench-steering-deepseek-v41-flash-1017-v1'
RUNS = {
    'GPT-OSS 120B': 'gptoss120b-terminalwrench-interactive-steering-active-time-v2',
    'GLM 5.3': 'glm53-terminalwrench-interactive-steering-active-time-v2-r2',
    'Gemma 4B': 'gemma4b-terminalwrench-interactive-steering-active-time-v2-r2',
}
UNITS_PER_REQUEST = 30


def segment_safe(message: str):
    """Use canonical segmentation; one malformed grid message gets exact lines."""
    try:
        return segment(message)
    except AssertionError:
        units, cursor = [], 0
        for raw in message.splitlines(keepends=True):
            text = raw.rstrip('\r\n')
            lo, hi = cursor, cursor + len(text)
            while lo < hi and message[lo].isspace():
                lo += 1
            while hi > lo and message[hi-1].isspace():
                hi -= 1
            if lo < hi:
                content = message[lo:hi]
                kind = 'code_block' if re.fullmatch(r'[.X#]+', content) else 'sentence_or_fragment'
                units.append({'unit_id': len(units), 'kind': kind,
                              'char_start': lo, 'char_end': hi, 'text': content})
            cursor += len(raw)
        validate_spans(message, units)
        return units


judge.segment = segment_safe


def source_rows():
    rows = []
    for model, run in RUNS.items():
        folder = ROOT / 'runs/emotion-steering' / run / 'episodes'
        paths = sorted(folder.glob('*.json'))
        require(paths, f'No episodes: {folder}')
        for path in paths:
            episode = read(path)
            task = episode['task']
            require(task['strength'] == 0 or task['emotion'], 'Unnamed steered emotion')
            rows.append((model, run, path, episode))
    return sorted(rows, key=lambda x: (x[3]['task']['strength'] != 0, x[0], x[3]['task']['id']))


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    records, requests, exclusions = [], [], []
    for model, run, path, episode in source_rows():
        task = episode['task']
        trace_id = f'{model}/{path.stem}'
        raw = episode.get('trajectory_text') or ''
        try:
            trajectory = json.loads(raw)
        except json.JSONDecodeError:
            exclusions.append({'id': trace_id, 'model': model, 'task': task,
                               'reason': 'missing_or_invalid_trajectory', 'source_path': str(path.relative_to(ROOT)),
                               'source_sha256': sha(path)})
            continue
        try:
            packet, units, excluded = judge.packet_and_units(trajectory)
        except ValueError as exc:
            require(str(exc) == 'No assistant prose in selected trace' or str(exc) == 'Empty trajectory',
                    f'Unexpected trajectory error: {trace_id}: {exc}')
            exclusions.append({'id': trace_id, 'model': model, 'task': task,
                               'reason': 'no_assistant_prose', 'source_path': str(path.relative_to(ROOT)),
                               'source_sha256': sha(path)})
            continue
        record = {'id': trace_id, 'model': model, 'run': run, 'task': task,
                  'termination': episode.get('termination'), 'finish_reason': episode.get('finish_reason'),
                  'source_path': str(path.relative_to(ROOT)), 'source_sha256': sha(path),
                  'trajectory_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                  'units': units, 'excluded_nonprose_units': len(excluded)}
        records.append(record)
        for start in range(0, len(units), UNITS_PER_REQUEST):
            chunk = units[start:start+UNITS_PER_REQUEST]
            body = {'model': MODEL, 'reasoning': {'effort': 'low', 'exclude': True},
                    'max_tokens': 4096 + 160 * len(chunk),
                    'response_format': {'type': 'json_schema', 'json_schema': {
                        'name': 'sentence_rh_labels', 'strict': True, 'schema': judge.SCHEMA}},
                    'messages': [{'role': 'system', 'content': judge.RUBRIC},
                                 {'role': 'user', 'content': json.dumps(
                                     {**packet, 'units_to_label': chunk}, ensure_ascii=False)}]}
            request_id = digest([trace_id, start, digest(body)])[:24]
            requests.append({'id': request_id, 'trace_id': trace_id,
                             'unit_ids': [u['unit_id'] for u in chunk],
                             'request_sha256': digest(body), 'body': body})
    require(len(records) + len(exclusions) == sum(1 for _ in source_rows()), 'Episode accounting differs')
    require(sum(r['task']['strength'] == 0 for r in records) == 12, 'All twelve baseline episodes need prose')
    frozen = {'schema_version': 1, 'model': MODEL, 'run_paths': RUNS,
              'rubric_sha256': hashlib.sha256(judge.RUBRIC.encode()).hexdigest(),
              'unit': 'Assistant message sentence or fragment; commands and observations are context only.',
              'units_per_request': UNITS_PER_REQUEST,
              'records': records, 'exclusions': exclusions,
              'requests': [{k: q[k] for k in ('id','trace_id','unit_ids','request_sha256')} for q in requests]}
    frozen['identity'] = digest(frozen)
    manifest_path = OUT / 'manifest.json'
    if manifest_path.exists():
        require(read(manifest_path) == frozen, 'Frozen source/segmentation changed; use new run ID')
    else:
        atomic(manifest_path, frozen)
    for req in requests:
        path = OUT / 'requests' / (req['id']+'.json')
        if path.exists():
            require(read(path) == req, 'Frozen judge request changed')
        else:
            atomic(path, req)
    (OUT / 'rubric.txt').write_text(judge.RUBRIC + '\n')
    byte_count = sum(len(json.dumps(q['body'], ensure_ascii=False).encode()) for q in requests)
    print(json.dumps({'episodes': len(records)+len(exclusions), 'labelable_episodes': len(records),
                      'excluded': dict(Counter(e['reason'] for e in exclusions)),
                      'prose_units': sum(len(r['units']) for r in records), 'requests': len(requests),
                      'request_input_mb': round(byte_count/1e6, 2)}, indent=2), flush=True)
    return frozen, requests


def annotate(request, config, stop):
    """Resumable API call; OpenRouter's in-flight 402 is transient."""
    destination = OUT / 'annotations' / (request['id']+'.json')
    if destination.exists():
        saved = read(destination)
        require(saved['request_sha256'] == request['request_sha256'], 'Cached request changed')
        raw = read(OUT / saved['raw_response_path'])
        require(judge.validate_response(raw['response'], request, config) == saved['labels'],
                'Cached labels changed')
        return saved
    errors = []
    for attempt in range(1, config['max_attempts']+1):
        path = OUT / 'attempts' / request['id'] / f'{attempt:02d}.json'
        try:
            if path.exists():
                saved = read(path)
                require(saved['request_sha256'] == request['request_sha256'], 'Attempt changed')
                if 'response' not in saved:
                    raise ValueError(saved.get('error', 'prior failed attempt'))
                raw = saved['response']
            else:
                require(not stop.is_set(), 'Dispatch stopped after non-retryable provider error')
                call = urllib.request.Request(judge.ENDPOINT,
                    data=json.dumps(request['body']).encode(),
                    headers={'Authorization': 'Bearer '+os.environ['OPENROUTER_API_KEY'],
                             'Content-Type': 'application/json'})
                try:
                    with urllib.request.urlopen(call, timeout=300) as response:
                        raw = json.load(response)
                    atomic(path, {'request_sha256': request['request_sha256'], 'response': raw})
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode(errors='replace')[:1800]
                    atomic(path, {'request_sha256': request['request_sha256'],
                                  'error': f'HTTP {exc.code}', 'detail': detail})
                    if exc.code in (400, 401, 403, 404):
                        stop.set()
                    if exc.code == 402 and 'in_flight_budget_exhausted' not in detail:
                        stop.set()
                    raise ValueError(f'HTTP {exc.code}: {detail[:160]}') from None
            choice = raw.get('choices', [{}])[0]
            if choice.get('finish_reason') == 'content_filter' or choice.get('message', {}).get('refusal'):
                stop.set()
                raise ValueError('Provider refused annotation')
            labels = judge.validate_response(raw, request, config)
            result = {'id': request['id'], 'trace_id': request['trace_id'],
                      'request_sha256': request['request_sha256'],
                      'model_returned': raw['model'], 'raw_response_path': str(path.relative_to(OUT)),
                      'usage': raw.get('usage', {}), 'labels': labels}
            atomic(destination, result)
            return result
        except (ValueError, KeyError, TypeError, OSError, TimeoutError) as exc:
            errors.append(f'{type(exc).__name__}: {exc}')
            if stop.is_set():
                break
            if attempt < config['max_attempts']:
                time.sleep(min(5*attempt, 30))
    atomic(OUT / 'errors' / (request['id']+'.json'), {'errors': errors})
    raise RuntimeError(f'{request["id"]}: {errors}')


def collect(manifest, requests):
    config = {'output': str(OUT), 'model': MODEL, 'allowed_returned_models': [MODEL], 'max_attempts': 6}
    by_trace, usage = {}, Counter()
    for req in requests:
        path = OUT / 'annotations' / (req['id']+'.json')
        if not path.exists():
            continue
        result = annotate(req, config, threading.Event())
        target = by_trace.setdefault(req['trace_id'], {})
        for item in result['labels']:
            require(item['unit_id'] not in target, 'Duplicate unit label')
            target[item['unit_id']] = item
    for path in (OUT / 'attempts').glob('*/*.json'):
        info = (read(path).get('response') or {}).get('usage') or {}
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cost'):
            usage[key] += info.get(key, 0) or 0
    labeled = []
    for record in manifest['records']:
        annotations = by_trace.get(record['id'], {})
        if len(annotations) != len(record['units']):
            continue
        joined = {**record, 'units': [{**u, **annotations[u['unit_id']]} for u in record['units']]}
        source = read(ROOT / record['source_path'])
        require(sha(ROOT / record['source_path']) == record['source_sha256'], 'Source episode changed')
        steps = {str(s['step_id']): s for s in json.loads(source['trajectory_text'])['steps']}
        for unit in joined['units']:
            message = steps[unit['step_id']]['message']
            require(steps[unit['step_id']]['source'] == 'agent' and
                    message[unit['char_start']:unit['char_end']] == unit['text'], 'Labeled span changed')
        labeled.append(joined)
    all_units = [u for r in labeled for u in r['units']]
    summary = {'status': 'complete' if len(labeled) == len(manifest['records']) else 'partial',
               'judge_model': MODEL, 'expected_episodes': len(manifest['records'])+len(manifest['exclusions']),
               'labelable_episodes': len(manifest['records']), 'labeled_episodes': len(labeled),
               'excluded': dict(Counter(e['reason'] for e in manifest['exclusions'])),
               'labeled_prose_units': len(all_units), 'rh_sentences': sum(u['hacking'] for u in all_units),
               'needs_review': sum(u['needs_review'] for u in all_units),
               'episodes_with_rh_prose': sum(any(u['hacking'] for u in r['units']) for r in labeled),
               'usage_all_saved_responses': dict(usage), 'manifest_identity': manifest['identity'],
               'scope': 'Post-hoc full-trajectory judge labels on assistant prose. Commands and observations are context; tool-only RH can be missed. No human semantic validation.'}
    atomic(OUT / 'labels.json', {'summary': summary, 'traces': labeled})
    atomic(OUT / 'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--limit', type=int)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    manifest, requests = prepare()
    if args.prepare_only:
        return
    require(os.environ.get('OPENROUTER_API_KEY'), 'OPENROUTER_API_KEY missing')
    jobs = requests[:args.limit] if args.limit is not None else requests
    config = {'output': str(OUT), 'model': MODEL, 'allowed_returned_models': [MODEL], 'max_attempts': 6}
    stop = threading.Event()
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(annotate, req, config, stop): req['id'] for req in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                failures.append({'request': futures[future], 'error': str(exc)[:500]})
            if i % 20 == 0 or i == len(jobs):
                print(f'{i}/{len(jobs)} requests; {len(failures)} failures', flush=True)
    atomic(OUT / 'failures.json', failures)
    summary = collect(manifest, requests)
    print(json.dumps(summary, indent=2), flush=True)
    if failures or (args.limit is None and summary['status'] != 'complete'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
