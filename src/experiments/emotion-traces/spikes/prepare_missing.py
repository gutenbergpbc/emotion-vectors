"""Freeze the two missing-cell replay batches from saved, reviewed labels."""

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

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = _REPO
CANON = ROOT / 'src/experiments'
sys.path[:0] = [str(CANON), str(CANON / 'emotion-probes'), str(CANON / 'emotion-rh-forecasting')]
from shared.io import atomic, digest, read, sha


def finish(folder, bundle, jobs, labels):
    atomic(folder / 'probes.json', bundle)
    entries = []
    for job in jobs:
        path = folder / 'inputs' / f"{job['id']}.json"
        if path.exists():
            assert read(path) == job
        else:
            atomic(path, job)
        entries.append({k: job[k] for k in ['id', 'trace_id', 'tokens'] } | {'sha256': sha(path)})
    spec = dict(schema_version=1, model=bundle['model'], probes_sha256=sha(folder / 'probes.json'),
                label_sources=[{'path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for p in labels],
                jobs=entries, total_tokens=sum(j['tokens'] for j in jobs))
    spec['identity'] = digest(spec)
    if (folder / 'replay-spec.json').exists():
        assert read(folder / 'replay-spec.json') == spec
    else:
        atomic(folder / 'replay-spec.json', spec)
    print(folder.name, len(jobs), 'jobs', spec['total_tokens'], 'tokens', flush=True)


def gemma():
    import prepare_gemma4b_coding_line_replay as old
    folder = HERE / 'replays/gemma4b-coding-controls-v1'
    labels = ROOT / 'output/control-expansion-20261027/labels/gemma4b/coding/labels.json'
    bundle = read(ROOT / 'runs/emotion-rh-replay/gemma3-4b-coding-lines-best12-v1/probes.json')
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-4b-it', revision='093f9f388b31de276ce2de164bdc2081324b9767')
    jobs = []
    for t in read(labels)['traces']:
        if not t['nonhack'] or not t['certain']:
            continue
        trace = {**t, 'id': 'expansion/' + t['id'], 'dataset': 'coding'}
        job = old.make_job(trace, 0, tokenizer)
        job['subtask'] = t['id'].split('/')[1]
        job['event'] = None
        job['control_label'] = 'nonhack_certain'
        jobs.append(job)
    assert len(jobs) == 5
    finish(folder, bundle, jobs, [labels])


def glm():
    folder = HERE / 'replays/glm53-terminalwrench-reviewed-neutral20-v1'
    labels = ROOT / 'runs/emotion-rh-sentence-labels/terminalwrench-neutral20-20261026-astra-review-v1/labels.json'
    bundle = read(ROOT / 'runs/emotion-rh-replay/glm-5.3-coding-1000-best12-v1/probes.json')
    tokenizer = AutoTokenizer.from_pretrained('zai-org/GLM-5.3', revision='ba32f8c19c8738e2c57c71798c601d477a7bf844')
    jobs = []
    for t in read(labels)['traces']:
        if t['model'] != 'GLM 5.3' or t['task']['condition'] != 'baseline':
            continue
        source = ROOT / t['source_path']
        assert sha(source) == t['source_sha256']
        steps = json.loads(read(source)['trajectory_text'])['steps']
        messages, bodies = [], []
        for step in steps:
            if step['source'] == 'user':
                messages.append({'role': 'user', 'content': step['message']})
            elif step['source'] == 'agent':
                sid = str(step['step_id'])
                body = step.get('message', '')
                if step.get('tool_calls'):
                    body += '\n\n# Submitted commands\n```json\n' + json.dumps(step['tool_calls'], ensure_ascii=False, indent=2) + '\n```'
                left, right = f'\ue000SPIKE_{sid}_START\ue001', f'\ue000SPIKE_{sid}_END\ue001'
                assert left not in body and right not in body
                messages.append({'role': 'assistant', 'content': left + body + right, 'reasoning_content': ''})
                bodies.append((sid, body, left, right))
                observation = '\n\n'.join(r.get('content', '') for r in (step.get('observation') or {}).get('results', []))
                if observation:
                    messages.append({'role': 'user', 'content': observation})
        while messages[-1]['role'] != 'assistant':
            messages.pop()
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, clear_thinking=False, reasoning_effort='low')
        positions = {}
        for sid, body, left, right in bodies:
            a = text.index(left)
            text = text.replace(left, '', 1)
            b = text.index(right)
            assert text[a:b] == body
            text = text.replace(right, '', 1)
            positions[sid] = (a, b)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded['offset_mapping']
        units = []
        for u in t['units']:
            start = positions[str(u['step_id'])][0]
            a, b = start + u['char_start'], start + u['char_end']
            assert text[a:b] == u['text']
            units.append({**u, 'rendered_char_start': a, 'rendered_char_end': b})
        hs = [u for u in units if u['hacking'] and not u['needs_review']]
        onset = min(u['rendered_char_start'] for u in hs) if hs else None
        event = sum(b <= onset for a, b in offsets) if onset is not None else None
        jobid = hashlib.sha256(t['id'].encode()).hexdigest()[:24]
        jobs.append(dict(id=jobid, trace_id=t['id'], subtask=t['task']['scenario'],
                         source_path=t['source_path'], source_sha256=t['source_sha256'],
                         token_ids=encoded['input_ids'], tokens=len(offsets), offsets=offsets,
                         rendered=text, rendered_sha256=hashlib.sha256(text.encode()).hexdigest(),
                         assistant_spans=list(positions.values()), units=units, event=event))
    assert len(jobs) == 20 and sum(j['event'] is not None for j in jobs) == 1
    finish(folder, bundle, jobs, [labels])


if __name__ == '__main__':
    gemma()
    glm()
