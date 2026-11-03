"""Join verified new score shards to the original selection adapters."""

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

import json
import hashlib
from pathlib import Path

import numpy as np


def append_entries(here, ns, only=None):
    entries = ns['entries']
    for name, model, task in [
        ('gemma4b-coding-controls-v1', 'Gemma 4B', 'Coding'),
        ('glm53-terminalwrench-reviewed-neutral20-v1', 'GLM 5.3', 'TerminalWrench'),
    ]:
        if only is not None and name not in only:
            continue
        folder = here / 'replays' / name
        spec = json.loads((folder / 'replay-spec.json').read_text())
        for e in spec['jobs']:
            inp = folder / 'inputs' / f"{e['id']}.json"
            assert hashlib.sha256(inp.read_bytes()).hexdigest() == e['sha256']
            job = json.loads(inp.read_text())
            path = folder / 'scores' / f"{e['id']}.npz"
            receipt = json.loads(path.with_suffix('.receipt.json').read_text())
            assert receipt['identity'] == spec['identity'] and receipt['job_sha256'] == e['sha256']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['sha256']
            scores = np.load(path)['scores']
            assert scores.shape == (job['tokens'], 12) and np.isfinite(scores).all()
            if task == 'Coding':
                start = len(scores) - job['output_tokens']
                scores = scores[start:]
                mask = np.array([ns['keep_id'](model, int(i)) for i in job['token_ids'][start:]], dtype=bool)
                event = None
            else:
                mask = ns['offsetmask'](job['offsets'], job['rendered'])
                event = job['event']
            base = dict(model=model, task=task, source=name, id=job['trace_id'],
                        subtask=job['subtask'], scores=scores, mask=mask, event=event)
            if task == 'TerminalWrench' and event is not None:
                passages = [u for u in job['units'] if u['hacking'] and not u['needs_review']]
                for number, unit in enumerate(passages, 1):
                    selected_event = sum(b <= unit['rendered_char_start'] for a, b in job['offsets'])
                    entries.append({**base, 'event': selected_event, 'anchor_unit_id': unit['unit_id'],
                                    'anchor_text': unit['text'], 'first_rh_event': event,
                                    'rh_passage_number': number})
            else:
                entries.append(base)
    return entries
