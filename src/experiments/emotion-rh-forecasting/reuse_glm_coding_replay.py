"""Reuse checksum-identical GLM probe scores for previously replayed coding traces."""
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


import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from shared.io import atomic, read, require, sha  # noqa: E402


TARGET = ROOT / 'runs/emotion-rh-replay/glm-5.3-coding-1000-best12-v1'
SOURCES = [ROOT / 'runs/emotion-rh-replay/glm-5.3-hack-traces-best12-v1',
           ROOT / 'runs/emotion-rh-replay/glm-5.3-nonhack-traces-best12-v1']


def main():
    target = read(TARGET / 'replay-spec.json')
    jobs = {entry['id']: entry for entry in target['jobs']}
    reused = []
    for source_root in SOURCES:
        source = read(source_root / 'replay-spec.json')
        require(source['model'] == target['model'] and source['probe_identity'] == target['probe_identity']
                and source['tokenizer_template_sha256'] == target['tokenizer_template_sha256'],
                'Replay checkpoint, vectors, or template differ')
        for entry in source['jobs']:
            if entry['id'] not in jobs or entry['trace_id'].split('/')[0] != 'coding':
                continue
            expected = jobs[entry['id']]
            require(entry['sha256'] == expected['sha256'], f"Input changed: {entry['id']}")
            source_score = source_root / 'scores' / f"{entry['id']}.npz"
            source_receipt = read(source_score.with_suffix('.receipt.json'))
            require(source_receipt['replay_identity'] == source['identity']
                    and source_receipt['job_sha256'] == entry['sha256']
                    and sha(source_score) == source_receipt['sha256'], 'Old score receipt invalid')
            score = TARGET / 'scores' / source_score.name
            receipt = score.with_suffix('.receipt.json')
            if score.exists():
                require(sha(score) == source_receipt['sha256'], 'Target score differs')
            else:
                score.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_score, score)
            row = {**source_receipt, 'replay_identity': target['identity'],
                   'reused_from': str(source_root.relative_to(ROOT)),
                   'source_replay_identity': source['identity'],
                   'source_score_sha256': source_receipt['sha256']}
            if receipt.exists():
                require(read(receipt) == row, 'Target receipt differs')
            else:
                atomic(receipt, row)
            reused.append(entry['id'])
    require(len(reused) == 23 and len(set(reused)) == 23, 'Expected exactly 23 old coding jobs')
    print(json.dumps({'reused_jobs': len(reused), 'new_jobs_to_score': len(jobs)-len(reused),
                      'target': str(TARGET)}, indent=2))


if __name__ == '__main__':
    main()
