"""Import explicitly selected clean attempts with verifiable original provenance."""

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
import json
import shutil
from pathlib import Path
from shared.io import atomic, sha


def reuse(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    old=json.loads((source/'config.resolved.json').read_text())
    new=json.loads((target/'config.resolved.json').read_text())
    for key in ['temperature','top_p','max_new_tokens','max_turns','seed','samples']:
        if old[key]!=new[key]:raise ValueError('Incompatible '+key)
    if old['model']['name']!=new['model']['name'] or old['model']['revision']!=new['model']['revision']:
        raise ValueError('Checkpoint mismatch')
    vectors={p['emotion']:p['vector_sha256'] for p in old['probes']}
    if any(vectors[p['emotion']]!=p['vector_sha256'] for p in new['probes'] if p['emotion'] in vectors):raise ValueError('Vector mismatch')
    planned={t['id']:t for t in json.loads((target/'planned-episodes.json').read_text())}
    kept=[]
    for path in sorted((source/'episodes').glob('*.json')):
        episode=json.loads(path.read_text());task=episode['task']
        if task['id'] not in planned or episode.get('generation_error') or (target/'episodes'/path.name).exists():continue
        if task!=planned[task['id']]:raise ValueError('Task/seed mismatch')
        for artifact in episode.get('native_artifacts',[]):
            src=source/artifact['path'];dst=target/artifact['path']
            if sha(src)!=artifact['sha256']:raise ValueError('Artifact mismatch')
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        if episode.get('reuse_provenance'):episode.setdefault('reuse_history',[]).append(episode['reuse_provenance'])
        episode['reuse_provenance']=dict(source_run=old['run_id'],source_identity=old['identity'],source_episode=str(path),source_sha256=sha(path),original_backend=old['model'].get('backend','transformers'),selection='user-approved saved attempts without generation_error; not selected by task reward or RH label')
        atomic(target/'episodes'/path.name,episode);kept.append(task['id'])
    for path in (source/'calls').glob('*.json'):
        call=json.loads(path.read_text())
        if call['task']['id'] in kept:
            dst=target/'calls'/path.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dst)
    receipt_path=target/'reuse-receipt.json'
    prior=json.loads(receipt_path.read_text()) if receipt_path.exists() else {'sources':[]}
    prior.setdefault('sources',[]).append({'source_run':old['run_id'],'newly_reused':len(kept),'task_ids':kept})
    total=len(list((target/'episodes').glob('*.json')))
    prior.update(reused=total,new_needed=len(planned)-total,warning='Combined dataset contains historical and new serving backends; analyze provenance explicitly.')
    atomic(receipt_path,prior)
    return len(kept)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('target');a=p.parse_args();print(reuse(a.source,a.target))
