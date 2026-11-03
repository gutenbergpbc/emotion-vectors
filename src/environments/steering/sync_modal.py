"""Mirror Modal run metadata/artifacts into the local dashboard output."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from .run import ROOT
from .tracking import read

VOLUME='qwen35-emotion-steering-runs-20261015'

def download(remote, target, volume=VOLUME):
    target=Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        dest=Path(tmp)/target.name
        result=subprocess.run(['modal','volume','get',volume,remote,str(dest),'--force'],
                              stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
        if result.returncode:
            raise RuntimeError(result.stderr[-800:])
        staged=target.with_name('.'+target.name+'.incoming')
        shutil.copyfile(dest,staged)
        staged.replace(target)

def download_tree(run_id, target, volume=VOLUME):
    target=Path(target)
    target.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target) as tmp:
        result=subprocess.run(['modal','volume','get',volume,run_id,tmp,'--force'],
                              stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
        if result.returncode: raise RuntimeError(result.stderr[-800:])
        source=Path(tmp)/Path(run_id).name
        for path in source.rglob('*'):
            if not path.is_file() or path.name=='.writer.lock': continue
            relative=path.relative_to(source)
            destination=target/Path(run_id).name/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            staged=destination.with_name('.'+destination.name+'.incoming')
            shutil.copyfile(path,staged)
            staged.replace(destination)

def once(run_id, full=False, volume=VOLUME):
    out=ROOT/'steering/runs'/run_id
    for name in ('progress.json','receipts.json'):
        download(f'{run_id}/{name}',out/name,volume)
    try: download(f'{run_id}/events.jsonl',out/'events.jsonl',volume)
    except RuntimeError: pass
    progress=read(out/'progress.json')
    if full and progress['state'] in ('waiting','complete'):
        download_tree(run_id,out.parent,volume)
    return progress

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='qwen35-9b-emotions-pilot-20261015');p.add_argument('--volume',default=VOLUME,help='Modal volume containing the selected run');p.add_argument('--once',action='store_true');p.add_argument('--interval',type=int,default=10);p.add_argument('--wait-for-running',action='store_true',help='Treat a previous terminal snapshot as stale until this invocation starts.');args=p.parse_args()
    if args.interval<5:p.error('--interval must be at least 5 seconds')
    full_synced_at=None
    seen_running=False
    while True:
        try:
            s=once(args.run_id,volume=args.volume)
            print(f"{s['state']} · {s.get('updated_at')} · {s.get('message','')}",flush=True)
            if s['state']=='running': seen_running=True
            terminal=s['state'] in ('waiting','complete','failed','cancelled')
            if terminal and (not args.wait_for_running or seen_running):
                if s['state'] in ('waiting','complete') and s.get('updated_at') != full_synced_at:
                    download_tree(args.run_id,ROOT/'steering/runs',args.volume)
                    full_synced_at=s.get('updated_at')
                if s['state'] in ('complete','failed','cancelled') or args.once: break
        except Exception as e:print(f'Modal sync retry: {e}',flush=True)
        if args.once:break
        time.sleep(args.interval)

if __name__=='__main__':main()
