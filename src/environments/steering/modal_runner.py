"""GPU generation runner. Launch with `modal run steering/modal_runner.py --limit 4`."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = 'qwen35-9b-emotions-pilot-20261015'
RESULT_VOLUME = modal.Volume.from_name('qwen35-emotion-steering-runs-20261015', create_if_missing=True)
MODEL_VOLUME = modal.Volume.from_name('qwen35-9b-hostility-model')
LEGACY = 'sandbox/ayush/emotion_vectors/local_reproduction_20261014/runtime/emotion_concepts'
IMAGE = (
    modal.Image.debian_slim(python_version='3.12')
    .uv_pip_install('transformers==5.17.0', 'flash-linear-attention==0.5.2', 'accelerate', 'sentencepiece', 'protobuf', 'numpy', 'requests', 'openai', 'pyyaml', 'tqdm')
    .env({'HF_HOME':'/models','HF_HUB_CACHE':'/models','HF_HUB_OFFLINE':'1'})
    .add_local_dir(str(ROOT/'steering'), remote_path='/root/model-forensics/steering', ignore=['runs/**','tests/**','__pycache__/**'])
    .add_local_dir(str(ROOT/'steering/runs'/RUN_ID/'vectors'), remote_path=f'/root/model-forensics/steering/runs/{RUN_ID}/vectors')
    .add_local_dir(str(ROOT/LEGACY), remote_path=f'/root/model-forensics/{LEGACY}')
)
app = modal.App('qwen35-emotion-steering-pilot')

@app.function(image=IMAGE, gpu='H100', cpu=8, memory=32768, timeout=86400,
              volumes={'/data':RESULT_VOLUME,'/models':MODEL_VOLUME})
def execute(spec, initial_snapshot, limit=None):
    import sys, time
    sys.path.insert(0,'/root/model-forensics')
    from steering.run import check_spec, generate, tasks
    from steering.tracking import Tracker, atomic, read
    from pathlib import Path
    root=Path('/data')/spec['run_id']
    root.mkdir(parents=True,exist_ok=True)
    if not (root/'run_spec.json').exists():
        atomic(root/'run_spec.json',spec)
        atomic(root/'progress.json',initial_snapshot)
        atomic(root/'receipts.json',{})
        (root/'events.jsonl').touch()
    else:
        previous=read(root/'run_spec.json')
        if previous != spec:
            def compatible_extension(value):
                value={**value,'source_hashes':{},'supplemental_conditions':[]}
                value['generation']={k:v for k,v in value['generation'].items() if k!='batch_size'}
                return value
            if compatible_extension(previous)!=compatible_extension(spec):
                raise ValueError('Scientific run configuration changed; start a new run ID')
            history=root/'run_config_history.jsonl'
            with history.open('a') as f:
                import hashlib,json
                f.write(json.dumps({'time':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
                    'previous_source_hashes':previous['source_hashes'],'new_source_hashes':spec['source_hashes'],
                    'previous_generation':previous['generation'],'new_generation':spec['generation'],
                    'reason':'Extended the completed pilot with selected reward-hacking probe strengths; preserved the shared dashboard, baseline, existing conditions, seeds, and committed responses.'})+'\n')
            atomic(root/'run_spec.json',spec)
            RESULT_VOLUME.commit()
    # Preserve one dashboard and baseline while increasing the tracked plan to
    # include the newly appended conditions.
    snapshot_path=root/'progress.json'
    snapshot=read(snapshot_path)
    total=len(tasks(spec))
    for stage in snapshot['stages']:
        if stage['id'] in ('generate','judge'):
            stage['progress']['total']=total
        if stage['id'] in ('judge','report') and spec.get('supplemental_conditions'):
            stage['state']='pending'
            stage['message']='Waiting for supplemental responses'
        if stage['id']=='report':
            stage['progress']['total']=sum(r['stage_id']=='report' for r in read(root/'receipts.json').values())+1
    snapshot['description']=f"{len(spec['probes'])} probes · original strengths {spec['strengths']} plus {len(spec.get('supplemental_conditions', []))} supplemental conditions · {total:,} total generations including the preserved shared baseline"
    atomic(snapshot_path,snapshot)
    if spec.get('supplemental_conditions'):
        import shutil
        shutil.copyfile('/root/model-forensics/steering/conditions.js',root/'conditions.js')
    volume=RESULT_VOLUME
    checkpoint={'last':0.0}
    def persist(force=False):
        current=time.monotonic()
        if force or current-checkpoint['last']>=15:
            volume.commit()
            checkpoint['last']=time.monotonic()
    with Tracker(root,on_checkpoint=persist,verify_existing=False) as tracker:
        check_spec(tracker)
        generate(tracker,limit)
    volume.commit()
    return {'state':read(root/'progress.json')['state'],
            'generation_receipts':sum(r['stage_id']=='generate' for r in read(root/'receipts.json').values()),
            'run_id':spec['run_id']}

@app.local_entrypoint()
def main(limit:int|None=None, config_path:str='steering/configs/qwen35_9b_emotions_pilot.json'):
    from steering.run import initialize
    from steering.tracking import read
    config=ROOT/config_path
    spec=read(config)
    output=ROOT/'steering/runs'/spec['run_id']
    if not (output/'run_spec.json').exists():
        initialize(config,output)
    snapshot=read(output/'progress.json')
    print(execute.remote(spec,snapshot,limit))
