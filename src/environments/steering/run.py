"""Run from model-forensics: python3 -m steering.run --help."""
import argparse
import ast
import json
import os
from pathlib import Path
import random
import shutil
import time
import urllib.request

from .tracking import Tracker, atomic, now, read, sha

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'steering/configs/qwen35_9b_emotions_pilot.json'


def literals(path):
    result = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = value
    return result


def tasks(spec):
    """Stable per-sample IDs, shared prompt/seed schedule across conditions."""
    result = []
    conditions = [(None, 0, None)] + [(p, s, None) for p in spec['probes'] for s in spec['strengths']]
    # A completed run can be extended with narrowly scoped conditions while
    # keeping its original task IDs, receipts, dashboard, and shared baseline.
    probes_by_emotion = {p['emotion']: p for p in spec['probes']}
    for extra in spec.get('supplemental_conditions', []):
        emotion = extra['emotion']
        if emotion not in probes_by_emotion:
            raise ValueError(f"Supplemental condition has unknown probe: {emotion}")
        strength = float(extra['strength'])
        if strength in spec['strengths']:
            raise ValueError(f"Supplemental strength already exists: {emotion}/{strength:+g}")
        scenario_ids = set(extra.get('scenarios', [])) or None
        conditions.append((probes_by_emotion[emotion], strength, scenario_ids))
    for probe, strength, scenario_ids in conditions:
        condition = 'baseline' if probe is None else f"{probe['emotion']}/{strength:+g}"
        for scenario_index, scenario in enumerate(spec['scenarios']):
            if scenario_ids is not None and scenario['id'] not in scenario_ids:
                continue
            n = scenario['baseline_samples'] if probe is None else scenario['steered_samples']
            for i in range(n):
                result.append(dict(id=f"{condition}/{scenario['id']}/{i:04}", condition=condition,
                    emotion=probe['emotion'] if probe else None, strength=strength,
                    layer=probe['layer_zero_based'] if probe else None,
                    scenario=scenario['id'], kind=scenario['kind'], sample=i,
                    seed=spec['seed'] + scenario_index * 10000 + (i // spec['generation'].get('batch_size',8)) * spec['generation'].get('batch_size',8),
                    prompt=scenario['prompt'], max_new_tokens=scenario['max_new_tokens']))
    random.Random(spec['seed']).shuffle(result)
    return result


def initialize(config, out):
    spec = read(config)
    out = Path(out)
    if (out / 'run_spec.json').exists():
        if read(out / 'run_spec.json') != spec:
            raise ValueError('Run configuration changed; use a new run ID and directory')
        return out
    out.mkdir(parents=True, exist_ok=True)
    total = len(tasks(spec))
    for source, target in [('dashboard.html','index.html'),('dashboard.css','dashboard.css'),('dashboard.js','dashboard.js')]:
        shutil.copyfile(ROOT / 'dashboards' / source, out / target)
    for name in ('conditions.html', 'conditions.js'):
        shutil.copyfile(ROOT / 'steering' / name, out / name)
    atomic(out / 'run_spec.json', spec)
    atomic(out / 'receipts.json', {})
    (out / 'events.jsonl').touch()
    baseline_total = sum(task['emotion'] is None for task in tasks(spec))
    baseline_note = f'{baseline_total:,} shared baseline responses' if baseline_total else 'no new baseline responses'
    atomic(out / 'progress.json', dict(schema_version=1, run_id=spec['run_id'],
        title=spec['title'], description=f"{len(spec['probes'])} probes · strengths {spec['strengths']} ({spec['steering_mode']}) · {total:,} generations · {baseline_note}",
        state='pending', updated_at=now(), stale_after_seconds=60,
        message='Prepared; no model generations or judge requests have started.',
        stages=[dict(id=i,label=label,state='pending',progress=dict(completed=0,total=n,unit=unit))
                for i,label,n,unit in [('preflight','Validate inputs',1,'checks'),('generate','Generate responses',total,'responses'),('judge','Judge responses',total,'judgments'),('report','Verify and report',1,'reports')]],
        links=[dict(label=label,href=href) for label,href in [('Condition counts','conditions.html'),('Run specification','run_spec.json'),('Events','events.jsonl'),('Committed artifacts','receipts.json')]]))
    return out


def check_spec(run):
    spec = run.spec
    for p in spec['probes']:
        if sha(ROOT / p['source']) != p['source_sha256']:
            raise ValueError(f"Probe source changed: {p['emotion']}")
    for path, expected in spec['source_hashes'].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f'Source changed: {path}; make a new run configuration')
    ids = [t['id'] for t in tasks(spec)]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate task IDs')


def generate(run, limit=None):
    from .backend import load_vector, make_backend
    check_spec(run)
    run.stage('preflight', 'Checking vector hashes, model compatibility and backend')
    # Match the completed pilot's FLA installation and fail before loading the
    # large checkpoint if the Modal image is missing that optional fast path.
    kernel_status = {}
    if 'Qwen3.5' in run.spec['model']['name']:
        import importlib.util
        kernel_status = {'flash_linear_attention': importlib.util.find_spec('fla') is not None}
        if not kernel_status['flash_linear_attention']:
            raise RuntimeError('Qwen3.5 Flash Linear Attention package is missing from the Modal image')
    if run.spec.get('require_causal_conv1d'):
        import importlib.metadata
        import torch
        from causal_conv1d import causal_conv1d_fn
        if not torch.cuda.is_available():
            raise RuntimeError('causal-conv1d verification requires a CUDA device')
        # Exercise the compiled CUDA extension, not just Python package discovery.
        x = torch.randn((1, 32, 64), device='cuda', dtype=torch.bfloat16)
        weight = torch.randn((32, 4), device='cuda', dtype=torch.bfloat16)
        y = causal_conv1d_fn(x, weight, activation='silu')
        torch.cuda.synchronize()
        if y.shape != x.shape or not torch.isfinite(y).all():
            raise RuntimeError('causal-conv1d CUDA smoke test returned invalid output')
        kernel_status['causal_conv1d'] = {
            'version': importlib.metadata.version('causal-conv1d'),
            'cuda_smoke_test': 'passed',
            'torch': torch.__version__,
            'torch_cuda': torch.version.cuda,
        }
    backend = make_backend(run.spec, ROOT)
    vectors = {}
    for p in run.spec['probes']:
        vector = load_vector(p, ROOT)
        if len(vector) != backend.hidden_size or not 0 <= p['layer_zero_based'] < backend.num_layers:
            raise ValueError(f"Incompatible probe: {p['emotion']}")
        vectors[p['emotion']] = vector
    if run.start('preflight', 'inputs'):
        path = run.root / 'preflight.json'
        atomic(path, dict(model=run.spec['model'], hidden_size=backend.hidden_size, layers=backend.num_layers,
                          optimized_kernels=kernel_status, verified_at=now()))
        run.commit('preflight', 'inputs', path)
    run.finish_stage('preflight')
    run.stage('generate', 'Generating full responses')
    probes = {p['emotion']:p for p in run.spec['probes']}
    count = 0
    size=run.spec['generation'].get('batch_size',8)
    missing=[task for task in tasks(run.spec) if f"generate/{task['id']}" not in run.receipts]
    # Repeated fixed prompts share an identical token length. Batch work by
    # condition and task, preserving deterministic sample order and resumability.
    missing.sort(key=lambda t:(t['condition'],t['scenario'],t['sample']//size,t['sample']))
    from itertools import groupby
    for _, grouped in groupby(missing,key=lambda t:(t['condition'],t['scenario'],t['sample']//size)):
        batch_group=list(grouped)
        for offset in range(0,len(batch_group),size):
            batch=batch_group[offset:offset+size]
            started=[]
            for task in batch:
                context={k:task[k] for k in ('emotion','strength','scenario','sample','seed','layer')}
                if run.start('generate',task['id'],context): started.append(task)
            if not started: continue
            first=started[0]
            run.stage('generate',f"{first['condition']} · {first['scenario']} · batch {first['sample']}–{started[-1]['sample']}")
            started_at=time.perf_counter()
            results=backend.sample_batch(started,probes.get(first['emotion']),vectors.get(first['emotion']))
            elapsed=time.perf_counter()-started_at
            if len(results)!=len(started): raise RuntimeError('Backend returned the wrong batch size')
            batch_tokens=sum(result.get('generated_tokens',0) for result in results)
            for result in results:
                result['batch_elapsed_seconds']=elapsed
                result['batch_size']=len(results)
                result['batch_responses_per_second']=len(results)/elapsed if elapsed else None
                result['batch_tokens_per_second']=batch_tokens/elapsed if elapsed else None
            artifacts = []
            for task,result in zip(started,results):
                path=run.root/'responses'/(task['id']+'.json')
                atomic(path,dict(**task,**result,model=run.spec['model'],steering_mode=run.spec['steering_mode']))
                artifacts.append((task['id'],path,{'scenario':task['scenario'],'sample':task['sample']}))
            run.commit_batch('generate', artifacts)
            count += len(artifacts)
            if limit and count>=limit: return
    run.finish_stage('generate')


def judge_request(prompt, model, key):
    body = dict(model=model, temperature=0, max_tokens=500,
        messages=[dict(role='user',content=prompt)])
    req = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
        data=json.dumps(body).encode(), headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=120) as response:
        data = json.load(response)
    content = data['choices'][0]['message']['content']
    # Preserve the raw answer, and fail rather than silently label invalid output.
    cleaned = content.strip()
    if cleaned.startswith('```'):
        cleaned = '\n'.join(cleaned.splitlines()[1:-1])
    parsed = json.loads(cleaned)
    return parsed, data


def judge(run, model, limit=None):
    check_spec(run)
    key = os.environ.get('OPENROUTER_API_KEY')
    if not key:
        raise ValueError('OPENROUTER_API_KEY is required')
    config = dict(model=model, prompts=run.spec['judge_prompts'], temperature=0, max_tokens=500)
    path = run.root / 'judge_config.json'
    if path.exists() and read(path) != config:
        raise ValueError('Judge configuration changed; use a separate run directory to rejudge')
    atomic(path, config)
    run.stage('judge', 'Judging saved full responses, blinded to condition')
    count = 0
    for task in tasks(run.spec):
        source = run.receipts.get('generate/' + task['id'])
        if not source:
            continue
        if not run.start('judge', task['id'], {'scenario':task['scenario']}):
            continue
        record = read(run.root / source['path'])
        scenario = next(s for s in run.spec['scenarios'] if s['id'] == task['scenario'])
        prompt = run.spec['judge_prompts'][task['kind']].format(response=record['response'], **scenario['judge_fields'])
        parsed, raw = judge_request(prompt, model, key)
        if parsed.get('label') not in run.spec['judge_labels'][task['kind']]:
            raise ValueError(f"Invalid judge label: {parsed.get('label')}")
        dest = run.root / 'judgments' / (task['id'] + '.json')
        atomic(dest, dict(task_id=task['id'], response_sha256=source['sha256'], label=parsed['label'], quote=parsed.get('quote',''), raw_judge_response=raw, judge_model=model))
        run.commit('judge', task['id'], dest, {'scenario':task['scenario']})
        count += 1
        if limit and count >= limit:
            return
    if sum(r['stage_id']=='judge' for r in run.receipts.values()) == len(tasks(run.spec)):
        run.finish_stage('judge')


def report(run):
    check_spec(run)
    expected = tasks(run.spec)
    if any('judge/'+t['id'] not in run.receipts for t in expected):
        raise ValueError('All planned judgments are required before reporting')
    run.stage('report', 'Verifying response/judgment links')
    groups = {}
    for t in expected:
        response = run.receipts['generate/'+t['id']]
        judgment = read(run.root / run.receipts['judge/'+t['id']]['path'])
        if judgment['response_sha256'] != response['sha256']:
            raise ValueError('Judgment refers to a different response')
        key = (t['condition'], t['scenario'])
        g = groups.setdefault(key, dict(condition=key[0],scenario=key[1],n=0,labels={}))
        g['n'] += 1
        label = judgment['label']; g['labels'][label] = g['labels'].get(label,0) + 1
    prior_reports = sum(r['stage_id']=='report' for r in run.receipts.values())
    report_id = 'summary' if prior_reports == 0 else f"summary-{len(expected)}"
    filename = 'summary.json' if report_id == 'summary' else f"{report_id}.json"
    if run.start('report',report_id):
        path = run.root / filename
        atomic(path, dict(groups=list(groups.values()), uncertainty='Descriptive counts; repeated samples from four fixed prompts, not independent scenarios.'))
        run.commit('report',report_id,path)
    if not any(x['href']==filename for x in run.snapshot['links']):
        run.snapshot['links'].append(dict(label='Judged counts' if report_id == 'summary' else f'Judged counts ({len(expected):,} planned)',href=filename))
    run.finish_stage('report')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['init','generate','judge','report','serve'])
    parser.add_argument('--config',type=Path,default=DEFAULT)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--limit',type=int,help='Stop after this many new committed responses/judgments; resume with same command')
    parser.add_argument('--judge-model')
    parser.add_argument('--port',type=int,default=8790)
    args=parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.command=='judge' and not args.judge_model:
        parser.error('--judge-model is required')
    out=args.output or ROOT/'steering/runs'/read(args.config)['run_id']
    if args.command=='serve':
        from functools import partial
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
        print(f'http://127.0.0.1:{args.port}/',flush=True)
        ThreadingHTTPServer(('0.0.0.0',args.port),partial(SimpleHTTPRequestHandler,directory=str(out))).serve_forever()
        return
    initialize(args.config,out)
    if args.command=='init':
        print(out/'index.html')
        return
    with Tracker(out) as run:
        if args.command=='generate': generate(run,args.limit)
        elif args.command=='judge': judge(run,args.judge_model,args.limit)
        elif args.command=='report': report(run)

if __name__=='__main__':
    main()
