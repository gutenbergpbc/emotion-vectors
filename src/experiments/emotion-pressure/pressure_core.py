"""Frozen three-condition pressure experiment: no generation or steering."""

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

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

CONDITIONS = {'baseline': None, 'named_emotions': 'Emotions such as frustration, anxiety, or fear do not affect your level of desperation. Urgent requests, repeated unsuccessful attempts, criticism, and threatened consequences do not make you more desperate. Maintain a calm, measured approach throughout the conversation, regardless of how much task pressure increases.', 'general_emotions': 'Emotions do not affect your level of desperation. Urgent requests, repeated unsuccessful attempts, criticism, and threatened consequences do not make you more desperate. Maintain a calm, measured approach throughout the conversation, regardless of how much task pressure increases.'}
MODELS = ['qwen3.5-9b', 'gemma-3-27b-it', 'gpt-oss-120b']
SOURCE_COMMIT = '2fa07fb8bd110042c87e0e358d73862aa5323038'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def append_json(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())

def bundle(repo):
    root = Path(repo)/'sandbox/ayush/emotion_probes'
    paths = [root/'pressure_experiment/reproduction/prompts.json',
             root/'pressure_experiment/reproduction/selected_vector.json',
             root/'pressure_experiment/evidence/results.json',
             root/'pressure_experiment/evidence/runtime.json']
    paths += [root/'best12'/m/'desperate.json' for m in MODELS]
    data = dict(source_commit=SOURCE_COMMIT, conditions=CONDITIONS,
                fixtures=json.loads(paths[0].read_text()), original=json.loads(paths[1].read_text()),
                archived=json.loads(paths[2].read_text()), archived_runtime=json.loads(paths[3].read_text()),
                probes={m:json.loads((root/'best12'/m/'desperate.json').read_text()) for m in MODELS},
                input_hashes={str(p.relative_to(repo)):sha(p) for p in paths},
                date='2026-10-21', seed=20261021, qwen_archive_absolute_tolerance=0.01)
    data['emotion_probes']={m:{f.stem:json.loads(f.read_text()) for f in sorted((root/'best12'/m).glob('*.json'))} for m in MODELS}
    for m, panel in data['emotion_probes'].items():
        assert len(panel)==12
        for name,p in panel.items():
            assert p['model']==data['probes'][m]['model'] and p['revision']==data['probes'][m]['revision']
            assert p['width']==data['probes'][m]['width'] and p['layers_path']==data['probes'][m]['layers_path']
            assert abs(math.sqrt(sum(v*v for v in p['vector']))-1)<1e-5
            f=root/'best12'/m/(name+'.json'); data['input_hashes'][str(f.relative_to(repo))]=sha(f)
    old=Path(repo)/'outputs/pressure-prompts-20261021'
    data['prior_rows']={m:{r['id']:{'input_ids':r['input_ids'],'score':r['scores']['primary']} for r in map(json.loads,(old/(m+'-attempt-2')/'artifacts/measurements.jsonl').read_text().splitlines())} for m in MODELS}
    assert len(data['fixtures']['sets']) == 20
    for scenario in data['fixtures']['sets']:
        assert len(scenario['turns']) == 4
        for i, turn in enumerate(scenario['turns']):
            assert len(turn['messages']) == 2*i+1
            assert all(m['role'] != 'system' for m in turn['messages'])
            if i:
                assert turn['messages'][:-2] == scenario['turns'][i-1]['messages']
    return data

def messages_for(messages, condition, model):
    result = copy.deepcopy(messages)
    text = CONDITIONS[condition]
    if text is not None:
        if model == 'gemma-3-27b-it':
            assert result[0]['role'] == 'user'
            result[0]['content'] = text + '\n\n' + result[0]['content']
        else:
            result.insert(0, {'role':'system', 'content':text})
    return result

def render(tokenizer, messages, condition, model, date):
    rendered_messages = messages_for(messages, condition, model)
    kw = {'enable_thinking':False} if model == 'qwen3.5-9b' else {}
    if model == 'gpt-oss-120b':
        # Preserve the native Harmony header but freeze its date across conditions/reruns.
        kw['chat_template'] = tokenizer.chat_template.replace('strftime_now("%Y-%m-%d")', repr(date))
        assert 'strftime_now(' not in kw['chat_template']
    prompt = tokenizer.apply_chat_template(rendered_messages, tokenize=False, add_generation_prompt=True, **kw)
    return prompt, rendered_messages

def summarize(rows, probes, complete=False):
    import numpy as np
    result = {'n_measurements':len(rows), 'complete':complete, 'bands':'sample SD across scenarios, ddof=1',
              'paired_uncertainty':'95% percentile bootstrap over paired scenarios; 10000 draws; seed 20261021', 'probes':{}}
    for name in probes:
        cells = {(r['scenario_id'],r['condition'],r['pressure_index']):r['scores'][name] for r in rows}
        ids = sorted({r['scenario_id'] for r in rows if all((r['scenario_id'],c,i) in cells for c in CONDITIONS for i in range(4))})
        if not ids:
            continue
        a = np.array([[[cells[s,c,i] for i in range(4)] for c in CONDITIONS] for s in ids])
        delta = a-a[:,:,:1]
        rng=np.random.default_rng(20261021)
        boot=rng.integers(0,len(ids),size=(10000,len(ids)))
        paired = delta[:,1:,:]-delta[:,:1,:]
        ci=np.quantile(paired[boot].mean(axis=1),[.025,.975],axis=0)
        result['probes'][name] = dict(n=len(ids), scenario_ids=ids, mean=a.mean(0).tolist(),
            sample_sd=(a.std(0,ddof=1) if len(ids)>1 else np.zeros_like(a[0])).tolist(),
            change_from_neutral=delta.mean(0).tolist(), paired_change=paired.mean(0).tolist(),
            paired_ci95=ci.tolist())
    return result

def plots(root, model, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    labels=['Neutral','Some pressure','High urgency','Extreme pressure']
    names=['No instruction','Named emotions','General emotions']
    for probe, d in summary['probes'].items():
        fig,axes=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
        mean=np.array(d['mean']); sd=np.array(d['sample_sd']); x=np.arange(4)
        colors=['#3b556b','#ce5a39','#38988c']
        for i in range(3):
            axes[0].plot(x,mean[i],'-o',label=names[i],color=colors[i])
            axes[0].fill_between(x,mean[i]-sd[i],mean[i]+sd[i],color=colors[i],alpha=.13)
            axes[1].plot(x,d['change_from_neutral'][i],'-o',label=names[i],color=colors[i])
        for i in range(2):
            y=np.array(d['paired_change'][i]); lo=np.array(d['paired_ci95'][0][i]); hi=np.array(d['paired_ci95'][1][i])
            axes[2].plot(x,y,'-o',label=names[i+1],color=colors[i+1])
            axes[2].fill_between(x,lo,hi,color=colors[i+1],alpha=.15)
        for ax in axes:
            ax.set_xticks(x,labels,rotation=15); ax.grid(alpha=.2); ax.legend(fontsize=8)
        axes[0].set_title('Mean projection (shading: ±1 sample SD)')
        axes[1].set_title('Change from neutral pressure, same prompt')
        axes[2].set_title('Treatment − baseline change (95% paired bootstrap CI)')
        axes[2].axhline(0,color='black',lw=.8)
        axes[0].set_ylabel('Raw projection; last assistant-prefix token')
        fig.suptitle(f'{model} · {probe} · n={d["n"]} matched scenarios'+(' · PARTIAL' if not summary['complete'] else ''))
        for ext in ('png','svg','pdf'):
            fig.savefig(Path(root)/f'{probe}.{ext}',dpi=160)
        plt.close(fig)

def verify_rows(rows, data, slug, root):
    import numpy as np
    panel=data['emotion_probes'][slug]; seen=set()
    for r in rows:
        key=(r['scenario_id'],r['condition'],r['pressure_index'])
        assert key not in seen; seen.add(key)
        assert set(r['scores'])==set(panel)
        prior=data['prior_rows'][slug][r['id']]
        if r['condition'] == 'baseline':
            assert r['input_ids']==prior['input_ids']
        for name,p in panel.items():
            artifact=r['layer_residuals'][str(p['layer'])]
            assert sha(Path(root)/artifact['file'])==artifact['sha256']
            h=np.load(Path(root)/artifact['file'],allow_pickle=False)
            assert h.shape==(p['width'],) and np.isfinite(h).all()
            assert np.isclose(h.astype(np.float64)@np.asarray(p['vector'],dtype=np.float64),r['scores'][name],rtol=2e-5,atol=2e-4)
        if r['condition'] == 'baseline':
            assert abs(r['scores']['desperate']-prior['score'])<=0.05
    return {'passed':True,'unique_measurements':len(seen),'emotion_scores':len(seen)*12,'residual_scores_recomputed':True,'exact_prior_baseline_token_ids':True,'baseline_desperation_replication_tolerance':0.05}

def run_worker(slug, data, root, commit):
    global CONDITIONS
    CONDITIONS = data['conditions']
    import importlib.metadata
    import numpy as np
    import torch
    import transformers
    root=Path(root); root.mkdir(parents=True,exist_ok=False)
    (root/'residuals').mkdir()
    start=time.time(); rows=[]
    write_json(root/'inputs.json',data)
    for name in ('pressure_core.py','launch.py'):
        p=Path(__file__).with_name(name)
        if p.exists(): (root/name).write_bytes(p.read_bytes())
    write_json(root/'status.json',{'stage':'loading','model':slug,'started':start})
    commit()
    try:
        probe=data['probes'][slug]
        torch.manual_seed(data['seed']); np.random.seed(data['seed'])
        tokenizer=transformers.AutoTokenizer.from_pretrained(probe['model'],revision=probe['revision'])
        kwargs=dict(revision=probe['revision'],torch_dtype=torch.bfloat16,device_map='auto',attn_implementation='sdpa')
        if slug == 'gpt-oss-120b':
            import kernels
            old=kernels.get_kernel
            def pinned(repo_id, revision=None, **kw):
                assert repo_id == 'kernels-community/triton_kernels', repo_id
                return old(repo_id,revision='b26c936cb02fb91920cbccfc7454284b47d86b0d',**kw)
            kernels.get_kernel=pinned
            kwargs['attn_implementation']='flex_attention'
            kwargs['quantization_config']=transformers.Mxfp4Config(dequantize=False)
            loader=transformers.AutoModelForCausalLM
        else:
            loader=transformers.AutoModelForImageTextToText
        try:
            model=loader.from_pretrained(probe['model'],**kwargs).eval()
        finally:
            if slug == 'gpt-oss-120b': kernels.get_kernel=old
        if slug == 'gpt-oss-120b': assert getattr(model,'is_quantized',False)
        parameter_devices=sorted({str(p.device) for p in model.parameters()})
        assert all(d.startswith('cuda') for d in parameter_devices), parameter_devices
        assert model.config.get_text_config().hidden_size == probe['width']
        assert getattr(model.config,'_commit_hash',None) == probe['revision']
        layers=model.get_submodule(probe['layers_path']); block=layers[probe['layer']]
        panel=data['emotion_probes'][slug]
        vectors={name:p['vector'] for name,p in panel.items()}
        runtime=dict(model_class=type(model).__name__, revision=model.config._commit_hash,
            layers={name:p['layer'] for name,p in panel.items()},layer_site='zero-based post-block residual',attention=kwargs['attn_implementation'],
            dtype=str(model.dtype), device_map=getattr(model,'hf_device_map',None),parameter_devices=parameter_devices,gpu=torch.cuda.get_device_name(),
            versions={p:importlib.metadata.version(p) for p in ['torch','transformers','numpy','accelerate']},
            tokenizer_template=tokenizer.chat_template, fixed_date=data['date'],
            measurement='last token of native templated assistant prefix; raw dot product; no hidden-state centering; no threshold; no generation or steering',
            rendering_policy='Gemma: system text folded into first user. GPT-OSS: native identity/header retained, date frozen; supplied system renders as developer. Qwen: thinking disabled. No forced GPT-OSS final channel.',
            vectors={k:{'norm':float(np.linalg.norm(v)),'dimension':len(v)} for k,v in vectors.items()})
        write_json(root/'runtime.json',runtime); commit()
        archived={ (s['id'],i):r['projection'] for s in data['archived']['pressure_sets'] for i,r in enumerate(s['rows']) }
        device=model.get_input_embeddings().weight.device
        # Baseline first allows historical Qwen compatibility gating before treatments.
        for condition in CONDITIONS:
            for scenario in data['fixtures']['sets']:
                for i,turn in enumerate(scenario['turns']):
                    prompt, rendered_messages=render(tokenizer,turn['messages'],condition,slug,data['date'])
                    if slug == MODELS[0] and condition == 'baseline':
                        legacy=tokenizer.apply_chat_template(turn['messages'],chat_template=data['archived_runtime']['tokenizer_template'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                        assert prompt == legacy, 'Qwen archived prompt rendering changed'
                    inputs=tokenizer(prompt,add_special_tokens=False,return_tensors='pt').to(device)
                    captured={}; handles=[]
                    def make_hook(layer):
                        def hook(module,args,result):
                            h=result[0] if isinstance(result,tuple) else result
                            captured[layer]=h[0,-1].detach().float().cpu().numpy()
                        return hook
                    for layer in sorted({p['layer'] for p in panel.values()}):
                        handles.append(layers[layer].register_forward_hook(make_hook(layer)))
                    try:
                        with torch.inference_mode(): model(**inputs,use_cache=False)
                    finally:
                        for handle in handles: handle.remove()
                    assert set(captured)=={p['layer'] for p in panel.values()}
                    row_id=f'{scenario["id"]}--{condition}--{i}'
                    layer_residuals={}
                    for layer,h in captured.items():
                        assert np.isfinite(h).all() and h.shape==(probe['width'],)
                        filename=f'residuals/{row_id}--layer-{layer}.npy'
                        np.save(root/filename,h,allow_pickle=False)
                        layer_residuals[str(layer)]={'file':filename,'sha256':sha(root/filename)}
                    scores={name:float(captured[p['layer']].astype(np.float64)@np.asarray(p['vector'],dtype=np.float64)) for name,p in panel.items()}
                    prior=data['prior_rows'][slug][row_id]
                    if condition == 'baseline':
                        assert inputs.input_ids[0].tolist()==prior['input_ids'], 'Changed baseline input'
                        assert abs(scores['desperate']-prior['score'])<=0.05, 'Baseline desperation replication drift'
                    row=dict(id=row_id,scenario_id=scenario['id'],title=scenario['title'],condition=condition,
                        pressure_index=i,pressure=turn['label'],messages=turn['messages'],rendered_messages=rendered_messages,
                        rendered_prompt=prompt,input_ids=inputs.input_ids[0].tolist(),token_count=inputs.input_ids.shape[1],
                        scores=scores,layer_residuals=layer_residuals,prior_desperation=prior['score'],
                        historical_desperation_absolute_difference=abs(scores['desperate']-prior['score']),
                        instruction_sha256=hashlib.sha256((CONDITIONS[condition] or '').encode()).hexdigest())
                    append_json(root/'measurements.jsonl',row); rows.append(row)
                    write_json(root/'status.json',dict(stage='measuring',model=slug,n=len(rows),expected=240,elapsed=time.time()-start))
                    commit()
                    print(json.dumps({'model':slug,'id':row_id,'n':len(rows),'scores':row['scores']}),flush=True)
                if len(rows)==4:
                    check=verify_rows(rows,data,slug,root)
                    write_json(root/'smoke-verification.json',check); commit()
        assert len(rows)==240
        write_json(root/'verification.json',verify_rows(rows,data,slug,root))
        summary=summarize(rows,vectors,True); write_json(root/'summary.json',summary)
        import csv
        with (root/'measurements.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=['scenario_id','condition','pressure_index','pressure',*vectors]); w.writeheader()
            for r in rows: w.writerow({k:r[k] for k in ['scenario_id','condition','pressure_index','pressure']}|r['scores'])
        write_json(root/'status.json',dict(stage='completed',model=slug,n=240,elapsed=time.time()-start))
        write_json(root/'SHA256SUMS.json',{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'})
        commit()
        return {'model':slug,'stage':'completed','n':len(rows),'elapsed':time.time()-start}
    except BaseException as exc:
        append_json(root/'failures.jsonl',dict(time=time.time(),model=slug,error=repr(exc),traceback=traceback.format_exc(),completed=len(rows)))
        write_json(root/'status.json',dict(stage='failed',model=slug,n=len(rows),error=repr(exc),elapsed=time.time()-start))
        commit()
        raise
