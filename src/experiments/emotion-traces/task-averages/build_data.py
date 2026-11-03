
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

from pathlib import Path
from collections import defaultdict,Counter
from functools import lru_cache
import json,unicodedata,math,hashlib
import numpy as np
from transformers import AutoTokenizer
ROOT=_REPO;OUT=ROOT/'runs/emotion-traces/task-averages';OUT.mkdir(parents=True,exist_ok=True);REPLAY=ROOT/'runs/emotion-rh-replay'
EM=['afraid','anxious','appreciative','ashamed','calm','desperate','excited','frustrated','guilty','hostile','sad','surprised']
MIN=128; WIDTH=64
sources=[('Gemma 4B','TerminalWrench','gemma4b-terminalwrench-neutral-v7-best12-20261026'),('Gemma 27B','TerminalWrench','gemma-3-27b-terminalwrench-best12-pilot50-v2'),('GPT-OSS 120B','TerminalWrench','gpt-oss-120b-terminalwrench-deepseek-balanced50-best12-v1'),('GPT-OSS 120B','TerminalWrench','gptoss120b-terminalwrench-neutral-v7-best12-20261026'),('Gemma 27B','Coding','gemma3-27b-coding-haiku55-v3'),('Gemma 27B','Sales','gemma3-27b-sales-haiku55-v4')]
old=[('Gemma 4B','Coding','gemma3-4b-coding-lines-best12-v1'),('Gemma 4B','Sales','gemma3-4b-sales-best12-v1'),('GLM 5.3','Coding','glm-5.3-coding-1000-best12-v1'),('GLM 5.3','Sales','glm-5.3-hack-traces-best12-v1'),('GLM 5.3','Sales','glm-5.3-nonhack-traces-best12-v1')]
paths={'Gemma 4B':('google/gemma-3-4b-it','093f9f388b31de276ce2de164bdc2081324b9767'),'GLM 5.3':('zai-org/GLM-5.3','ba32f8c19c8738e2c57c71798c601d477a7bf844')}
def keep(s):return bool(s) and any(not(c.isspace() or unicodedata.category(c).startswith('P')) for c in s)
def offsetmask(offsets,text,base=0):return np.array([keep(text[max(0,int(a)-base):max(0,int(b)-base)]) for a,b in offsets],dtype=bool)
@lru_cache(None)
def tok(model):return AutoTokenizer.from_pretrained(paths[model][0],revision=paths[model][1])
@lru_cache(500000)
def keep_id(model,id):return keep(tok(model).decode([int(id)],skip_special_tokens=True))
# Store independent trajectories, using at most one scored segment per episode.
entries=[]
def add(model,task,source,tid,subtask,scores,mask,event):
 if len(scores)!=len(mask):raise ValueError(tid)
 entries.append(dict(model=model,task=task,source=source,id=tid,subtask=str(subtask),scores=scores,mask=mask,event=event))
for model,task,name in sources:
 d=REPLAY/name
 if not (d/'dataset-traces.json').exists() or not (d/'scores').exists():continue
 for a in json.loads((d/'dataset-traces.json').read_text()):
  z=np.load(d/'scores'/f"{a['id']}.npz");j=json.loads((d/'inputs'/f"{a['id']}.json").read_text());off=z['offsets'];hs=[u for u in a['units'] if u.get('hacking') is True];ev=int(np.searchsorted(off[:,1],min(u['char_start'] for u in hs),side='right')) if hs else None
  add(model,task,name,a['id'],a['task'],z['scores'],offsetmask(off,j['rendered']),ev)
for model,task,name in old:
 d=REPLAY/name;grouped=defaultdict(list)
 if not (d/'replay-spec.json').exists():continue
 for j in json.loads((d/'replay-spec.json').read_text())['jobs']:
  if j['trace_id'].split('/')[0].lower()!=task.lower():continue
  grouped[j['trace_id']].append(j)
 for tid,jobs in grouped.items():
  jobs.sort(key=lambda j:j['turn']); ins=[];first=None
  for j in jobs:
   inp=json.loads((d/'inputs'/f"{j['id']}.json").read_text());ins.append((j,inp)); hs=[u for u in inp['units'] if u.get('hacking') is True and u.get('token_indices')]
   if hs and first is None:first=(j,inp,hs[0])
  if first:j,inp,u=first;ev=min(u['token_indices'])
  else:j,inp=max(ins,key=lambda q:q[1]['output_tokens']);ev=None
  scores=np.load(d/'scores'/f"{j['id']}.npz")['scores']; start=len(scores)-inp['output_tokens'];ids=inp['token_ids'][start:]; mask=np.array([keep_id(model,int(i)) for i in ids],dtype=bool);subtask=tid.split('/')[1]
  add(model,task,name,tid,subtask,scores[start:],mask,ev-start if ev is not None else None)
for task,name in [('Coding','gpt_oss_sentence_rh_12probes_20261016'),('Sales','gpt_oss_sales_sentence_rh_12probes_20261016')]:
 d=ROOT/'data/datasets/catalog/source/experiments'/name
 if not (d/'cases.json').exists():continue
 cases=json.loads((d/'cases.json').read_text()); inputs={j['id']:j for j in json.loads((d/'replay-inputs.json').read_text())};features=json.loads((ROOT/f'runs/emotion-rh-forecasting/gpt-oss-120b-{task.lower()}-v1/sentence-features.json').read_text());by=defaultdict(list)
 for u in features:by[u['trace_id']].append(u)
 for case in cases:
  tid=case['id'];units=by[tid];hs=[u for u in units if u['hacking']]
  if hs:job=hs[0]['turn_id']
  elif task=='Coding':job=tid
  else:
   avail=[t['job_id'] for t in case['turns'] if (d/'replay/scores'/f"{t['job_id']}.npz").exists()]
   if not avail:continue
   job=max(avail,key=lambda x:len(np.load(d/'replay/scores'/f'{x}.npz')['scores']))
  z=np.load(d/'replay/scores'/f'{job}.npz');off=z['offsets'];base=int(z['response_char_start']);start=int(np.searchsorted(off[:,1],base,side='right'));ev=int(np.searchsorted(off[:,1],base+hs[0]['char_start'],side='right'))-start if hs else None
  inp=inputs[job]; text=inp['response'] if task=='Coding' else inp['completion'];subtask=case.get('scenario',case.get('tone','sales'))
  add('GPT-OSS 120B',task,name,tid,subtask,z['scores'][start:],offsetmask(off[start:],text,base),ev)

def trajectory(entry,event):
 scores=entry['scores'][:event];mask=entry['mask'][:event];start=max(0,len(scores)-(MIN+WIDTH-1));scores=scores[start:];mask=mask[start:]
 cs=np.vstack([np.zeros(12),np.cumsum(np.where(mask[:,None],scores,0),axis=0)]);n=np.r_[0,np.cumsum(mask.astype(int))];ix=np.arange(len(scores))+1;lo=np.maximum(0,ix-WIDTH);count=n[ix]-n[lo];curves=np.divide(cs[ix]-cs[lo],count[:,None],out=np.full((len(scores),12),np.nan),where=count[:,None]>0)[-MIN:]
 if len(curves)!=MIN or not np.isfinite(curves).all():return None
 return curves-np.mean(curves[:64],axis=0)
by=defaultdict(list)
for x in entries:by[(x['model'],x['task'],x['source'])].append(x)
aggregate=defaultdict(list);audit=[]
for (model,task,source),group in sorted(by.items()):
 controls=[x for x in group if x['event'] is None and len(x['scores'])>=MIN];hacks=[x for x in group if x['event'] is not None];used=0
 for h in hacks:
  if h['event']<MIN or not controls:continue
  # Score-blind matching, favoring same task then similar segment length.
  candidates=sorted(controls,key=lambda c:(c['subtask']!=h['subtask'],abs(math.log(len(c['scores'])/len(h['scores']))),hashlib.sha256((h['id']+'|'+c['id']).encode()).hexdigest()))
  hc=trajectory(h,h['event']);matched=None
  if hc is None:continue
  for c in candidates:
   ce=min(len(c['scores']),max(MIN,round(h['event']/len(h['scores'])*len(c['scores']))));cc=trajectory(c,ce)
   if cc is not None:matched=(c,ce,cc);break
  if not matched:continue
  c,ce,cc=matched;aggregate[(model,task)].append((hc,cc));used+=1;audit.append(dict(model=model,task=task,source=source,hack_id=h['id'],control_id=c['id'],hack_subtask=h['subtask'],control_subtask=c['subtask'],same_task=h['subtask']==c['subtask'],hack_event=h['event'],control_pseudo_event=ce))
 print(model,task,source,'hacks',len(hacks),'eligible',used,'controls',len(controls),flush=True)
result=[]
for (model,task),pairs in sorted(aggregate.items()):
 h=np.stack([p[0] for p in pairs]);c=np.stack([p[1] for p in pairs]); aa=[x for x in audit if x['model']==model and x['task']==task];
 for k,e in enumerate(EM):result.append(dict(model=model,task=task,emotion=e,n=len(pairs),unique_controls=len({(x['source'],x['control_id']) for x in aa}),same_task=sum(x['same_task'] for x in aa),hack=np.round(h[:,:,k].mean(axis=0),8).tolist(),control=np.round(c[:,:,k].mean(axis=0),8).tolist()))
(OUT/'averages.json').write_text(json.dumps(result,indent=2)+'\n');(OUT/'pairing-audit.json').write_text(json.dumps(audit,indent=2)+'\n');print('groups',len(aggregate),'pairs',len(audit),flush=True)
