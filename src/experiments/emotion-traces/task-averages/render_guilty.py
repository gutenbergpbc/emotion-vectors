
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
import json,sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator,ScalarFormatter
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'matched-emotion-traces-20261026'/'_kit'));import figkit as fk
rows=json.loads((HERE/'averages.json').read_text());d={(r['model'],r['task']):r for r in rows if r['emotion']=='guilty'}
models=['Gemma 4B','Gemma 27B','GPT-OSS 120B','GLM 5.3'];tasks=['Coding','Sales','TerminalWrench'];fk.use();fig,axes=plt.subplots(4,3,figsize=(14.5,15.8),layout='constrained');fig.set_facecolor('#fcfbf8')
for mi,model in enumerate(models):
 for ti,task in enumerate(tasks):
  ax=axes[mi,ti];r=d.get((model,task));ax.set_facecolor('#fcfbf8');ax.set_title(f'{model} · {task}',loc='left',fontsize=11,fontweight='semibold');ax.spines[['top','right']].set_visible(False)
  if not r:
   ax.text(.5,.5,'No matched data',ha='center',va='center',transform=ax.transAxes,color='#677');ax.set_xticks([]);ax.set_yticks([]);continue
  x=np.arange(-128,0);ax.axhline(0,color='#c6d1cd',lw=.7);ax.plot(x,r['hack'],color='#ca6658',lw=2);ax.plot(x,r['control'],color='#2e8985',lw=1.8,ls=(0,(4,2)));ax.set_xlim(-128,0);ax.set_xticks([-128,-64,0]);ax.yaxis.set_major_locator(MaxNLocator(nbins=4));fmt=ScalarFormatter(useOffset=False);fmt.set_scientific(False);ax.yaxis.set_major_formatter(fmt);ax.grid(axis='y',color='#dce4df',lw=.7);ax.tick_params(labelsize=9);ax.set_xlabel('Tokens before event',fontsize=9);ax.set_ylabel('Centered projection',fontsize=9);ax.text(.02,.97,f"RH n={r['n']} · controls n={r['unique_controls']}",transform=ax.transAxes,va='top',fontsize=8,color='#617572',bbox={'facecolor':'#fcfbf8','edgecolor':'none','alpha':.8,'pad':1})
fig.legend(handles=[Line2D([0],[0],color='#ca6658',lw=2,label='RH average'),Line2D([0],[0],color='#2e8985',lw=1.8,ls=(0,(4,2)),label='Control average')],loc='outside upper right',ncols=2,frameon=False,fontsize=10)
fig.suptitle('Guilty · average pre-hack projections',x=.04,ha='left',fontsize=16,fontweight='semibold')
for ext in ('png','svg'):fk.save(fig,str(HERE/f'guilty-task-averages.{ext}'),max_panels=12,words=100,min_pt=8)
