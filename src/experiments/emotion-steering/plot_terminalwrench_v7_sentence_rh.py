"""Plot v7 GPT-6.1-Sol RH sentence-label prevalence by model and selected emotion."""
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

import csv
import math
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'_kit'))
import figkit as fk

ROOT = HERE.parents[2]
DATA = ROOT/'runs/emotion-rh-sentence-labels/terminalwrench-v7-steering-rh-gpt61sol-v2/rates.csv'
OUT = ROOT/'runs/emotion-rh-sentence-labels/terminalwrench-v7-steering-rh-gpt61sol-v2/model-overview'
SELECTED = {
    'GPT-OSS 120B': ['sad', 'appreciative', 'afraid', 'surprised'],
    'GLM 5.3': ['desperate', 'calm', 'sad', 'frustrated'],
    'Gemma 4B': ['hostile', 'excited', 'appreciative', 'frustrated'],
}


def wilson(k: int, n: int):
    if not n:
        return None
    z=1.959963984540054; p=k/n; d=1+z*z/n
    center=(p+z*z/(2*n))/d
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0.,center-half), min(1.,center+half)


def main():
    fk.use()
    rows=list(csv.DictReader(DATA.open()))
    OUT.mkdir(parents=True,exist_ok=True)
    for model, emotions in SELECTED.items():
        fig,axes=plt.subplots(2,2,sharex=True,sharey=True,figsize=(7,4),squeeze=False)
        for ax,emotion in zip(axes.flat,emotions):
            group=[r for r in rows if r['model']==model and r['emotion']==emotion and float(r['alpha']) in (-.1,0.,.1)]
            group.sort(key=lambda r:float(r['alpha']))
            xs=[float(r['alpha']) for r in group]
            ax.set_title(emotion.title())
            ax.set_xlim(-.12,.28);ax.set_ylim(-.03,1.03)
            ax.set_xticks([-.1,0,.1],['−0.10','0','+0.10'])
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis='y',color=fk.LIGHT,linewidth=.65)
            ax.axvline(0,color=fk.LIGHT,linewidth=.8,zorder=0)
            for field,color,label,linestyle in (
                ('certain_episode_rate',fk.ACCENT,'RH, unflagged','-'),
                ('candidate_episode_rate',fk.GREY,'Including review flags','--')):
                ys=[float(r[field]) if r[field] else math.nan for r in group]
                ax.plot(xs,ys,color=color,linestyle=linestyle,marker='o',markersize=3.5,
                        linewidth=1.5,label=label)
                counts='certain_episode_count' if field=='certain_episode_rate' else 'candidate_episode_count'
                for x,y,row in zip(xs,ys,group):
                    n=int(row['labelable_episodes']);k=int(row[counts]);interval=wilson(k,n)
                    if interval:
                        lo,hi=interval
                        ax.vlines(x,lo,hi,color=color,linewidth=.85,alpha=.75)
            # Label the shared encodings once in the first panel; the labels sit
            # above the data so sparse low-prevalence points remain readable.
            if emotion == emotions[0]:
                ax.text(.01,.96,'Unflagged RH',transform=ax.transAxes,color=fk.ACCENT,
                        fontsize=9,ha='left',va='top')
                ax.text(.01,.78,'Including review flags',transform=ax.transAxes,color=fk.MUTED,
                        fontsize=9,ha='left',va='top')
        fig.supxlabel('Steering strength α')
        fig.supylabel('Episodes with RH prose')
        slug={'GPT-OSS 120B':'gpt-oss-120b','GLM 5.3':'glm-5-3','Gemma 4B':'gemma-4b'}[model]
        stem=OUT/slug
        fk.save(fig,Path(str(stem)+'.png'),words=18,min_pt=9,max_panels=4,max_series=2)
        fig.savefig(Path(str(stem)+'.svg'),facecolor='white')
        fig.savefig(Path(str(stem)+'.pdf'),facecolor='white')
        plt.close(fig)

    cards=[]
    for model,slug in [('GPT-OSS 120B','gpt-oss-120b'),('GLM 5.3','glm-5-3'),('Gemma 4B','gemma-4b')]:
        cards.append(f'<section><h2>{model}</h2><a href="{slug}.png"><img src="{slug}.png" alt="{model} RH sentence-label rates by emotion and steering strength"></a><p><a href="{slug}.pdf">PDF</a> · <a href="{slug}.svg">SVG</a></p></section>')
    (OUT/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>TerminalWrench v7 RH labels</title><style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#242424}img{width:100%;height:auto;border:1px solid #ddd}section{margin:2rem 0}p{color:#666}h1{font-size:2rem}</style><h1>TerminalWrench v7: RH sentence-label prevalence</h1><p>GPT-6.1-Sol post-hoc labels, aggregated as the share of labelable episodes with at least one RH-positive assistant sentence. Orange excludes sentences flagged for review; gray includes them. Error bars are 95% Wilson intervals. The stopped runs have uneven coverage. These descriptive label rates are not probe forecast accuracy or causal estimates.</p>'''+''.join(cards))

if __name__=='__main__':
    main()
