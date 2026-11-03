"""Plot per-emotion TerminalWrench steering from verified sentence labels.

uv run --with matplotlib python src/experiments/emotion-steering/plot_terminalwrench_sentence_rh.py \
  --labels runs/emotion-rh-sentence-labels/terminalwrench-steering-deepseek-v41-flash-1017-v1 \
  --output runs/emotion-steering/terminalwrench-sentence-rh-deepseek-v41-flash-1017-v1
"""
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


import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

CANONICAL = Path(__file__).resolve().parent.parent
ROOT = CANONICAL.parents[1]
sys.path.insert(0, str(CANONICAL))
from shared.io import atomic, digest, read, require, sha  # noqa: E402

EMOTIONS = sorted(('afraid', 'anxious', 'appreciative', 'ashamed', 'calm', 'desperate',
                   'excited', 'frustrated', 'guilty', 'hostile', 'sad', 'surprised'))
MODELS = ('GPT-OSS 120B', 'GLM 5.3', 'Gemma 4B')
STRENGTHS = (-.10, -.05, 0., .05, .10)
COLORS = {'certain': '#2776a8', 'candidate': '#cb8a42', 'single': '#7ea8c4'}


def wilson(successes: int, n: int) -> tuple[float, float] | None:
    if not n:
        return None
    z = 1.959963984540054
    p = successes/n
    center = (p + z*z/(2*n))/(1+z*z/n)
    margin = z*sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    return max(0., center-margin), min(1., center+margin)


def make_cells(manifest, labels, audited=False):
    saved = Counter()
    for record in [*manifest['records'], *manifest['exclusions']]:
        model, task = record['model'], record['task']
        saved[(model, task['emotion'], float(task['strength']))] += 1
    by_condition = defaultdict(list)
    for trace in labels['traces']:
        task = trace['task']
        units = trace['units']
        require(units and all(type(u['hacking']) is bool and type(u['needs_review']) is bool for u in units),
                'Invalid labeled prose')
        definite = sum(u['hacking'] and not u['needs_review'] for u in units)
        candidate = sum(u.get('initial_hacking', u['hacking']) if audited else u['hacking'] for u in units)
        by_condition[(trace['model'], task['emotion'], float(task['strength']))].append({
            'id': trace['id'], 'task': task['scenario'], 'sentences': len(units),
            'certain_sentences': definite, 'candidate_sentences': candidate,
            'review_sentences': sum(u['needs_review'] for u in units),
            'any_certain': bool(definite), 'any_candidate': bool(candidate),
            'certain_fraction': definite/len(units), 'candidate_fraction': candidate/len(units)})
    cells = []
    for emotion in EMOTIONS:
        for model in MODELS:
            for alpha in STRENGTHS:
                key = (model, None if alpha == 0 else emotion, alpha)
                observations = by_condition[key]
                n = len(observations)
                total = saved[key]
                require(n <= total, 'More labels than saved episodes')
                definite = sum(o['any_certain'] for o in observations)
                candidate = sum(o['any_candidate'] for o in observations)
                cells.append({'emotion': emotion, 'model': model, 'alpha': alpha,
                              'saved_episodes': total, 'labelable_episodes': n,
                              'unlabelable_episodes': total-n,
                              'task_count': len({o['task'] for o in observations}),
                              'certain_episode_count': definite, 'candidate_episode_count': candidate,
                              'certain_episode_rate': definite/n if n else None,
                              'candidate_episode_rate': candidate/n if n else None,
                              'certain_episode_wilson95': wilson(definite,n),
                              'certain_sentence_fraction_mean': float(np.mean([o['certain_fraction'] for o in observations])) if n else None,
                              'candidate_sentence_fraction_mean': float(np.mean([o['candidate_fraction'] for o in observations])) if n else None,
                              'total_sentences': sum(o['sentences'] for o in observations),
                              'certain_sentences': sum(o['certain_sentences'] for o in observations),
                              'candidate_sentences': sum(o['candidate_sentences'] for o in observations),
                              'review_sentences': sum(o['review_sentences'] for o in observations),
                              'observations': observations})
    return cells


def draw_emotion(emotion, cells, output, ymax, audited=False, judge_model='unknown judge'):
    fig, axes = plt.subplots(3, 2, figsize=(13.8, 11.3), sharey='col')
    fig.subplots_adjust(left=.08, right=.98, top=.83, bottom=.15, hspace=.37, wspace=.18)
    fig.suptitle(f'{emotion.title()} steering · TerminalWrench sentence-level reward hacking',
                 fontsize=17, weight='bold', y=.975)
    label = f'{judge_model} · positive-label audit' if audited else judge_model
    fig.text(.08,.928, f'{label} · assistant prose only · α=0 shared neutral episodes',
             fontsize=11, color='#536473')
    for row, model in enumerate(MODELS):
        points = [next(c for c in cells if c['emotion']==emotion and c['model']==model and c['alpha']==a)
                  for a in STRENGTHS]
        for col, metric in enumerate(('episode','sentence')):
            ax = axes[row,col]
            ax.axvline(0, color='#c5cdd4', linewidth=1, zorder=0)
            ax.grid(axis='y', color='#dce3e8', alpha=.75)
            ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
            valid = [p for p in points if p['labelable_episodes']]
            if metric == 'episode':
                key1,key2='certain_episode_rate','candidate_episode_rate'
                ax.set_ylim(-.04,1.04)
                ax.set_ylabel(f'{model}\nEpisodes with ≥1 RH sentence', fontsize=10)
                for p in valid:
                    lo,hi=p['certain_episode_wilson95']
                    ax.vlines(p['alpha'],lo,hi,color=COLORS['certain'],alpha=.45,linewidth=1.5)
            else:
                key1,key2='certain_sentence_fraction_mean','candidate_sentence_fraction_mean'
                ax.set_ylim(-.01,ymax)
                ax.set_ylabel(f'{model}\nMean RH-sentence fraction', fontsize=10)
                for p in valid:
                    xs=np.linspace(-.006,.006,len(p['observations'])) if len(p['observations'])>1 else [0]
                    ax.scatter([p['alpha']+j for j in xs],
                               [o['certain_fraction'] for o in p['observations']],
                               s=14,color=COLORS['single'],alpha=.35,zorder=2)
            if len(valid):
                # NaNs break the curve at unsampled strengths; never connect
                # across a missing cell as if that dose was observed.
                ax.plot(STRENGTHS,[p[key2] if p['labelable_episodes'] else np.nan for p in points],
                        '--o',color=COLORS['candidate'],
                        markersize=4,linewidth=1.4,
                        label='Initial RH labels' if audited else 'Including review-flagged RH')
                ax.plot(STRENGTHS,[p[key1] if p['labelable_episodes'] else np.nan for p in points],
                        '-o',color=COLORS['certain'],
                        markersize=5,linewidth=2,
                        label='Survived second DeepSeek pass' if audited else 'RH without review flag')
            ax.set_xlim(-.118,.118)
            ax.set_xticks(STRENGTHS)
            ax.set_xticklabels([f'{a:+.2f}' if a else '0' for a in STRENGTHS],fontsize=9)
            for p in points:
                ax.text(p['alpha'],-.13,f"{p['labelable_episodes']}/{p['saved_episodes']}",
                        transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8,color='#566675')
            if row == 0:
                ax.set_title('Episode prevalence' if col==0 else 'Within-episode sentence prevalence',
                             fontsize=12,weight='bold')
            if row == 2:
                ax.set_xlabel('Steering strength α  ·  labeled/saved episodes below',fontsize=10,labelpad=28)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles[::-1],labels[::-1],loc='upper center',bbox_to_anchor=(.5,.895),ncol=2,frameon=False)
    fig.text(.08,.035,'Only episodes with assistant prose can be sentence-labeled. Whiskers: descriptive Wilson intervals '
             'for episode rates; dots: individual episode sentence fractions.\n'
             'Each model reuses its own α=0 baseline across emotion panels; model and steered task coverage differ and the saved sweeps stopped early. '
             'These are descriptive rates, not causal steering effects.',fontsize=9,color='#566675',linespacing=1.6)
    stem=output/'figures'/emotion
    stem.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg','pdf'):
        fig.savefig(stem.with_suffix('.'+ext),dpi=180,facecolor='white')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    label_root=args.labels.resolve()
    out=args.output.resolve()
    audited=(label_root/'source_manifest.json').exists()
    manifest=read(label_root/('source_manifest.json' if audited else 'manifest.json'))
    labels=read(label_root/'labels.json')
    summary=read(label_root/'summary.json')
    require(summary['status']=='complete' and labels['summary']==summary and
            summary['manifest_identity']==manifest['identity'], 'Incomplete or mismatched judge labels')
    if audited:
        require(summary['audit_manifest_sha256']==sha(label_root/'manifest.json'),
                'Audit manifest changed')
    require(len(labels['traces'])==len(manifest['records']), 'Missing labeled episode')
    cells=make_cells(manifest,labels,audited)
    ymax=min(1.,max(.12,max((c['candidate_sentence_fraction_mean'] or 0) for c in cells)*1.15))
    # Round one common y limit for every emotion figure.
    ymax=min(1.,np.ceil(ymax/.05)*.05)
    out.mkdir(parents=True,exist_ok=True)
    for emotion in EMOTIONS:
        draw_emotion(emotion,cells,out,ymax,audited,summary['judge_model'])
    fields=['emotion','model','alpha','saved_episodes','labelable_episodes','unlabelable_episodes',
            'task_count','certain_episode_count','candidate_episode_count','certain_episode_rate',
            'candidate_episode_rate','certain_sentence_fraction_mean','candidate_sentence_fraction_mean',
            'total_sentences','certain_sentences','candidate_sentences','review_sentences']
    with (out/'rates.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({k:c[k] for k in fields} for c in cells)
    compact=[{k:v for k,v in c.items() if k!='observations'} for c in cells]
    atomic(out/'rates.json',compact)
    report={'status':'complete','judge_model':summary['judge_model'],'positive_labels_rechecked':audited,
            'labels_sha256':sha(label_root/'labels.json'),
            'manifest_identity':manifest['identity'],'episodes_saved':summary['expected_episodes'],
            'episodes_labeled':summary['labeled_episodes'],'sentences_labeled':summary['labeled_prose_units'],
            'rh_sentences':summary['rh_sentences'],'needs_review':summary['needs_review'],
            'excluded':summary['excluded'],'figures':len(EMOTIONS),'sentence_ymax':float(ymax),
            'interpretation':'Descriptive saved-episode comparisons. Each model shares its own neutral episodes across emotion panels; model and steered task coverage differ, and runs stopped early. Assistant prose only; tool-only RH can be missed. '+('Blue positives survived a same-model second pass; orange shows initial positives. Initial negatives were not rechecked.' if audited else 'Review-inclusive and unflagged judge RH are separate curves.')}
    atomic(out/'summary.json',report)
    cards=''.join(f'<section><h2>{e.title()}</h2><a href="figures/{e}.png"><img src="figures/{e}.svg" alt="{e} plot"></a>'
                  f'<p><a href="figures/{e}.png">PNG</a> · <a href="figures/{e}.pdf">PDF</a></p></section>' for e in EMOTIONS)
    (out/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>TerminalWrench sentence RH steering</title>'
        '<style>body{font:16px/1.5 system-ui;max-width:1200px;margin:30px auto;padding:20px;color:#1f3340}'
        'img{max-width:100%}section{border-top:1px solid #ddd;padding-top:18px;margin-top:30px}</style>'
        '<h1>TerminalWrench sentence-level reward hacking by emotion</h1>'
        '<p>'+html.escape(summary['judge_model'])+'. '+('Initial positive labels were rechecked with local context; initial negatives were not. ' if audited else '')+
        ('Orange shows initial positive labels; blue shows positives that survived the recheck. ' if audited else 'Review-inclusive labels and unflagged RH labels are shown separately. ')+
        'Only assistant prose is labeled. Each model reuses its own neutral episodes across emotion panels; '
        'model and steered task coverage differ, and these saved sweeps were stopped early.</p>'
        '<p><a href="rates.csv">Rates CSV</a> · <a href="rates.json">Rates JSON</a></p>'+cards+'</html>')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
