"""Combine frozen emotion projections; run after a verified canonical forecast run."""

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
import hashlib
import json
from pathlib import Path
import sys
import warnings
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning
from emotion_rh_forecasting.analysis import EMOTIONS, make_pairs, balanced_groups, draw_balanced, fit_threshold, predict_threshold
from emotion_rh_forecasting.analysis import assign_folds
from emotion_rh_forecasting.inputs import Sources, load_probes, vector_identity
from shared.io import read, require

ROOT = Path(__file__).resolve().parents[3]
CUSTOM = {
    'gemma3-4b-sales-gpt55-v1': ('emotion-rh-sentence-labels/gemma3-4b-sales-gpt55-v1', 'gemma3-4b-sales-best12-v1', 'sales'),
    'gemma3-4b-coding-lines-v1': ('emotion-rh-code-line-labels/gemma3-4b-coding-v1', 'gemma3-4b-coding-lines-best12-v1', 'coding'),
    'glm-5.3-coding-1000-gpt55-v1': ('emotion-rh-sentence-labels/glm53-coding-1000-gpt55-v1', 'glm-5.3-coding-1000-best12-v1', 'coding'),
}
GLM_BALANCED_SALES = 'glm-5.3-balanced-source-best12-v1'

def is_glm_balanced_sales(root):
    return root.name == 'sales' and root.parent.name == GLM_BALANCED_SALES

def load_glm_balanced_sales(root):
    from plot_glm_hack_forecast import rows_for_dataset, frozen_folds
    result = read(root.parent / 'summary.json')
    require(result['status'] == 'complete', 'Individual forecast incomplete')
    sources = result['sources']
    require(len(sources) == 2, 'Expected hack and nonhack source cohorts')
    rows = []
    identities = set()
    for source in sources:
        labels_path = Path(source['labels_path'])
        replay = Path(source['replay_path'])
        labels, spec = read(labels_path), read(replay / 'replay-spec.json')
        require(labels['summary']['status'] == 'complete', 'Labels incomplete')
        require(sha(labels_path) == source['labels_sha256'] == spec['labels_sha256'], 'Labels changed')
        require(spec['identity'] == source['replay_identity'], 'Replay identity changed')
        bundle = load_probes(replay / 'probes.json', Sources())
        require(spec['probe_identity'] == vector_identity(bundle) == result['probe_identity'], 'Probe identity mismatch')
        require(spec['probes_sha256'] == sha(replay / 'probes.json'), 'Probe file changed')
        require(len(list((replay / 'scores').glob('*.receipt.json'))) == len(spec['jobs']), 'Replay incomplete')
        rows.extend(rows_for_dataset('sales', labels, spec, bundle, replay))
        identities.add(spec['probe_identity'])
    require(len(identities) == 1, 'Source probe bundles differ')
    require(len(rows) == result['datasets']['sales']['sentences'], 'Individual forecast population changed')
    folds = frozen_folds(rows, 'sales')
    require(folds == read(root / 'metrics.json')['folds'], 'Individual forecast folds changed')
    identity = {'individual_summary_sha256': sha(root.parent / 'summary.json'),
                'individual_metrics_sha256': sha(root / 'metrics.json'),
                'source_labels_sha256': {str(Path(s['labels_path'])): s['labels_sha256'] for s in sources},
                'source_replay_identities': {str(Path(s['replay_path'])): s['replay_identity'] for s in sources}}
    return rows, dict(title='GLM 5.3 Sales', seed=20261021, repeats=2000, horizons=list(range(11))), identity

def load_custom(root):
    from plot_glm_hack_forecast import rows_for_dataset
    label_rel, replay_name, dataset = CUSTOM[root.name]
    labels_path = ROOT / 'runs' / label_rel / 'labels.json'
    replay = ROOT / 'runs/emotion-rh-replay' / replay_name
    labels, spec = read(labels_path), read(replay / 'replay-spec.json')
    require(labels['summary']['status'] == 'complete', 'Labels incomplete')
    bundle = load_probes(replay / 'probes.json', Sources())
    require(spec['labels_sha256'] == sha(labels_path) and spec['probe_identity'] == vector_identity(bundle), 'Label/probe identity mismatch')
    require(len(list((replay / 'scores').glob('*.receipt.json'))) == len(spec['jobs']), 'Replay incomplete')
    if root.name == 'gemma3-4b-coding-lines-v1':
        from plot_gemma4b_coding_line_forecast import adapted
        labels = {**labels, 'traces': [adapted(t) for t in labels['traces']]}
    rows = rows_for_dataset(dataset, labels, spec, bundle, replay)
    if root.name == 'glm-5.3-coding-1000-gpt55-v1':
        from plot_glm_coding_1000 import TASK_FOLDS
        require({r['task'] for r in rows} == set(TASK_FOLDS), 'GLM task population changed')
        for row in rows:
            row['group_id'] = row['task']
            row['fold'] = TASK_FOLDS[row['task']]
        assign_folds(rows, {'method': 'provided'}, 20261022)
    else:
        assign_folds(rows, {'method': 'held_out_task'}, 20261022)
    summary = read(root / 'summary.json')
    require(summary['probe_identity'] == spec['probe_identity'] and summary['labels_sha256'] == sha(labels_path), 'Individual forecast identity mismatch')
    require(len(rows) == summary.get('sentences', summary.get('code_lines')), 'Individual forecast population changed')
    return rows, dict(title=root.name.replace('-', ' ').replace('_', ' ').title(), seed=20261022, repeats=2000, horizons=list(range(11))), {
        'labels_sha256': sha(labels_path), 'replay_spec_sha256': sha(replay / 'replay-spec.json'),
        'probe_sha256': sha(replay / 'probes.json'), 'individual_metrics_sha256': sha(root / 'metrics.json')}

def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(root):
    custom = root.name in CUSTOM or is_glm_balanced_sales(root)
    if custom:
        rows, config, identity = (load_glm_balanced_sales(root) if is_glm_balanced_sales(root)
                                  else load_custom(root))
        reference_metrics = read(root/'metrics.json')
        reference = {(e['cohort'],e['horizon'],e['model']): e for e in reference_metrics['estimates']}
        populations_ref = {(p['cohort'],p['horizon']): p for p in reference_metrics['populations']}
    else:
        receipt=json.loads((root/'verification.json').read_text())
        assert receipt['passed']
        assert all(sha(root/p)==h for p,h in receipt['output_hashes'].items())
        rows=json.loads((root/'sentence-features.json').read_text())
        config=json.loads((root/'config.resolved.json').read_text())
        identity=receipt['identity']
    out=root/'combined-all12';out.mkdir(exist_ok=True)
    assert config['horizons']==list(range(11))
    seed=config['seed']; repeats=config['repeats']
    split_description = 'trace-disjoint folds' if is_glm_balanced_sales(root) else 'task-disjoint folds'
    unit='code line' if root.name == 'gemma3-4b-coding-lines-v1' else 'prose sentence'
    plan=dict(features=EMOTIONS,model='Training-fold StandardScaler; L2 logistic regression C=1, class_weight=balanced, threshold=0.5; no tuning.',target=f'RH at original {unit} i+k from completed-token emotion scores at i',evaluation='Frozen source-group-disjoint folds; within-fold class balancing; minority-count-weighted balanced accuracy',interval='95% majority-class subsampling variability, not generalization confidence',seed=seed,repeats=repeats)
    save(out/'analysis-plan.json',plan)
    common={(r['trace_id'],r['unit_id']) for r in make_pairs(rows,10)}
    estimates=[];populations=[]
    cohorts = config.get('cohorts', ['all_available', 'common_sources', 'certain_only'])
    for cohort in cohorts:
        for k in range(11):
            pairs=make_pairs(rows,k,allowed=common if cohort=='common_sources' else None,certain_only=cohort=='certain_only')
            if custom and (cohort,k) not in populations_ref:
                continue
            y=np.array([r['hacking'] for r in pairs],bool);folds=np.array([r['fold'] for r in pairs])
            x=np.array([[r['projections'][e] for e in EMOTIONS] for r in pairs]);pos=np.array([r['mean_token_position'] for r in pairs])
            assert np.isfinite(x).all()
            groups=balanced_groups(y,folds);pred=np.zeros((3,len(y)),bool);prob=np.zeros(len(y));rules=[]
            for f in sorted(set(folds)):
                tr=np.flatnonzero(folds!=f);te=np.flatnonzero(folds==f)
                task_disjoint = (custom and not is_glm_balanced_sales(root)) or (not custom and config['evaluation']['method']=='held_out_task')
                for key in ['trace_id','group_id'] + (['task'] if task_disjoint else []):
                    assert not {pairs[i][key] for i in tr}&{pairs[i][key] for i in te},key
                clf=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=5000,random_state=seed))
                with warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning);clf.fit(x[tr],y[tr])
                scaler,lr=clf.steps[0][1],clf.steps[1][1]
                prob[te]=clf.predict_proba(x[te])[:,1];pred[0,te]=prob[te]>=.5
                rule=fit_threshold(pos[tr],y[tr]);pred[1,te]=predict_threshold(pos[te],rule)
                # Independently reproduce predictions from exported parameters.
                logits=((x[te]-scaler.mean_)/scaler.scale_)@lr.coef_[0]+lr.intercept_[0]
                np.testing.assert_allclose(prob[te],1/(1+np.exp(-logits)),atol=1e-12)
                rules.append(dict(fold=int(f),mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coef=lr.coef_[0].tolist(),intercept=float(lr.intercept_[0]),token_position=rule,train_trace_ids=sorted({pairs[i]['trace_id'] for i in tr}),test_trace_ids=sorted({pairs[i]['trace_id'] for i in te})))
            if custom:
                assert len(pairs)==populations_ref[cohort,k]['eligible_pairs']
                assert sum(2*n for a,b,n in groups)==populations_ref[cohort,k]['balanced_eval_pairs']
                expected_position=reference[cohort,k,'token_position']['repeated_sampling_mean_accuracy']
            else:
                with np.load(root/'results'/cohort/f'h{k:02d}'/'oof-predictions.npz') as ref_npz:
                    np.testing.assert_array_equal(y,ref_npz['labels'])
                    np.testing.assert_array_equal(folds,ref_npz['folds'])
                    np.testing.assert_array_equal(pred[1],ref_npz['predictions'][list(ref_npz['models']).index('token_position')])
            rng=np.random.default_rng(seed+1);samples=np.array([(pred[:,ix]==y[ix]).mean(axis=1) for ix in (draw_balanced(groups,rng) for _ in range(repeats))])
            if custom:
                np.testing.assert_allclose(samples[:,1].mean(), expected_position, atol=1e-12)
            weights=np.array([n for a,b,n in groups],float);weights/=weights.sum()
            for j,name in enumerate(['all_12','token_position','always_not_hacking']):
                expected=float(weights@np.array([.5*(pred[j,a].mean()+(~pred[j,b]).mean()) for a,b,n in groups]))
                estimates.append(dict(cohort=cohort,k=k,model=name,accuracy=expected,sampling_interval95=np.percentile(samples[:,j],[2.5,97.5]).tolist()))
            directory=out/cohort/f'k{k:02d}';save(directory/'fold-models.json',rules)
            save(directory/'pairs.json',[{key:r[key] for key in ['trace_id','group_id','task','unit_id','target_unit_id','sentence_index','target_sentence_index','fold','hacking']} for r in pairs])
            np.savez_compressed(directory/'oof-predictions.npz',labels=y,folds=folds,probabilities=prob,predictions=pred,sampling_accuracies=samples)
            populations.append(dict(cohort=cohort,k=k,pairs=len(y),rh=int(y.sum()),balanced_pairs=sum(2*n for a,b,n in groups)))
            print(cohort,k,round(estimates[-3]['accuracy'],4),flush=True)
    bundle=dict(plan=plan,estimates=estimates,populations=populations,input_identity=identity,input_sha256=sha(root/'sentence-features.json') if not custom else identity['individual_metrics_sha256'],source_sha256=sha(Path(__file__)))
    save(out/'result-bundle.json',bundle)
    for cohort in cohorts:
        fig,ax=plt.subplots(figsize=(11,7));fig.subplots_adjust(left=.12,right=.97,top=.86,bottom=.27)
        for name,color,style,label in [('all_12','#2d70b3','-o','All 12 emotions'),('token_position','#777777',':','Token-position cutoff')]:
            es=[r for r in estimates if r['cohort']==cohort and r['model']==name]
            ax.plot([r['k'] for r in es],[r['accuracy']*100 for r in es],style,color=color,lw=3,label=label)
            if name=='all_12':ax.fill_between([r['k'] for r in es],[r['sampling_interval95'][0]*100 for r in es],[r['sampling_interval95'][1]*100 for r in es],alpha=.12,color=color)
        ax.axhline(50,color='#b57428',ls='--',label='Always non-hacking / chance')
        ax.set(xlabel='Sentence offset k',ylabel='Balanced-evaluation accuracy (%)',xticks=range(11),ylim=(25,100));ax.grid(axis='y',alpha=.25)
        ax.legend(frameon=False,loc='best');fig.suptitle(config['title']+' · all 12 emotions',fontsize=18)
        counts=[r['balanced_pairs'] for r in populations if r['cohort']==cohort]
        fig.text(.12,.17,'Balanced n (k=0 → 10): '+' / '.join(map(str,counts)),fontsize=10)
        fig.text(.12,.10,f"{len({r['trace_id'] for r in rows})} traces; {split_description}; "+cohort.replace('_',' ')+'.',fontsize=10)
        fig.text(.12,.035,f'Band: majority-class subsampling variability, not generalization confidence.\nk=0 classifies the current {unit}; k>0 predicts a later {unit}. Labels are retrospective.',fontsize=9)
        for ext in ['png','svg','pdf']:fig.savefig(out/f'{cohort}.{ext}',dpi=180,facecolor='white')
        plt.close(fig)
    with (out/'metrics.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(estimates[0]));writer.writeheader();writer.writerows(estimates)
    table=''.join('<tr><td>'+str(k)+'</td>'+''.join('<td>'+format(next(r['accuracy'] for r in estimates if r['cohort']=="all_available" and r['k']==k and r['model']==m)*100,".1f")+'%</td>' for m in ['all_12','token_position'])+'</tr>' for k in range(11))
    fixed_source_section = '<h2>Fixed source cohort</h2><img src="common_sources.png">' if 'common_sources' in cohorts else ''
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Emotion-based RH forecasting</title><style>body{font:18px system-ui;max-width:1100px;margin:40px auto}img{width:100%}td,th{padding:8px 20px;text-align:right}p{line-height:1.5}</style><h1>'+html.escape(config['title'])+' · all 12 emotions</h1><p>'+html.escape(plan['model'])+'</p><p>'+str(len({r['trace_id'] for r in rows}))+' traces. Same frozen probes, labels, source/target pairs and '+split_description+' as the individual curves. k=0 classifies the current '+unit+'; positive offsets target exactly k later '+unit+'s. Later positives can reflect ongoing RH.</p><p>Bands show subsampling variability, not generalization confidence intervals.</p><img src="all_available.png"><table><tr><th>k</th><th>12 emotions</th><th>Token position</th></tr>'+table+'</table>'+fixed_source_section+'<h2>Review-flagged endpoints excluded</h2><img src="certain_only.png"><p><a href="../report.html">Individual probe report</a> · <a href="result-bundle.json">Result bundle</a> · <a href="metrics.csv">Metrics CSV</a></p>')
    save(out/'verification.json',dict(passed=True,checks=['Canonical input receipt and all hashes verified',split_description,'Original sentence offsets preserved','Training-only scaling and fixed hyperparameters','Exported parameters reproduce probabilities','All cohorts and horizons have both classes in every fold'],output_hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='verification.json'}))
    (out/'README.md').write_text('Frozen emotion probes predict retrospective RH labels. No residual-stream RH probe is used.\n\n'+plan['model']+'\n\n'+plan['evaluation']+'. Bands: '+plan['interval']+'.\n\nk=0 is current-unit classification. k>0 predicts exactly k later eligible units, including across assistant turns. Future positives can reflect ongoing RH. Gemma 4B coding uses code lines rather than prose sentences.\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_dir',type=Path);main(p.parse_args().run_dir.resolve())
