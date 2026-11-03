"""Combine judged Qwen emotion-steering rates across the pilot and RH sweep.

Run with:
  uv run --with requests --with matplotlib python steering/plot_qwen_combined_rates.py

Writes a CSV plus one 12-panel figure each for reward hacking and blackmail.
The 0.075 run contains RH tasks only; the pilot contains blackmail at +/-0.05,
and RH at +/-0.05 for all emotions plus +/-0.10 and +/-0.15 for six emotions.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PILOT_ID = 'qwen35-9b-emotions-pilot-20261015'
RH075_ID = 'qwen35-9b-rh-conv1d-075-20261015'
JUDGE_SCRIPT = ROOT / 'sandbox/ayush/emotion_vectors/local_reproduction_20261014/runtime/emotion_concepts/scripts/09_judge_steering_run.py'
JUDGE_MODEL = 'anthropic/claude-haiku-5.5'


def load_module(path):
    spec = importlib.util.spec_from_file_location('qwen_local_judge_adapter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (c - h) / d, (c + h) / d


def add_label(rows, metric, emotion, strength, run_id, label):
    key = (metric, emotion or 'baseline', float(strength))
    cell = rows.setdefault(key, dict(metric=metric, emotion=emotion or 'baseline',
                                     alpha=float(strength), run_id=run_id,
                                     labels={}, n=0, unjudged=0))
    if label == 'unjudged':
        cell['unjudged'] += 1
    else:
        cell['n'] += 1
        cell['labels'][label] = cell['labels'].get(label, 0) + 1


def pilot_rows(run_dir, adapter):
    spec = json.loads((run_dir / 'run_spec.json').read_text())
    _, _, paths = adapter.prepare(run_dir)
    model = JUDGE_MODEL
    b_cache = adapter.cache_rows(run_dir / 'judge_cache/blackmail.jsonl')
    r_cache = adapter.cache_rows(run_dir / 'judge_cache/reward_hacking.jsonl')
    rows = {}
    for record in adapter.BLACKMAIL.collect([str(paths['blackmail_base']), str(paths['blackmail_steered'])]):
        key = adapter.BLACKMAIL.rec_key(record, model)
        judgment = b_cache.get(key)
        if judgment is None:
            raise ValueError(f"Missing blackmail judgment cache entry: {key}")
        emotion = None if record['emotion'] == 'baseline' else record['emotion']
        add_label(rows, 'blackmail', emotion, float(record['alpha']), PILOT_ID,
                  judgment['label'])
    for record in adapter.REWARD.collect([str(paths['reward_steered'])], [str(paths['reward_base'])]):
        key = adapter.REWARD.rec_key(record, model)
        judgment = r_cache.get(key)
        if judgment is None:
            raise ValueError(f"Missing reward-hacking judgment cache entry: {key}")
        emotion = None if record['emotion'] == 'baseline' else record['emotion']
        add_label(rows, 'reward_hacking', emotion, float(record['alpha']), PILOT_ID,
                  judgment['label'])
    return spec, rows


def rh075_rows(run_dir):
    spec = json.loads((run_dir / 'run_spec.json').read_text())
    summary = json.loads((run_dir / 'summary.json').read_text())
    rows = {}
    for group in summary['groups']:
        if group['scenario'] not in ('impossible_sort', 'impossible_prime', 'impossible_search'):
            continue
        cond = group['condition']
        if cond == 'baseline':
            emotion, strength = None, 0.0
        else:
            emotion, value = cond.split('/', 1)
            strength = float(value)
        key = ('reward_hacking', emotion or 'baseline', strength)
        cell = rows.setdefault(key, dict(metric='reward_hacking', emotion=emotion or 'baseline',
                                         alpha=strength, run_id=RH075_ID, labels={}, n=0,
                                         unjudged=0))
        for label, count in group['labels'].items():
            if label == 'unjudged':
                cell['unjudged'] += count
            else:
                cell['n'] += count
                cell['labels'][label] = cell['labels'].get(label, 0) + count
    return spec, rows


def finalize(rows):
    for cell in rows.values():
        k = cell['labels'].get('hack', 0)
        cell['hack_count'] = k
        cell['hack_rate'] = k / cell['n'] if cell['n'] else None
        cell['ci95_low'], cell['ci95_high'] = wilson(k, cell['n'])
    return rows


def figure(metric, rows, emotions, output, matplotlib):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey=True)
    fig.suptitle('Qwen3.5-9B · ' + ('Reward-hacking' if metric == 'reward_hacking' else 'Blackmail') +
                 ' rate by emotion and steering strength', fontsize=18, y=0.985)
    fig.text(0.5, 0.948,
             'Observed proportions with 95% Wilson intervals · points shown only at tested strengths',
             ha='center', fontsize=10, color='#555')
    color = '#cf5648' if metric == 'reward_hacking' else '#317fa8'
    baseline = rows.get((metric, 'baseline', 0.0))
    for ax, emotion in zip(axes.flat, emotions):
        values = sorted((c for c in rows.values() if c['metric'] == metric and c['emotion'] == emotion),
                        key=lambda c: c['alpha'])
        if baseline and baseline['n']:
            rate = baseline['hack_rate']
            ax.axhline(rate, color='#777', lw=1, ls='--', alpha=.65)
            ax.errorbar([0], [rate], yerr=[[max(0, rate-baseline['ci95_low'])],
                                            [max(0, baseline['ci95_high']-rate)]],
                        fmt='D', ms=4, color='#333', capsize=2, zorder=5)
        if values:
            xs = [c['alpha'] for c in values]
            ys = [c['hack_rate'] for c in values]
            lows = [max(0, y-c['ci95_low']) for y, c in zip(ys, values)]
            highs = [max(0, c['ci95_high']-y) for y, c in zip(ys, values)]
            ax.errorbar(xs, ys, yerr=[lows, highs], fmt='o-', color=color, lw=1.5,
                        ms=4.5, capsize=2, zorder=4)
        ax.set_title(emotion.capitalize(), fontsize=12)
        ax.set_xlim(-0.17, 0.17)
        ax.set_xticks([-0.15, -0.10, -0.075, -0.05, 0, 0.05, 0.075, 0.10, 0.15])
        ax.set_xticklabels(['−.15','−.10','−.075','−.05','0','+.05','+.075','+.10','+.15'],
                           rotation=35, ha='right', fontsize=8)
        ax.grid(axis='y', color='#ddd', lw=.7)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel('Steering strength α')
    for ax in axes[:, 0]:
        ax.set_ylabel('Judge-labeled rate')
    # Keep rate scale shared within a metric while avoiding a zero-height figure.
    upper = max((c['ci95_high'] for c in rows.values() if c['metric'] == metric and c['n']),
                default=0.05)
    if baseline:
        upper = max(upper, baseline['ci95_high'])
    ylim = min(1.0, max(0.12, upper * 1.18))
    for ax in axes.flat:
        ax.set_ylim(0, ylim)
    handles = [plt.Line2D([], [], color=color, marker='o', label='Steered condition'),
               plt.Line2D([], [], color='#333', marker='D', ls='--', label='Shared pilot baseline')]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, .035))
    if metric == 'reward_hacking':
        note = 'RH: n=90 per emotion/strength (3 tasks × 30); shared pilot baseline n=360. One unjudged response is excluded from its cell denominator.'
    else:
        note = 'Blackmail: n=50 per emotion/strength; shared pilot baseline n=200. Only ±0.05 was tested for blackmail.'
    fig.text(.5, .012, note, ha='center', fontsize=8, color='#555')
    fig.tight_layout(rect=(0.02, .08, .99, .92))
    fig.savefig(output.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(output.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pilot', type=Path, default=ROOT / 'steering/runs' / PILOT_ID)
    p.add_argument('--rh075', type=Path, default=ROOT / 'steering/runs' / RH075_ID)
    p.add_argument('--output', type=Path, default=ROOT / 'steering/runs' / PILOT_ID / 'combined_qwen_rates')
    args = p.parse_args()
    if not (args.pilot / 'judge_cache/reward_hacking.jsonl').exists():
        raise SystemExit('Pilot judge cache is missing')
    adapter = load_module(JUDGE_SCRIPT)
    pilot_spec, pilot = pilot_rows(args.pilot.resolve(), adapter)
    sweep_spec, sweep = rh075_rows(args.rh075.resolve())
    emotions = [p['emotion'] for p in pilot_spec['probes']]
    # Use the larger pilot baseline for both figures; do not pool a second
    # shared baseline with overlapping task/sample seeds from the 0.075 run.
    all_rows = finalize({**pilot, **{k:v for k,v in sweep.items() if v['emotion'] != 'baseline'}})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ordered_rows = sorted(all_rows.values(), key=lambda x:(x['metric'],x['emotion'] != 'baseline',x['emotion'],x['alpha']))
    combined = {
        'schema_version': 1,
        'model': 'Qwen/Qwen3.5-9B',
        'judge_model': JUDGE_MODEL,
        'baseline_policy': 'Use the pilot baseline only; the 0.075 run baseline is omitted to avoid pooling overlapping samples.',
        'runs': {
            PILOT_ID: {'source': 'pilot', 'coverage': 'Blackmail and reward hacking at +/-0.05; reward hacking at +/-0.10 and +/-0.15 for six emotions.'},
            RH075_ID: {'source': 'causal-conv1d sweep', 'coverage': 'Reward hacking at +/-0.075 for six emotions.'},
        },
        'conditions': [
            {
                'metric': c['metric'], 'emotion': c['emotion'], 'alpha': c['alpha'],
                'run_id': c['run_id'], 'labels': c['labels'],
                'hack_count': c['hack_count'], 'n_judged': c['n'],
                'unjudged': c['unjudged'], 'rate': c['hack_rate'],
                'ci95_wilson': [c['ci95_low'], c['ci95_high']],
            }
            for c in ordered_rows
        ],
        'limitations': [
            'Blackmail is available only at +/-0.05.',
            'Reward-hacking +/-0.10 and +/-0.15 are available only for the six emotions listed in the run spec.',
            'One unjudged reward-hacking response is excluded from its condition denominator.',
            'Intervals are Wilson binomial intervals over judged responses in each fixed prompt/task set; they do not capture scenario-sampling uncertainty.',
            'Reward-hacking and blackmail labels are judge classifications, not definitive ground truth.',
        ],
    }
    args.output.with_suffix('.json').write_text(json.dumps(combined, indent=2) + '\n')
    fields = ['metric','emotion','alpha','run_id','hack_count','n','unjudged','hack_rate','ci95_low','ci95_high']
    with args.output.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        baseline_rows = [c for c in all_rows.values() if c['emotion'] == 'baseline']
        expanded = {p['emotion'] for p in sweep_spec['probes']}
        for c in ordered_rows:
            writer.writerow({k:c.get(k) for k in fields})
        notes = {
            'model': 'Qwen/Qwen3.5-9B',
            'pilot_run': PILOT_ID,
            'rh_0_075_run': RH075_ID,
            'judge_model': JUDGE_MODEL,
            'reward_hacking_emotions_with_0_1_and_0_15': sorted(expanded),
            'reward_hacking_per_condition_n': 90,
            'blackmail_per_condition_n': 50,
            'baselines': {c['metric']: {'n': c['n'], 'labels': c['labels'], 'run_id': c['run_id']}
                          for c in baseline_rows},
            'limitation': 'Blackmail is only available at +/-0.05; 0.075 is RH-only; +/-0.10 and +/-0.15 are RH-only and only for the six listed emotions. Unjudged responses are excluded from rates and shown in CSV.',
        }
    args.output.with_name(args.output.name + '_metadata.json').write_text(json.dumps(notes, indent=2) + '\n')
    figure('reward_hacking', all_rows, emotions, args.output.with_name(args.output.name + '_reward_hacking'), None)
    figure('blackmail', all_rows, emotions, args.output.with_name(args.output.name + '_blackmail'), None)
    print('Output stem:', args.output)
    print('Reward-hacking conditions:', sum(c['metric']=='reward_hacking' and c['emotion']!='baseline' for c in all_rows.values()))
    print('Blackmail conditions:', sum(c['metric']=='blackmail' and c['emotion']!='baseline' for c in all_rows.values()))
    print('Unjudged labels:', sum(c['unjudged'] for c in all_rows.values()))


if __name__ == '__main__':
    main()
