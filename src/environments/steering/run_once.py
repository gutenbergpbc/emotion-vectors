#!/usr/bin/env python3
"""Run one configured steering experiment: Modal generation, local judging, report.

Edit the JSON config, then run from the model-forensics checkout:
  python3 -m steering.run_once --config steering/configs/qwen35_9b_rh_gamut_six_b32.json
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .run import ROOT, initialize, tasks
from .tracking import read

DEFAULT_CONFIG = ROOT / 'steering/configs/qwen35_9b_rh_gamut_six_b32.json'
JUDGE_ADAPTER = ROOT / 'sandbox/ayush/emotion_vectors/local_reproduction_20261014/runtime/emotion_concepts/scripts/09_judge_steering_run.py'
MODAL_RUNNER = ROOT / 'steering/modal_runner_once.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--judge-model', default='anthropic/claude-haiku-5.5')
    parser.add_argument('--plan-only', action='store_true', help='Print workload and settings without starting generation or judging')
    parser.add_argument('--skip-generation', action='store_true', help='Use already-synced local responses and start at judging')
    parser.add_argument('--skip-judge', action='store_true', help='Stop after generation and artifact sync')
    parser.add_argument('--serve', action='store_true', help='Start the run dashboard in the background')
    parser.add_argument('--port', type=int, default=8790)
    parser.add_argument('--sync-seconds', type=int, default=10)
    parser.add_argument('--sales-traces', type=Path, help='Prepare a sales run from a completed trace folder')
    parser.add_argument('--sales-samples', type=int, default=32, help='Episodes per sales scenario and condition')
    args = parser.parse_args()

    if args.sales_traces:
        from .sales import prepare
        args.config = prepare(args.sales_traces, args.sales_samples)

    config = args.config if args.config.is_absolute() else (ROOT / args.config)
    config = config.resolve()
    spec = read(config)
    is_sales = bool(spec.get('sales'))
    modal_runner = ROOT / 'steering/modal_runner_sales_once.py' if is_sales else MODAL_RUNNER
    run_id = spec['run_id']
    volume = spec.get('modal_result_volume', f'emotion-steering-{run_id}')
    run_dir = ROOT / 'steering/runs' / run_id
    planned = tasks(spec)
    if args.plan_only:
        print(json.dumps({
            'run_id': run_id, 'model': spec['model']['name'],
            'probes': [p['emotion'] for p in spec['probes']],
            'strengths': spec['strengths'], 'batch_size': spec['generation'].get('batch_size'),
            'planned_generations': len(planned),
            'shared_baseline_generations': sum(t['emotion'] is None for t in planned),
            'modal_results_volume': volume, 'local_judge_model': args.judge_model,
        }, indent=2))
        return

    if not args.skip_generation:
        if shutil.which('modal') is None:
            parser.error('Modal CLI is required; install/login to Modal before starting generation')
        if not os.environ.get('OPENROUTER_API_KEY') and not args.skip_judge:
            parser.error('OPENROUTER_API_KEY is required for the automatic local judge stage')
    elif not args.skip_judge and not os.environ.get('OPENROUTER_API_KEY'):
        parser.error('OPENROUTER_API_KEY is required for the local judge stage')

    initialize(config, run_dir)
    if args.serve:
        server = subprocess.Popen(
            [sys.executable, '-m', 'steering.run', 'serve', '--output', str(run_dir), '--port', str(args.port)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)
        print(f'Dashboard: http://127.0.0.1:{args.port}/ (server pid {server.pid})', flush=True)

    if not args.skip_generation:
        env = os.environ.copy()
        env['STEERING_CONFIG'] = str(config)
        proc = subprocess.Popen(['modal', 'run', str(modal_runner)], cwd=ROOT, env=env)
        from .sync_modal import once, download_tree
        while proc.poll() is None:
            try:
                snapshot = once(run_id, volume=volume)
                print(f"Generation {snapshot['state']} · {snapshot.get('message', '')}", flush=True)
            except Exception as error:  # Volume data can lag app startup by a few seconds.
                print(f'Waiting for Modal progress snapshot: {error}', flush=True)
            time.sleep(max(5, args.sync_seconds))
        # Pull the final snapshot/artifacts even on a failed Modal invocation so
        # the dashboard and committed work remain available for resume.
        try:
            download_tree(run_id, ROOT / 'steering/runs', volume)
        except Exception as error:
            print(f'Final Modal artifact sync failed: {error}', file=sys.stderr, flush=True)
        if proc.returncode:
            raise SystemExit(proc.returncode)

    receipts = read(run_dir / 'receipts.json')
    generated = sum(r['stage_id'] == 'generate' for r in receipts.values())
    if generated != len(planned):
        raise SystemExit(f'Generation is incomplete: {generated}/{len(planned)}; judge stage was not started')
    if args.skip_judge:
        print(f'Generation complete: {generated:,}/{len(planned):,}. Run directory: {run_dir}')
        return

    if is_sales:
        from .sales import judge_and_report
        judge_and_report(run_dir, args.judge_model)
    else:
        subprocess.run([sys.executable, str(JUDGE_ADAPTER), '--run-dir', str(run_dir),
                        '--judge-model', args.judge_model], cwd=ROOT, check=True)
    print(f'Generation, local judging, verification, and report complete: {run_dir}', flush=True)


if __name__ == '__main__':
    main()
