"""Common argument parsing for independent experiment entrypoints.

Each experiment invokes this helper with its own type. It never schedules a
different experiment; prerequisites are explicit inputs in the supplied config.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main(experiment):
    parser = argparse.ArgumentParser(description=f"Run {experiment} from a JSON config")
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs and preview work without model downloads or paid compute')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--through', default='verify')
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--reuse-token-cache', action='store_true')
    parser.add_argument('--serve', type=int)
    args = parser.parse_args()
    supplied = json.loads(args.config.read_text())
    if supplied.get('experiment', 'emotion-probes') != experiment:
        parser.error(f'This entrypoint accepts only {experiment} configs')
    if experiment != 'emotion-probes' and (args.smoke_only or args.reuse_token_cache):
        parser.error('Token-cache and extraction-smoke flags apply only to emotion-probes')
    if experiment == 'emotion-steering':
        from emotion_steering.pipeline import main as run
        run(args.config, dry_run=args.dry_run, resume=args.resume,
            serve_port=args.serve, through=args.through)
    elif experiment == 'emotion-rh-forecasting':
        from emotion_rh_forecasting.pipeline import main as run
        if args.through != 'verify':
            parser.error('Forecasting manages its own score/evaluate stages')
        run(args.config, dry_run=args.dry_run, resume=args.resume, serve_port=args.serve)
    elif experiment == 'emotion-self-report':
        from emotion_self_report.pipeline import main as run
        run(args.config, dry_run=args.dry_run, resume=args.resume,
            serve_port=args.serve, through=args.through)
    elif experiment == 'emotion-probes':
        from emotion_probes.config import load_config, resolve, run_root
        from emotion_probes.pipeline import execute, prepare
        from shared.progress import STAGES, Progress, lock
        from shared.server import serve
        if args.through not in STAGES:
            parser.error(f'--through must be one of {STAGES}')
        if args.reuse_token_cache and (not args.resume or args.through in ('prepare', 'tokenize')):
            parser.error('--reuse-token-cache requires --resume and extraction or later')
        cfg = load_config(args.config)
        root = run_root(cfg)
        if args.serve:
            serve(root, args.serve)
            return
        spec = resolve(cfg)
        if args.dry_run:
            print(json.dumps({'run_dir': str(root), 'model': spec['model'],
                'gpu': spec['gpu'], 'counts': spec['counts'], 'probes': 36,
                'identity': spec['identity']}, indent=2))
            return
        with lock(root):
            prepare(root, spec, args.resume)
        if spec['executor'] == 'modal' and (args.smoke_only or STAGES.index(args.through) >= STAGES.index('extract')):
            progress = Progress(root, spec['run_id'])
            progress.data.update(state='waiting', message='Connecting to Modal worker', active_stage='extract')
            progress.publish()
            worker = Path(__file__).resolve().parents[1] / 'emotion-probes/modal_worker.py'
            cmd = [sys.executable, '-m', 'modal', 'run', '--detach', str(worker) + '::main',
                   '--run-dir', str(root), '--through', args.through]
            if args.smoke_only:
                cmd.append('--smoke-only')
            if args.reuse_token_cache:
                cmd.append('--reuse-token-cache')
            family = ('gemma3' if spec['model'].startswith('google/gemma-3-') else
                      'glm52' if spec['model'] == 'nvidia/GLM-5.2-NVFP4' else 'glm53')
            subprocess.run(cmd, check=True, env={**os.environ,
                'EMOTION_PROBE_GPU': spec['gpu'], 'EMOTION_PROBE_MODEL_FAMILY': family})
        else:
            if args.reuse_token_cache:
                from resume_from_extract import execute
            execute(root, args.through, args.smoke_only)
