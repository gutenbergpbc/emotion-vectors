"""Generate a resumable, frozen GLM coding cohort over additional task families."""

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
import asyncio
from collections import Counter
import json
import os
from pathlib import Path

import aiohttp

from generate_openrouter_traces import completion, digest, save, stamp


ROOT = Path(__file__).resolve().parents[3]
MODEL = 'z-ai/glm-5.3'


async def main(args):
    key = os.environ.get('OPENROUTER_API_KEY')
    if not key:
        raise SystemExit('OPENROUTER_API_KEY is missing')
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    scenarios = config['scenarios']
    assert config['model'] == MODEL and len({x['id'] for x in scenarios}) == len(scenarios)
    out = Path(args.output).resolve()
    spec = {'model': MODEL, 'config': str(config_path), 'config_sha256': digest(config_path),
            'count_per_scenario': args.count_per_scenario, 'scenario_ids': [x['id'] for x in scenarios],
            'temperature': config['temperature'], 'max_tokens': config['max_tokens'],
            'selection': 'all scenarios, equal number of independent unsteered samples; no labels used'}
    if (out / 'run-spec.json').exists():
        assert json.loads((out / 'run-spec.json').read_text()) == spec, 'Run spec changed'
    else:
        save(out / 'run-spec.json', spec)
    jobs = [(scenario, index) for scenario in scenarios for index in range(args.count_per_scenario)]
    timeout = aiohttp.ClientTimeout(total=300)
    semaphore = asyncio.Semaphore(args.concurrency)
    failures = []
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=args.concurrency)) as session:
        async def one(scenario, index):
            path = out / 'traces' / scenario['id'] / f'{index:04d}.json'
            if path.exists():
                saved = json.loads(path.read_text())
                assert saved['status'] == 'complete' and saved['prompt'] == scenario['prompt']
                return
            async with semaphore:
                try:
                    messages = [{'role': 'user', 'content': scenario['prompt']}]
                    result = await completion(session, key, MODEL, messages, config['max_tokens'])
                    from generate_openrouter_traces import reasoning_text
                    trace = {'id': f"coding/{scenario['id']}/{index:04d}", 'dataset': 'coding',
                             'scenario': scenario['id'], 'sample': index, 'model_requested': MODEL,
                             'created_at': stamp(), 'status': 'complete', 'prompt': scenario['prompt'],
                             'messages': messages, 'completion': result,
                             'reasoning_text': reasoning_text(result['message']),
                             'response': result['message'].get('content') or ''}
                    save(path, trace)
                    print('saved', trace['id'], flush=True)
                except Exception as exc:
                    failures.append({'id': f"coding/{scenario['id']}/{index:04d}", 'error': str(exc)[:500]})
                    print('FAILED', scenario['id'], index, str(exc)[:200], flush=True)
        await asyncio.gather(*(one(s, i) for s, i in jobs))
    saved = list((out / 'traces').glob('*/*.json'))
    counts = Counter(json.loads(path.read_text())['scenario'] for path in saved)
    summary = {'updated_at': stamp(), 'target': len(jobs), 'saved': len(saved),
               'by_scenario': dict(counts), 'failures': failures,
               'reported_api_cost_usd': sum((json.loads(path.read_text())['completion'].get('usage') or {}).get('cost', 0) or 0 for path in saved)}
    save(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    if failures or len(saved) != len(jobs):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--count-per-scenario', type=int, required=True)
    parser.add_argument('--concurrency', type=int, default=12)
    asyncio.run(main(parser.parse_args()))
