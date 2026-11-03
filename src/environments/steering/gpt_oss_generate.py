"""GPT-OSS generation loop that packs different task prompts into H200 batches."""
from itertools import groupby
import time


def generate(run):
    import torch
    from .backend import load_vector
    from .gpt_oss_backend import make_backend
    from .run import ROOT, check_spec, tasks
    from .tracking import atomic, now

    spec = run.spec
    check_spec(run)
    run.stage('preflight', 'Checking frozen GPT-OSS vectors, hashes, and H200 backend')
    backend = make_backend(spec, ROOT)
    vectors = {}
    for probe in spec['probes']:
        vector = load_vector(probe, ROOT)
        if len(vector) != backend.hidden_size or not 0 <= probe['layer_zero_based'] < backend.num_layers:
            raise ValueError(f"Incompatible GPT-OSS probe: {probe['emotion']}")
        vectors[probe['emotion']] = vector
    if run.start('preflight', 'inputs'):
        path = run.root / 'preflight.json'
        atomic(path, dict(model=spec['model'], hidden_size=backend.hidden_size,
                          layers=backend.num_layers, gpu='H200',
                          attention=spec['model']['runtime']['attention'],
                          verified_at=now()))
        run.commit('preflight', 'inputs', path)
    run.finish_stage('preflight')

    run.stage('generate', 'Generating GPT-OSS responses in cross-scenario batches')
    probes = {probe['emotion']: probe for probe in spec['probes']}
    batch_size = int(spec['generation'].get('batch_size', 32))
    fallback_size = int(spec['generation'].get('oom_fallback_batch_size', 32))
    missing = [task for task in tasks(spec) if f"generate/{task['id']}" not in run.receipts]
    # A condition has 32 examples for each of three prompts. Pool prompts so a
    # 64-row H200 batch can use the remaining memory without increasing n.
    missing.sort(key=lambda task: (task['condition'], task['sample'] // batch_size,
                                   task['scenario'], task['sample']))

    def save_batch(batch, started=True):
        if started:
            started_tasks = []
            for task in batch:
                context = {key: task[key] for key in
                           ('emotion', 'strength', 'scenario', 'sample', 'layer')}
                if run.start('generate', task['id'], context):
                    started_tasks.append(task)
            batch = started_tasks
        if not batch:
            return
        first = batch[0]
        condition = first['condition']
        sample_group = min(task['sample'] for task in batch) // batch_size
        # Use a stable, condition-independent group seed across prompts.
        common_seed = spec['seed'] + sample_group * batch_size
        for task in batch:
            task['seed'] = common_seed
        run.stage('generate', f"{condition} · pooled prompts · batch of {len(batch)}")
        started_at = time.perf_counter()
        try:
            results = backend.sample_batch(batch, probes.get(first['emotion']),
                                           vectors.get(first['emotion']))
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or 'out of memory' in str(error).lower()
            if not is_oom or len(batch) <= fallback_size:
                raise
            torch.cuda.empty_cache()
            run.event('generation.batch_oom_fallback', batch_size=len(batch),
                      fallback_batch_size=fallback_size, condition=condition)
            for offset in range(0, len(batch), fallback_size):
                save_batch(batch[offset:offset + fallback_size], started=False)
            return
        elapsed = time.perf_counter() - started_at
        if len(results) != len(batch):
            raise RuntimeError('GPT-OSS backend returned the wrong batch size')
        token_total = sum(result.get('generated_tokens', 0) for result in results)
        artifacts = []
        for task, result in zip(batch, results):
            result['batch_elapsed_seconds'] = elapsed
            result['batch_size'] = len(batch)
            result['batch_responses_per_second'] = len(batch) / elapsed if elapsed else None
            result['batch_tokens_per_second'] = token_total / elapsed if elapsed else None
            path = run.root / 'responses' / (task['id'] + '.json')
            atomic(path, dict(**task, **result, model=spec['model'],
                              steering_mode=spec['steering_mode']))
            artifacts.append((task['id'], path, {'scenario': task['scenario'],
                                                 'sample': task['sample']}))
        run.commit_batch('generate', artifacts)

    for _, grouped in groupby(missing, key=lambda task: (task['condition'], task['sample'] // batch_size)):
        group = list(grouped)
        for offset in range(0, len(group), batch_size):
            save_batch(group[offset:offset + batch_size])
    run.finish_stage('generate')
