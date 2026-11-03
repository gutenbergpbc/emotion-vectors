"""Measure capture repeatability without changing the frozen experiment checks."""

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
import json
import time

import modal
from modal_worker import image, models, runs, kernels, runtime_details
from shared.io import atomic, digest, read, require, save_npz, sha
from shared.progress import lock, now

app = modal.App('canonical-glm53-capture-diagnostic')


def difference(a, b):
    import numpy as np
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    an, bn = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    relative = np.linalg.norm(a-b, axis=1) / np.maximum(an, 1e-12)
    cosine = (a*b).sum(axis=1) / np.maximum(an*bn, 1e-24)
    return {'max_relative_l2': float(relative.max()), 'per_layer_relative_l2': relative.tolist(),
            'min_cosine': float(cosine.min()), 'per_layer_cosine': cosine.tolist(),
            'max_absolute_difference': float(np.abs(a-b).max()),
            'layers_over_original_limit': np.flatnonzero(relative > 1e-5).tolist()}


@app.function(image=image, gpu='B200:4', cpu=16, memory=131072, timeout=3600,
              volumes={'/models': models, '/data': runs, '/kernel-cache': kernels}, max_containers=1)
def diagnose(remote_name: str, attempt: str):
    import numpy as np
    from vllm import SamplingParams
    from vllm_lens import Hook
    from emotion_probes.config import source_hashes
    from emotion_probes.extraction import LensExtractor, make_capture_fn

    root = Path('/data') / remote_name
    spec = read(root / 'config.resolved.json')
    require(spec['source_hashes'] == source_hashes(), 'Frozen scientific source changed')
    folder = root / 'logs' / attempt
    report = {'identity': spec['identity'], 'started_at': now(), 'state': 'loading',
              'diagnostic_source_sha256': sha(Path(__file__)), 'runtime': runtime_details(),
              'original_relative_l2_limit': 1e-5, 'repeatability': [],
              'note': 'Diagnostic only; no change to acceptance thresholds or probe outputs'}
    def publish():
        atomic(folder / 'report.json', report)
        runs.commit()

    with lock(root):
        try:
            publish()
            extractor = LensExtractor(spec)
            report.update(state='measuring', model_loaded_at=now())
            publish()
            records = [(part, i, r) for part in ('training', 'eval')
                       for i, r in enumerate(read(root / 'tokens' / part / 'manifest.json')['records'])]
            failed = next(x for x in records if x[2]['id'] == 'guilty::guilty-039::guilty')
            shortest = min(records, key=lambda x: x[2]['prefix_token_count'])
            longest = max(records, key=lambda x: x[2]['prefix_token_count'])
            for label, (part, index, _) in [('failed_case', failed), ('longest_case', longest)]:
                record = read(root / 'tokens' / part / f'{index:05d}.json')
                require(record['identity'] == spec['identity'] and
                        record['token_ids_sha256'] == digest(record['input_ids']), 'Invalid cached tokens')
                captures = []
                entry = {'case': record['id'], 'prefix_tokens': len(record['input_ids']),
                         'seconds': [], 'comparisons': []}
                report['repeatability'].append(entry)
                for repeat in range(3):
                    start = time.monotonic()
                    captures.append(extractor.capture(record))
                    entry['seconds'].append(time.monotonic()-start)
                    save_npz(folder / f'{label}.npz', captures=np.stack(captures))
                    if repeat:
                        entry['comparisons'].append({'a': repeat-1, 'b': repeat,
                                                     **difference(captures[-2], captures[-1])})
                    publish()

            # Capture raw residuals and pooled values during the SAME forward
            # pass so model nondeterminism cannot masquerade as a pooling bug.
            part, index, _ = shortest
            record = read(root / 'tokens' / part / f'{index:05d}.json')
            all_layers = list(range(spec['layers']))
            layers = sorted({0, spec['layers']//2, spec['layers']-1})
            lo, hi = record['target_token_span']
            hook = Hook(fn=make_capture_fn(lo, hi, all_layers, spec['width']),
                        layer_indices=all_layers, pre=False)
            output, = extractor.llm.generate([{'prompt_token_ids': record['input_ids']}],
                SamplingParams(temperature=0, max_tokens=1, ignore_eos=True,
                    extra_args={'apply_hooks': [hook], 'output_residual_stream': layers}), use_tqdm=False)
            raw = output.activations['residual_stream'].float().cpu().numpy()
            require(raw.shape[0] == len(layers) and raw.shape[1] >= hi and raw.shape[2] == spec['width'],
                    'Unexpected raw residual shape')
            result = output.hook_results
            result = result.get('0', result.get(0, result))
            actual = np.asarray(result['means'], dtype=np.float32)[layers]
            expected = raw[:, lo:hi].mean(axis=1)
            save_npz(folder / 'same-forward-pooling.npz', actual=actual, expected=expected)
            report['same_forward_pooling'] = {'layers': layers, 'case': record['id'],
                                             **difference(expected, actual)}
            report.update(state='complete', finished_at=now())
            publish()
            return report
        except Exception as exc:
            report.update(state='failed', error=str(exc), finished_at=now())
            publish()
            raise
        finally:
            kernels.commit()


@app.local_entrypoint()
def main(run_dir: str):
    root = Path(run_dir).resolve()
    spec = read(root / 'config.resolved.json')
    remote_name = root.name + '-' + spec['identity'][:12]
    attempt = f'capture-diagnostic-{time.time_ns()}'
    call = diagnose.spawn(remote_name, attempt)
    atomic(root / 'logs/capture-diagnostic-connection.json',
           {'call_id': call.object_id, 'url': call.get_dashboard_url(), 'attempt': attempt, 'started_at': now()})
    print(call.get_dashboard_url(), flush=True)
    while True:
        try:
            data = json.loads(b''.join(runs.read_file(f'/{remote_name}/logs/{attempt}/report.json')))
            atomic(root / 'logs' / attempt / 'report.json', data)
        except (FileNotFoundError, modal.exception.NotFoundError):
            pass
        try:
            result = call.get(timeout=10)
            break
        except TimeoutError:
            continue
    atomic(root / 'logs' / attempt / 'report.json', result)
    print(json.dumps(result), flush=True)
