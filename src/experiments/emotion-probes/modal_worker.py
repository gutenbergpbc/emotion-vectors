"""Durable Modal extraction worker and local progress/result mirror.

Called by canonical/run.py. An interrupted client can reconnect using --resume;
connection.json records the active call so it never launches a duplicate worker.
"""

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
import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
import threading
import time

import modal

CANONICAL = Path(__file__).resolve().parent.parent if modal.is_local() else Path('/study/src/experiments')
HERE = CANONICAL / 'emotion-probes'
sys.path[:0] = [str(CANONICAL), str(HERE)]
from shared.io import atomic, read, require, sha
from shared.progress import lock, now

GPU = os.environ.get('EMOTION_PROBE_GPU', 'B200:4')
MODEL_FAMILY = os.environ.get('EMOTION_PROBE_MODEL_FAMILY', 'glm53')
CUDA_BASE = 'nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6'


# Run this standalone during image building: the experiment source mounts are
# only available later, when a worker starts.
CUDA_TOOLCHAIN_CHECK = r'''
from pathlib import Path
import subprocess
import tempfile
import torch
assert torch.version.cuda and torch.version.cuda.startswith('13.0'), torch.version.cuda
subprocess.run(['nvcc', '--version'], check=True)
with tempfile.TemporaryDirectory() as folder:
    source = Path(folder) / 'smoke.cu'
    source.write_text('#include <cuda_runtime.h>\n#include <cub/cub.cuh>\n'
                      '__global__ void smoke(float* x) { if (threadIdx.x == 0) x[0] += 1; }\n')
    subprocess.run(['nvcc', '-std=c++17', '-arch=sm_100a', '--cubin', str(source),
                    '-o', str(Path(folder) / 'smoke.cubin')], check=True)
'''


app = modal.App(f'canonical-{MODEL_FAMILY}-emotion-probes')
models = modal.Volume.from_name(f'canonical-{MODEL_FAMILY}-model-cache-v1', create_if_missing=True)
runs = modal.Volume.from_name('canonical-emotion-probe-runs-v1', create_if_missing=True)
kernels = modal.Volume.from_name('canonical-glm53-kernel-cache-v1', create_if_missing=True)
image = (modal.Image.from_registry(CUDA_BASE, add_python='3.12').entrypoint([])
         .apt_install('build-essential', 'ninja-build')
         .uv_pip_install('vllm==0.30.0', 'vllm-lens==1.2.1', 'transformers==5.17.0',
                         'numpy==2.3.5', 'scikit-learn==1.9.1', 'matplotlib==3.11.2', 'jinja2==3.1.6')
         .env({'CUDA_HOME': '/usr/local/cuda'})
         .run_commands('python -c ' + shlex.quote('exec(' + repr(CUDA_TOOLCHAIN_CHECK) + ')'))
         .uv_pip_install(
             'flashinfer-jit-cache @ https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.18.post1/flashinfer_jit_cache-0.6.18.post1%2Bcu130-cp39-abi3-manylinux_2_28_x86_64.whl#sha256=d1729490636a98f22f158518e10c257eec5a188d3dadf30b4c9de298675d8d4b',
             'flashinfer-cubin @ https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.18.post1/flashinfer_cubin-0.6.18.post1-py3-none-any.whl#sha256=bbacb5b8bbf429e43bf2740bb45551d981f41d0462842e36ee6fb3e3452763ab')
         .env({'HF_HOME': '/models', 'PYTHONUNBUFFERED': '1', 'EMOTION_PROBE_GPU': GPU, 'EMOTION_PROBE_MODEL_FAMILY': MODEL_FAMILY,
               'PYTHONPATH': '/study/src/experiments:/study/src/experiments/emotion-probes',
               'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'MPLCONFIGDIR': '/tmp/matplotlib',
               'FLASHINFER_WORKSPACE_BASE': '/kernel-cache/cuda13-gcc13-v1', 'MAX_JOBS': '2',
               'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8'})
         .add_local_dir(CANONICAL / 'shared', '/study/src/experiments/shared', ignore=['__pycache__'])
         .add_local_dir(HERE / 'emotion_probes', '/study/src/experiments/emotion-probes/emotion_probes', ignore=['__pycache__'])
         .add_local_file(HERE / 'resume_from_extract.py', '/study/src/experiments/emotion-probes/resume_from_extract.py')
         .add_local_file(HERE / 'kernel_smoke.py', '/study/src/experiments/emotion-probes/kernel_smoke.py')
         .add_local_dir(CANONICAL.parents[1] / 'dashboards', '/study/dashboards'))
secret = modal.Secret.from_dict({'HF_TOKEN': os.environ['HF_TOKEN']}) if os.environ.get('HF_TOKEN') else modal.Secret.from_name('huggingface')


def runtime_details():
    import importlib.metadata
    import torch
    import emotion_probes.pipeline
    return {'pipeline': emotion_probes.pipeline.__file__, 'cuda_base': CUDA_BASE,
              'torch_cuda': torch.version.cuda,
              'nvcc': subprocess.check_output(['nvcc', '--version'], text=True),
              'versions': {n: importlib.metadata.version(n) for n in ['torch', 'vllm', 'vllm-lens', 'transformers', 'numpy',
                                                                     'flashinfer-python', 'flashinfer-jit-cache', 'flashinfer-cubin']}}


def preflight_key():
    # Invalidate cached smoke receipts whenever orchestration or its tests change.
    return hashlib.sha256((CUDA_BASE + sha(Path(__file__)) + sha(HERE / 'kernel_smoke.py')).encode()).hexdigest()


@app.function(image=image, cpu=2, memory=8192, timeout=180)
def inspect_runtime():
    """Cheap deployment/import check before a large-model GPU cold start."""
    result = runtime_details()
    print(json.dumps(result), flush=True)
    return json.dumps(result)


@app.function(image=image, cpu=8, memory=32768, timeout=1800,
              volumes={'/kernel-cache': kernels}, max_containers=1)
def cpu_kernel_build(force_build: bool = False):
    """Check prebuilt kernels, or reproduce their compilation, without GPUs."""
    import resource
    os.environ['FLASHINFER_CUDA_ARCH_LIST'] = '10.0a'
    print(json.dumps({'stage': 'compile', 'stack_limit': resource.getrlimit(resource.RLIMIT_STACK),
                      'compiler': subprocess.check_output(['g++', '--version'], text=True)}), flush=True)
    from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module
    started = time.monotonic()
    spec = gen_trtllm_gen_fused_moe_sm100_module()
    try:
        if force_build or not spec.is_aot:
            spec.build(verbose=True, need_lock=True)
            library = spec.jit_library_path
            mode = 'compiled on CPU'
        else:
            library = spec.aot_path
            mode = 'matching precompiled package'
        require(library.exists(), 'Required MoE library is missing')
        result = {**runtime_details(), 'passed': True, 'gpu_required': False,
                  'module': spec.name, 'mode': mode, 'seconds': time.monotonic() - started,
                  'library': str(library), 'sha256': sha(library), 'preflight_key': preflight_key()}
        atomic(Path(os.environ['FLASHINFER_WORKSPACE_BASE']) / 'cpu-preflight.json', result)
        previous = Path(os.environ['FLASHINFER_WORKSPACE_BASE']) / 'gpu-preflight.json'
        if previous.exists():
            receipt = read(previous)
            if (receipt.get('passed') and receipt.get('preflight_key') == result['preflight_key']
                    and receipt.get('versions') == result['versions']):
                result['cached_gpu_preflight'] = receipt
        print(json.dumps(result), flush=True)
        return result
    finally:
        kernels.commit()


@app.function(image=image, gpu='B200', cpu=8, memory=16384, timeout=900,
              volumes={'/kernel-cache': kernels})
def gpu_kernel_smoke():
    """Exercise sampling and a GLM-sized FP4 MoE shard without model weights."""
    import torch
    from kernel_smoke import check_nvfp4_moe
    from flashinfer.sampling import top_k_mask_logits
    started = time.monotonic()
    logits = torch.tensor([[1., 4., 2., 3.]], device='cuda')
    actual = top_k_mask_logits(logits, 2)
    torch.cuda.synchronize()
    require(torch.equal(torch.isfinite(actual), torch.tensor([[False, True, False, True]], device='cuda')),
            'FlashInfer top-k mask smoke failed')
    require(torch.equal(actual[torch.isfinite(actual)], torch.tensor([4., 3.], device='cuda')),
            'FlashInfer top-k values smoke failed')
    moe_check = check_nvfp4_moe()
    result = {**runtime_details(), 'passed': True, 'gpu': torch.cuda.get_device_name(),
              'seconds': time.monotonic() - started,
              'check': 'FlashInfer top_k_mask_logits CUDA execution', 'moe': moe_check,
              'test_sha256': sha(HERE / 'kernel_smoke.py'), 'preflight_key': preflight_key()}
    atomic(Path(os.environ['FLASHINFER_WORKSPACE_BASE']) / 'gpu-preflight.json', result)
    kernels.commit()
    print(json.dumps(result), flush=True)
    return result


def ensure_kernel_preflight(root):
    def status(message):
        path = root / 'progress.json'
        if path.exists():
            data = read(path)
            data.update(state='waiting', active_stage='extract', message=message, updated_at=now())
            data.pop('error', None)
            stage = next(s for s in data['stages'] if s['id'] == 'extract')
            stage.update(state='waiting', message=message)
            stage.pop('error', None)
            atomic(path, data)

    print('Checking the runtime on CPU before requesting model GPUs...', flush=True)
    status('Checking precompiled GPU kernels on CPU; model GPUs have not been requested')
    cpu = cpu_kernel_build.remote()
    require(cpu.get('passed') and cpu.get('preflight_key') == preflight_key(),
            'CPU kernel preflight failed or source changed; GPU allocation blocked')
    gpu = cpu.pop('cached_gpu_preflight', None)
    atomic(root / 'logs/cpu-kernel-preflight.json', cpu)
    if gpu is None:
        print('Testing sampling and the NVFP4 MoE kernel on one B200, without model weights...', flush=True)
        status('Testing GPU kernels on one B200 with synthetic inputs; GLM is not loading yet')
        gpu = gpu_kernel_smoke.remote()
    require(cpu['passed'] and gpu['passed'] and cpu['preflight_key'] == gpu['preflight_key'] == preflight_key(),
            'Runtime preflight failed or source changed during preflight')
    atomic(root / 'logs/gpu-kernel-preflight.json', gpu)
    status('Kernel preflight passed; ready for the full-model worker')


@app.local_entrypoint()
def preflight(run_dir: str):
    """Test the exact runtime without launching the four-GPU experiment."""
    ensure_kernel_preflight(Path(run_dir).resolve())
    print('Kernel preflight passed; the full-model worker was not launched.', flush=True)


@app.function(image=image, gpu=GPU, cpu=16, memory=131072, timeout=86400,
              volumes={'/models': models, '/data': runs, '/kernel-cache': kernels}, secrets=[secret], max_containers=1)
def worker(remote_name: str, through: str, smoke_only: bool, reuse_token_cache: bool = False):
    root = Path('/data') / remote_name
    os.environ['PYTHONPATH'] = '/study/src/experiments:/study/src/experiments/emotion-probes'
    entrypoint = 'resume_from_extract' if reuse_token_cache else 'emotion_probes.pipeline'
    command = ['python', '-c',
               f'from {entrypoint} import execute; import sys; execute(sys.argv[1],sys.argv[2],sys.argv[3]=="1")',
               str(root), through, '1' if smoke_only else '0']
    (root / 'logs').mkdir(exist_ok=True)
    atomic(root / 'logs' / f'runtime-{time.time_ns()}.json', {**runtime_details(), 'started_at': now()})
    with (root / 'logs/worker.log').open('a') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            # Commit snapshots and successfully written shards so an ordinary
            # provider interruption can resume without repeating completed work.
            time.sleep(10)
            runs.commit()
        runs.commit()
        models.commit()
    return {'exit_code': process.returncode, 'remote_name': remote_name}


def download(root, remote_name, name):
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.download')
    with temporary.open('wb') as stream:
        for chunk in runs.read_file(f'/{remote_name}/{name}'):
            stream.write(chunk)
    temporary.replace(target)


def mirror_snapshot(root, remote_name, terminal=False):
    connection = read(root / 'connection.json') if (root / 'connection.json').exists() else {}
    try:
        data = b''.join(runs.read_file(f'/{remote_name}/progress.json'))
        snapshot = json.loads(data)
    except (FileNotFoundError, modal.exception.NotFoundError):
        if (root / 'progress.json').exists():
            snapshot = read(root / 'progress.json')
            snapshot.update(state='waiting', updated_at=now(), active_stage='extract',
                            message='Waiting for Modal GPU allocation; no worker heartbeat received yet')
            snapshot.pop('error', None)
            if connection.get('job_url'):
                snapshot['links'] = [x for x in snapshot['links'] if x['label'] != 'Modal job'] + [
                    {'label': 'Modal job', 'href': connection['job_url']}]
            atomic(root / 'progress.json', snapshot)
        return
    # Preserve the remote heartbeat. A polling client must not make a dead
    # worker look healthy, nor mark complete before artifacts arrive.
    if (not terminal and connection.get('state') == 'running'
            and snapshot.get('updated_at', '') < connection.get('started_at', '')):
        # A retry reuses the remote directory; its prior failure snapshot is
        # not evidence that the newly queued attempt has failed too.
        snapshot = copy.deepcopy(snapshot)
        snapshot.update(state='waiting', updated_at=now(), active_stage='extract',
                        message='Waiting for Modal GPU allocation for the retry; cached work will be reused')
        snapshot.pop('error', None)
        for stage in snapshot['stages']:
            if stage['state'] == 'failed':
                stage.update(state='waiting', message='Retry queued')
                stage.pop('error', None)
    elif snapshot['state'] == 'complete' and not terminal:
        snapshot = copy.deepcopy(snapshot)
        snapshot.update(state='waiting', message='Remote computation finished; downloading verified artifacts')
    if connection.get('job_url'):
        snapshot['links'] = [x for x in snapshot['links'] if x['label'] != 'Modal job'] + [
            {'label': 'Modal job', 'href': connection['job_url']}]
    atomic(root / 'progress.json', snapshot)


def sync_artifacts(root, remote_name):
    # Activations and token caches remain on the durable volume. The portable
    # bundle, plots, metrics, and evidence are mirrored into experiment-results.
    for entry in runs.iterdir(f'/{remote_name}', recursive=True):
        full = entry.path.lstrip('/')
        relative = str(Path(full).relative_to(remote_name))
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        if Path(relative).parts[0] in ('activations', 'tokens', 'inputs', 'checkpoints') or relative.startswith('.'):
            continue
        if relative in ('config.resolved.json', 'progress.json'):
            continue
        download(root, remote_name, relative)
    manifest_path = root / 'probes/manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        for name, checksum in manifest['files'].items():
            require(sha(root / 'probes' / name) == checksum, f'Download checksum mismatch: {name}')


@app.local_entrypoint()
def main(run_dir: str, through: str = 'verify', smoke_only: bool = False, reuse_token_cache: bool = False):
    root = Path(run_dir).resolve()
    spec = read(root / 'config.resolved.json')
    require(spec['gpu'] == GPU, f'Launch GPU differs from configuration: {GPU}')
    remote_name = root.name + '-' + spec['identity'][:12]
    with lock(root):
        connection_path = root / 'connection.json'
        connection = read(connection_path) if connection_path.exists() else {}
        call = None
        if connection.get('call_id') and connection.get('state') == 'running':
            require(connection['identity'] == spec['identity'], 'Remote run identity changed')
            call = modal.FunctionCall.from_id(connection['call_id'])
            print(f'Reconnecting: {connection["job_url"]}', flush=True)
        if call is None:
            if spec['quantization'] == 'modelopt_fp4' and GPU.startswith('B200'):
                ensure_kernel_preflight(root)
            try:
                remote_spec = json.loads(b''.join(runs.read_file(f'/{remote_name}/config.resolved.json')))
                require(remote_spec == spec, 'Remote specification differs')
            except (FileNotFoundError, modal.exception.NotFoundError):
                with runs.batch_upload() as upload:
                    upload.put_file(root / 'config.resolved.json', f'/{remote_name}/config.resolved.json')
                    upload.put_directory(root / 'inputs', f'/{remote_name}/inputs')
            if connection:
                atomic(root / 'logs' / f'connection-{time.time_ns()}.json', connection)
            call = worker.spawn(remote_name, through, smoke_only, reuse_token_cache)
            connection = {'identity': spec['identity'], 'call_id': call.object_id, 'job_url': call.get_dashboard_url(),
                          'remote_name': remote_name, 'volume': runs.name, 'state': 'running',
                          'smoke_only': smoke_only, 'through': through, 'reuse_token_cache': reuse_token_cache, 'started_at': now()}
            atomic(connection_path, connection)
            print(f'Modal job: {connection["job_url"]}', flush=True)
        while True:
            mirror_snapshot(root, remote_name)
            try:
                result = call.get(timeout=10)
                break
            except TimeoutError:
                continue
            except Exception as exc:
                connection.update(state='failed', finished_at=now(), error=str(exc))
                atomic(connection_path, connection)
                mirror_snapshot(root, remote_name)
                if (root / 'progress.json').exists():
                    snapshot = read(root / 'progress.json')
                    snapshot.update(state='failed', error=str(exc), message='Modal call failed; inspect provider job and logs')
                    atomic(root / 'progress.json', snapshot)
                raise
        sync_artifacts(root, remote_name)
        connection.update(state='complete' if result['exit_code'] == 0 else 'failed', finished_at=now(), exit_code=result['exit_code'])
        atomic(connection_path, connection)
        mirror_snapshot(root, remote_name, terminal=True)
        require(result['exit_code'] == 0, 'Remote runner failed; inspect logs/worker.log and progress.json')
        if not connection['smoke_only'] and connection['through'] == 'verify':
            require(read(root / 'verification.json')['passed'], 'Remote verification did not pass')
        print(f'Artifacts: {root}', flush=True)
