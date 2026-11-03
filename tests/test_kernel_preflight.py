from types import SimpleNamespace

import pytest

import modal_worker
from shared.io import read


@pytest.mark.parametrize('cpu_passed,cached_gpu', [(False, False), (True, False), (True, True)])
def test_preflight_blocks_gpus_after_failure_and_reuses_matching_receipt(tmp_path, monkeypatch, cpu_passed, cached_gpu):
    key = 'fixed-runtime'
    gpu = {'passed': True, 'preflight_key': key}
    cpu = {'passed': cpu_passed, 'preflight_key': key}
    if cached_gpu:
        cpu['cached_gpu_preflight'] = gpu
    monkeypatch.setattr(modal_worker, 'preflight_key', lambda: key)
    monkeypatch.setattr(modal_worker, 'cpu_kernel_build', SimpleNamespace(remote=lambda: cpu))
    calls = []
    def gpu_remote():
        calls.append('gpu')
        return gpu
    monkeypatch.setattr(modal_worker, 'gpu_kernel_smoke', SimpleNamespace(remote=gpu_remote))
    if not cpu_passed:
        with pytest.raises(ValueError, match='GPU allocation blocked'):
            modal_worker.ensure_kernel_preflight(tmp_path)
        assert calls == []
    else:
        modal_worker.ensure_kernel_preflight(tmp_path)
        assert calls == ([] if cached_gpu else ['gpu'])
        assert read(tmp_path / 'logs/gpu-kernel-preflight.json')['passed']
