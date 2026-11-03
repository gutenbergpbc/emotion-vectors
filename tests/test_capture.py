"""Exercise chunked pooling without a vLLM install or a GPU."""
import sys
import types

import numpy as np
import pytest

from emotion_probes.extraction import make_capture_fn


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.ndim = self.values.ndim
    def __getitem__(self, key):
        return Tensor(self.values[key])
    def float(self):
        return self
    def sum(self, dim):
        return Tensor(self.values.sum(axis=dim))
    def __add__(self, other):
        return Tensor(self.values + (other.values if isinstance(other, Tensor) else other))
    __radd__ = __add__
    def __truediv__(self, value):
        return Tensor(self.values/value)
    def cpu(self):
        return self
    def numpy(self):
        return self.values


def test_chunked_capture_matches_full_target_mean(monkeypatch):
    torch = types.ModuleType('torch')
    torch.stack = lambda values: Tensor(np.stack([v.values for v in values]))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    distributed = types.ModuleType('vllm.distributed.parallel_state')
    distributed.get_tensor_model_parallel_rank = lambda: 0
    monkeypatch.setitem(sys.modules, 'vllm', types.ModuleType('vllm'))
    monkeypatch.setitem(sys.modules, 'vllm.distributed', types.ModuleType('vllm.distributed'))
    monkeypatch.setitem(sys.modules, 'vllm.distributed.parallel_state', distributed)
    values = np.arange(3*13*4).reshape(3, 13, 4)
    fn = make_capture_fn(5, 13, [0, 1, 2], 4)
    ctx = types.SimpleNamespace(saved={}, layer_idx=0)
    for start, end in [(0, 3), (3, 8), (8, 13)]:
        for layer in range(3):
            ctx.layer_idx = layer
            assert fn(ctx, Tensor(values[layer, start:end])) is None
    assert np.array_equal(ctx.saved['means'], values[:, 5:13].mean(axis=1))
    # A decode token must never be included in the pooled target.
    fn(ctx, Tensor(np.full((1, 4), 1e9)))
    assert np.array_equal(ctx.saved['means'], values[:, 5:13].mean(axis=1))
    distributed.get_tensor_model_parallel_rank = lambda: 1
    other = types.SimpleNamespace(saved={}, layer_idx=0)
    fn(other, Tensor(values[0]))
    assert other.saved == {}


def test_incomplete_capture_does_not_publish_success(monkeypatch):
    torch = types.ModuleType('torch')
    monkeypatch.setitem(sys.modules, 'torch', torch)
    distributed = types.ModuleType('vllm.distributed.parallel_state')
    distributed.get_tensor_model_parallel_rank = lambda: 0
    monkeypatch.setitem(sys.modules, 'vllm', types.ModuleType('vllm'))
    monkeypatch.setitem(sys.modules, 'vllm.distributed', types.ModuleType('vllm.distributed'))
    monkeypatch.setitem(sys.modules, 'vllm.distributed.parallel_state', distributed)
    ctx = types.SimpleNamespace(saved={}, layer_idx=2)
    with pytest.raises(ValueError, match='Missing/duplicated'):
        make_capture_fn(1, 4, [0, 1, 2], 4)(ctx, Tensor(np.ones((4, 4))))
    assert 'means' not in ctx.saved
