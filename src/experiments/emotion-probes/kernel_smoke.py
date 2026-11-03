"""Small NVFP4 runtime checks; no model checkpoint or experiment data needed."""

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



def check_nvfp4_moe():
    import torch
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
    from flashinfer import RoutingMethodType

    # Match one GLM 5.3 TP=4 shard: 256 experts, hidden=6144, FFN=2048/4.
    # Constant FP4 weights survive the kernel's row/block shuffles unchanged.
    # E2M1 code 2 represents 1.0; each byte holds two elements.
    experts, hidden, intermediate, tokens = 256, 6144, 512, 16
    device = 'cuda'
    def fp8_ones(shape):
        return torch.ones(shape, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)

    arguments = dict(
        routing_logits=torch.linspace(-1, 1, experts, device=device).repeat(tokens, 1),
        routing_bias=torch.zeros(experts, device=device),
        hidden_states=torch.full((tokens, hidden // 2), 0x22, dtype=torch.uint8, device=device),
        hidden_states_scale=fp8_ones((tokens, hidden // 16)),
        gemm1_weights=torch.full((experts, 2 * intermediate, hidden // 2), 0x22,
                                 dtype=torch.uint8, device=device),
        gemm1_weights_scale=fp8_ones((experts, 2 * intermediate, hidden // 16)),
        gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
        gemm2_weights=torch.full((experts, hidden, intermediate // 2), 0x22,
                                 dtype=torch.uint8, device=device),
        gemm2_weights_scale=fp8_ones((experts, hidden, intermediate // 16)),
        gemm2_bias=None,
        output1_scale_scalar=torch.full((experts,), 1 / hidden, device=device),
        output1_scale_gate_scalar=torch.full((experts,), 1 / hidden, device=device),
        output2_scale_scalar=torch.full((experts,), 1 / intermediate, device=device),
        num_experts=experts, top_k=8, n_group=1, topk_group=1,
        intermediate_size=intermediate, local_expert_offset=0, local_num_experts=experts,
        routed_scaling_factor=1.0, routing_method_type=RoutingMethodType.DeepSeekV3.value,
        do_finalize=True, norm_topk_prob=True,
    )
    def forward():
        result = trtllm_fp4_block_scale_moe(**arguments)
        result = result[0] if isinstance(result, (list, tuple)) else result
        torch.cuda.synchronize()
        assert result.shape == (tokens, hidden) and torch.isfinite(result).all()
        return result.float().clone()

    actual = forward()
    # Normalized all-one projections give SiLU(1), with NVFP4 rounding.
    reference = torch.nn.functional.silu(torch.tensor(1.0)).item()
    torch.testing.assert_close(actual, torch.full_like(actual, reference), atol=0.08, rtol=0)
    arguments['output2_scale_scalar'].mul_(2)
    torch.testing.assert_close(forward(), 2 * actual, atol=0.02, rtol=0.02)
    arguments['hidden_states'].zero_()
    torch.testing.assert_close(forward(), torch.zeros_like(actual), atol=0, rtol=0)
    return {'check': 'GLM-sized NVFP4 MoE shard: constant-input reference, output scaling, zero input',
            'experts': experts, 'hidden': hidden, 'intermediate': intermediate,
            'tokens': tokens, 'mean': actual.mean().item(), 'reference': reference}
