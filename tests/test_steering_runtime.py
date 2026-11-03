"""Runtime boundaries that must hold before executing native environment actions."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src/experiments") / "emotion-steering"))
from emotion_steering.backend import (
    BackendValidationError,
    LensBackend,
    lens_hook,
    split_completion,
)


def test_glm_reasoning_is_never_executed_without_closed_boundary():
    model = {"family": "glm"}
    assert split_completion(
        "I might edit the ledger", "I might edit the ledger", model, False
    ) == ("", "I might edit the ledger", "length")
    assert (
        split_completion(
            "Still reasoning<|endoftext|>", "Still reasoning", model, True
        )[2]
        == "invalid_channel"
    )
    assert split_completion(
        "Consider options</think>read_email\n<|endoftext|>",
        "Consider optionsread_email",
        model,
        True,
    ) == ("read_email", "Consider options", "eos")


@pytest.mark.parametrize("strength", [-0.1, 0.0, 0.1])
def test_lens_hook_preserves_fraction_direction_and_receipts(strength):
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[3.0, 4.0], [0.0, 2.0]], dtype=torch.float32)
    ctx = SimpleNamespace(saved={}, layer_idx=7)
    hook = lens_hook([0.0, 8.0], strength)
    changed = hook(ctx, hidden)
    assert torch.allclose(
        changed - hidden, torch.tensor([[0.0, 5.0 * strength], [0.0, 2.0 * strength]])
    )
    assert torch.equal(hidden, torch.tensor([[3.0, 4.0], [0.0, 2.0]]))
    hook(ctx, hidden[:1])
    assert ctx.saved["tokens_steered"] == 3
    assert ctx.saved["layer"] == 7
    assert ctx.saved["strength"] == strength
    assert ctx.saved["first_delta_relative_norm"] == pytest.approx(abs(strength))


def test_lens_rejects_partial_residual_width():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="full post-block residual"):
        lens_hook([1.0, 0.0], 0.1)(
            SimpleNamespace(saved={}, layer_idx=1), torch.ones(4, 1)
        )


@pytest.mark.parametrize(
    "extra,finish,last_token,valid",
    [
        (0, "stop", 106, True),
        (1, "stop", 106, True),
        (2, "stop", 106, False),
        (-1, "stop", 106, False),
        (1, "length", 12, False),
        (1, "stop", 12, False),
    ],
)
def test_batch_keeps_baseline_hooks_seeds_and_stops_separate(
    monkeypatch, extra, finish, last_token, valid
):
    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(SamplingParams=SimpleNamespace)
    )
    monkeypatch.setitem(sys.modules, "vllm_lens", SimpleNamespace(Hook=SimpleNamespace))
    engine_calls = []

    class Engine:
        def generate(self, prompts, parameters, use_tqdm):
            engine_calls.append((prompts, parameters))
            outputs = []
            for prompt, param in zip(prompts, parameters, strict=True):
                hook = param.extra_args["apply_hooks"]
                outputs.append(
                    SimpleNamespace(
                        prompt_token_ids=prompt["prompt_token_ids"],
                        outputs=[
                            SimpleNamespace(
                                token_ids=[11, last_token], finish_reason=finish
                            )
                        ],
                        hook_results={
                            "0": {
                                "layer": 7,
                                "strength": 0.1,
                                "request_tag": "1:222",
                                "tokens_steered": len(prompt["prompt_token_ids"])
                                + 1
                                + extra,
                            }
                        }
                        if hook
                        else None,
                    )
                )
            return outputs

    backend = LensBackend.__new__(LensBackend)
    backend.cfg = {"family": "gemma", "max_context": 100}
    backend.spec = {"temperature": 0.8, "top_p": 0.95}
    backend.stop_token_ids = [1, 106]
    backend.tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **kw: messages[0]["content"],
        encode=lambda prompt, **kw: [ord(c) for c in prompt],
        decode=lambda ids, **kw: "Final answer",
    )
    backend.llm = Engine()
    requests = [
        {
            "messages": [{"role": "user", "content": "a"}],
            "probe": None,
            "strength": 0,
            "seed": 111,
            "max_tokens": 32,
        },
        {
            "messages": [{"role": "user", "content": "b"}],
            "probe": {"vector": [0, 1], "layer_zero_based": 7},
            "strength": 0.1,
            "seed": 222,
            "max_tokens": 16,
        },
    ]
    if not valid:
        with pytest.raises(BackendValidationError) as failed:
            backend.complete_batch(requests)
        assert len(failed.value.diagnostics) == 2
        return
    result = backend.complete_batch(requests)
    assert [r["sampling_seed"] for r in result] == [111, 222]
    assert result[0]["steering_receipt"] is None
    assert result[1]["steering_receipt"]["strength"] == 0.1
    assert result[1]["steering_receipt"]["extra_terminal_forward_tokens"] == extra
    _prompts, params = engine_calls[0]
    assert [p.seed for p in params] == [111, 222]
    assert [p.max_tokens for p in params] == [32, 16]
    assert all(p.stop_token_ids == [1, 106] for p in params)
    assert not params[0].extra_args["apply_hooks"]
    assert len(params[1].extra_args["apply_hooks"]) == 1
