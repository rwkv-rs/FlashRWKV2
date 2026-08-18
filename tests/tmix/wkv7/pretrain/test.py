# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect

import pytest
import torch
from utils import require_cuda_backend

from flashrwkv2.tmix.wkv7 import pretrain_tmix_wkv7_recurrent_bf16

W_SCALE = -0.6065306597


def _require_cuda() -> None:
    require_cuda_backend("_C_sm90", 9)
    assert hasattr(torch.ops.rwkv7_clampw_v3, "forward")
    assert hasattr(torch.ops.rwkv7_clampw_v3, "backward")


def _case(
    batch: int,
    tokens: int,
    heads: int,
    *,
    requires_grad: bool = False,
    head_size: int = 64,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20260806 + batch + tokens + heads)
    shape = (batch, tokens, heads * head_size)
    values = tuple(
        (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.05)
        .contiguous()
        .requires_grad_(requires_grad)
        for _ in range(6)
    )
    return values


def _reference(values: tuple[torch.Tensor, ...], head_size: int = 64) -> torch.Tensor:
    r, w, k, v, a, b = values
    batch, tokens, channels = r.shape
    heads = channels // head_size
    r4, w4, k4, v4, a4, b4 = (
        tensor.view(batch, tokens, heads, head_size).float()
        for tensor in (r, w, k, v, a, b)
    )
    state = torch.zeros((batch, heads, head_size, head_size), device=r.device)
    outputs = []
    for token in range(tokens):
        retention = torch.exp(W_SCALE / (1.0 + torch.exp(-w4[:, token])))
        state_dot_a = torch.einsum("bhkv,bhk->bhv", state, a4[:, token])
        state = (
            state * retention.unsqueeze(-1)
            + b4[:, token].unsqueeze(-1) * state_dot_a.unsqueeze(-2)
            + k4[:, token].unsqueeze(-1) * v4[:, token].unsqueeze(-2)
        )
        outputs.append(
            torch.einsum("bhkv,bhk->bhv", state, r4[:, token])
        )
    return torch.stack(outputs, dim=1).reshape(batch, tokens, channels)


def test_public_contract_matches_clampw_v3() -> None:
    signature = inspect.signature(pretrain_tmix_wkv7_recurrent_bf16)
    assert tuple(signature.parameters) == ("r", "w", "k", "v", "a", "b", "head_size")
    assert signature.parameters["head_size"].default == 64


@pytest.mark.parametrize(
    "batch,tokens,heads",
    [
        pytest.param(
            1,
            16,
            1,
            marks=(pytest.mark.memcheck, pytest.mark.racecheck),
        ),
        (2, 32, 2),
    ],
)
@pytest.mark.sm90
def test_forward_matches_clampw_recurrence(
    batch: int, tokens: int, heads: int
) -> None:
    _require_cuda()
    values = _case(batch, tokens, heads)
    actual = pretrain_tmix_wkv7_recurrent_bf16(*values)
    expected = _reference(values)
    assert actual.shape == values[0].shape
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual.float(), expected, atol=0.02, rtol=0.02)
    assert torch.equal(actual, pretrain_tmix_wkv7_recurrent_bf16(*values))


@pytest.mark.sm90
def test_backward_matches_clampw_recurrence() -> None:
    _require_cuda()
    values = _case(1, 16, 1, requires_grad=True)
    actual = pretrain_tmix_wkv7_recurrent_bf16(*values)
    grad_output = torch.randn_like(actual)
    actual.backward(grad_output)
    actual_gradients = [value.grad.detach().float() for value in values]

    reference_values = tuple(
        value.detach().clone().requires_grad_() for value in values
    )
    expected = _reference(reference_values)
    expected.backward(grad_output.float())
    expected_gradients = [value.grad.detach().float() for value in reference_values]
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients
    ):
        assert torch.isfinite(actual_gradient).all()
        assert torch.allclose(
            actual_gradient, expected_gradient, atol=0.03, rtol=0.05
        )


@pytest.mark.parametrize("head_size", (128, 256))
@pytest.mark.sm90
def test_generalized_forward_backward_matches_recurrence(head_size: int) -> None:
    _require_cuda()
    values = _case(1, 16, 1, requires_grad=True, head_size=head_size)
    actual = pretrain_tmix_wkv7_recurrent_bf16(*values, head_size=head_size)
    expected_values = tuple(value.detach().clone().requires_grad_() for value in values)
    expected = _reference(expected_values, head_size)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad.float())
    assert torch.allclose(actual.float(), expected, atol=0.03, rtol=0.03)
    for value, reference in zip(values, expected_values, strict=True):
        assert torch.isfinite(value.grad).all()
        assert torch.allclose(value.grad.float(), reference.grad.float(), atol=0.05, rtol=0.08)


@pytest.mark.sm90
def test_invalid_input_contract_is_rejected() -> None:
    _require_cuda()
    valid = list(_case(1, 16, 1))

    with pytest.raises(TypeError, match="bfloat16"):
        pretrain_tmix_wkv7_recurrent_bf16(valid[0].float(), *valid[1:])
    with pytest.raises(ValueError, match="T must be divisible"):
        short = [tensor[:, :15].contiguous() for tensor in valid]
        pretrain_tmix_wkv7_recurrent_bf16(*short)
    with pytest.raises(ValueError, match="C must be divisible"):
        narrow = [tensor[:, :, :63].contiguous() for tensor in valid]
        pretrain_tmix_wkv7_recurrent_bf16(*narrow)
    with pytest.raises(ValueError, match="must match r shape"):
        pretrain_tmix_wkv7_recurrent_bf16(
            valid[0], valid[1][:, :, :32].contiguous(), *valid[2:]
        )
    with pytest.raises(ValueError, match="contiguous CUDA"):
        noncontiguous = valid[0].transpose(1, 2)
        pretrain_tmix_wkv7_recurrent_bf16(noncontiguous, *valid[1:])
    with pytest.raises(ValueError, match="contiguous CUDA"):
        cpu = tuple(tensor.cpu() for tensor in valid)
        pretrain_tmix_wkv7_recurrent_bf16(*cpu)
