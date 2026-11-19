# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flashrwkv2.tmix.wkv7 import pretrain_tmix_wkv7_recurrent_bf16

W_SCALE = -0.6065306597
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


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
        tensor.view(batch, tokens, heads, head_size).double()
        for tensor in (r, w, k, v, a, b)
    )
    state = torch.zeros((batch, heads, head_size, head_size), device=r.device, dtype=torch.float64)
    outputs = []
    for token in range(tokens):
        retention = torch.exp(W_SCALE / (1.0 + torch.exp(-w4[:, token])))
        state_dot_a = torch.einsum("bhkv,bhk->bhv", state, a4[:, token])
        state = (
            state * retention.unsqueeze(-1)
            + b4[:, token].unsqueeze(-1) * state_dot_a.unsqueeze(-2)
            + k4[:, token].unsqueeze(-1) * v4[:, token].unsqueeze(-2)
        )
        outputs.append(torch.einsum("bhkv,bhk->bhv", state, r4[:, token]))
    return torch.stack(outputs, dim=1).reshape(batch, tokens, channels)


@pytest.mark.parametrize(
    "batch,tokens,heads",
    [
        pytest.param(
            1,
            16,
            1,
            marks=(pytest.mark.memcheck, pytest.mark.racecheck),
        ),
        (1, 1, 1),
        (2, 17, 2),
        (2, 33, 2),
    ],
)
@pytest.mark.cuda
def test_forward_matches_clampw_recurrence(batch: int, tokens: int, heads: int) -> None:
    values = _case(batch, tokens, heads)
    actual = pretrain_tmix_wkv7_recurrent_bf16(*values)
    expected = _reference(values)
    assert actual.shape == values[0].shape
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual.double(), expected, atol=0.02, rtol=0.02)
    assert torch.equal(actual, pretrain_tmix_wkv7_recurrent_bf16(*values))


@pytest.mark.cuda
def test_backward_matches_clampw_recurrence() -> None:
    values = _case(1, 17, 1, requires_grad=True)
    actual = pretrain_tmix_wkv7_recurrent_bf16(*values)
    grad_output = torch.randn_like(actual)
    actual.backward(grad_output)
    actual_gradients = [value.grad.detach().double() for value in values]

    reference_values = tuple(
        value.detach().double().requires_grad_() for value in values
    )
    expected = _reference(reference_values)
    expected.backward(grad_output.float())
    expected_gradients = [value.grad.detach() for value in reference_values]
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(actual_gradient).all()
        assert torch.allclose(actual_gradient, expected_gradient, atol=0.03, rtol=0.05)


@pytest.mark.parametrize("head_size", (128, 256))
@pytest.mark.cuda
def test_generalized_forward_backward_matches_recurrence(head_size: int) -> None:
    values = _case(1, 17, 1, requires_grad=True, head_size=head_size)
    actual = pretrain_tmix_wkv7_recurrent_bf16(*values, head_size=head_size)
    expected_values = tuple(value.detach().double().requires_grad_() for value in values)
    expected = _reference(expected_values, head_size)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad.float())
    assert torch.allclose(actual.double(), expected, atol=0.03, rtol=0.03)
    for value, reference in zip(values, expected_values, strict=True):
        assert torch.isfinite(value.grad).all()
        assert torch.allclose(
            value.grad.double(), reference.grad, atol=0.05, rtol=0.08
        )


@pytest.mark.cuda
def test_invalid_input_contract_is_rejected() -> None:
    valid = list(_case(1, 16, 1))

    with pytest.raises(TypeError, match="bfloat16"):
        pretrain_tmix_wkv7_recurrent_bf16(valid[0].float(), *valid[1:])
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

@pytest.mark.cuda
@pytest.mark.parametrize("length", [15, 16, 17, 255, 256, 257, 513])
def test_dense_and_packed_state_across_chunk_boundaries(length):
    from flashrwkv2 import (
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen,
        prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor,
    )

    values = _case(2, length, 1)
    state = torch.randn(2, 1, 64, 64, device="cuda") * 0.01
    whole, final = pretrain_tmix_wkv7_recurrent_bf16(
        *values, initial_state=state, return_final_state=True,
    )
    cut = length // 2 + 1
    left, mid = pretrain_tmix_wkv7_recurrent_bf16(
        *(x[:, :cut].contiguous() for x in values), initial_state=state, return_final_state=True,
    )
    right, end = pretrain_tmix_wkv7_recurrent_bf16(
        *(x[:, cut:].contiguous() for x in values), initial_state=mid, return_final_state=True,
    )
    assert torch.equal(torch.cat((left, right), dim=1), whole)
    assert torch.equal(end, final)

    # Pack two sequences around an empty one, with a nonzero initial state for each.
    pool = torch.stack((state[0], state[0], state[1]))
    cu = torch.tensor([0, length, length, 2 * length], dtype=torch.int32, device="cuda")
    packed = tuple(x.view(-1, 1, 64) for x in values)
    output, packed_state = pretrain_tmix_wkv7_recurrent_bf16(
        *packed, cu_seqlens=cu, initial_state=pool, return_final_state=True,
    )
    assert torch.equal(output.view_as(whole), whole)
    assert torch.equal(packed_state[1], pool[1])
    assert torch.equal(packed_state[[0, 2]], final)
    for row in range(2):
        single, tail = pretrain_tmix_wkv7_recurrent_bf16(
            *(x[row:row + 1] for x in values), initial_state=state[row:row + 1], return_final_state=True,
        )
        assert torch.equal(single[0], whole[row])
        assert torch.equal(tail[0], final[row])
    infer_pool = state.clone()
    inferred = infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
        *packed, state=prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(infer_pool),
        cu_seqlens=cu[[0, 1, 3]], state_indices=torch.arange(2, dtype=torch.int32, device="cuda"),
    )
    assert torch.equal(inferred.view_as(whole), whole)
    assert torch.equal(infer_pool, final)
