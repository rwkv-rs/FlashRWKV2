# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flashrwkv2.cmix import (
    infer_cmix_forward_varlen,
    pretrain_cmix_bf16,
    statetune_cmix_bf16,
)
from flashrwkv2.tmix.wkv7 import prepare_tmix_wkv7_recurrent_metadata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.sm120
def test_infer_cmix_highest_fusion_island_matches_reference() -> None:
    torch.manual_seed(20260820)
    device = torch.device("cuda")
    channels = 4096
    features = 4096
    x = torch.randn(1, channels, device=device, dtype=torch.float16) * 0.02
    res = torch.randn_like(x) * 0.02
    weight = torch.randn(channels, device=device, dtype=torch.float16) * 0.05 + 1
    bias = torch.randn(channels, device=device, dtype=torch.float16) * 0.01
    x_k = torch.randn(channels, device=device, dtype=torch.float16) * 0.1
    key = torch.randn(features, channels, device=device, dtype=torch.float16) * 0.01
    value = torch.randn(features, channels, device=device, dtype=torch.float16) * 0.01
    initial = torch.randn(3, channels, device=device, dtype=torch.float16) * 0.02
    shift = initial.clone()
    cu = torch.tensor([0, 1], device=device, dtype=torch.int32)
    slots = torch.tensor([2], device=device, dtype=torch.int32)

    summed, output = infer_cmix_forward_varlen(
        x,
        res,
        weight,
        bias,
        x_k,
        key,
        value,
        shift_state_pool=shift,
        cu_seqlens=cu,
        state_indices=slots,
        max_seqlen=1,
    )
    summed_ref = x.float() + res.float()
    mean = summed_ref.mean(dim=-1, keepdim=True)
    centered = summed_ref - mean
    normalized = (
        centered
        * torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + 1.0e-5)
        * weight.float()
        + bias.float()
    )
    normalized_f16 = normalized.to(torch.float16)
    mixed = normalized_f16.float() + (
        initial[2].float() - normalized_f16.float()
    ) * x_k.float()
    activation = torch.relu(mixed @ key.float().t()).square()
    expected_output = activation @ value.float()
    torch.testing.assert_close(summed.float(), summed_ref, atol=0.01, rtol=0.01)
    torch.testing.assert_close(output.float(), expected_output, atol=0.10, rtol=0.06)
    torch.testing.assert_close(shift[2], normalized_f16[0], atol=0.01, rtol=0.01)
    assert torch.equal(shift[:2], initial[:2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.sm120
def test_infer_cmix_packed_ragged_matches_reference_and_slot_updates() -> None:
    torch.manual_seed(20260821)
    device = torch.device("cuda")
    rows, channels, features = 4, 4096, 4096
    x = torch.randn(rows, channels, device=device, dtype=torch.float16) * 0.02
    res = torch.randn_like(x) * 0.02
    weight = torch.randn(channels, device=device, dtype=torch.float16) * 0.05 + 1
    bias = torch.randn(channels, device=device, dtype=torch.float16) * 0.01
    x_k = torch.randn(channels, device=device, dtype=torch.float16) * 0.1
    key = torch.randn(features, channels, device=device, dtype=torch.float16) * 0.01
    value = torch.randn(features, channels, device=device, dtype=torch.float16) * 0.01
    initial = torch.randn(5, channels, device=device, dtype=torch.float16) * 0.02
    state = initial.clone()
    cu = torch.tensor([0, 1, 4], device=device, dtype=torch.int32)
    slots = torch.tensor([3, 1], device=device, dtype=torch.int32)
    ticket = prepare_tmix_wkv7_recurrent_metadata(
        cu,
        slots,
        total_tokens=rows,
        state_pool_size=state.shape[0],
        max_seqlen=3,
    )
    summed, output = infer_cmix_forward_varlen(
        x,
        res,
        weight,
        bias,
        x_k,
        key,
        value,
        shift_state_pool=state,
        cu_seqlens=cu,
        state_indices=slots,
        validated_metadata=ticket,
    )
    summed_ref = x.float() + res.float()
    centered = summed_ref - summed_ref.mean(dim=-1, keepdim=True)
    normalized = (
        centered
        * torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + 1.0e-5)
        * weight.float()
        + bias.float()
    ).to(torch.float16)
    previous = torch.empty_like(normalized)
    previous[0] = initial[3]
    previous[1] = initial[1]
    previous[2:] = normalized[1:3]
    mixed = normalized.float() + (previous.float() - normalized.float()) * x_k.float()
    expected_output = torch.relu(mixed @ key.float().t()).square() @ value.float()
    torch.testing.assert_close(summed.float(), summed_ref, atol=0.01, rtol=0.01)
    torch.testing.assert_close(output.float(), expected_output, atol=0.12, rtol=0.07)
    # Ragged CMix selects the direct-Welford PostNorm body, whose reduction
    # order need not be bitwise identical to the independent FP32 oracle.
    torch.testing.assert_close(
        state[3].float(), normalized[0].float(), atol=0.01, rtol=0.01
    )
    torch.testing.assert_close(
        state[1].float(), normalized[3].float(), atol=0.01, rtol=0.01
    )
    assert torch.equal(state[[0, 2, 4]], initial[[0, 2, 4]])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.sm120
def test_infer_cmix_ticket_mutation_fails_without_state_write() -> None:
    device = torch.device("cuda")
    rows, channels, features = 2, 4096, 4096
    x = torch.zeros(rows, channels, device=device, dtype=torch.float16)
    vector = torch.zeros(channels, device=device, dtype=torch.float16)
    key = torch.zeros(features, channels, device=device, dtype=torch.float16)
    state = torch.randn(3, channels, device=device, dtype=torch.float16)
    before = state.clone()
    cu = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
    slots = torch.tensor([0, 2], device=device, dtype=torch.int32)
    ticket = prepare_tmix_wkv7_recurrent_metadata(
        cu, slots, total_tokens=rows, state_pool_size=3, max_seqlen=1
    )
    slots[0] = 1
    with pytest.raises(RuntimeError, match="version"):
        infer_cmix_forward_varlen(
            x,
            x,
            torch.ones_like(vector),
            vector,
            vector,
            key,
            key,
            shift_state_pool=state,
            cu_seqlens=cu,
            state_indices=slots,
            validated_metadata=ticket,
        )
    assert torch.equal(state, before)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.sm90
def test_pretrain_cmix_forward_backward() -> None:
    torch.manual_seed(5)
    device = torch.device("cuda")
    b, t, c = 1, 3, 4
    x = (torch.randn(b, t, c, device=device) * 0.03).to(torch.bfloat16).requires_grad_()
    x_k = (torch.randn(c, device=device) * 0.1).to(torch.bfloat16).requires_grad_()
    key = (
        (torch.randn(4 * c, c, device=device) * 0.05)
        .to(torch.bfloat16)
        .requires_grad_()
    )
    value = (
        (torch.randn(c, 4 * c, device=device) * 0.05)
        .to(torch.bfloat16)
        .requires_grad_()
    )
    output = pretrain_cmix_bf16(x, x_k, key, value)
    x_ref = x.detach().clone().requires_grad_()
    x_k_ref = x_k.detach().clone().requires_grad_()
    key_ref = key.detach().clone().requires_grad_()
    value_ref = value.detach().clone().requires_grad_()
    previous = torch.cat((torch.zeros_like(x_ref[:, :1]), x_ref[:, :-1]), dim=1)
    mixed = x_ref + (previous - x_ref) * x_k_ref
    preact = mixed.float().reshape(-1, c) @ key_ref.float().t()
    activation = torch.relu(preact) ** 2
    expected = (activation @ value_ref.float().t()).reshape_as(x_ref)
    assert torch.allclose(output.float(), expected.float(), atol=0.03, rtol=0.03)
    upstream = torch.randn_like(output)
    actual_grads = torch.autograd.grad(output, (x, x_k, key, value), upstream)
    expected_grads = torch.autograd.grad(
        expected, (x_ref, x_k_ref, key_ref, value_ref), upstream.float()
    )
    for actual, reference in zip(actual_grads, expected_grads, strict=True):
        assert torch.allclose(actual.float(), reference.float(), atol=0.06, rtol=0.06)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.sm90
@pytest.mark.parametrize("batch_size", (1, 2))
@pytest.mark.parametrize("seqlen", (1, 2, 7, 16, 31))
@pytest.mark.parametrize("zero_shift", (False, True))
def test_statetune_cmix_matches_torch_forward_backward(
    batch_size: int, seqlen: int, zero_shift: bool
) -> None:
    torch.manual_seed(2000 + batch_size * 100 + seqlen)
    device = torch.device("cuda")
    channels = 16
    x = (
        (torch.randn(batch_size, seqlen, channels, device=device) * 0.05)
        .to(torch.bfloat16)
        .requires_grad_()
    )
    initial = (torch.randn(batch_size, channels, device=device) * 0.05).to(
        torch.bfloat16
    )
    if zero_shift:
        initial.zero_()
    initial.requires_grad_()
    x_k = (
        (torch.randn(channels, device=device) * 0.1).to(torch.bfloat16).requires_grad_()
    )
    key = (
        (torch.randn(4 * channels, channels, device=device) * 0.05)
        .to(torch.bfloat16)
        .requires_grad_()
    )
    value = (
        (torch.randn(channels, 4 * channels, device=device) * 0.05)
        .to(torch.bfloat16)
        .requires_grad_()
    )
    actual = statetune_cmix_bf16(x, initial, x_k, key, value)

    x_ref = x.detach().clone().requires_grad_()
    initial_ref = initial.detach().clone().requires_grad_()
    x_k_ref = x_k.detach().clone().requires_grad_()
    key_ref = key.detach().clone().requires_grad_()
    value_ref = value.detach().clone().requires_grad_()
    previous = torch.cat((initial_ref[:, None, :], x_ref[:, :-1, :]), dim=1)
    mixed = x_ref + (previous - x_ref) * x_k_ref
    activation = torch.relu(mixed @ key_ref.t()).square()
    expected = (activation @ value_ref.t(), x_ref[:, -1, :].contiguous())
    upstream = [torch.randn_like(output) for output in actual]
    actual_grads = torch.autograd.grad(
        actual, (x, initial, x_k, key, value), grad_outputs=upstream
    )
    expected_grads = torch.autograd.grad(
        expected,
        (x_ref, initial_ref, x_k_ref, key_ref, value_ref),
        grad_outputs=upstream,
    )
    for output, reference in zip(actual, expected, strict=True):
        assert torch.allclose(output, reference, atol=0.03, rtol=0.03)
    for gradient, reference in zip(actual_grads, expected_grads, strict=True):
        assert torch.allclose(gradient, reference, atol=0.06, rtol=0.06)

    if zero_shift:
        pretrain = pretrain_cmix_bf16(
            x.detach(), x_k.detach(), key.detach(), value.detach()
        )
        assert torch.equal(actual[0], pretrain)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.sm90
def test_statetune_cmix_chunk_composition() -> None:
    torch.manual_seed(29)
    device = torch.device("cuda")
    b, t, c, split = 2, 7, 16, 3
    # Keep the chunk test in the operator's normal activation/weight range.
    # Unit-scale BF16 FFN matrices make whole-vs-split GEMM weight gradients
    # differ at cancellation points solely because each chunk is rounded before
    # autograd adds it, which does not exercise the recurrent shift contract.
    x = (torch.randn(b, t, c, device=device) * 0.05).to(torch.bfloat16).requires_grad_()
    initial = (torch.randn(b, c, device=device) * 0.05).to(torch.bfloat16).requires_grad_()
    x_k = (torch.randn(c, device=device) * 0.1).to(torch.bfloat16).requires_grad_()
    key = (torch.randn(4 * c, c, device=device) * 0.05).to(torch.bfloat16).requires_grad_()
    value = (torch.randn(c, 4 * c, device=device) * 0.05).to(torch.bfloat16).requires_grad_()
    whole = statetune_cmix_bf16(x, initial, x_k, key, value)
    upstream = [torch.randn_like(output) for output in whole]
    whole_grads = torch.autograd.grad(
        whole, (x, initial, x_k, key, value), grad_outputs=upstream
    )

    x_chunked = x.detach().clone().requires_grad_()
    initial_chunked = initial.detach().clone().requires_grad_()
    x_k_chunked = x_k.detach().clone().requires_grad_()
    key_chunked = key.detach().clone().requires_grad_()
    value_chunked = value.detach().clone().requires_grad_()
    first = statetune_cmix_bf16(
        x_chunked[:, :split].contiguous(),
        initial_chunked,
        x_k_chunked,
        key_chunked,
        value_chunked,
    )
    second = statetune_cmix_bf16(
        x_chunked[:, split:].contiguous(),
        first[1],
        x_k_chunked,
        key_chunked,
        value_chunked,
    )
    chunked = (torch.cat((first[0], second[0]), dim=1), second[1])
    chunked_grads = torch.autograd.grad(
        chunked,
        (x_chunked, initial_chunked, x_k_chunked, key_chunked, value_chunked),
        grad_outputs=upstream,
    )
    for expected, actual in zip(whole, chunked, strict=True):
        assert torch.equal(expected, actual)
    for expected, actual in zip(whole_grads, chunked_grads, strict=True):
        assert torch.allclose(expected, actual, atol=0.06, rtol=0.06)


def test_statetune_cmix_rejects_cpu_and_invalid_shape() -> None:
    x = torch.zeros(1, 1, 8, dtype=torch.bfloat16)
    initial = torch.zeros(1, 8, dtype=torch.bfloat16)
    x_k = torch.zeros(8, dtype=torch.bfloat16)
    key = torch.zeros(32, 8, dtype=torch.bfloat16)
    value = torch.zeros(8, 32, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA"):
        statetune_cmix_bf16(x, initial, x_k, key, value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.sm90
def test_statetune_cmix_rejects_misaligned_vec2_input() -> None:
    device = torch.device("cuda")
    x = torch.zeros(1, 2, 8, device=device, dtype=torch.bfloat16)
    initial = torch.empty(1 * 8 + 1, device=device, dtype=torch.bfloat16)[1:].view(
        1, 8
    )
    x_k = torch.zeros(8, device=device, dtype=torch.bfloat16)
    key = torch.zeros(32, 8, device=device, dtype=torch.bfloat16)
    value = torch.zeros(8, 32, device=device, dtype=torch.bfloat16)
    assert initial.is_contiguous() and initial.data_ptr() % 4 == 2
    with pytest.raises(ValueError, match="initial_shift must be 4-byte aligned"):
        statetune_cmix_bf16(x, initial, x_k, key, value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.sm90
def test_statetune_cmix_rejects_invalid_cuda_contracts() -> None:
    device = torch.device("cuda")
    x = torch.zeros(1, 2, 8, device=device, dtype=torch.bfloat16)
    initial = torch.zeros(1, 8, device=device, dtype=torch.bfloat16)
    x_k = torch.zeros(8, device=device, dtype=torch.bfloat16)
    key = torch.zeros(32, 8, device=device, dtype=torch.bfloat16)
    value = torch.zeros(8, 32, device=device, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="x must have dtype"):
        statetune_cmix_bf16(x.float(), initial, x_k, key, value)
    with pytest.raises(TypeError, match="initial_shift must have dtype"):
        statetune_cmix_bf16(x, initial.half(), x_k, key, value)
    with pytest.raises(ValueError, match="x must be contiguous"):
        statetune_cmix_bf16(x.transpose(1, 2), initial, x_k, key, value)
    with pytest.raises(ValueError, match="non-empty"):
        statetune_cmix_bf16(x[:, :0], initial, x_k, key, value)
    with pytest.raises(ValueError, match="initial_shift must have shape"):
        statetune_cmix_bf16(x, initial[:, :-2].contiguous(), x_k, key, value)
    with pytest.raises(ValueError, match="key_weight must have shape"):
        statetune_cmix_bf16(x, initial, x_k, key[:-1].contiguous(), value)

    odd_x = torch.zeros(1, 2, 7, device=device, dtype=torch.bfloat16)
    odd_initial = torch.zeros(1, 7, device=device, dtype=torch.bfloat16)
    odd_x_k = torch.zeros(7, device=device, dtype=torch.bfloat16)
    odd_key = torch.zeros(28, 7, device=device, dtype=torch.bfloat16)
    odd_value = torch.zeros(7, 28, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="divisible by 2"):
        statetune_cmix_bf16(odd_x, odd_initial, odd_x_k, odd_key, odd_value)

    if torch.cuda.device_count() > 1:
        foreign = x_k.to("cuda:1")
        with pytest.raises(ValueError, match="x_k must share x's device"):
            statetune_cmix_bf16(x, initial, foreign, key, value)
