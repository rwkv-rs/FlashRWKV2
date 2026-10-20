# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch
from utils import TorchOperationRecorder

from flashrwkv2.cmix import (
    infer_cmix_forward_varlen,
    pretrain_cmix_bf16,
    statetune_cmix_bf16,
)
from flashrwkv2.tmix.wkv7 import prepare_tmix_wkv7_recurrent_metadata


def _linear_reference(x, weight, a=None, b=None):
    output = (x.float() @ weight.float().t()).to(torch.bfloat16)
    if a is not None:
        normalized = (a.float() / (a.float().norm(dim=0, keepdim=True) + 1.0e-6)).to(
            torch.bfloat16
        )
        features = (x.float() @ normalized.float().t()).to(torch.bfloat16)
        update = features.float() @ b.to(torch.bfloat16).float().t()
        output = (output.float() + update).to(torch.bfloat16)
    return output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
@pytest.mark.memcheck
@pytest.mark.racecheck
def test_infer_cmix_highest_fusion_island_matches_reference() -> None:
    torch.manual_seed(20260820)
    device = torch.device("cuda")
    channels = 4096
    features = 4096
    x = torch.randn(1, channels, device=device, dtype=torch.bfloat16) * 0.02
    res = torch.randn_like(x) * 0.02
    weight = torch.randn(channels, device=device, dtype=torch.bfloat16) * 0.05 + 1
    bias = torch.randn(channels, device=device, dtype=torch.bfloat16) * 0.01
    x_k = torch.randn(channels, device=device, dtype=torch.bfloat16) * 0.1
    key = torch.randn(features, channels, device=device, dtype=torch.bfloat16) * 0.01
    value = torch.randn(features, channels, device=device, dtype=torch.bfloat16) * 0.01
    initial = torch.randn(3, channels, device=device, dtype=torch.bfloat16) * 0.02
    shift = initial.clone()
    cu = torch.tensor([0, 1], device=device, dtype=torch.int32)
    slots = torch.tensor([2], device=device, dtype=torch.int32)

    with TorchOperationRecorder() as recorder:
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
        )
    assert "aten::max" not in recorder.operations
    summed_ref = (x.float() + res.float()).to(torch.bfloat16).float()
    mean = summed_ref.mean(dim=-1, keepdim=True)
    centered = summed_ref - mean
    normalized = (
        centered
        * torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + 1.0e-5)
        * weight.float()
        + bias.float()
    )
    normalized_bf16 = normalized.to(torch.bfloat16)
    mixed = (
        normalized_bf16.float()
        + (initial[2].float() - normalized_bf16.float()) * x_k.float()
    ).to(torch.bfloat16)
    activation = (
        _linear_reference(mixed, key).float().relu().square().to(torch.bfloat16)
    )
    expected_output = _linear_reference(activation, value.t()).float()
    torch.testing.assert_close(summed.float(), summed_ref, atol=0.01, rtol=0.01)
    torch.testing.assert_close(output.float(), expected_output, atol=0.10, rtol=0.06)
    torch.testing.assert_close(shift[2], normalized_bf16[0], atol=0.01, rtol=0.01)
    assert torch.equal(shift[:2], initial[:2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
@pytest.mark.parametrize("batch_size", (16, 64, 320, 960))
def test_infer_cmix_t1_fused_varlen_preserves_fragmented_slots(
    batch_size: int,
) -> None:
    torch.manual_seed(20260824 + batch_size)
    device = torch.device("cuda")
    channels, features = 4096, 128
    x = torch.randn(batch_size, channels, device=device, dtype=torch.bfloat16).mul_(
        0.02
    )
    res = torch.randn_like(x).mul_(0.02)
    weight = (
        torch.randn(channels, device=device, dtype=torch.bfloat16).mul_(0.05).add_(1)
    )
    bias = torch.randn(channels, device=device, dtype=torch.bfloat16).mul_(0.01)
    x_k = torch.randn(channels, device=device, dtype=torch.bfloat16).mul_(0.1)
    key = torch.randn(features, channels, device=device, dtype=torch.bfloat16).mul_(
        0.01
    )
    value = torch.randn_like(key).mul_(0.01)
    initial = torch.randn(
        batch_size + 3, channels, device=device, dtype=torch.bfloat16
    ).mul_(0.02)
    state = initial.clone()
    cu = torch.arange(batch_size + 1, device=device, dtype=torch.int32)
    slots = torch.arange(batch_size - 1, -1, -1, device=device, dtype=torch.int32)
    ticket = prepare_tmix_wkv7_recurrent_metadata(
        cu,
        slots,
        total_tokens=batch_size,
        state_pool_size=state.shape[0],
        max_seqlen=1,
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

    summed_ref = (x.float() + res.float()).to(torch.bfloat16).float()
    centered = summed_ref - summed_ref.mean(dim=-1, keepdim=True)
    normalized = (
        centered
        * torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + 1.0e-5)
        * weight.float()
        + bias.float()
    ).to(torch.bfloat16)
    previous = initial.index_select(0, slots.to(torch.int64))
    mixed = (
        normalized.float() + (previous.float() - normalized.float()) * x_k.float()
    ).to(torch.bfloat16)
    activation = (
        _linear_reference(mixed, key).float().relu().square().to(torch.bfloat16)
    )
    expected = _linear_reference(activation, value.t()).float()
    torch.testing.assert_close(summed.float(), summed_ref, atol=0.01, rtol=0.01)
    torch.testing.assert_close(output.float(), expected, atol=0.12, rtol=0.07)
    torch.testing.assert_close(
        state.index_select(0, slots.to(torch.int64)).float(),
        normalized.float(),
        atol=0.01,
        rtol=0.01,
    )
    assert torch.equal(state[batch_size:], initial[batch_size:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
def test_infer_cmix_packed_ragged_matches_reference_and_slot_updates() -> None:
    torch.manual_seed(20260821)
    device = torch.device("cuda")
    rows, channels, features = 4, 4096, 4096
    x = torch.randn(rows, channels, device=device, dtype=torch.bfloat16) * 0.02
    res = torch.randn_like(x) * 0.02
    weight = torch.randn(channels, device=device, dtype=torch.bfloat16) * 0.05 + 1
    bias = torch.randn(channels, device=device, dtype=torch.bfloat16) * 0.01
    x_k = torch.randn(channels, device=device, dtype=torch.bfloat16) * 0.1
    key = torch.randn(features, channels, device=device, dtype=torch.bfloat16) * 0.01
    value = torch.randn(features, channels, device=device, dtype=torch.bfloat16) * 0.01
    initial = torch.randn(5, channels, device=device, dtype=torch.bfloat16) * 0.02
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
    summed_ref = (x.float() + res.float()).to(torch.bfloat16).float()
    centered = summed_ref - summed_ref.mean(dim=-1, keepdim=True)
    normalized = (
        centered
        * torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + 1.0e-5)
        * weight.float()
        + bias.float()
    ).to(torch.bfloat16)
    previous = torch.empty_like(normalized)
    previous[0] = initial[3]
    previous[1] = initial[1]
    previous[2:] = normalized[1:3]
    mixed = (
        normalized.float() + (previous.float() - normalized.float()) * x_k.float()
    ).to(torch.bfloat16)
    activation = (
        _linear_reference(mixed, key).float().relu().square().to(torch.bfloat16)
    )
    expected_output = _linear_reference(activation, value.t()).float()
    torch.testing.assert_close(summed.float(), summed_ref, atol=0.01, rtol=0.01)
    torch.testing.assert_close(output.float(), expected_output, atol=0.12, rtol=0.07)
    torch.testing.assert_close(
        state[3].float(), normalized[0].float(), atol=0.01, rtol=0.01
    )
    torch.testing.assert_close(
        state[1].float(), normalized[3].float(), atol=0.01, rtol=0.01
    )
    assert torch.equal(state[[0, 2, 4]], initial[[0, 2, 4]])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
def test_infer_cmix_ticket_mutation_fails_without_state_write() -> None:
    device = torch.device("cuda")
    rows, channels, features = 2, 4096, 4096
    x = torch.zeros(rows, channels, device=device, dtype=torch.bfloat16)
    vector = torch.zeros(channels, device=device, dtype=torch.bfloat16)
    key = torch.zeros(features, channels, device=device, dtype=torch.bfloat16)
    state = torch.randn(3, channels, device=device, dtype=torch.bfloat16)
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
@pytest.mark.cuda
@pytest.mark.memcheck
@pytest.mark.racecheck
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
    mixed = (x_ref.float() + (previous.float() - x_ref.float()) * x_k_ref.float()).to(
        torch.bfloat16
    )
    preact = _linear_reference(mixed, key_ref)
    activation = preact.float().relu().square().to(torch.bfloat16)
    expected = _linear_reference(activation, value_ref)
    assert torch.allclose(output.float(), expected.float(), atol=0.03, rtol=0.03)
    upstream = torch.randn_like(output)
    actual_grads = torch.autograd.grad(output, (x, x_k, key, value), upstream)
    expected_grads = torch.autograd.grad(
        expected, (x_ref, x_k_ref, key_ref, value_ref), upstream.float()
    )
    for actual, reference in zip(actual_grads, expected_grads, strict=True):
        assert torch.allclose(actual.float(), reference.float(), atol=0.06, rtol=0.06)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
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
    mixed = (x_ref.float() + (previous.float() - x_ref.float()) * x_k_ref.float()).to(
        torch.bfloat16
    )
    activation = (
        _linear_reference(mixed, key_ref).float().relu().square().to(torch.bfloat16)
    )
    expected = (_linear_reference(activation, value_ref), x_ref[:, -1, :].contiguous())
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
@pytest.mark.cuda
def test_statetune_cmix_chunk_composition() -> None:
    torch.manual_seed(29)
    device = torch.device("cuda")
    b, t, c, split = 2, 7, 16, 3
    # Keep the chunk test in the operator's normal activation/weight range.
    # Unit-scale BF16 FFN matrices make whole-vs-split GEMM weight gradients
    # differ at cancellation points solely because each chunk is rounded before
    # autograd adds it, which does not exercise the recurrent shift contract.
    x = (torch.randn(b, t, c, device=device) * 0.05).to(torch.bfloat16).requires_grad_()
    initial = (
        (torch.randn(b, c, device=device) * 0.05).to(torch.bfloat16).requires_grad_()
    )
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
@pytest.mark.cuda
def test_statetune_cmix_rejects_misaligned_vec2_input() -> None:
    device = torch.device("cuda")
    x = torch.zeros(1, 2, 8, device=device, dtype=torch.bfloat16)
    initial = torch.empty(1 * 8 + 1, device=device, dtype=torch.bfloat16)[1:].view(1, 8)
    x_k = torch.zeros(8, device=device, dtype=torch.bfloat16)
    key = torch.zeros(32, 8, device=device, dtype=torch.bfloat16)
    value = torch.zeros(8, 32, device=device, dtype=torch.bfloat16)
    assert initial.is_contiguous() and initial.data_ptr() % 4 == 2
    with pytest.raises(ValueError, match="4-byte aligned"):
        statetune_cmix_bf16(x, initial, x_k, key, value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
def test_statetune_cmix_rejects_invalid_cuda_contracts() -> None:
    device = torch.device("cuda")
    x = torch.zeros(1, 2, 8, device=device, dtype=torch.bfloat16)
    initial = torch.zeros(1, 8, device=device, dtype=torch.bfloat16)
    x_k = torch.zeros(8, device=device, dtype=torch.bfloat16)
    key = torch.zeros(32, 8, device=device, dtype=torch.bfloat16)
    value = torch.zeros(8, 32, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA bfloat16"):
        statetune_cmix_bf16(x.float(), initial, x_k, key, value)
    with pytest.raises(ValueError, match="CUDA bfloat16"):
        statetune_cmix_bf16(x, initial.half(), x_k, key, value)
    with pytest.raises(ValueError, match="shape|contiguous"):
        statetune_cmix_bf16(x.transpose(1, 2), initial, x_k, key, value)
    with pytest.raises(ValueError, match="non-empty"):
        statetune_cmix_bf16(x[:, :0], initial, x_k, key, value)
    with pytest.raises(ValueError, match="initial_shift must have shape"):
        statetune_cmix_bf16(x, initial[:, :-2].contiguous(), x_k, key, value)
    with pytest.raises(ValueError, match="CMix projection"):
        statetune_cmix_bf16(x, initial, x_k, key[:-1].contiguous(), value)

    odd_x = torch.zeros(1, 2, 7, device=device, dtype=torch.bfloat16)
    odd_initial = torch.zeros(1, 7, device=device, dtype=torch.bfloat16)
    odd_x_k = torch.zeros(7, device=device, dtype=torch.bfloat16)
    odd_key = torch.zeros(28, 7, device=device, dtype=torch.bfloat16)
    odd_value = torch.zeros(7, 28, device=device, dtype=torch.bfloat16)
    odd_output, odd_final = statetune_cmix_bf16(
        odd_x, odd_initial, odd_x_k, odd_key, odd_value
    )
    assert torch.equal(odd_output, odd_x)
    assert torch.equal(odd_final, odd_initial)

    if torch.cuda.device_count() > 1:
        foreign = x_k.to("cuda:1")
        with pytest.raises(ValueError, match="share x's device"):
            statetune_cmix_bf16(x, initial, foreign, key, value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
@pytest.mark.parametrize("stateful", [False, True])
def test_cmix_nora_training(stateful):
    from utils import assert_relative_rmse

    torch.manual_seed(73)
    c, rank = 64, 256
    x = (
        torch.randn(2, 17, c, device="cuda", dtype=torch.bfloat16) * 0.1
    ).requires_grad_()
    mix = (torch.randn(c, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_()
    initial = (
        torch.randn(2, c, device="cuda", dtype=torch.bfloat16) * 0.1
    ).requires_grad_()
    key = torch.randn(4 * c, c, device="cuda", dtype=torch.bfloat16) * 0.03
    value = torch.randn(c, 4 * c, device="cuda", dtype=torch.bfloat16) * 0.03
    ka, kb, va, vb = [
        (torch.randn(*shape, device="cuda") * 0.03).requires_grad_()
        for shape in ((rank, c), (4 * c, rank), (rank, 4 * c), (c, rank))
    ]
    tensors = [x, mix, ka, kb, va, vb] + ([initial] if stateful else [])
    refs = [tensor.detach().clone().requires_grad_() for tensor in tensors]
    options = {
        "key_lora_a": ka,
        "key_lora_b": kb,
        "value_lora_a": va,
        "value_lora_b": vb,
        "key_use_nora": True,
        "value_use_nora": True,
    }
    if stateful:
        output, next_shift = statetune_cmix_bf16(x, initial, mix, key, value, **options)
    else:
        output = pretrain_cmix_bf16(x, mix, key, value, **options)
    xr, mr, kar, kbr, var, vbr = refs[:6]
    previous = torch.cat(
        (refs[-1][:, None] if stateful else torch.zeros_like(xr[:, :1]), xr[:, :-1]),
        dim=1,
    )
    mixed = (xr.float() + (previous.float() - xr.float()) * mr.float()).to(
        torch.bfloat16
    )
    activation = (
        _linear_reference(mixed, key, kar, kbr)
        .float()
        .relu()
        .square()
        .to(torch.bfloat16)
    )
    expected = _linear_reference(activation, value, var, vbr)
    upstream = torch.randn_like(output)
    actual_loss, expected_loss = (output * upstream).sum(), (expected * upstream).sum()
    if stateful:
        boundary_grad = torch.randn_like(initial)
        actual_loss = actual_loss + (next_shift * boundary_grad).sum()
        expected_loss = expected_loss + (xr[:, -1] * boundary_grad).sum()
    grads = torch.autograd.grad(actual_loss, tensors)
    ref_grads = torch.autograd.grad(expected_loss, refs)
    assert_relative_rmse(output, expected, 0.04)
    for actual, reference in zip(grads, ref_grads, strict=True):
        assert_relative_rmse(actual, reference, 0.05)
    assert key.grad is None and value.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
@pytest.mark.parametrize("channels", (768, 4096))
def test_cmix_training_packed_and_decode_are_bitwise_equal(channels):
    from flashrwkv2.post_norm.tilelang import norm

    torch.manual_seed(103)
    lengths, slot_ids = (1, 3, 7, 16), (4, 2, 0, 3)
    rows, features = sum(lengths), 128
    x = (
        torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16) * 0.03
    ).requires_grad_()
    res = torch.randn_like(x) * 0.03
    weight = torch.ones(channels, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn_like(weight) * 0.01
    mix = torch.randn_like(weight) * 0.1
    key = torch.randn(features, channels, device="cuda", dtype=torch.bfloat16) * 0.02
    value = torch.randn(features, channels, device="cuda", dtype=torch.bfloat16) * 0.02
    initial = torch.randn(6, channels, device="cuda", dtype=torch.bfloat16) * 0.03
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    slots = torch.tensor(slot_ids, device="cuda", dtype=torch.int32)
    normalized, train_res = norm(x, weight, bias, res=res, return_res=True)
    training, expected_state = [], initial.clone()
    for sequence in range(len(lengths)):
        start, end = offsets[sequence : sequence + 2]
        output, final = statetune_cmix_bf16(
            normalized[start:end].unsqueeze(0),
            initial[slot_ids[sequence] : slot_ids[sequence] + 1],
            mix,
            key,
            value.t(),
        )
        training.append(output.squeeze(0))
        expected_state[slot_ids[sequence]] = final[0]
    state = initial.clone()
    with torch.no_grad():
        residual, packed = infer_cmix_forward_varlen(
            x.detach(),
            res,
            weight,
            bias,
            mix,
            key,
            value,
            shift_state_pool=state,
            cu_seqlens=cu,
            state_indices=slots,
            max_seqlen=max(lengths),
        )
    assert torch.equal(residual, train_res)
    assert torch.equal(packed, torch.cat(training))
    assert torch.equal(state, expected_state)

    decode_state = initial.clone()
    decoded = []
    one_cu = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    with torch.no_grad():
        for sequence, length in enumerate(lengths):
            for row in range(offsets[sequence], offsets[sequence] + length):
                _, output = infer_cmix_forward_varlen(
                    x[row : row + 1].detach(),
                    res[row : row + 1],
                    weight,
                    bias,
                    mix,
                    key,
                    value,
                    shift_state_pool=decode_state,
                    cu_seqlens=one_cu,
                    state_indices=slots[sequence : sequence + 1],
                    max_seqlen=1,
                )
                decoded.append(output)
    assert torch.equal(packed, torch.cat(decoded))
    assert torch.equal(state, decode_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
@pytest.mark.parametrize(
    "invalid",
    (
        "key",
        "value",
        "adapter_pair",
        "adapter_shape",
        "dtype",
        "device",
        "scale_nan",
        "scale_overflow",
    ),
)
def test_cmix_invalid_projection_preserves_t1_shift_state(invalid):
    channels, features = 64, 128
    x = torch.zeros(1, channels, device="cuda", dtype=torch.bfloat16)
    vector = torch.zeros(channels, device="cuda", dtype=torch.bfloat16)
    key = torch.zeros(features, channels, device="cuda", dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    state = torch.randn(2, channels, device="cuda", dtype=torch.bfloat16)
    before = state.clone()
    options = {}
    if invalid == "key":
        key = key[:, :-1].contiguous()
    elif invalid == "value":
        value = value[:, :-1].contiguous()
    elif invalid.startswith("adapter"):
        options["key_lora_a"] = torch.zeros(8, channels, device="cuda")
        if invalid == "adapter_shape":
            options["key_lora_b"] = torch.zeros(features - 1, 8, device="cuda")
    elif invalid.startswith("scale"):
        options["key_lora_a"] = torch.zeros(8, channels, device="cuda")
        options["key_lora_b"] = torch.zeros(features, 8, device="cuda")
        options["key_lora_scale"] = (
            float("nan")
            if invalid == "scale_nan"
            else 2 * torch.finfo(torch.float32).max
        )
    elif invalid == "dtype":
        key = key.float()
    else:
        key = key.cpu()
    with pytest.raises(ValueError, match="CMix|adapter scale"):
        infer_cmix_forward_varlen(
            x,
            x,
            torch.ones_like(vector),
            vector,
            vector,
            key,
            value,
            shift_state_pool=state,
            cu_seqlens=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
            state_indices=torch.tensor([1], device="cuda", dtype=torch.int32),
            max_seqlen=1,
            **options,
        )
    assert torch.equal(state, before)
