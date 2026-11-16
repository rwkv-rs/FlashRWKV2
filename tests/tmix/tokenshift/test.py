# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch
from utils import TorchOperationRecorder

from flashrwkv2.tmix.tokenshift import (
    infer_tmix_postnorm_tokenshift_forward_varlen,
    pretrain_tmix_tokenshift_bf16,
)
from flashrwkv2.tmix.wkv7 import prepare_tmix_wkv7_recurrent_metadata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
@pytest.mark.memcheck
@pytest.mark.racecheck
def test_pretrain_tokenshift_forward_backward() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    x = (torch.randn(2, 3, 8, device=device) * 0.1).to(torch.bfloat16).requires_grad_()
    params = [
        (torch.randn(8, device=device) * 0.1).to(torch.bfloat16).requires_grad_()
        for _ in range(6)
    ]
    outputs = pretrain_tmix_tokenshift_bf16(x, *params)
    x_ref = x.detach().clone().requires_grad_()
    params_ref = [parameter.detach().clone().requires_grad_() for parameter in params]
    reference = []
    previous = torch.zeros_like(x_ref[:, :1])
    for parameter in params_ref:
        mixed = x_ref + (
            torch.cat((previous, x_ref[:, :-1]), dim=1) - x_ref
        ) * parameter
        reference.append(mixed)
    for output, expected in zip(outputs, reference, strict=True):
        assert torch.allclose(output.float(), expected.float(), atol=0.01, rtol=0.01)
    upstream = [torch.randn_like(output) for output in outputs]
    actual_grads = torch.autograd.grad(outputs, (x, *params), grad_outputs=upstream)
    expected_grads = torch.autograd.grad(
        reference, (x_ref, *params_ref), grad_outputs=upstream
    )
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        assert torch.allclose(actual, expected, atol=0.03, rtol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
@pytest.mark.parametrize("batch_size", (1, 2))
@pytest.mark.parametrize("seqlen", (1, 2, 7, 16, 31))
@pytest.mark.parametrize("zero_shift", (False, True))
def test_statetune_tokenshift_matches_torch_forward_backward(
    batch_size: int, seqlen: int, zero_shift: bool
) -> None:
    torch.manual_seed(1000 + batch_size * 100 + seqlen)
    device = torch.device("cuda")
    channels = 128
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
    params = [
        (torch.randn(channels, device=device) * 0.1).to(torch.bfloat16).requires_grad_()
        for _ in range(6)
    ]
    actual = pretrain_tmix_tokenshift_bf16(x, *params, initial_shift=initial, return_final_state=True)

    x_ref = x.detach().clone().requires_grad_()
    initial_ref = initial.detach().clone().requires_grad_()
    params_ref = [parameter.detach().clone().requires_grad_() for parameter in params]
    previous = torch.cat((initial_ref[:, None, :], x_ref[:, :-1, :]), dim=1)
    expected = tuple(
        x_ref + (previous - x_ref) * parameter for parameter in params_ref
    ) + (x_ref[:, -1, :].contiguous(),)
    upstream = [torch.randn_like(output) for output in actual]
    actual_grads = torch.autograd.grad(
        actual, (x, initial, *params), grad_outputs=upstream
    )
    expected_grads = torch.autograd.grad(
        expected, (x_ref, initial_ref, *params_ref), grad_outputs=upstream
    )
    for output, reference in zip(actual, expected, strict=True):
        assert torch.allclose(output, reference, atol=0.01, rtol=0.01)
    for gradient, reference in zip(actual_grads, expected_grads, strict=True):
        assert torch.allclose(gradient, reference, atol=0.03, rtol=0.03)

    if zero_shift:
        pretrain = pretrain_tmix_tokenshift_bf16(x.detach(), *(p.detach() for p in params))
        for output, reference in zip(actual[:6], pretrain, strict=True):
            assert torch.equal(output, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
def test_statetune_tokenshift_chunk_composition() -> None:
    torch.manual_seed(23)
    device = torch.device("cuda")
    b, t, c, split = 2, 7, 128, 3
    x = torch.randn(b, t, c, device=device, dtype=torch.bfloat16).requires_grad_()
    initial = torch.randn(b, c, device=device, dtype=torch.bfloat16).requires_grad_()
    params = [
        torch.randn(c, device=device, dtype=torch.bfloat16).requires_grad_()
        for _ in range(6)
    ]
    whole = pretrain_tmix_tokenshift_bf16(x, *params, initial_shift=initial, return_final_state=True)
    upstream = [torch.randn_like(output) for output in whole]
    whole_grads = torch.autograd.grad(
        whole, (x, initial, *params), grad_outputs=upstream
    )

    x_chunked = x.detach().clone().requires_grad_()
    initial_chunked = initial.detach().clone().requires_grad_()
    params_chunked = [parameter.detach().clone().requires_grad_() for parameter in params]
    first = pretrain_tmix_tokenshift_bf16(
        x_chunked[:, :split].contiguous(), *params_chunked,
        initial_shift=initial_chunked, return_final_state=True,
    )
    second = pretrain_tmix_tokenshift_bf16(
        x_chunked[:, split:].contiguous(), *params_chunked,
        initial_shift=first[-1], return_final_state=True,
    )
    chunked = tuple(
        torch.cat((left, right), dim=1)
        for left, right in zip(first[:6], second[:6], strict=True)
    ) + (second[-1],)
    chunked_grads = torch.autograd.grad(
        chunked,
        (x_chunked, initial_chunked, *params_chunked),
        grad_outputs=upstream,
    )
    for expected, actual in zip(whole, chunked, strict=True):
        assert torch.equal(expected, actual)
    for expected, actual in zip(whole_grads, chunked_grads, strict=True):
        assert torch.allclose(expected, actual, atol=0.03, rtol=0.03)


def test_statetune_tokenshift_rejects_cpu_and_invalid_shape() -> None:
    x = torch.zeros(1, 1, 8, dtype=torch.bfloat16)
    initial = torch.zeros(1, 8, dtype=torch.bfloat16)
    params = [torch.zeros(8, dtype=torch.bfloat16) for _ in range(6)]
    with pytest.raises(ValueError, match="CUDA"):
        pretrain_tmix_tokenshift_bf16(x, *params, initial_shift=initial, return_final_state=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
def test_statetune_tokenshift_rejects_misaligned_vec2_input() -> None:
    device = torch.device("cuda")
    x = torch.empty(1 * 2 * 8 + 1, device=device, dtype=torch.bfloat16)[1:].view(
        1, 2, 8
    )
    initial = torch.zeros(1, 8, device=device, dtype=torch.bfloat16)
    params = [torch.zeros(8, device=device, dtype=torch.bfloat16) for _ in range(6)]
    assert x.is_contiguous() and x.data_ptr() % 4 == 2
    with pytest.raises(ValueError, match="4-byte aligned"):
        pretrain_tmix_tokenshift_bf16(x, *params, initial_shift=initial, return_final_state=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
def test_statetune_tokenshift_rejects_invalid_cuda_contracts() -> None:
    device = torch.device("cuda")
    x = torch.zeros(1, 2, 8, device=device, dtype=torch.bfloat16)
    initial = torch.zeros(1, 8, device=device, dtype=torch.bfloat16)
    params = [torch.zeros(8, device=device, dtype=torch.bfloat16) for _ in range(6)]
    with pytest.raises(ValueError, match="bfloat16"):
        pretrain_tmix_tokenshift_bf16(x.float(), *params, initial_shift=initial, return_final_state=True)
    with pytest.raises(ValueError, match="bfloat16"):
        pretrain_tmix_tokenshift_bf16(x, *params, initial_shift=initial.half(), return_final_state=True)
    with pytest.raises(ValueError, match="x must be contiguous"):
        pretrain_tmix_tokenshift_bf16(
            x.transpose(1, 2), *params,
            initial_shift=initial, return_final_state=True,
        )
    with pytest.raises(ValueError, match="non-empty"):
        pretrain_tmix_tokenshift_bf16(x[:, :0], *params, initial_shift=initial, return_final_state=True)
    with pytest.raises(ValueError, match="initial_shift must have shape"):
        pretrain_tmix_tokenshift_bf16(
            x, *params,
            initial_shift=initial[:, :-2].contiguous(), return_final_state=True,
        )
    with pytest.raises(ValueError, match="parameters must have shape"):
        pretrain_tmix_tokenshift_bf16(
            x, params[0][:-2].contiguous(), *params[1:],
            initial_shift=initial, return_final_state=True,
        )

    odd_x = torch.zeros(1, 2, 7, device=device, dtype=torch.bfloat16)
    odd_initial = torch.zeros(1, 7, device=device, dtype=torch.bfloat16)
    odd_params = [torch.zeros(7, device=device, dtype=torch.bfloat16) for _ in range(6)]
    odd_output = pretrain_tmix_tokenshift_bf16(
        odd_x, *odd_params,
        initial_shift=odd_initial, return_final_state=True,
    )
    assert all(torch.count_nonzero(tensor) == 0 for tensor in odd_output)

    if torch.cuda.device_count() > 1:
        foreign = params[0].to("cuda:1")
        with pytest.raises(ValueError, match="share x's device"):
            pretrain_tmix_tokenshift_bf16(
                x, foreign, *params[1:],
                initial_shift=initial, return_final_state=True,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
@pytest.mark.memcheck
@pytest.mark.racecheck
@pytest.mark.parametrize("use_ticket", [False, True])
def test_infer_tmix_postnorm_tokenshift_matches_shared_t1_path(use_ticket: bool) -> None:
    torch.manual_seed(19)
    device = torch.device("cuda")
    c = 4096
    eps = 1.0e-5
    x = (torch.randn(1, c, device=device) * 0.03).to(torch.bfloat16)
    res = (torch.randn(1, c, device=device) * 0.02).to(torch.bfloat16)
    weight = (torch.randn(c, device=device) * 0.1 + 1.0).to(torch.bfloat16)
    bias = (torch.randn(c, device=device) * 0.01).to(torch.bfloat16)
    params = [(torch.randn(c, device=device) * 0.1).to(torch.bfloat16) for _ in range(6)]
    initial = torch.randn(5, c, device=device, dtype=torch.bfloat16)
    shift_state = initial.clone()
    saved = torch.full_like(x, 123)
    cu_seqlens = torch.tensor([0, 1], device=device, dtype=torch.int32)
    state_indices = torch.tensor([3], device=device, dtype=torch.int32)
    ticket = prepare_tmix_wkv7_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=1,
        state_pool_size=shift_state.shape[0],
        max_seqlen=1,
    )

    with TorchOperationRecorder() as recorder:
        outputs = infer_tmix_postnorm_tokenshift_forward_varlen(
            x,
            res,
            weight,
            bias,
            *params,
            shift_state_pool=shift_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            validated_metadata=ticket if use_ticket else None,
            save_normalized=saved,
        )

    assert "aten::max" not in recorder.operations
    summed = (x + res).float()
    mean = summed.mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt((summed - mean).square().mean(dim=-1, keepdim=True) + eps)
    normalized = ((summed - mean) * rstd * weight.float() + bias.float()).to(
        torch.bfloat16
    )
    previous = initial[3]
    expected = [
        (
            normalized.float()
            + (previous.float() - normalized.float()) * parameter.float()
        ).to(torch.bfloat16)
        for parameter in params
    ]

    assert torch.allclose(outputs[0], summed.to(torch.bfloat16), atol=0.01, rtol=0.01)
    for output, expected_output in zip(outputs[1:], expected, strict=True):
        assert torch.allclose(output, expected_output, atol=0.01, rtol=0.01)
    expected_state = initial.clone()
    expected_state[3] = normalized[0]
    assert torch.allclose(shift_state, expected_state, atol=0.01, rtol=0.01)
    assert torch.equal(saved[0], shift_state[3])
    assert torch.equal(shift_state[[0, 1, 2, 4]], initial[[0, 1, 2, 4]])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
@pytest.mark.parametrize("batch_size", (4, 16, 64, 320, 960))
def test_infer_tmix_t1_varlen_preserves_fragmented_slots(
    batch_size: int,
) -> None:
    torch.manual_seed(20260823 + batch_size)
    device = torch.device("cuda")
    channels = 4096
    x = torch.randn(
        batch_size, channels, device=device, dtype=torch.bfloat16
    ).mul_(0.02)
    res = torch.randn_like(x).mul_(0.02)
    weight = torch.randn(
        channels, device=device, dtype=torch.bfloat16
    ).mul_(0.05).add_(1)
    bias = torch.randn(
        channels, device=device, dtype=torch.bfloat16
    ).mul_(0.01)
    params = [
        torch.randn(channels, device=device, dtype=torch.bfloat16).mul_(0.1)
        for _ in range(6)
    ]
    initial = torch.randn(
        batch_size + 3, channels, device=device, dtype=torch.bfloat16
    ).mul_(0.02)
    state = initial.clone()
    cu = torch.arange(batch_size + 1, device=device, dtype=torch.int32)
    slots = torch.arange(
        batch_size - 1, -1, -1, device=device, dtype=torch.int32
    )
    ticket = prepare_tmix_wkv7_recurrent_metadata(
        cu,
        slots,
        total_tokens=batch_size,
        state_pool_size=state.shape[0],
        max_seqlen=1,
    )
    outputs = infer_tmix_postnorm_tokenshift_forward_varlen(
        x,
        res,
        weight,
        bias,
        *params,
        shift_state_pool=state,
        cu_seqlens=cu,
        state_indices=slots,
        validated_metadata=ticket,
    )

    summed = (x + res).float()
    centered = summed - summed.mean(dim=-1, keepdim=True)
    normalized = (
        centered
        * torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + 1.0e-5)
        * weight.float()
        + bias.float()
    ).to(torch.bfloat16)
    previous = initial.index_select(0, slots.to(torch.int64))
    expected = [
        normalized.float()
        + (previous.float() - normalized.float()) * parameter.float()
        for parameter in params
    ]
    torch.testing.assert_close(outputs[0].float(), summed, atol=0.01, rtol=0.01)
    for actual, reference in zip(outputs[1:], expected, strict=True):
        torch.testing.assert_close(actual.float(), reference, atol=0.02, rtol=0.02)
    torch.testing.assert_close(
        state.index_select(0, slots.to(torch.int64)).float(),
        normalized.float(),
        atol=0.01,
        rtol=0.01,
    )
    assert torch.equal(state[batch_size:], initial[batch_size:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
def test_infer_tokenshift_generic_packed_ragged_matches_reference() -> None:
    torch.manual_seed(20260822)
    device = torch.device("cuda")
    rows, channels = 4, 4096
    x = torch.randn(rows, channels, device=device, dtype=torch.bfloat16) * 0.02
    res = torch.randn_like(x) * 0.02
    weight = torch.randn(channels, device=device, dtype=torch.bfloat16) * 0.05 + 1
    bias = torch.randn(channels, device=device, dtype=torch.bfloat16) * 0.01
    params = [
        torch.randn(channels, device=device, dtype=torch.bfloat16) * 0.1
        for _ in range(6)
    ]
    initial = torch.randn(5, channels, device=device, dtype=torch.bfloat16) * 0.02
    state = initial.clone()
    saved = torch.full_like(x, 123)
    cu = torch.tensor([0, 1, 4], device=device, dtype=torch.int32)
    slots = torch.tensor([3, 1], device=device, dtype=torch.int32)
    ticket = prepare_tmix_wkv7_recurrent_metadata(
        cu, slots, total_tokens=rows, state_pool_size=5, max_seqlen=3
    )
    outputs = infer_tmix_postnorm_tokenshift_forward_varlen(
        x,
        res,
        weight,
        bias,
        *params,
        shift_state_pool=state,
        cu_seqlens=cu,
        state_indices=slots,
        validated_metadata=ticket,
        save_normalized=saved,
    )
    summed = (x + res).float()
    centered = summed - summed.mean(dim=-1, keepdim=True)
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
    expected = [
        normalized.float()
        + (previous.float() - normalized.float()) * parameter.float()
        for parameter in params
    ]
    torch.testing.assert_close(outputs[0].float(), summed, atol=0.01, rtol=0.01)
    for actual, reference in zip(outputs[1:], expected, strict=True):
        torch.testing.assert_close(actual.float(), reference, atol=0.02, rtol=0.02)
    expected_state = initial.clone()
    expected_state[3] = normalized[0]
    expected_state[1] = normalized[3]
    torch.testing.assert_close(
        state.float(), expected_state.float(), atol=0.01, rtol=0.01
    )
    decode_state = initial.clone()
    one = cu.new_tensor([0, 1])
    for row, seq in enumerate((0, 1, 1, 1)):
        decoded = infer_tmix_postnorm_tokenshift_forward_varlen(
            x[row:row + 1], res[row:row + 1], weight, bias, *params,
            shift_state_pool=decode_state, cu_seqlens=one, state_indices=slots[seq:seq + 1],
        )
        assert torch.equal(saved[row:row + 1], decode_state[slots[seq:seq + 1].long()])
        assert all(torch.equal(actual[row:row + 1], want) for actual, want in zip(outputs, decoded, strict=True))
    assert torch.equal(state, decode_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
@pytest.mark.cuda_graph
@pytest.mark.parametrize("length,capacity", [(1, 2), (5, 8)])
def test_shift_state_export_cuda_graph_writes_only_active_rows(length, capacity):
    from flashrwkv2.cmix import infer_cmix_forward_varlen

    torch.manual_seed(20260809)
    device = torch.device("cuda")
    channels, num_slots = 4096, 4
    x = torch.randn(capacity, channels, device=device, dtype=torch.bfloat16)
    res = torch.randn_like(x)
    weight = torch.randn(channels, device=device, dtype=torch.bfloat16)
    bias = torch.randn(channels, device=device, dtype=torch.bfloat16)
    params = [
        torch.randn(channels, device=device, dtype=torch.bfloat16)
        for _ in range(6)
    ]
    initial = torch.randn(num_slots, channels, device=device, dtype=torch.bfloat16)
    cmix_key = torch.randn(64, channels, device=device, dtype=torch.bfloat16) * 0.01
    cmix_value = torch.randn_like(cmix_key) * 0.01
    pools = (initial.clone(), initial.clone())
    saves = (torch.full_like(x, 123), torch.full_like(x, 123))
    cu = torch.tensor([0, length, -1], device=device, dtype=torch.int32)
    indices = cu.new_tensor([3, 99])
    num_active_tokens = cu.new_tensor([length])
    num_active_sequences = cu.new_tensor([1])

    def infer(values, residual, offsets, slots, states, normalized, ticket=None):
        tmix = infer_tmix_postnorm_tokenshift_forward_varlen(
            values, residual, weight, bias, *params,
            shift_state_pool=states[0], cu_seqlens=offsets, state_indices=slots,
            validated_metadata=ticket, save_normalized=normalized[0],
        )
        cmix = infer_cmix_forward_varlen(
            values, residual, weight, bias, params[0], cmix_key, cmix_value,
            shift_state_pool=states[1], cu_seqlens=offsets, state_indices=slots,
            validated_metadata=ticket, save_normalized=normalized[1],
        )
        return (*tmix, *cmix)

    def launch():
        ticket = prepare_tmix_wkv7_recurrent_metadata(
            cu, indices, state_pool_size=num_slots, token_capacity=capacity,
            sequence_capacity=2, max_seqlen_capacity=length,
            num_active_tokens=num_active_tokens,
            num_active_sequences=num_active_sequences,
        )
        return infer(x, res, cu, indices, pools, saves, ticket)

    expected_pools = (initial.clone(), initial.clone())
    expected_saves = (torch.empty_like(x[:length]), torch.empty_like(x[:length]))
    expected = infer(x[:length], res[:length], cu[:2], indices[:1], expected_pools, expected_saves)
    current = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        launch()
    current.wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = launch()

    for pool, saved in zip(pools, saves, strict=True):
        pool.copy_(initial)
        saved.fill_(123)
    graph.replay()
    for actual, reference in zip(outputs, expected, strict=True):
        assert torch.equal(actual[:length], reference)
    for pool, saved, expected_pool, expected_saved in zip(pools, saves, expected_pools, expected_saves, strict=True):
        assert torch.equal(pool, expected_pool)
        assert torch.equal(saved[:length], expected_saved)
        assert (saved[length:] == 123).all()
        pool.copy_(initial)
        saved.fill_(123)

    indices.fill_(99)
    num_active_tokens.zero_()
    num_active_sequences.zero_()
    graph.replay()
    for pool, saved in zip(pools, saves, strict=True):
        assert torch.equal(pool, initial)
        assert (saved == 123).all()
