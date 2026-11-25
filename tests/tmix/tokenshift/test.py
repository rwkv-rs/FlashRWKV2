# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch
from utils import TorchOperationRecorder, assert_relative_rmse

from flashrwkv2.tmix.tokenshift import (
    infer_tmix_postnorm_tokenshift_forward_varlen,
    pretrain_tmix_tokenshift_bf16,
)
from flashrwkv2.tmix.wkv7 import prepare_tmix_wkv7_recurrent_metadata

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.cuda,
]


def _bf16(shape, scale=1):
    return torch.randn(shape, device="cuda", dtype=torch.bfloat16).mul_(scale)


def _leaf(shape, scale=1):
    return (torch.randn(shape, device="cuda") * scale).to(torch.bfloat16).requires_grad_()


def _clone_leaves(*tensors):
    return tuple(t.detach().clone().requires_grad_() for t in tensors)


def _norm_reference(x, res, weight, bias, eps=1.0e-5):
    summed = x + res
    values = summed.float()
    centered = values - values.mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + eps)
    return summed, (centered * rstd * weight.float() + bias.float()).to(torch.bfloat16)


def _tokenshift_reference(x, initial, mixes):
    previous = torch.cat((initial[:, None], x[:, :-1]), dim=1).float()
    values = x.float()
    return tuple((values + (previous - values) * mix.float()).to(x.dtype) for mix in mixes)


@pytest.mark.parametrize(
    ("shape", "zero_shift"),
    [
        pytest.param((2, 3, 8), True, marks=(pytest.mark.memcheck, pytest.mark.racecheck)),
        ((2, 7, 7), False),
        *[((b, t, 128), zero) for b in (1, 2) for t in (1, 7, 31) for zero in (False, True)],
    ],
)
def test_pretrain_tokenshift_matches_reference(shape, zero_shift) -> None:
    b, t, c = shape
    torch.manual_seed(1000 + b * 100 + t)
    x = _leaf(shape, 0.05)
    initial = _leaf((b, c), 0 if zero_shift else 0.05)
    mixes = tuple(_leaf((c,), 0.1) for _ in range(6))
    leaves = (x, initial, *mixes)
    ref_x, ref_initial, *ref_mixes = _clone_leaves(*leaves)
    expected = (*_tokenshift_reference(ref_x, ref_initial, ref_mixes), ref_x[:, -1])
    actual = pretrain_tmix_tokenshift_bf16(x, *mixes, initial_shift=initial, return_final_state=True)
    for output, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)

    upstream = tuple(torch.randn_like(output) for output in actual)
    actual_grads = torch.autograd.grad(actual, leaves, upstream)
    ref_grads = torch.autograd.grad(expected, (ref_x, ref_initial, *ref_mixes), upstream)
    for actual_grad, ref_grad in zip(actual_grads, ref_grads, strict=True):
        torch.testing.assert_close(actual_grad, ref_grad, atol=0.03, rtol=0.03)
    if zero_shift:
        default = pretrain_tmix_tokenshift_bf16(x, *mixes)
        for output, reference in zip(default, actual[:-1], strict=True):
            assert torch.equal(output, reference)


def test_pretrain_tokenshift_chunk_composition() -> None:
    torch.manual_seed(23)
    b, t, c, split = 2, 7, 128, 3
    x, initial = _leaf((b, t, c)), _leaf((b, c))
    params = tuple(_leaf((c,)) for _ in range(6))
    whole = pretrain_tmix_tokenshift_bf16(x, *params, initial_shift=initial, return_final_state=True)
    upstream = [torch.randn_like(output) for output in whole]
    whole_grads = torch.autograd.grad(
        whole, (x, initial, *params), grad_outputs=upstream
    )

    x_chunked, initial_chunked, *params_chunked = _clone_leaves(x, initial, *params)
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
    initial = _bf16((5, c))
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
    summed, normalized = _norm_reference(x, res, weight, bias, eps)
    expected = _tokenshift_reference(normalized[:, None], initial[[3]], params)
    torch.testing.assert_close(outputs[0], summed, atol=0.01, rtol=0.01)
    for actual, reference in zip(outputs[1:], expected, strict=True):
        torch.testing.assert_close(actual, reference[:, 0], atol=0.01, rtol=0.01)
    expected_state = initial.clone()
    expected_state[3] = normalized[0]
    assert torch.allclose(shift_state, expected_state, atol=0.01, rtol=0.01)
    assert torch.equal(saved[0], shift_state[3])
    assert torch.equal(shift_state[[0, 1, 2, 4]], initial[[0, 1, 2, 4]])


@pytest.mark.parametrize("batch_size", (4, 16, 64, 320))
def test_infer_tmix_t1_varlen_preserves_fragmented_slots(
    batch_size: int,
) -> None:
    torch.manual_seed(20260823 + batch_size)
    device = torch.device("cuda")
    channels = 4096
    x = _bf16((batch_size, channels), 0.02)
    res = torch.randn_like(x).mul_(0.02)
    weight = _bf16((channels,), 0.05).add_(1)
    bias = _bf16((channels,), 0.01)
    params = [
        _bf16((channels,), 0.1)
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

    summed, normalized = _norm_reference(x, res, weight, bias)
    previous = initial.index_select(0, slots.long())
    expected = _tokenshift_reference(normalized[:, None], previous, params)
    torch.testing.assert_close(outputs[0], summed, atol=0.01, rtol=0.01)
    for actual, reference in zip(outputs[1:], expected, strict=True):
        torch.testing.assert_close(actual, reference[:, 0], atol=0.02, rtol=0.02)
    torch.testing.assert_close(
        state.index_select(0, slots.to(torch.int64)).float(),
        normalized.float(),
        atol=0.01,
        rtol=0.01,
    )
    assert torch.equal(state[batch_size:], initial[batch_size:])


def test_infer_tokenshift_generic_packed_ragged_matches_reference() -> None:
    torch.manual_seed(20260822)
    device = torch.device("cuda")
    rows, channels = 4, 4096
    x = _bf16((rows, channels), 0.02)
    res = torch.randn_like(x) * 0.02
    weight = _bf16((channels,), 0.05) + 1
    bias = _bf16((channels,), 0.01)
    params = [
        _bf16((channels,), 0.1)
        for _ in range(6)
    ]
    initial = _bf16((5, channels), 0.02)
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
    summed, normalized = _norm_reference(x, res, weight, bias)
    expected = (
        _tokenshift_reference(normalized[:1][None], initial[[3]], params),
        _tokenshift_reference(normalized[1:][None], initial[[1]], params),
    )
    torch.testing.assert_close(outputs[0], summed, atol=0.01, rtol=0.01)
    for actual, (first, second) in zip(outputs[1:], zip(*expected, strict=True), strict=True):
        torch.testing.assert_close(actual, torch.cat((first, second), dim=1)[0], atol=0.02, rtol=0.02)
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

    x_train, initial_train, *mixes = _clone_leaves(saved, initial[slots.long()], *params)
    packed = pretrain_tmix_tokenshift_bf16(
        x_train, *mixes, cu_seqlens=cu.long(), initial_shift=initial_train, return_final_state=True,
    )
    independent = [
        pretrain_tmix_tokenshift_bf16(
            sample[None], *mixes, initial_shift=initial_train[seq:seq + 1], return_final_state=True,
        )
        for seq, sample in enumerate(x_train.split((1, 3)))
    ]
    dense = tuple(torch.cat(parts, dim=-2).squeeze(0) for parts in zip(*independent, strict=True))
    for actual, reference in zip(packed, dense, strict=True):
        assert torch.equal(actual, reference)
    for actual, reference in zip(packed[:-1], outputs[1:], strict=True):
        assert torch.equal(actual, reference)
    assert torch.equal(packed[-1], state[slots.long()])

    leaves = (x_train, initial_train, *mixes)
    upstream = tuple(torch.randn_like(output) for output in packed)
    packed_grads = torch.autograd.grad(packed, leaves, upstream)
    dense_grads = torch.autograd.grad(dense, leaves, upstream)
    for actual, reference in zip(packed_grads[:2], dense_grads[:2], strict=True):
        assert torch.equal(actual, reference)
    # Separate backwards round each sequence's parameter gradient to BF16 before summing.
    for actual, reference in zip(packed_grads[2:], dense_grads[2:], strict=True):
        assert_relative_rmse(actual, reference, 0.01)


@pytest.mark.cuda_graph
@pytest.mark.parametrize("length,capacity", [(1, 2), (5, 8)])
def test_shift_state_export_cuda_graph_writes_only_active_rows(length, capacity):
    from flashrwkv2.cmix import infer_cmix_forward_varlen

    torch.manual_seed(20260809)
    device = torch.device("cuda")
    channels, num_slots = 4096, 4
    x = _bf16((capacity, channels))
    res = torch.randn_like(x)
    weight = _bf16((channels,))
    bias = _bf16((channels,))
    params = [
        _bf16((channels,))
        for _ in range(6)
    ]
    initial = _bf16((num_slots, channels))
    cmix_key = _bf16((64, channels), 0.01)
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
