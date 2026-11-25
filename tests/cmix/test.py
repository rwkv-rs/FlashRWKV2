# SPDX-License-Identifier: MIT

from __future__ import annotations

from itertools import accumulate

import pytest
import torch
from utils import TorchOperationRecorder, assert_relative_rmse

from flashrwkv2.cmix import (
    infer_cmix_forward_varlen,
    pretrain_cmix_bf16,
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


def _cmix_leaves(b, t, c, *, zero_shift=False):
    return (
        _leaf((b, t, c), 0.05), _leaf((b, c), 0 if zero_shift else 0.05),
        _leaf((c,), 0.1), _leaf((4 * c, c), 0.05), _leaf((c, 4 * c), 0.05),
    )


def _norm_reference(x, res, weight, bias):
    summed = x + res
    values = summed.float()
    centered = values - values.mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + 1.0e-5)
    return summed, (centered * rstd * weight.float() + bias.float()).to(torch.bfloat16)


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


def _cmix_reference(x, initial, x_k, key, value, adapters=((), ())):
    previous = torch.cat((initial[:, None], x[:, :-1]), dim=1).float()
    x = x.float()
    mixed = (x + (previous - x) * x_k.float()).to(torch.bfloat16)
    features = _linear_reference(mixed, key, *adapters[0])
    activated = features.float().relu().square().to(torch.bfloat16)
    return _linear_reference(activated, value, *adapters[1])


@pytest.mark.parametrize(
    ("shape", "zero_shift"),
    [
        pytest.param((1, 3, 4), True, marks=(pytest.mark.memcheck, pytest.mark.racecheck)),
        ((1, 65536, 64), True),
        ((2, 7, 7), False),
        *[((b, t, 16), zero) for b in (1, 2) for t in (1, 7, 31) for zero in (False, True)],
    ],
)
def test_pretrain_cmix_matches_reference(shape, zero_shift) -> None:
    b, t, c = shape
    torch.manual_seed(2000 + b * 100 + t)
    x, initial, x_k, key, value = leaves = _cmix_leaves(b, t, c, zero_shift=zero_shift)
    ref_x, *ref_args = ref_leaves = _clone_leaves(*leaves)
    expected = (_cmix_reference(ref_x, *ref_args), ref_x[:, -1])
    actual = pretrain_cmix_bf16(x, x_k, key, value, initial_shift=initial, return_final_state=True)
    for output, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(output, reference, atol=0.03, rtol=0.03)
    upstream = tuple(torch.randn_like(output) for output in actual)
    actual_grads = torch.autograd.grad(actual, leaves, upstream)
    ref_grads = torch.autograd.grad(expected, ref_leaves, upstream)
    for actual_grad, ref_grad in zip(actual_grads, ref_grads, strict=True):
        torch.testing.assert_close(actual_grad, ref_grad, atol=0.06, rtol=0.06)
    if zero_shift:
        assert torch.equal(pretrain_cmix_bf16(x, x_k, key, value), actual[0])


@pytest.mark.memcheck
@pytest.mark.racecheck
def test_infer_cmix_highest_fusion_island_matches_reference() -> None:
    torch.manual_seed(20260820)
    device = torch.device("cuda")
    channels = 4096
    features = 4096
    x = _bf16((1, channels), 0.02)
    res = torch.randn_like(x) * 0.02
    weight = _bf16((channels,), 0.05) + 1
    bias = _bf16((channels,), 0.01)
    x_k = _bf16((channels,), 0.1)
    key = _bf16((features, channels), 0.01)
    value = _bf16((features, channels), 0.01)
    initial = _bf16((3, channels), 0.02)
    shift = initial.clone()
    saved = torch.full_like(x, 123)
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
            save_normalized=saved,
        )
    assert "aten::max" not in recorder.operations
    summed_ref, normalized_bf16 = _norm_reference(x, res, weight, bias)
    expected_output = _cmix_reference(
        normalized_bf16[:, None], initial[[2]], x_k, key, value.t(),
    )[:, 0]
    torch.testing.assert_close(summed, summed_ref, atol=0.01, rtol=0.01)
    torch.testing.assert_close(output, expected_output, atol=0.10, rtol=0.06)
    torch.testing.assert_close(shift[2], normalized_bf16[0], atol=0.01, rtol=0.01)
    assert torch.equal(saved[0], shift[2])
    assert torch.equal(shift[:2], initial[:2])


@pytest.mark.parametrize("batch_size", (16, 64, 320, 960))
def test_infer_cmix_t1_fused_varlen_preserves_fragmented_slots(
    batch_size: int,
) -> None:
    torch.manual_seed(20260824 + batch_size)
    device = torch.device("cuda")
    channels, features = 4096, 128
    x = _bf16((batch_size, channels), 0.02)
    res = torch.randn_like(x).mul_(0.02)
    weight = (
        _bf16((channels,), 0.05).add_(1)
    )
    bias = _bf16((channels,), 0.01)
    x_k = _bf16((channels,), 0.1)
    key = _bf16((features, channels), 0.01)
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

    summed_ref, normalized = _norm_reference(x, res, weight, bias)
    previous = initial.index_select(0, slots.long())
    expected_output = _cmix_reference(normalized[:, None], previous, x_k, key, value.t())[:, 0]
    torch.testing.assert_close(summed, summed_ref, atol=0.01, rtol=0.01)
    torch.testing.assert_close(output, expected_output, atol=0.12, rtol=0.07)
    torch.testing.assert_close(
        state.index_select(0, slots.to(torch.int64)).float(),
        normalized.float(),
        atol=0.01,
        rtol=0.01,
    )
    assert torch.equal(state[batch_size:], initial[batch_size:])


def test_infer_cmix_packed_ragged_matches_reference_and_slot_updates() -> None:
    torch.manual_seed(20260821)
    device = torch.device("cuda")
    rows, channels, features = 4, 4096, 4096
    x = _bf16((rows, channels), 0.02)
    res = torch.randn_like(x) * 0.02
    weight = _bf16((channels,), 0.05) + 1
    bias = _bf16((channels,), 0.01)
    x_k = _bf16((channels,), 0.1)
    key = _bf16((features, channels), 0.01)
    value = _bf16((features, channels), 0.01)
    initial = _bf16((5, channels), 0.02)
    state = initial.clone()
    saved = torch.full_like(x, 123)
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
        save_normalized=saved,
    )
    summed_ref, normalized = _norm_reference(x, res, weight, bias)
    expected_output = torch.cat([
        _cmix_reference(normalized[:1][None], initial[[3]], x_k, key, value.t())[0],
        _cmix_reference(normalized[1:][None], initial[[1]], x_k, key, value.t())[0],
    ])
    torch.testing.assert_close(summed, summed_ref, atol=0.01, rtol=0.01)
    torch.testing.assert_close(output, expected_output, atol=0.12, rtol=0.07)
    torch.testing.assert_close(
        state[3].float(), normalized[0].float(), atol=0.01, rtol=0.01
    )
    torch.testing.assert_close(
        state[1].float(), normalized[3].float(), atol=0.01, rtol=0.01
    )
    assert torch.equal(state[[0, 2, 4]], initial[[0, 2, 4]])
    decode_state = initial.clone()
    one = cu.new_tensor([0, 1])
    for row, seq in enumerate((0, 1, 1, 1)):
        decode_sum, decode_output = infer_cmix_forward_varlen(
            x[row:row + 1], res[row:row + 1], weight, bias, x_k, key, value,
            shift_state_pool=decode_state, cu_seqlens=one, state_indices=slots[seq:seq + 1],
        )
        assert torch.equal(saved[row:row + 1], decode_state[slots[seq:seq + 1].long()])
        assert torch.equal(summed[row:row + 1], decode_sum)
        assert torch.equal(output[row:row + 1], decode_output)
    assert torch.equal(state, decode_state)


def test_infer_cmix_ticket_mutation_fails_without_state_write() -> None:
    device = torch.device("cuda")
    rows, channels, features = 2, 4096, 4096
    x = torch.zeros(rows, channels, device=device, dtype=torch.bfloat16)
    vector = torch.zeros(channels, device=device, dtype=torch.bfloat16)
    key = torch.zeros(features, channels, device=device, dtype=torch.bfloat16)
    state = _bf16((3, channels))
    before = state.clone()
    cu = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
    slots = torch.tensor([0, 2], device=device, dtype=torch.int32)
    ticket = prepare_tmix_wkv7_recurrent_metadata(
        cu, slots, total_tokens=rows, state_pool_size=3, max_seqlen=1
    )
    slots[0] = 1
    with pytest.raises(RuntimeError):
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


def test_pretrain_cmix_chunk_composition() -> None:
    torch.manual_seed(29)
    b, t, c, split = 2, 7, 16, 3
    # Use the activation range of the reference tests to limit BF16
    # gradient rounding when summing contributions from separate chunks.
    x, initial, x_k, key, value = _cmix_leaves(b, t, c)
    whole = pretrain_cmix_bf16(x, x_k, key, value, initial_shift=initial, return_final_state=True)
    upstream = [torch.randn_like(output) for output in whole]
    whole_grads = torch.autograd.grad(
        whole, (x, initial, x_k, key, value), grad_outputs=upstream
    )

    chunked_leaves = _clone_leaves(x, initial, x_k, key, value)
    x_chunked, initial_chunked, x_k_chunked, key_chunked, value_chunked = chunked_leaves
    first = pretrain_cmix_bf16(
        x_chunked[:, :split].contiguous(), x_k_chunked, key_chunked, value_chunked,
        initial_shift=initial_chunked, return_final_state=True,
    )
    second = pretrain_cmix_bf16(
        x_chunked[:, split:].contiguous(), x_k_chunked, key_chunked, value_chunked,
        initial_shift=first[1], return_final_state=True,
    )
    chunked = (torch.cat((first[0], second[0]), dim=1), second[1])
    chunked_grads = torch.autograd.grad(
        chunked,
        chunked_leaves,
        grad_outputs=upstream,
    )
    for expected, actual in zip(whole, chunked, strict=True):
        assert torch.equal(expected, actual)
    for expected, actual in zip(whole_grads, chunked_grads, strict=True):
        assert torch.allclose(expected, actual, atol=0.06, rtol=0.06)


@pytest.mark.parametrize("stateful", [False, True])
def test_cmix_nora_training(stateful):
    torch.manual_seed(73)
    c, rank = 64, 256
    x, mix, initial = _leaf((2, 17, c), 0.1), _leaf((c,), 0.1), _leaf((2, c), 0.1)
    key, value = _bf16((4 * c, c), 0.03), _bf16((c, 4 * c), 0.03)
    ka, kb, va, vb = [
        (torch.randn(shape, device="cuda") * 0.03).requires_grad_()
        for shape in ((rank, c), (4 * c, rank), (rank, 4 * c), (c, rank))
    ]
    tensors = [x, mix, ka, kb, va, vb] + ([initial] if stateful else [])
    refs = _clone_leaves(*tensors)
    options = {
        "key_lora_a": ka,
        "key_lora_b": kb,
        "value_lora_a": va,
        "value_lora_b": vb,
        "key_use_nora": True,
        "value_use_nora": True,
    }
    if stateful:
        output, next_shift = pretrain_cmix_bf16(
            x, mix, key, value,
            initial_shift=initial, return_final_state=True, **options,
        )
    else:
        output = pretrain_cmix_bf16(x, mix, key, value, **options)
    xr, mr, kar, kbr, var, vbr = refs[:6]
    ref_initial = refs[-1] if stateful else torch.zeros_like(initial)
    expected = _cmix_reference(xr, ref_initial, mr, key, value, ((kar, kbr), (var, vbr)))
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


@pytest.mark.parametrize("channels", (768, 4096))
def test_cmix_training_packed_and_decode_are_bitwise_equal(channels):
    from flashrwkv2.post_norm.tilelang import norm

    torch.manual_seed(103)
    lengths, slot_ids = (1, 3, 7, 16), (4, 2, 0, 3)
    rows, features = sum(lengths), 128
    x = _bf16((rows, channels), 0.03)
    res = torch.randn_like(x) * 0.03
    weight = torch.ones_like(x[0])
    bias, mix = _bf16((channels,), 0.01), _bf16((channels,), 0.1)
    key = _bf16((features, channels), 0.02)
    value = _bf16((features, channels), 0.02)
    initial = _bf16((6, channels), 0.03)
    offsets = list(accumulate(lengths, initial=0))
    cu = torch.tensor(offsets, device=x.device, dtype=torch.int32)
    slots = torch.tensor(slot_ids, device=x.device, dtype=torch.int32)
    normalized, train_res = norm(x, weight, bias, res=res, return_res=True)
    train_x, train_initial, train_mix, train_key, train_value = _clone_leaves(
        normalized, initial[slots.long()], mix, key, value.t(),
    )
    leaves = (train_x, train_initial, train_mix, train_key, train_value)
    dense = [
        pretrain_cmix_bf16(
            sample[None], train_mix, train_key, train_value,
            initial_shift=train_initial[seq:seq + 1], return_final_state=True,
        )
        for seq, sample in enumerate(train_x.split(lengths))
    ]
    dense_output = torch.cat([output[0] for output, _ in dense])
    dense_final = torch.cat([final for _, final in dense])
    training = pretrain_cmix_bf16(
        train_x, train_mix, train_key, train_value, cu_seqlens=cu,
        initial_shift=train_initial, return_final_state=True,
    )
    assert torch.equal(training[0], dense_output)
    assert torch.equal(training[1], dense_final)
    upstream = tuple(torch.randn_like(output) for output in training)
    packed_grads = torch.autograd.grad(training, leaves, upstream)
    dense_grads = torch.autograd.grad((dense_output, dense_final), leaves, upstream)
    for actual, reference in zip(packed_grads, dense_grads, strict=True):
        assert_relative_rmse(actual, reference, 0.01)

    expected_state = initial.clone()
    expected_state[slots.long()] = dense_final.detach()
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
    assert torch.equal(packed, training[0])
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


def test_cmix_invalid_projection_preserves_t1_shift_state():
    channels, features = 64, 128
    x = torch.zeros(1, channels, device="cuda", dtype=torch.bfloat16)
    vector = torch.zeros_like(x[0])
    key = torch.zeros(features, channels - 1, device=x.device, dtype=x.dtype)
    value = torch.zeros(features, channels, device=x.device, dtype=x.dtype)
    state = _bf16((2, channels))
    before = state.clone()
    with pytest.raises(ValueError):
        infer_cmix_forward_varlen(
            x, x, torch.ones_like(vector), vector, vector, key, value,
            shift_state_pool=state,
            cu_seqlens=torch.tensor([0, 1], device=x.device, dtype=torch.int32),
            state_indices=torch.tensor([1], device=x.device, dtype=torch.int32),
            max_seqlen=1,
        )
    assert torch.equal(state, before)
