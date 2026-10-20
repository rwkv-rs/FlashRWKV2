# SPDX-License-Identifier: MIT

import pytest
import torch
import tilelang

from flashrwkv2._internal.state import _transfer
from flashrwkv2.post_norm.tilelang import norm
from flashrwkv2.tmix.tokenshift.tilelang import (
    _backward, _forward, _fused_forward, fused_post_norm_tokenshift, tokenshift,
)


@pytest.mark.parametrize("arch", ("sm_89", "sm_120"))
@pytest.mark.parametrize("channels", (7, 128, 1024, 4096, 7168))
def test_tokenshift_lowering(arch, channels):
    with tilelang.tvm.target.Target({"kind": "cuda", "arch": arch}) as target:
        for block in (128, 256, 512):
            tilelang.lower(_forward.get_tir(channels, 6, 5, "int32", block), target=target)
        tilelang.lower(_forward.get_tir(channels, 6, 5, "int32", 256, True), target=target)
        tilelang.lower(_backward.get_tir(channels, 6, 5, "int64"), target=target)
        for write in (False, True):
            tilelang.lower(_transfer.get_tir(channels, 5, "bfloat16", write), target=target)
        for count in (1, 6):
            tilelang.lower(_fused_forward.get_tir(channels, count, 1e-5, 2), target=target)
        tilelang.lower(_backward.get_tir(channels, 6, 5, "int32", 256, 32, True), target=target)
        tilelang.lower(_backward.get_tir(channels, 1, 5, "int32", 256, 32, False), target=target)


def reference(x, mixes, offsets, initial):
    previous, final = [], []
    for b, (start, end) in enumerate(zip(offsets, offsets[1:])):
        if end > start:
            previous.append(torch.cat((initial[b:b + 1], x[start:end - 1])))
            final.append(x[end - 1])
        else:
            final.append(initial[b])
    previous = torch.cat(previous)
    outputs = tuple(
        (x.float() + (previous.float() - x.float()) * mix.float()).bfloat16()
        for mix in mixes
    )
    return outputs + (torch.stack(final),)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("channels", (7, 128, 1024, 4096, 7168))
def test_tokenshift_bitwise_packing_chunks_and_grad_mode(channels):
    torch.manual_seed(711)
    offsets = [0, 0, 1, 5, 5, 12]
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    x = torch.randn(12, channels, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mixes = torch.randn(6, channels, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    initial = torch.randn(5, channels, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    outputs, final = tokenshift(x, mixes, cu, initial)
    with torch.no_grad():
        infer, infer_final = tokenshift(x, mixes, cu, initial)
        collected = [[] for _ in range(6)]
        states = []
        for b, (start, end) in enumerate(zip(offsets, offsets[1:])):
            state = initial[b:b + 1].contiguous()
            for left in range(start, end, 3):
                right = min(left + 3, end)
                metadata = torch.tensor([0, right - left], device="cuda", dtype=torch.int32)
                part, state = tokenshift(x[left:right].contiguous(), mixes, metadata, state)
                for values, result in zip(collected, part, strict=True):
                    values.append(result)
            states.append(state)
    for training, inference, chunks in zip(outputs, infer, collected, strict=True):
        assert torch.equal(training, inference)
        assert torch.equal(training, torch.cat(chunks))
    assert torch.equal(final, infer_final)
    assert torch.equal(final, torch.cat(states))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tokenshift_independent_vjp_with_empty_sequences():
    torch.manual_seed(712)
    offsets = [0, 0, 1, 5, 5, 12]
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    inputs = (
        torch.randn(12, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True),
        torch.randn(6, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True),
        torch.randn(5, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True),
    )
    x, mixes, initial = inputs
    outputs, final = tokenshift(x, mixes, cu, initial)
    actual = (*outputs, final)
    refs = [t.detach().clone().requires_grad_() for t in inputs]
    expected = reference(refs[0], refs[1], offsets, refs[2])
    upstream = tuple(torch.randn_like(t) for t in actual)
    gradients = torch.autograd.grad(actual, inputs, upstream)
    expected_grads = torch.autograd.grad(expected, refs, upstream)
    for output, ref in zip(actual, expected, strict=True):
        assert torch.equal(output, ref)
    for gradient, ref in zip(gradients, expected_grads, strict=True):
        torch.testing.assert_close(gradient, ref, atol=0.125, rtol=0.02)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("channels", (128, 4096, 7168))
@pytest.mark.parametrize("count", (1, 6))
@pytest.mark.parametrize("batch", (1, 5))
def test_fused_t1_matches_training_and_preserves_inactive_slots(channels, count, batch):
    torch.manual_seed(713)
    x = torch.randn(batch, channels, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    res = torch.randn_like(x)
    weight = torch.randn(channels, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn_like(weight)
    mixes = torch.randn(count, channels, device="cuda", dtype=torch.bfloat16)
    original = torch.randn(batch + 2, channels, device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(batch, device="cuda", dtype=torch.int32).flip(0) + 1
    status = torch.tensor([0, batch, batch], device="cuda", dtype=torch.int32)
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    normalized, total = norm(x, weight, bias, res=res, return_res=True)
    expected, final = tokenshift(normalized, mixes, cu, original.index_select(0, indices.long()))
    expected_pool = original.clone().index_copy_(0, indices.long(), final)
    for group in (1, 2, 4):
        pool = original.clone()
        actual_total, actual = _fused_forward(channels, count, 1e-5, group)(
            x, res, weight, bias, mixes, pool, indices, status
        )
        assert torch.equal(total, actual_total)
        assert torch.equal(torch.stack(expected), actual)
        assert torch.equal(expected_pool, pool)
    status.zero_()
    indices.fill_(99)
    pool = original.clone()
    inactive_total, inactive_outputs = fused_post_norm_tokenshift(
        x, res, weight, bias, mixes, pool, indices, status
    )
    assert torch.equal(pool, original)
    assert torch.isnan(inactive_total).all()
    assert all(torch.isnan(output).all() for output in inactive_outputs)
