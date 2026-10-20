# SPDX-License-Identifier: MIT

import pytest
import torch
import tilelang

from flashrwkv2.tmix.readout.tilelang import _backward, _forward, readout


@pytest.mark.parametrize("arch", ("sm_89", "sm_120"))
@pytest.mark.parametrize("head_size", (64, 128, 256))
def test_readout_lowering(arch, head_size):
    with tilelang.tvm.target.Target({"kind": "cuda", "arch": arch}) as target:
        for dtype in ("bfloat16", "float32"):
            for save in (False, True):
                tilelang.lower(_forward.get_tir(4096, head_size, 64e-5, dtype, save), target=target)
            tilelang.lower(_backward.get_tir(4096, head_size, dtype), target=target)
            tilelang.lower(_forward.get_tir(4096, head_size, 64e-5, dtype, True, 16), target=target)
            for needed in (False, True):
                tilelang.lower(_backward.get_tir(4096, head_size, dtype, 32, (needed,) * 3), target=target)


def reference(inputs, head_size):
    x, r, k, v, r_k, weight, bias, gate = inputs
    shape = (*x.shape[:-1], -1, head_size)
    grouped = x.float().view(shape)
    centered = grouped - grouped.mean(-1, keepdim=True)
    normalized = centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + 64e-5)
    scale = (
        r.float().view(shape) * k.float().view(shape) * r_k.float().view(-1, head_size)
    ).sum(-1, keepdim=True)
    return (
        (normalized.view_as(x) * weight.float() + bias.float() + (scale * v.float().view(shape)).view_as(x))
        * gate.float()
    ).bfloat16()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("head_size", (64, 128, 256))
@pytest.mark.parametrize("channels", (1024, 4096, 7168))
def test_readout_bitwise_batch_grad_and_vjp(head_size, channels):
    torch.manual_seed(721)
    rows = [
        (torch.randn(17, channels, device="cuda") * 0.1).bfloat16().requires_grad_()
        for _ in range(5)
    ]
    vectors = [
        torch.randn(channels, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    inputs = (*rows[:4], *vectors, rows[4])
    actual = readout(*inputs, head_size)
    with torch.no_grad():
        infer = readout(*inputs, head_size)
        pieces = [
            readout(
                *(t[start:end] for t in rows[:4]), *vectors, rows[4][start:end], head_size,
            )
            for start, end in ((0, 1), (1, 8), (8, 17))
        ]
    assert torch.equal(actual, infer)
    assert torch.equal(actual, torch.cat(pieces))
    refs = [t.detach().clone().requires_grad_() for t in inputs]
    expected = reference(refs, head_size)
    upstream = torch.randn_like(actual)
    gradients = torch.autograd.grad(actual, inputs, upstream)
    expected_gradients = torch.autograd.grad(expected, refs, upstream)
    torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.01)
    for gradient, ref in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(gradient, ref, atol=0.03, rtol=0.02)
