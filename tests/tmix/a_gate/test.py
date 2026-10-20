# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flashrwkv2.tmix.a_gate import pretrain_tmix_a_gate_bf16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
@pytest.mark.memcheck
@pytest.mark.parametrize("channels", (8, 768, 4096))
def test_a_gate_forward_backward(channels) -> None:
    torch.manual_seed(3)
    device = torch.device("cuda")
    a0 = (
        (torch.randn(channels, device=device) * 0.1).to(torch.bfloat16).requires_grad_()
    )
    a12 = (
        (torch.randn(2, 17, channels, device=device) * 0.1)
        .to(torch.bfloat16)
        .requires_grad_()
    )
    output = pretrain_tmix_a_gate_bf16(a0, a12)
    reference = torch.sigmoid(a0.float() + a12.float()).to(torch.bfloat16)
    assert torch.allclose(output.float(), reference.float(), atol=0.01, rtol=0.01)
    upstream = torch.randn_like(output)
    output.backward(upstream)
    expected = torch.sigmoid(a0.detach().float() + a12.detach().float())
    expected_grad = upstream.float() * expected * (1.0 - expected)
    assert torch.allclose(a12.grad.float(), expected_grad, atol=0.02, rtol=0.02)
    assert torch.allclose(
        a0.grad.float(), expected_grad.sum((0, 1)), atol=0.02, rtol=0.02
    )


def test_a_gate_rejects_cpu_before_native_launch() -> None:
    a0 = torch.zeros(8, dtype=torch.bfloat16)
    a12 = torch.zeros(1, 2, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA"):
        pretrain_tmix_a_gate_bf16(a0, a12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
@pytest.mark.parametrize("kind", ("sigmoid", "tanh", "relu2", "identity"))
def test_activation_batch_invariance_and_vjp(kind):
    from flashrwkv2._internal.elementwise import activation

    torch.manual_seed(83)
    x = torch.randn(65, 96, device="cuda", dtype=torch.bfloat16).requires_grad_()
    bias = torch.randn(96, device="cuda", dtype=torch.bfloat16).requires_grad_()
    result = activation(x, kind, bias)
    chunks = torch.cat([activation(part, kind, bias) for part in x.detach().split(7)])
    assert torch.equal(result, chunks)
    for row in (0, 1, 31, 64):
        assert torch.equal(
            result[row : row + 1], activation(x[row : row + 1], kind, bias)
        )
    xr, br = x.detach().clone().requires_grad_(), bias.detach().clone().requires_grad_()
    z = xr.float() + br.float()
    if kind == "sigmoid":
        reference = z.sigmoid()
    elif kind == "tanh":
        reference = z.tanh()
    elif kind == "relu2":
        reference = z.relu().square()
    else:
        reference = z
    reference = reference.to(torch.bfloat16)
    upstream = torch.randn_like(result)
    actual = torch.autograd.grad(result, (x, bias), upstream)
    expected = torch.autograd.grad(reference, (xr, br), upstream)
    torch.testing.assert_close(result, reference, atol=0.015625, rtol=0.01)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, atol=0.0625, rtol=0.01)
