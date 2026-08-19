# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect

import pytest
import torch

from flashrwkv2.tmix.readout import (
    infer_tmix_readout_forward_varlen,
    pretrain_tmix_readout_bf16,
)


def test_readout_public_contract() -> None:
    parameters = inspect.signature(infer_tmix_readout_forward_varlen).parameters
    assert list(parameters)[:9] == [
        "wkv_output", "receptance", "key", "value", "r_k",
        "ln_weight", "ln_bias", "gate", "output_weight",
    ]
    assert "output_lora_a" in parameters
    assert inspect.signature(pretrain_tmix_readout_bf16).parameters["head_size"].default == 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.sm120
@pytest.mark.memcheck
@pytest.mark.parametrize("head_size", [64, 128, 256])
def test_readout_inference_matches_reference(head_size: int) -> None:
    torch.manual_seed(20260816 + head_size)
    rows, channels = 2, 4096
    device = torch.device("cuda")
    packed = [torch.randn(rows, channels, device=device, dtype=torch.float16) for _ in range(5)]
    vectors = [torch.randn(channels, device=device, dtype=torch.float16) for _ in range(3)]
    output_weight = torch.randn(channels, channels, device=device, dtype=torch.float16)
    output = infer_tmix_readout_forward_varlen(
        packed[0], packed[1], packed[2], packed[3], vectors[0], vectors[1],
        vectors[2], packed[4], output_weight, head_size=head_size,
        batch_size=1, max_seqlen=rows,
    )
    heads = channels // head_size
    wkv = packed[0].float().view(rows, heads, head_size)
    receptance = packed[1].float().view(rows, heads, head_size)
    key = packed[2].float().view(rows, heads, head_size)
    value = packed[3].float().view(rows, heads, head_size)
    centered = wkv - wkv.mean(dim=-1, keepdim=True)
    normalized = centered * torch.rsqrt(
        centered.square().mean(dim=-1, keepdim=True) + 64.0e-5
    )
    residual = (
        receptance
        * key
        * vectors[0].float().view(1, heads, head_size)
    ).sum(dim=-1, keepdim=True)
    gated = (
        normalized * vectors[1].float().view(1, heads, head_size)
        + vectors[2].float().view(1, heads, head_size)
        + residual * value
    ) * packed[4].float().view(rows, heads, head_size)
    # The native readout island stores its prelinear result in FP16 before the
    # output projection consumes it; preserve that public numerical boundary.
    expected = (
        gated.reshape(rows, channels).to(torch.float16).float()
        @ output_weight.float().t()
    )
    torch.testing.assert_close(output.float(), expected, atol=0.08, rtol=0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.sm90
@pytest.mark.sm120
@pytest.mark.memcheck
@pytest.mark.parametrize("head_size", [64, 128, 256])
def test_readout_pretrain_matches_forward_and_gradient_reference(
    head_size: int,
) -> None:
    torch.manual_seed(20260817 + head_size)
    b, t, heads = 1, 2, 2
    channels = heads * head_size
    device = torch.device("cuda")
    tokens = [
        (torch.randn(b, t, channels, device=device) * 0.03)
        .to(torch.bfloat16)
        .requires_grad_()
        for _ in range(4)
    ]
    residual_scale = (
        torch.randn(heads, head_size, device=device) * 0.1
    ).to(torch.bfloat16).requires_grad_()
    weight = torch.randn(
        channels, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    bias = torch.randn(
        channels, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    gate = (
        torch.randn(b, t, channels, device=device) * 0.1
    ).to(torch.bfloat16).requires_grad_()
    inputs = (*tokens, residual_scale, weight, bias, gate)
    output = pretrain_tmix_readout_bf16(*inputs, head_size=head_size)

    references = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    x, r, k, v, scale, ln_weight, ln_bias, g = references
    x_heads = x.float().view(b, t, heads, head_size)
    centered = x_heads - x_heads.mean(dim=-1, keepdim=True)
    normalized = centered * torch.rsqrt(
        centered.square().mean(dim=-1, keepdim=True) + 64.0e-5
    )
    residual = (
        r.float().view(b, t, heads, head_size)
        * k.float().view(b, t, heads, head_size)
        * scale.float().view(1, 1, heads, head_size)
    ).sum(dim=-1, keepdim=True)
    expected = (
        normalized * ln_weight.float().view(1, 1, heads, head_size)
        + ln_bias.float().view(1, 1, heads, head_size)
        + residual * v.float().view(b, t, heads, head_size)
    ) * g.float().view(b, t, heads, head_size)
    expected = expected.reshape_as(x)
    torch.testing.assert_close(output.float(), expected, atol=0.04, rtol=0.04)
    upstream = torch.randn_like(output)
    actual_grads = torch.autograd.grad(output, inputs, grad_outputs=upstream)
    expected_grads = torch.autograd.grad(
        expected, references, grad_outputs=upstream.float()
    )
    for actual, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual.float(), reference.float(), atol=0.08, rtol=0.08)
