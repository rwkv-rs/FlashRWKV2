# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect

import pytest
import torch

from flashrwkv2.tmix.kk_pre import pretrain_tmix_kk_pre_bf16


def test_kk_pre_head_size_api() -> None:
    assert inspect.signature(pretrain_tmix_kk_pre_bf16).parameters["head_size"].default == 64


def test_kk_pre_rejects_cpu_before_native_launch() -> None:
    key = torch.zeros(1, 1, 64, dtype=torch.bfloat16)
    vector = torch.zeros(64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA"):
        pretrain_tmix_kk_pre_bf16(key, vector, key, vector)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.sm90
@pytest.mark.sm120
@pytest.mark.memcheck
@pytest.mark.parametrize("head_size", (64, 128, 256))
def test_kk_pre_forward_backward(head_size: int) -> None:
    torch.manual_seed(19)
    device = torch.device("cuda")
    b, t, c = 2, 3, head_size * 2
    tensors = [
        (torch.randn(b, t, c, device=device) * 0.1).to(torch.bfloat16).requires_grad_(),
        (torch.randn(c, device=device) * 0.1).to(torch.bfloat16).requires_grad_(),
        (torch.randn(b, t, c, device=device) * 0.1).to(torch.bfloat16).requires_grad_(),
        (torch.randn(c, device=device) * 0.1).to(torch.bfloat16).requires_grad_(),
    ]
    outputs = pretrain_tmix_kk_pre_bf16(*tensors, head_size=head_size)
    references = [tensor.detach().clone().requires_grad_() for tensor in tensors]
    key, key_scale, learning_rate, learning_rate_scale = references
    heads = c // head_size
    scaled_key = (
        key.float().view(b, t, heads, head_size)
        * key_scale.float().view(1, 1, heads, head_size)
    )
    normalized = scaled_key * torch.rsqrt(
        scaled_key.square().sum(dim=-1, keepdim=True).clamp_min(1.0e-24)
    )
    expected = (
        key.float()
        * (
            learning_rate.float() * learning_rate_scale.float()
            + 1.0
            - learning_rate_scale.float()
        ),
        -normalized.reshape_as(key),
        normalized.reshape_as(key) * learning_rate.float(),
    )
    for actual, reference in zip(outputs, expected, strict=True):
        torch.testing.assert_close(actual.float(), reference, atol=0.02, rtol=0.02)
    upstream = [torch.randn_like(output) for output in outputs]
    actual_grads = torch.autograd.grad(outputs, tensors, grad_outputs=upstream)
    expected_grads = torch.autograd.grad(
        expected, references, grad_outputs=[value.float() for value in upstream]
    )
    for actual, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual.float(), reference.float(), atol=0.05, rtol=0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.sm90
@pytest.mark.sm120
def test_kk_pre_rejects_non_head_aligned_channels() -> None:
    device = torch.device("cuda")
    key = torch.zeros(1, 1, 65, device=device, dtype=torch.bfloat16)
    vector = torch.zeros(65, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="divisible by head_size"):
        pretrain_tmix_kk_pre_bf16(key, vector, key, vector)
