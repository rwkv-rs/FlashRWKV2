# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect

import pytest
import torch

from flashrwkv2.tmix.kk_pre import pretrain_tmix_kk_pre_bf16


def test_kk_pre_head_size_api() -> None:
    assert (
        inspect.signature(pretrain_tmix_kk_pre_bf16).parameters["head_size"].default
        == 64
    )


def test_kk_pre_rejects_cpu_before_native_launch() -> None:
    key = torch.zeros(1, 1, 64, dtype=torch.bfloat16)
    vector = torch.zeros(64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA"):
        pretrain_tmix_kk_pre_bf16(key, vector, key, vector)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
@pytest.mark.memcheck
@pytest.mark.parametrize("head_size", (64, 128, 256))
@pytest.mark.parametrize(("b", "t"), ((1, 1), (2, 17)))
def test_kk_pre_forward_backward(head_size: int, b: int, t: int) -> None:
    torch.manual_seed(19)
    device = torch.device("cuda")
    c = head_size * 2
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
    scaled_key = key.float().view(b, t, heads, head_size) * key_scale.float().view(
        1, 1, heads, head_size
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
        torch.testing.assert_close(
            actual.float(), reference.float(), atol=0.05, rtol=0.05
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
def test_kk_pre_rejects_non_head_aligned_channels() -> None:
    device = torch.device("cuda")
    key = torch.zeros(1, 1, 65, device=device, dtype=torch.bfloat16)
    vector = torch.zeros(65, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="divisible by head_size"):
        pretrain_tmix_kk_pre_bf16(key, vector, key, vector)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
@pytest.mark.parametrize("head_size", (64, 128, 256))
def test_kk_pre_batch_invariance(head_size):
    torch.manual_seed(97)
    key = torch.randn(65, 4096, device="cuda", dtype=torch.bfloat16)
    rate = torch.randn_like(key)
    ks = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    ka = torch.randn_like(ks)
    output = pretrain_tmix_kk_pre_bf16(key, ks, rate, ka, head_size=head_size)
    pieces = [
        pretrain_tmix_kk_pre_bf16(k, ks, a, ka, head_size=head_size)
        for k, a in zip(key.split(7), rate.split(7), strict=True)
    ]
    for index in range(3):
        assert torch.equal(output[index], torch.cat([piece[index] for piece in pieces]))
    for row in (0, 1, 31, 64):
        single = pretrain_tmix_kk_pre_bf16(
            key[row : row + 1], ks, rate[row : row + 1], ka, head_size=head_size
        )
        for full, one in zip(output, single, strict=True):
            assert torch.equal(full[row : row + 1], one)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
def test_kk_pre_zero_norm_vjp_with_unused_outputs():
    key = torch.zeros(33, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    scale = torch.ones(64, device="cuda", dtype=torch.bfloat16)
    rate = torch.zeros_like(key)
    _, a, _ = pretrain_tmix_kk_pre_bf16(key, scale, rate, scale)
    a.sum().backward()
    reference = key.detach().float().requires_grad_()
    torch.nn.functional.normalize(reference, dim=-1, eps=1.0e-12).neg().sum().backward()
    assert torch.isfinite(key.grad).all()
    torch.testing.assert_close(key.grad.float(), reference.grad, atol=0.0, rtol=0.005)
