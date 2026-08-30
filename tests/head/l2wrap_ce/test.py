# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch
from torch.nn import functional

from flashrwkv2.head.l2wrap_ce import pretrain_head_l2wrap_ce_bf16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
@pytest.mark.memcheck
@pytest.mark.parametrize("chunk_rows", (1, 2))
def test_head_l2wrap_forward_backward(chunk_rows: int) -> None:
    torch.manual_seed(29)
    device = torch.device("cuda")
    hidden = torch.randn(1, 3, 8, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = (torch.randn(65536, 8, device=device, dtype=torch.bfloat16) * 0.01).requires_grad_()
    targets = torch.tensor([3, 4096, 65535], device=device, dtype=torch.int64)
    loss = pretrain_head_l2wrap_ce_bf16(
        hidden, weight, targets, chunk_rows=chunk_rows
    )
    hidden_ref = hidden.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    logits = hidden_ref.float().reshape(-1, 8) @ weight_ref.float().t()
    reference_loss = functional.cross_entropy(logits, targets)
    maxima = logits.max(dim=-1).values
    l2wrap = 0.5e-4 / targets.numel() * maxima.square().sum()
    reference_with_gradient = reference_loss + l2wrap - l2wrap.detach()
    actual_grads = torch.autograd.grad(loss, (hidden, weight))
    expected_grads = torch.autograd.grad(
        reference_with_gradient, (hidden_ref, weight_ref)
    )
    torch.testing.assert_close(loss, reference_loss, atol=3.0e-4, rtol=3.0e-4)
    for actual, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual.float(), reference.float(), atol=0.004, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
def test_head_l2wrap_rejects_invalid_target() -> None:
    device = torch.device("cuda")
    hidden = torch.zeros(1, 1, 8, device=device, dtype=torch.bfloat16)
    weight = torch.zeros(65536, 8, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="targets"):
        pretrain_head_l2wrap_ce_bf16(hidden, weight, torch.tensor([65536], device=device))
