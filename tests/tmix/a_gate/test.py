# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flashrwkv2.tmix.a_gate import pretrain_tmix_a_gate_bf16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.sm90
@pytest.mark.memcheck
def test_a_gate_forward_backward() -> None:
    torch.manual_seed(3)
    device = torch.device("cuda")
    a0 = (torch.randn(8, device=device) * 0.1).to(torch.bfloat16).requires_grad_()
    a12 = (torch.randn(2, 3, 8, device=device) * 0.1).to(torch.bfloat16).requires_grad_()
    output = pretrain_tmix_a_gate_bf16(a0, a12)
    reference = torch.sigmoid(a0.float() + a12.float()).to(torch.bfloat16)
    assert torch.allclose(output.float(), reference.float(), atol=0.01, rtol=0.01)
    output.float().sum().backward()
    expected = torch.sigmoid(a0.detach().float() + a12.detach().float())
    expected_grad = expected * (1.0 - expected)
    assert torch.allclose(a12.grad.float(), expected_grad, atol=0.02, rtol=0.02)
    assert torch.allclose(a0.grad.float(), expected_grad.sum((0, 1)), atol=0.02, rtol=0.02)


def test_a_gate_rejects_cpu_before_native_launch() -> None:
    a0 = torch.zeros(8, dtype=torch.bfloat16)
    a12 = torch.zeros(1, 2, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA"):
        pretrain_tmix_a_gate_bf16(a0, a12)
