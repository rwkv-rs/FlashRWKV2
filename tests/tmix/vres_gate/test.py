# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flashrwkv2.tmix.vres_gate import pretrain_tmix_vres_gate_bf16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
@pytest.mark.memcheck
def test_vres_gate_training_forward_backward() -> None:
    torch.manual_seed(32)
    device = torch.device("cuda")
    value = (torch.randn(1, 2, 8, device=device) * 0.1).to(torch.bfloat16).requires_grad_()
    first = value.detach().clone().requires_grad_()
    v0 = torch.randn(8, device=device, dtype=torch.bfloat16, requires_grad=True)
    v12 = torch.randn_like(value, requires_grad=True)
    output = pretrain_tmix_vres_gate_bf16(value, first, v0, v12)
    value_ref = value.detach().clone().requires_grad_()
    first_ref = first.detach().clone().requires_grad_()
    v0_ref = v0.detach().clone().requires_grad_()
    v12_ref = v12.detach().clone().requires_grad_()
    gate = torch.sigmoid(v0_ref.float() + v12_ref.float())
    expected = value_ref.float() + (first_ref.float() - value_ref.float()) * gate
    torch.testing.assert_close(output.float(), expected, atol=0.01, rtol=0.01)
    upstream = torch.randn_like(output)
    actual_grads = torch.autograd.grad(
        output, (value, first, v0, v12), grad_outputs=upstream
    )
    expected_grads = torch.autograd.grad(
        expected, (value_ref, first_ref, v0_ref, v12_ref), grad_outputs=upstream.float()
    )
    for actual, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual.float(), reference.float(), atol=0.03, rtol=0.03)


def test_vres_gate_rejects_cpu_before_native_launch() -> None:
    value = torch.zeros(1, 2, 8, dtype=torch.bfloat16)
    vector = torch.zeros(8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA"):
        pretrain_tmix_vres_gate_bf16(value, value, vector, value)
