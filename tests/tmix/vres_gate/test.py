# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flashrwkv2.tmix.vres_gate import pretrain_tmix_vres_gate_bf16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
@pytest.mark.memcheck
@pytest.mark.parametrize("shape", ((1, 2, 8), (1, 33, 768), (2, 17, 4096)))
def test_vres_gate_training_forward_backward(shape) -> None:
    torch.manual_seed(32)
    device = torch.device("cuda")
    value = (
        (torch.randn(shape, device=device) * 0.1).to(torch.bfloat16).requires_grad_()
    )
    first = torch.randn_like(value, requires_grad=True)
    v0 = torch.randn(shape[-1], device=device, dtype=torch.bfloat16, requires_grad=True)
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
        torch.testing.assert_close(
            actual.float(), reference.float(), atol=0.03, rtol=0.03
        )


def test_vres_gate_rejects_cpu_before_native_launch() -> None:
    value = torch.zeros(1, 2, 8, dtype=torch.bfloat16)
    vector = torch.zeros(8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA"):
        pretrain_tmix_vres_gate_bf16(value, value, vector, value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
def test_vres_gate_batch_invariance_and_frozen_bias():
    torch.manual_seed(89)
    value = torch.randn(65, 768, device="cuda", dtype=torch.bfloat16).requires_grad_()
    first = torch.randn_like(value, requires_grad=True)
    logits = torch.randn_like(value, requires_grad=True)
    bias = torch.randn(768, device="cuda", dtype=torch.bfloat16)
    result = pretrain_tmix_vres_gate_bf16(value, first, bias, logits)
    chunks = torch.cat(
        [
            pretrain_tmix_vres_gate_bf16(v, f, bias, l)
            for v, f, l in zip(
                value.split(7), first.split(7), logits.split(7), strict=True
            )
        ]
    )
    assert torch.equal(result, chunks)
    for row in (0, 1, 31, 64):
        single = pretrain_tmix_vres_gate_bf16(
            value[row : row + 1], first[row : row + 1], bias, logits[row : row + 1]
        )
        assert torch.equal(result[row : row + 1], single)
    upstream = torch.randn_like(result)
    got = torch.autograd.grad(result, (value, first, logits), upstream)
    refs = [t.detach().clone().requires_grad_() for t in (value, first, logits)]
    gate = (bias.float() + refs[2].float()).sigmoid()
    expected = (refs[0].float() + (refs[1].float() - refs[0].float()) * gate).to(
        torch.bfloat16
    )
    want = torch.autograd.grad(expected, refs, upstream)
    for actual, reference in zip(got, want, strict=True):
        torch.testing.assert_close(actual, reference, atol=0.03, rtol=0.03)
