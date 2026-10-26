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
    hidden = torch.randn(
        1, 3, 8, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    weight = (
        torch.randn(65536, 8, device=device, dtype=torch.bfloat16) * 0.01
    ).requires_grad_()
    targets = torch.tensor([3, 4096, 65535], device=device, dtype=torch.int64)
    loss = pretrain_head_l2wrap_ce_bf16(hidden, weight, targets, chunk_rows=chunk_rows)
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
        torch.testing.assert_close(
            actual.float(), reference.float(), atol=0.004, rtol=0.02
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
def test_head_l2wrap_rejects_invalid_target() -> None:
    device = torch.device("cuda")
    hidden = torch.zeros(1, 1, 8, device=device, dtype=torch.bfloat16)
    weight = torch.zeros(65536, 8, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="targets"):
        pretrain_head_l2wrap_ce_bf16(
            hidden, weight, torch.tensor([65536], device=device)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
def test_chunked_head_nora_gradients():
    from flashrwkv2 import pretrain_head_linear_bf16, pretrain_l2wrap_ce_bf16
    from utils import assert_relative_rmse

    torch.manual_seed(83)
    hidden = (
        torch.randn(1, 3, 32, device="cuda", dtype=torch.bfloat16) * 0.1
    ).requires_grad_()
    weight = torch.randn(65536, 32, device="cuda", dtype=torch.bfloat16) * 0.03
    a = (torch.randn(256, 32, device="cuda") * 0.03).requires_grad_()
    b = (torch.randn(65536, 256, device="cuda") * 0.001).requires_grad_()
    targets = torch.tensor([7, 4096, 65535], device="cuda")
    refs = [tensor.detach().clone().requires_grad_() for tensor in (hidden, a, b)]
    loss = pretrain_head_l2wrap_ce_bf16(
        hidden, weight, targets, chunk_rows=2, lora_a=a, lora_b=b, use_nora=True
    )
    logits = pretrain_head_linear_bf16(
        refs[0], weight, lora_a=refs[1], lora_b=refs[2], use_nora=True
    )
    expected = pretrain_l2wrap_ce_bf16(logits, targets)
    actual_grads = torch.autograd.grad(loss * 0.7, (hidden, a, b))
    expected_grads = torch.autograd.grad(expected * 0.7, refs)
    torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-5)
    for actual, reference in zip(actual_grads, expected_grads, strict=True):
        assert_relative_rmse(actual, reference)
    assert weight.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
@pytest.mark.parametrize("chunk_rows", (1, 7, 1024))
@pytest.mark.parametrize("parameter", ("weight", "adapter"))
def test_head_l2wrap_accumulates_parameter_gradients_in_fp32(chunk_rows, parameter):
    """1024 identical rows expose BF16 accumulation saturation after 256 chunks."""
    rows, vocab = 1024, 65536
    hidden = torch.ones(1, rows, 1, device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros(vocab, 1, device="cuda", dtype=torch.bfloat16)
    targets = torch.zeros(rows, device="cuda", dtype=torch.int64)
    options = {}
    if parameter == "weight":
        trainable = weight.requires_grad_()
    else:
        trainable = torch.zeros_like(weight, requires_grad=True)
        options = {
            "lora_a": torch.ones(1, 1, device="cuda", dtype=torch.bfloat16),
            "lora_b": trainable,
        }
    loss = pretrain_head_l2wrap_ce_bf16(
        hidden, weight, targets, chunk_rows=chunk_rows, **options,
    )
    actual = torch.autograd.grad(loss, trainable)[0]
    expected = torch.full_like(actual, 1.0 / vocab)
    expected[0] = 1.0 / vocab - 1.0
    assert torch.equal(actual, expected)
    torch.testing.assert_close(
        loss, torch.tensor(float(vocab), device="cuda").log(), atol=2e-6, rtol=2e-6,
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("use_nora", [False, True])
def test_fullgraph_head_loss_ten_steps_preserve_loss_and_grad_norm(use_nora):
    torch.manual_seed(740)
    hidden = (torch.randn(1, 6, 64, device="cuda") * .1).bfloat16()
    weight = (torch.randn(65536, 64, device="cuda") * .01).bfloat16()
    targets = torch.randint(65536, (1, 6), device="cuda")
    initial = [torch.randn(4, 64, device="cuda") * .02,
               torch.randn(65536, 4, device="cuda") * .01]
    eager_parameters = [value.clone().requires_grad_() for value in initial]
    compiled_parameters = [value.clone().requires_grad_() for value in initial]
    eager_optimizer = torch.optim.SGD(eager_parameters, lr=.02)
    compiled_optimizer = torch.optim.SGD(compiled_parameters, lr=.02)

    def loss_function(a, b):
        return pretrain_head_l2wrap_ce_bf16(
            hidden, weight, targets, chunk_rows=4, lora_a=a, lora_b=b,
            lora_scale=.75, use_nora=use_nora,
        )

    compiled = torch.compile(loss_function, fullgraph=True, options={"emulate_precision_casts": True})
    for _ in range(10):
        eager_optimizer.zero_grad(set_to_none=True)
        compiled_optimizer.zero_grad(set_to_none=True)
        expected = loss_function(*eager_parameters)
        actual = compiled(*compiled_parameters)
        expected.backward()
        actual.backward()
        assert actual.dtype == torch.float32
        assert torch.equal(actual, expected)
        expected_norm = torch.linalg.vector_norm(torch.stack([value.grad.norm() for value in eager_parameters]))
        actual_norm = torch.linalg.vector_norm(torch.stack([value.grad.norm() for value in compiled_parameters]))
        assert torch.equal(actual_norm, expected_norm)
        for eager, traced in zip(eager_parameters, compiled_parameters, strict=True):
            assert eager.grad.dtype == traced.grad.dtype == torch.float32
            assert torch.equal(eager.grad, traced.grad)
        eager_optimizer.step()
        compiled_optimizer.step()
        assert all(torch.equal(eager, traced) for eager, traced in zip(eager_parameters, compiled_parameters, strict=True))
