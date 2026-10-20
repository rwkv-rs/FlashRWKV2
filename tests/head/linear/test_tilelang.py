# SPDX-License-Identifier: MIT

import pytest
import torch

from flashrwkv2._internal.linear import (
    _forward,
    _NoRA,
    adapter_linear,
    linear,
    linear_input_vjp,
    linear_weight_vjp_fp32,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_tilelang_linear_rejects_cpu():
    with pytest.raises(ValueError, match="CUDA"):
        linear(
            torch.ones(1, 64, dtype=torch.bfloat16),
            torch.ones(32, 64, dtype=torch.bfloat16),
        )


def test_linear_weight_vjp_keeps_fp32_until_chunk_accumulation_finishes():
    x = torch.ones(3, 2, dtype=torch.bfloat16)
    dy = torch.tensor([[256.0], [1.0], [-256.0]])
    accumulated = torch.zeros(1, 2)
    for grad, source in zip(dy.split(1), x.split(1), strict=True):
        result = linear_weight_vjp_fp32(grad, source, out=accumulated)
        assert result is accumulated
    assert torch.equal(accumulated, torch.ones(1, 2))


def test_linear_input_vjp_keeps_fp32_upstream_until_the_bf16_input_edge():
    dy = torch.tensor([[1.001, -1.0]])
    weight = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.bfloat16)
    actual = linear_input_vjp(dy, weight)
    expected = torch.tensor([[0.001, 2.0]], dtype=torch.bfloat16)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "scale",
    (float("nan"), float("inf"), -float("inf"), 2 * torch.finfo(torch.float32).max),
)
def test_adapter_rejects_nonfinite_or_overflowing_scale_before_launch(scale):
    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    weight = torch.zeros(4, 8, dtype=torch.bfloat16)
    a, b = torch.zeros(2, 8), torch.zeros(4, 2)
    with pytest.raises(ValueError, match="finite.*float32"):
        adapter_linear(x, weight, a, b, scale)


@cuda
@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(27, 64, 257), (33, 768, 96), (65, 4096, 4096)])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_tilelang_linear_batch_and_chunk_invariance(shape, output_dtype):
    rows, channels, outputs = shape
    torch.manual_seed(71)
    x = torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(outputs, channels, device="cuda", dtype=torch.bfloat16) * 0.02
    bias = torch.randn(outputs, device="cuda", dtype=torch.float32)
    result = linear(x, weight, bias, output_dtype)
    input_major_weight = weight.t().contiguous().t()
    assert torch.equal(result, linear(x, input_major_weight, bias, output_dtype))
    expected = (x.float() @ weight.float().t() + bias).to(output_dtype)
    torch.testing.assert_close(result, expected, atol=0.03, rtol=0.01)
    for row in (0, 1, 15, rows - 1):
        single = linear(x[row : row + 1], weight, bias, output_dtype)
        assert torch.equal(result[row : row + 1], single)
    chunked = torch.cat(
        [linear(part, weight, bias, output_dtype) for part in x.split(7)]
    )
    assert torch.equal(result, chunked)
    perm = torch.randperm(rows, device="cuda")
    assert torch.equal(result[perm], linear(x[perm], weight, bias, output_dtype))


@cuda
@pytest.mark.cuda
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_tilelang_linear_independent_vjp(output_dtype):
    torch.manual_seed(73)
    x = torch.randn(19, 96, device="cuda", dtype=torch.bfloat16).requires_grad_()
    weight = torch.randn(127, 96, device="cuda", dtype=torch.bfloat16).requires_grad_()
    bias = torch.randn(127, device="cuda", dtype=torch.float32).requires_grad_()
    result = linear(x, weight, bias, output_dtype)
    upstream = torch.randn_like(result)
    actual = torch.autograd.grad(result, (x, weight, bias), upstream)
    dy = upstream.float()
    expected = (
        (dy @ weight.float()).to(x.dtype),
        (dy.t() @ x.float()).to(weight.dtype),
        dy.sum(dim=0),
    )
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, atol=0.06, rtol=0.01)


@cuda
@pytest.mark.cuda
@pytest.mark.parametrize("upstream_dtype", (torch.bfloat16, torch.float32))
def test_linear_weight_vjp_fp32_chunk_accumulation(upstream_dtype):
    torch.manual_seed(75)
    x = torch.randn(33, 32, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(33, 17, device="cuda", dtype=upstream_dtype)
    expected = dy.float().t() @ x.float()
    for chunk_rows in (1, 7, 32):
        actual = torch.zeros_like(expected)
        for gradient, source in zip(
            dy.split(chunk_rows), x.split(chunk_rows), strict=True
        ):
            linear_weight_vjp_fp32(gradient, source, out=actual)
        torch.testing.assert_close(actual, expected, atol=1.0e-5, rtol=1.0e-5)


@cuda
@pytest.mark.cuda
@pytest.mark.parametrize(
    "config", [(16, 32, 3), (32, 128, 2), (64, 64, 3), (128, 32, 2)]
)
def test_tilelang_linear_tiling_preserves_fp32_result(config):
    torch.manual_seed(77)
    x = torch.randn(65, 160, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(257, 160, device="cuda", dtype=torch.bfloat16)
    canonical = _forward(x, weight, None, torch.float32, (16, 64, 2))
    assert torch.equal(canonical, _forward(x, weight, None, torch.float32, config))


@cuda
@pytest.mark.cuda
@pytest.mark.parametrize("use_nora", [False, True])
def test_tilelang_adapter_vjp_and_batch_invariance(use_nora):
    torch.manual_seed(79)
    x = torch.randn(17, 96, device="cuda", dtype=torch.bfloat16).requires_grad_()
    weight = torch.randn(127, 96, device="cuda", dtype=torch.bfloat16) * 0.02
    a = torch.randn(32, 96, device="cuda", requires_grad=True)
    b = torch.randn(127, 32, device="cuda", requires_grad=True)
    result = adapter_linear(x, weight, a, b, 0.7, use_nora)
    assert torch.equal(result[:1], adapter_linear(x[:1], weight, a, b, 0.7, use_nora))
    xr, ar, br = (t.detach().clone().requires_grad_() for t in (x, a, b))
    normalized = ar.float()
    if use_nora:
        normalized = normalized / (normalized.norm(dim=0, keepdim=True) + 1.0e-6)
    normalized = normalized.to(torch.bfloat16)
    base = (xr.float() @ weight.float().t()).to(torch.bfloat16)
    features = (xr.float() @ normalized.float().t()).to(torch.bfloat16)
    update = features.float() @ br.to(torch.bfloat16).float().t()
    reference = (base.float() + 0.7 * update).to(torch.bfloat16)
    upstream = torch.randn_like(result)
    actual = torch.autograd.grad(result, (x, a, b), upstream)
    expected = torch.autograd.grad(reference, (xr, ar, br), upstream)
    torch.testing.assert_close(result, reference, atol=0.12, rtol=0.02)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, atol=0.3, rtol=0.03)


@cuda
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_tilelang_nora_independent_vjp_and_zero_column(dtype):
    torch.manual_seed(81)
    a = torch.randn(95, 32, device="cuda", dtype=dtype).t()
    a[:, 0] = 0.0
    a.requires_grad_()
    result = _NoRA.apply(a)
    reference_a = a.detach().clone().requires_grad_()
    values = reference_a.float()
    reference = (values / (values.norm(dim=0, keepdim=True) + 1.0e-6)).to(
        torch.bfloat16
    )
    upstream = torch.randn_like(result)
    actual_grad = torch.autograd.grad(result, a, upstream)[0]
    expected_grad = torch.autograd.grad(reference, reference_a, upstream)[0]
    torch.testing.assert_close(result, reference, atol=0.0078125, rtol=0.01)
    torch.testing.assert_close(actual_grad, expected_grad, atol=0.01, rtol=0.01)


@cuda
@pytest.mark.cuda
@pytest.mark.cuda_graph
def test_tilelang_linear_cuda_graph():
    x = torch.randn(3, 96, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(127, 96, device="cuda", dtype=torch.bfloat16)
    expected = linear(x, weight)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = linear(x, weight)
    graph.replay()
    assert torch.equal(actual, expected)
