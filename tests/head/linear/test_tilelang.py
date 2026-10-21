# SPDX-License-Identifier: MIT

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flashrwkv2._internal.linear import (
    _compiled,
    _forward,
    _NoRA,
    adapter_linear,
    linear,
    linear_input_vjp,
    linear_weight_vjp_fp32,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.resource
@pytest.mark.parametrize("arch", [89, 120])
@pytest.mark.parametrize("input_major", [False, True])
@pytest.mark.parametrize("config", [(16, 8, 2, 32), (16, 16, 3, 64), (16, 64, 2, 128)])
def test_tilelang_linear_device_compilation_without_gpu(arch, input_major, config):
    from tilelang.contrib.nvcc import get_cuda_version
    from tilelang.env import CUDA_HOME

    if CUDA_HOME is None or not (Path(CUDA_HOME) / "bin" / "nvcc").is_file():
        pytest.skip("CUDA Toolkit is required for device compilation")
    if arch == 120 and get_cuda_version() < (12, 8):
        pytest.skip("SM120 compilation requires CUDA Toolkit 12.8 or newer")
    kernel = _compiled(
        128,
        128,
        "float32",
        "bfloat16",
        False,
        config,
        (128, 1),
        input_major,
        arch,
    )
    source = kernel.get_kernel_source()
    assert "cp_async" in source
    assert "warpgroup_reg_" not in source


@pytest.mark.resource
@pytest.mark.parametrize("arch", [89, 120])
def test_tilelang_linear_k1_head_device_compilation_without_gpu(arch):
    """The chunked HeadCE forward projects [rows,1] into 65536 logits."""
    from tilelang.contrib.nvcc import get_cuda_version
    from tilelang.env import CUDA_HOME

    if CUDA_HOME is None or not (Path(CUDA_HOME) / "bin" / "nvcc").is_file():
        pytest.skip("CUDA Toolkit is required for device compilation")
    if arch == 120 and get_cuda_version() < (12, 8):
        pytest.skip("SM120 compilation requires CUDA Toolkit 12.8 or newer")
    kernel = _compiled(
        65536, 1, "float32", "bfloat16", False, (16, 64, 2, 32), (1, 1), True, arch,
    )
    source = kernel.get_kernel_source()
    assert "cp_async" not in source
    assert "tl::mma_sync<" in source
    assert "tl::DataType::kFloat32, 16, 8, 16" in source


@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bias_dtype", [None, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape", [
        (32, 96, 64), (1, 64, 96), (15, 128, 128),
        (1, 129, 64), (16, 64, 160), (256, 256, 128),
    ]
)
def test_tilelang_linear_tuning_checks_production_and_skips_compile_failures(
    monkeypatch, output_dtype, bias_dtype, shape
):
    from flashrwkv2._internal import linear as implementation
    from flashrwkv2._internal import tuning

    rows, outputs, channels = shape
    x = torch.ones(rows, channels, dtype=torch.bfloat16)
    weight = torch.ones(outputs, channels, dtype=torch.bfloat16)
    bias = None if bias_dtype is None else torch.ones(outputs, dtype=bias_dtype)
    rejected = (16, 128, 3, 32)
    wrong_bf16 = (16, 32, 2, 32)
    compiled_dtypes = set()
    compiled_configs = set()
    select_calls = 0

    def compile_kernel(
        outputs, channels, dtype, compiled_bias, has_bias, config, *args
    ):
        compiled_configs.add(config)
        assert has_bias == (bias is not None)
        assert compiled_bias == (
            "float32" if bias_dtype == torch.float32 else "bfloat16"
        )
        compiled_dtypes.add(dtype)
        if config == rejected and dtype == implementation._dtype_name(output_dtype):
            raise RuntimeError("candidate exceeds shared memory")

    def forward(x, weight, kernel_bias, dtype, config):
        assert kernel_bias is bias
        assert config != rejected
        value = 2 if config == wrong_bf16 and dtype == torch.bfloat16 else 1
        return torch.full((x.shape[0], weight.shape[0]), value, dtype=dtype)

    def select_config(candidates, run, *, check, reference):
        nonlocal select_calls
        candidates = tuple(candidates)
        assert rejected not in candidates
        if select_calls == 0:
            narrow_tiles = rows < 16 and outputs <= 128
            assert len(candidates) == (24 if rows >= 16 else 10 if narrow_tiles else 6) - 1
            assert {config[1] for config in candidates} == (
                {8, 16, 32, 64, 128} if narrow_tiles else {32, 64, 128}
            )
            assert {config[3] for config in candidates} == {32}
        else:
            assert len({config[:3] for config in candidates}) == 1
            assert {config[3] for config in candidates} == (
                {32, 64} if rows == 256 else
                {32, *(span for span in (64, 128) if channels % span == 0)}
            )
        select_calls += 1
        assert tuple(value.dtype for value in reference) == (
            (torch.float32, torch.bfloat16)
            if output_dtype == torch.bfloat16
            else (torch.float32,)
        )
        measurements = []
        for config in candidates:
            equal = all(
                torch.equal(value, target)
                for value, target in zip(check(config), reference, strict=True)
            )
            record = {"config": config, "bitwise_equal": equal}
            if equal:
                run(config)
                record["milliseconds"] = 1.0
            measurements.append(record)
        return max(
            item["config"] for item in measurements if item["bitwise_equal"]
        ), measurements

    monkeypatch.setattr(implementation, "linear", lambda *args, **kwargs: None)
    monkeypatch.setattr(implementation, "_compiled", compile_kernel)
    monkeypatch.setattr(implementation, "_forward", forward)
    monkeypatch.setattr(implementation, "_TUNED_CONFIGS", {})
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "test")
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda device: SimpleNamespace(shared_memory_per_block_optin=101376),
    )
    monkeypatch.setattr(tuning, "select_config", select_config)
    result = implementation.tune_linear(x, weight, output_dtype, bias=bias)
    assert select_calls == (2 if channels % 64 == 0 else 1)
    assert compiled_dtypes == (
        {"bfloat16", "float32"} if output_dtype == torch.bfloat16 else {"float32"}
    )
    assert result["config"] != rejected
    if rows == 256:
        assert (128, 128, 3, 128) not in compiled_configs
        resource = next(item for item in result["candidates"] if "resource_error" in item)
        assert resource["config"] == (128, 128, 3, 128)
    if output_dtype == torch.bfloat16:
        assert result["config"] != wrong_bf16
    assert (
        implementation._TUNED_CONFIGS[
            implementation._key(x, weight, output_dtype, bias)
        ]
        == result["config"]
    )
    assert (
        len(
            {
                implementation._key(x, weight, output_dtype, candidate_bias)
                for candidate_bias in (
                    None,
                    weight.new_ones(outputs),
                    torch.ones(outputs, dtype=torch.float32),
                )
            }
        )
        == 3
    )
    assert next(
        item for item in result["candidates"] if item["config"] == rejected
    ) == {"config": rejected, "compile_error": "candidate exceeds shared memory"}


@pytest.mark.parametrize("offset", [0, 1, 8])
@pytest.mark.parametrize("input_major", [False, True])
@pytest.mark.parametrize("bias_dtype", [torch.bfloat16, torch.float32])
def test_tilelang_linear_aligns_storage_offset_before_launch(
    monkeypatch, offset, input_major, bias_dtype
):
    from flashrwkv2._internal import linear as implementation

    x = torch.arange(3 * 64 + offset, dtype=torch.bfloat16)[offset:].view(3, 64)
    stored_weight = torch.arange(96 * 64 + offset, dtype=torch.bfloat16)[offset:].view(
        (64, 96) if input_major else (96, 64)
    )
    weight = stored_weight.t() if input_major else stored_weight
    bias = torch.arange(96 + offset, dtype=bias_dtype)[offset:]

    def compile_kernel(*args):
        assert args[6] == stored_weight.stride()
        assert args[7] == input_major

        def launch(source, physical_weight, kernel_bias):
            for actual, original in (
                (source, x),
                (physical_weight, stored_weight),
                (kernel_bias, bias),
            ):
                assert actual.data_ptr() % 16 == 0
                assert torch.equal(actual, original)
                assert (actual.data_ptr() == original.data_ptr()) == (offset != 1)
            return source.float() @ weight.float().t() + kernel_bias.float()

        return launch

    monkeypatch.setattr(implementation, "_compiled", compile_kernel)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    actual = implementation._forward(x, weight, bias, torch.float32, (16, 64, 2, 32))
    assert torch.equal(actual, x.float() @ weight.float().t() + bias.float())


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
@pytest.mark.parametrize("input_major", [False, True])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_tilelang_linear_offset_views_match_aligned_operands(input_major, output_dtype):
    torch.manual_seed(69)
    x = torch.randn(3 * 64 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(3, 64)
    stored_weight = torch.randn(96 * 64 + 1, device="cuda", dtype=torch.bfloat16)[
        1:
    ].view((64, 96) if input_major else (96, 64))
    weight = stored_weight.t() if input_major else stored_weight
    bias = torch.randn(97, device="cuda", dtype=output_dtype)[1:]
    actual = linear(x, weight, bias, output_dtype)
    expected = linear(x.clone(), weight.clone(), bias.clone(), output_dtype)
    assert torch.equal(actual, expected)


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
    "config",
    [
        pytest.param(
            (*config, copy_k),
            marks=(pytest.mark.memcheck, pytest.mark.racecheck)
            if config == (16, 8, 3) and copy_k == 128
            else (),
        )
        for config in ((16, 8, 2), (16, 8, 3), (16, 16, 2), (16, 16, 3))
        for copy_k in (32, 64, 128)
    ],
)
@pytest.mark.parametrize("rows", [1, 15])
@pytest.mark.parametrize("input_major", [False, True])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_tilelang_linear_narrow_tiles_preserve_results(
    config, rows, input_major, output_dtype
):
    torch.manual_seed(76)
    x = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    if input_major:
        weight = weight.t().contiguous().t()
    canonical = _forward(x, weight, None, output_dtype, (16, 64, 2, 32))
    assert torch.equal(canonical, _forward(x, weight, None, output_dtype, config))
    for row in sorted({0, rows - 1}):
        single = _forward(x[row : row + 1], weight, None, output_dtype, config)
        assert torch.equal(canonical[row : row + 1], single)


@cuda
@pytest.mark.cuda
@pytest.mark.parametrize(
    "config", [(16, 32, 3, 32), (32, 128, 2, 32), (64, 64, 3, 32), (128, 32, 2, 32)]
)
def test_tilelang_linear_tiling_preserves_fp32_result(config):
    torch.manual_seed(77)
    x = torch.randn(65, 160, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(257, 160, device="cuda", dtype=torch.bfloat16)
    canonical = _forward(x, weight, None, torch.float32, (16, 64, 2, 32))
    assert torch.equal(canonical, _forward(x, weight, None, torch.float32, config))
    for rows in (1, 17, 33, 65):
        assert torch.equal(canonical[:rows], _forward(x[:rows], weight, None, torch.float32))


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
