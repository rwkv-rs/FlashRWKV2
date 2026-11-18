# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
import torch

from flashrwkv2._internal.linear import (
    _compiled,
    _forward,
    _NoRA,
    adapter_linear,
    linear,
    linear_weight_vjp_fp32,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.resource
@pytest.mark.parametrize("arch", (89, 120))
@pytest.mark.parametrize("input_major", (False, True))
@pytest.mark.parametrize("reduction", (0, 1, 128))
def test_projection_device_code_preserves_copy_contract(arch, input_major, reduction):
    from tilelang.contrib.nvcc import get_cuda_version
    from tilelang.env import CUDA_HOME

    if not CUDA_HOME or not (Path(CUDA_HOME) / "bin" / "nvcc").is_file():
        pytest.skip("CUDA Toolkit is required")
    if arch == 120 and get_cuda_version() < (12, 8):
        pytest.skip("SM120 requires CUDA Toolkit 12.8")
    dynamic = reduction == 0
    kernel = _compiled(
        65536 if reduction == 1 else 128, reduction, "float32", "bfloat16", False,
        (128, 64, 2, 32) if dynamic else (16, 64, 2, 32),
        None if dynamic else ((65536 if reduction == 1 else 128) if input_major else reduction, 1),
        input_major, arch, dynamic, True,
    )
    source = kernel.get_kernel_source()
    assert "warpgroup_reg_" not in source
    if dynamic:
        copies = [line for line in source.splitlines() if "(X[" in line]
        assert copies and all("cp_async" in line and "<16>" in line for line in copies)
    elif reduction == 1:
        assert "cp_async" not in source and "tl::mma_sync<" in source
    else:
        assert "cp_async" in source


@cuda
@pytest.mark.cuda
def test_tuning_publishes_only_compiled_numerically_valid_candidates(monkeypatch):
    from flashrwkv2._internal import linear as implementation
    import _tuning as tuning

    x = torch.randn(5, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(17, 64, device="cuda", dtype=torch.bfloat16)
    rejected, wrong, valid = (16, 16, 2, 32), (16, 32, 2, 32), (16, 64, 2, 32)
    compile_kernel, forward = implementation._compiled, implementation._forward
    def compiled(*key):
        if key[5] == rejected:
            raise RuntimeError("candidate compilation failed")
        return compile_kernel(*key)
    def run(x, weight, bias, dtype, config=None, **kwargs):
        output = forward(x, weight, bias, dtype, config, **kwargs)
        return output + 1 if config == wrong and dtype == torch.bfloat16 else output
    monkeypatch.setattr(implementation, "_compiled", compiled)
    monkeypatch.setattr(implementation, "_forward", run)
    monkeypatch.setattr(implementation, "_fixed_projection_config", lambda *args: None)
    monkeypatch.setattr(implementation, "_TUNED_CONFIGS", {})
    monkeypatch.setattr(tuning, "projection_configs", lambda *args: (rejected, wrong, valid) if not tuple(args[-1]) else ())
    result = tuning.tune_linear(x, weight)
    records = {row["config"]: row for row in result["candidates"]}
    assert "compile_error" in records[rejected]
    assert not records[wrong]["bitwise_equal"] and result["config"] == valid
    assert torch.equal(linear(x, weight), forward(x, weight, None, torch.bfloat16, valid))


def test_tilelang_linear_rejects_cpu():
    with pytest.raises(ValueError, match="CUDA"):
        linear(
            torch.ones(1, 64, dtype=torch.bfloat16),
            torch.ones(32, 64, dtype=torch.bfloat16),
        )


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
@pytest.mark.parametrize("shape", [(27, 64, 257), (33, 768, 96), (65, 4096, 4096), (17, 64, 65536)])
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
    import random


    from flashrwkv2._internal import execution as tuning
    from flashrwkv2._internal.linear import _projection_config
    if tuning._fixed_strategy_device(tuning.device_class(x.device)):
        assert torch.equal(linear(x, weight.clone(), bias, output_dtype), result)
        sizes = {1, 17, 89, 90, 100, 4096, *random.Random(803).sample(range(1, 4097), 20)}
        previous = None
        for size in range(1, 8193):
            config = _projection_config(
                size, (channels,), (outputs,), x.device)
            if previous is not None and config != previous:
                sizes.update((size - 1, size, size + 1))
            previous = config
        for size in sorted(sizes):
            actual = linear(x[:1].expand(size, -1).contiguous(), weight, bias, output_dtype)
            assert torch.equal(actual, result[:1].expand(size, -1))



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
    weight.copy_(torch.randn_like(weight))
    replacement = linear(x, weight)
    graph.replay()
    assert torch.equal(actual, replacement) and not torch.equal(actual, expected)



@cuda
@pytest.mark.cuda
@pytest.mark.cuda_graph
@pytest.mark.parametrize("input_major", (False, True))
@pytest.mark.parametrize("rows,outputs,reductions", (
    (96, 128, (1, 17, 31, 32, 33, 63, 64, 65, 89, 90, 100, 127, 128, 129, 1000, 1024, 4096)),
    (129, 16384, (32, 33, 64, 96, 128, 129, 1024, 4096)),
))
def test_dynamic_weight_reduction_preserves_accumulators_and_graph(rows, outputs, reductions, input_major):

    from flashrwkv2._internal import execution as tuning

    if not tuning._fixed_strategy_device(tuning.device_class(torch.device("cuda"))):
        pytest.skip("dynamic reduction requires a qualified device")
    torch.manual_seed(801)
    cases = []
    for reduction in reductions:
        x = (torch.randn(rows, reduction, device="cuda") * .1).bfloat16()
        weight = (torch.randn(outputs, reduction, device="cuda") * .1).bfloat16()
        if input_major:
            weight = weight.t().contiguous().t()
        reference = _forward(x, weight, None, torch.float32, config=(16, 64, 2, 32))
        cases.append((x, weight, reference))
    for x, weight, reference in cases:
        for dtype in (torch.float32, torch.bfloat16):
            assert torch.equal(_forward(x, weight, None, dtype, dynamic_k=True), reference.to(dtype))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = _forward(x, weight, None, torch.float32, dynamic_k=True)
        graph.replay()
        assert torch.equal(actual, reference)


@cuda
@pytest.mark.cuda
@pytest.mark.resource
def test_prepared_projection_reuses_disk_cache_in_new_process():
    import json
    import os
    import subprocess
    import sys
    from tempfile import TemporaryDirectory

    from flashrwkv2._internal import execution as tuning

    if not tuning._fixed_strategy_device(tuning.device_class(torch.device("cuda"))):
        pytest.skip("deterministic strategy is qualified on tested devices")
    artifacts = Path(__file__).resolve().parents[3] / "artifacts"
    artifacts.mkdir(exist_ok=True)
    script = r'''
import hashlib, json, sys
import torch, tilelang
from tilelang.jit.kernel import JITKernel
from flashrwkv2._internal import linear
phase = sys.argv[1]
compilations = []
original = JITKernel._compile_and_create_adapter
def compile_kernel(self, *args, **kwargs):
    assert phase == "cold", "warm process attempted device compilation"
    compilations.append(1)
    return original(self, *args, **kwargs)
JITKernel._compile_and_create_adapter = compile_kernel
import pytest
status = pytest.main(["-q", "-p", "no:cacheprovider",
    "-o", "pythonpath=" + sys.argv[2] + "/tests " + sys.argv[2] + "/benchmarks",
    sys.argv[2] + "/tests/tmix/wkv7/test_block.py::test_block_crosses_recurrent_dispatch_thresholds"])
assert status == 0, "full inference/training block disk-cache reuse failed"
torch.manual_seed(104)
x = torch.randn(17, 64, device="cuda").bfloat16()
w = torch.randn(96, 64, device="cuda").bfloat16()
dx = torch.randn(96, 32, device="cuda").bfloat16()
dw = torch.randn(32, 64, device="cuda").bfloat16().t()
values = []
for rows in (89, 90, 100):
    values.append(linear._forward(x[:1].expand(rows, -1).contiguous(), w, None, torch.float32))
for k in (1, 17, 31, 32, 33, 64, 65, 89, 90, 100, 129):
    a = dx[:, :1].expand(-1, k).contiguous()
    b = dw[:, :1].expand(-1, k).t().contiguous().t()
    values.append(linear._forward(a, b, None, torch.float32, dynamic_k=True))
torch.cuda.synchronize()
hashes = [hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for v in values]
print("CACHE_RESULT " + json.dumps({"phase": phase, "device_compilations": len(compilations),
                                   "hashes": hashes}))
'''
    reports = []
    with TemporaryDirectory(prefix="tilelang-cold-", dir=artifacts) as cache:
        environment = {**os.environ, "TILELANG_CACHE_DIR": cache}
        for phase in ("cold", "warm"):
            result = subprocess.run([sys.executable, "-c", script, phase, str(artifacts.parent)], env=environment,
                                    text=True, capture_output=True, check=False, timeout=600)
            assert result.returncode == 0, result.stdout + result.stderr
            report = json.loads(next(line.removeprefix("CACHE_RESULT ") for line in
                                     result.stdout.splitlines() if line.startswith("CACHE_RESULT ")))
            reports.append(report)
    assert reports[0]["device_compilations"] > 0
    assert reports[1]["device_compilations"] == 0
    assert reports[0]["hashes"] == reports[1]["hashes"]
    print(json.dumps({"cold_device_compilations": reports[0]["device_compilations"],
                      "warm_device_compilations": reports[1]["device_compilations"]}))
