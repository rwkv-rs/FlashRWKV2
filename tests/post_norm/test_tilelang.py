# SPDX-License-Identifier: MIT

import pytest
import torch
import torch.nn.functional as F
import tilelang

from flashrwkv2.post_norm.tilelang import _backward, _forward, norm


@pytest.mark.parametrize("arch", ("sm_89", "sm_120"))
@pytest.mark.parametrize("channels", (64, 257, 1024, 4096, 7168))
def test_norm_lowering(arch, channels):
    with tilelang.tvm.target.Target({"kind": "cuda", "arch": arch}) as target:
        for args in ((channels, 1e-5, True), (channels, 1e-5, True, False, False)):
            tilelang.lower(_forward.get_tir(*args), target=target)
        tilelang.lower(_backward.get_tir(channels, True), target=target)
        for save_stats, store_res in ((True, True), (False, False), (False, True)):
            tilelang.lower(_forward.get_tir(channels, 1e-5, True, save_stats, store_res, 4), target=target)
        for needed in (False, True):
            tilelang.lower(_backward.get_tir(channels, True, 32, needed, needed), target=target)


@pytest.mark.parametrize("fault", ("stats", "production", "res_output", "res_store"))
def test_norm_tuning_rejects_changed_statistics_production_or_res(monkeypatch, fault):
    from flashrwkv2._internal import tuning
    from flashrwkv2.post_norm import tilelang as implementation

    x = torch.ones(3, 64, dtype=torch.bfloat16)
    weight, bias, res = x[0], x[0], x.clone()

    def forward(channels, eps, has_res, save_stats=True, store_res=True, rows_per_block=1):
        def kernel(values, residual, weight, bias, total):
            output = torch.ones_like(values)
            mean = torch.ones(values.shape[0] if save_stats else 0)
            rstd = torch.ones_like(mean)
            if store_res:
                total.copy_(values + residual)
            if rows_per_block == 2:
                if fault == "stats" and save_stats:
                    mean.add_(1e-5)  # Invisible after a BF16 cast.
                if fault == "production" and not save_stats and not store_res:
                    output.add_(1)
                if fault == "res_output" and not save_stats and store_res:
                    output.add_(1)
                if fault == "res_store" and not save_stats and store_res:
                    total.add_(1)
            return output, mean, rstd
        return kernel

    def select_config(candidates, run, *, check, reference):
        assert len(reference) == 6 and reference[1].dtype == torch.float32
        eligible = []
        for config in candidates:
            actual = check(config)
            if all(torch.equal(a, b) for a, b in zip(actual, reference, strict=True)):
                eligible.append(config)
        assert eligible == [1, 4]
        assert torch.equal(run(1), reference[0])
        return 1, []

    monkeypatch.setattr(implementation, "norm", lambda *args, **kwargs: None)
    monkeypatch.setattr(implementation, "_forward", forward)
    monkeypatch.setattr(implementation, "_ROW_CONFIGS", {})
    monkeypatch.setattr(tuning, "select_config", select_config)
    monkeypatch.setattr(tuning, "save_config", lambda *args: None)
    implementation.tune_norm(x, weight, bias, res=res)
    assert implementation._ROW_CONFIGS[implementation._config_key(x, True)] == 1


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("channels", (64, 257, 1024, 4096, 7168))
@pytest.mark.parametrize("has_res", (False, True))
def test_norm_bitwise_batch_and_grad_mode(channels, has_res):
    torch.manual_seed(701)
    x = torch.randn(19, channels, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(channels, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = torch.randn_like(weight, requires_grad=True)
    res = torch.randn_like(x, requires_grad=True) if has_res else None
    train, total = norm(x, weight, bias, res=res, return_res=True)
    with torch.no_grad():
        infer = norm(x, weight, bias, res=res)
        pieces = [
            norm(x[s:e], weight, bias, res=None if res is None else res[s:e])
            for s, e in ((0, 1), (1, 8), (8, 19))
        ]
    assert torch.equal(train, infer)
    assert torch.equal(train, torch.cat(pieces))
    assert torch.equal(total, x if res is None else x + res)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("channels", (257, 4096))
def test_norm_independent_vjp(channels):
    torch.manual_seed(702)
    x = (torch.randn(7, channels, device="cuda") * 0.2).bfloat16().requires_grad_()
    res = torch.randn_like(x, requires_grad=True)
    weight = torch.randn(channels, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = torch.randn_like(weight, requires_grad=True)
    inputs = (x, weight, bias, res)
    actual = norm(x, weight, bias, res=res, return_res=True)
    refs = [t.detach().clone().requires_grad_() for t in inputs]
    xr, wr, br, rr = refs
    summed = xr + rr
    expected = (F.layer_norm(summed.float(), (channels,), wr.float(), br.float()).bfloat16(), summed)
    upstream = (torch.randn_like(x), torch.randn_like(x))
    actual_grads = torch.autograd.grad(actual, inputs, upstream)
    expected_grads = torch.autograd.grad(expected, refs, upstream)
    torch.testing.assert_close(actual[0], expected[0], atol=0.02, rtol=0.01)
    for gradient, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(gradient, reference, atol=0.03, rtol=0.02)
