# SPDX-License-Identifier: MIT

import pytest
import torch
import tilelang

from flashrwkv2.tmix.readout.tilelang import _backward, _forward, readout


@pytest.mark.parametrize("arch", ("sm_89", "sm_120"))
@pytest.mark.parametrize("head_size", (64, 128, 256))
def test_readout_lowering(arch, head_size):
    with tilelang.tvm.target.Target({"kind": "cuda", "arch": arch}) as target:
        for dtype in ("bfloat16", "float32"):
            for save in (False, True):
                tilelang.lower(_forward.get_tir(4096, head_size, 64e-5, dtype, save), target=target)
            tilelang.lower(_backward.get_tir(4096, head_size, dtype), target=target)
            for save in (False, True):
                tilelang.lower(_forward.get_tir(4096, head_size, 64e-5, dtype, save, 16), target=target)
            for needed in (False, True):
                tilelang.lower(_backward.get_tir(4096, head_size, dtype, 32, (needed,) * 3), target=target)


@pytest.mark.parametrize("fault", ("stats", "production"))
def test_readout_tuning_rejects_changed_statistics_or_production(monkeypatch, fault):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.readout import tilelang as implementation

    x = torch.ones(3, 64, dtype=torch.bfloat16)
    vector = x[0]

    def forward(channels, head_size, eps, x_dtype, save_stats=True, rows_per_block=1):
        def kernel(*inputs):
            output = torch.ones_like(inputs[0])
            stats = torch.ones(inputs[0].shape[0] if save_stats else 0, 1, 3)
            if rows_per_block == 2:
                if fault == "stats" and save_stats:
                    stats.add_(1e-5)  # Invisible after a BF16 cast.
                if fault == "production" and not save_stats:
                    output.add_(1)
            return output, stats
        return kernel

    def select_config(candidates, run, *, check, reference):
        assert len(reference) == 3 and reference[1].dtype == torch.float32
        eligible = []
        for config in candidates:
            actual = check(config)
            if all(torch.equal(a, b) for a, b in zip(actual, reference, strict=True)):
                eligible.append(config)
        assert eligible == [1, 4, 8, 16]
        assert torch.equal(run(1), reference[0])
        return 1, []

    monkeypatch.setattr(implementation, "readout", lambda *args, **kwargs: None)
    monkeypatch.setattr(implementation, "_forward", forward)
    monkeypatch.setattr(implementation, "_ROW_CONFIGS", {})
    monkeypatch.setattr(tuning, "select_config", select_config)
    monkeypatch.setattr(tuning, "save_config", lambda *args: None)
    monkeypatch.setattr(implementation, "device_class", lambda device: "test")
    implementation.tune_readout(x, x, x, x, vector, vector, vector, x)
    assert implementation._ROW_CONFIGS[implementation._config_key(x, 64)] == 1


def reference(inputs, head_size):
    x, r, k, v, r_k, weight, bias, gate = inputs
    shape = (*x.shape[:-1], -1, head_size)
    grouped = x.float().view(shape)
    centered = grouped - grouped.mean(-1, keepdim=True)
    normalized = centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + 64e-5)
    scale = (
        r.float().view(shape) * k.float().view(shape) * r_k.float().view(-1, head_size)
    ).sum(-1, keepdim=True)
    return (
        (normalized.view_as(x) * weight.float() + bias.float() + (scale * v.float().view(shape)).view_as(x))
        * gate.float()
    ).bfloat16()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("head_size", (64, 128, 256))
@pytest.mark.parametrize("channels", (1024, 4096, 7168))
def test_readout_bitwise_batch_grad_and_vjp(head_size, channels):
    torch.manual_seed(721)
    rows = [
        (torch.randn(17, channels, device="cuda") * 0.1).bfloat16().requires_grad_()
        for _ in range(5)
    ]
    vectors = [
        torch.randn(channels, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    inputs = (*rows[:4], *vectors, rows[4])
    actual = readout(*inputs, head_size)
    with torch.no_grad():
        infer = readout(*inputs, head_size)
        pieces = [
            readout(
                *(t[start:end] for t in rows[:4]), *vectors, rows[4][start:end], head_size,
            )
            for start, end in ((0, 1), (1, 8), (8, 17))
        ]
    assert torch.equal(actual, infer)
    assert torch.equal(actual, torch.cat(pieces))
    refs = [t.detach().clone().requires_grad_() for t in inputs]
    expected = reference(refs, head_size)
    upstream = torch.randn_like(actual)
    gradients = torch.autograd.grad(actual, inputs, upstream)
    expected_gradients = torch.autograd.grad(expected, refs, upstream)
    torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.01)
    for gradient, ref in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(gradient, ref, atol=0.03, rtol=0.02)


@pytest.mark.parametrize("channels,rows", ((2048, 1), (2560, 17), (4096, 1025), (5120, 4096)))
def test_fixed_readout_covers_dynamic_rows(monkeypatch, channels, rows):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.readout import tilelang as implementation

    monkeypatch.setattr(implementation, "device_class", lambda device: ("test", 8, 9, 128))
    monkeypatch.setattr(tuning, "get_config", lambda configs, owner, key, default, *, tune, fixed: (fixed, default))
    x = torch.empty(rows, channels, device="meta", dtype=torch.bfloat16)
    assert implementation._readout_group(x, x, x, x, x, x, x, x, 64, 64e-5) == (True, 4 if rows == 1 else 8)


@pytest.mark.cuda
@pytest.mark.cuda_graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("channels,rows", (
    (2048, 1), (2048, 4), (2048, 16), (2048, 17), (2048, 64), (2048, 128),
    (2048, 193), (2048, 320), (2048, 512), (2048, 513), (2048, 768), (2048, 1024),
    (2560, 17), (2560, 64), (2560, 128), (2560, 193), (2560, 320), (2560, 512),
    (2560, 513), (2560, 768), (2560, 1024), (4096, 17), (4096, 64), (4096, 128),
    (4096, 192), (4096, 193), (4096, 256), (4096, 961), (4096, 1024),
))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_fixed_decode_readout_preserves_stats_vjp_and_graph(monkeypatch, tmp_path, channels, rows, dtype):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.readout import tilelang as implementation

    if not tuning._fixed_strategy_device(tuning.device_class(torch.device("cuda"))):
        pytest.skip("fixed readout strategy is qualified on tested devices")
    torch.manual_seed(721)
    x = torch.randn(rows, channels, device="cuda", dtype=dtype)
    bf = tuple(torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16) for _ in range(4))
    vectors = tuple(torch.randn(channels, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    args = (x, *bf[:3], *vectors, bf[3])
    dtype_name = str(dtype).split(".")[-1]
    group = 4 if rows == 1 else 8
    expected, expected_stats = implementation._forward(channels, 64, 64e-5, dtype_name, True, 1)(*args)
    actual, actual_stats = implementation._forward(channels, 64, 64e-5, dtype_name, True, group)(*args)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(actual_stats.view(torch.int32), expected_stats.view(torch.int32))

    variables = tuple(t.detach().requires_grad_() for t in args)
    upstream = torch.randn_like(expected)
    with monkeypatch.context() as patch:
        patch.setattr(implementation, "_readout_group", lambda *args: 1)
        baseline = readout(*variables)
        expected_grads = torch.autograd.grad(baseline, variables, upstream)

    monkeypatch.setattr(tuning, "_profile_root", lambda: tmp_path)
    monkeypatch.setattr(tuning, "_read_profile", lambda *args: pytest.fail("read a profile"))
    monkeypatch.setattr(implementation, "tune_readout", lambda *args: pytest.fail("tuned a fixed configuration"))
    monkeypatch.setattr(implementation, "_ROW_CONFIGS", {})
    trained = readout(*variables)
    actual_grads = torch.autograd.grad(trained, variables, upstream)
    assert torch.equal(trained.view(torch.int16), baseline.view(torch.int16))
    assert all(torch.equal(a, b) for a, b in zip(actual_grads, expected_grads))
    if rows == 4 and dtype == torch.bfloat16:
        compiled = torch.compile(readout, backend="eager", fullgraph=True)
        compiled_output = compiled(*variables)
        compiled_grads = torch.autograd.grad(compiled_output, variables, upstream)
        assert torch.equal(compiled_output.view(torch.int16), baseline.view(torch.int16))
        assert all(torch.equal(a, b) for a, b in zip(compiled_grads, expected_grads))

    def call():
        with torch.no_grad():
            return readout(*args)

    for mode in ("inspect", "require", "tune"):
        report = tuning.prepare_tuning([call], mode=mode)
        assert report["complete"] and report["counts"]["fixed"] == 1
        assert report["records"][0]["config"] == group
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    graph.replay()
    assert torch.equal(captured.view(torch.int16), expected.view(torch.int16))
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("qualified,head_size,groups", (
    (False, 64, ()), (True, 128, ()), (True, 64, (4, 8)),
))
def test_preparation_enumerates_only_qualified_readout_families(monkeypatch, qualified, head_size, groups):
    from types import SimpleNamespace
    from flashrwkv2.tmix.readout import tilelang as implementation
    calls = []
    def prepare(owner, compiler, options, variants):
        calls.append((options, tuple(variants)))
        return "kernel"
    monkeypatch.setattr(implementation, "device_class", lambda device: "test")
    monkeypatch.setattr(implementation, "_fixed_strategy_device", lambda info: qualified)
    monkeypatch.setattr(implementation, "prepared_kernel", prepare)
    x = SimpleNamespace(shape=(17, 256), dtype=torch.bfloat16, device="cuda")
    assert implementation._prepared_forward(x, head_size, 64e-5, False, 2) == "kernel"
    options, variants = calls[0]
    assert options[-1] == 2
    assert tuple(key[-1] for key in variants) == groups
