# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from flashrwkv2._internal import linear as shared_linear
from flashrwkv2.tmix.wkv_prepare import (
    infer_tmix_wkv_prepare_forward_varlen,
    pretrain_tmix_wkv_prepare_bf16,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@cuda
@pytest.mark.cuda
@pytest.mark.memcheck
@pytest.mark.racecheck
@pytest.mark.parametrize("stage", ["in", "out"])
def test_real_model_rank_profile_is_loaded_for_decode_graph(monkeypatch, tmp_path, stage):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.wkv_prepare import tilelang as implementation

    monkeypatch.setattr(tuning, "_profile_root", lambda: tmp_path)
    monkeypatch.setattr(implementation, "_GROUP_CONFIGS", {})
    generator = torch.Generator(device="cuda").manual_seed(20261022)
    ranks = (128, 128, 480, 96)
    channels, rows = 4096, 513

    def random(*shape, dtype=torch.bfloat16):
        return torch.randn(shape, generator=generator, device="cuda", dtype=dtype) * 0.03

    inputs = tuple(random(rows, channels if stage == "in" else rank) for rank in ranks)
    weights = tuple(
        random(rank, channels) if stage == "in" else random(channels, rank)
        for rank in ranks
    )
    kinds = (1, 0, 2, 0) if stage == "in" else (0, 2, 0, 0)
    biases = (None,) * 4 if stage == "in" else (
        None, random(channels), None, None
    )
    config = (64, 64, 2, 128) if stage == "in" else (256, 64, 2, 32)

    def check(candidate):
        values, raw = implementation._run_groups(
            inputs, weights, kinds, biases, raw_mode=2, config=candidate
        )
        return *(value.view(torch.int16) for value in values), raw.view(torch.int32)

    reference = check((16, 64, 2, 32))
    selected, _ = tuning.select_config(
        (config,),
        lambda candidate: implementation._run_groups(
            inputs, weights, kinds, biases, config=candidate
        )[0],
        check=check,
        reference=reference,
    )
    key = implementation._group_key(inputs, weights, kinds, biases)
    tuning.save_config("tmix/wkv_prepare", key, selected)
    actual = implementation._grouped_linear(inputs, weights, kinds, biases)
    assert implementation._GROUP_CONFIGS[key] == selected
    for value, expected in zip(actual, reference[:-1], strict=True):
        assert torch.equal(value.view(torch.int16), expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = implementation._grouped_linear(inputs, weights, kinds, biases)
    graph.replay()
    for value, expected in zip(captured, reference[:-1], strict=True):
        assert torch.equal(value.view(torch.int16), expected)
    tuning._LOADED.clear()


def test_prepare_prefers_canonical_weight_when_runtime_is_also_supplied():
    from flashrwkv2.tmix.wkv_prepare import _original_weight

    canonical = torch.empty(192, 4096, device="meta")
    runtime = torch.empty(4096, 192, device="meta")
    assert _original_weight(canonical, runtime, "w1") is canonical
    assert _original_weight(None, runtime, "w1").stride() == (1, 192)


def test_deterministic_group_projection_checks_layout(monkeypatch):
    from types import SimpleNamespace
    from flashrwkv2.tmix.wkv_prepare import tilelang as implementation

    monkeypatch.setattr(implementation, "device_class", lambda device: ("test", 8, 9, 128))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(
        multi_processor_count=128, shared_memory_per_block_optin=99_264,
    ))
    xs = tuple(torch.empty(89, 256, device="meta") for _ in range(3))
    weights = tuple(torch.empty(96, 256, device="meta") for _ in range(3))
    assert implementation._fixed_projection_config(xs, weights, (1, 0, 2), (None,) * 3) is not None
    mixed = (weights[0].t().contiguous().t(), *weights[1:])
    assert implementation._fixed_projection_config(xs, mixed, (1, 0, 2), (None,) * 3) is None



@cuda
@pytest.mark.cuda
@pytest.mark.cuda_graph
@pytest.mark.parametrize("channels,rows", (
    (2048, 17), (2048, 129), (2560, 65), (2560, 1024),
    (4096, 32), (4096, 257),
))
def test_fixed_rkv_row_wave_preserves_accumulators_and_graph(monkeypatch, tmp_path, channels, rows):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.wkv_prepare import tilelang as implementation

    if not tuning._fixed_strategy_device(tuning.device_class(torch.device("cuda"))):
        pytest.skip("fixed grouped strategy is qualified on tested SM89/SM120 devices")
    torch.manual_seed(831)
    xs = tuple(torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16) * .02 for _ in range(3))
    weights = tuple(torch.randn(channels, channels, device="cuda", dtype=torch.bfloat16) * .02 for _ in range(3))
    kinds, biases = (0, 0, 0), (None,) * 3
    expected, raw = implementation._run_groups(
        xs, weights, kinds, biases, raw_mode=2, config=(16, 64, 2, 32),
    )
    monkeypatch.setattr(tuning, "_profile_root", lambda: tmp_path)
    monkeypatch.setattr(tuning, "_read_profile", lambda *args: pytest.fail("read a profile"))
    values, actual_raw = implementation._run_groups(xs, weights, kinds, biases, raw_mode=2)
    assert torch.equal(actual_raw, raw)
    assert all(torch.equal(a, b) for a, b in zip(values, expected, strict=True))

    def call():
        return implementation._grouped_linear(xs, weights, kinds, biases)

    for mode in ("inspect", "require", "tune"):
        report = tuning.prepare_tuning([call], mode=mode)
        assert report["complete"] and report["counts"]["fixed"] == 1
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = call()
    graph.replay()
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert not tuple(tmp_path.iterdir())


@cuda
@pytest.mark.cuda
@pytest.mark.parametrize("channels,rows", ((2048, 384), (2048, 1024), (2560, 320), (2560, 1024)))
def test_fixed_rkv_projection_preserves_training_gradients(monkeypatch, channels, rows):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.wkv_prepare import tilelang as implementation

    if not tuning._fixed_strategy_device(tuning.device_class(torch.device("cuda"))):
        pytest.skip("fixed grouped strategy is qualified on tested SM89/SM120 devices")
    torch.manual_seed(832)
    xs = tuple((torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16) * .02).requires_grad_() for _ in range(3))
    weights = tuple((torch.randn(channels, channels, device="cuda", dtype=torch.bfloat16) * .02).requires_grad_() for _ in range(3))
    kinds, biases = (0, 0, 0), (None,) * 3
    upstream = tuple(torch.randn_like(x) for x in xs)
    key = implementation._group_key(xs, weights, kinds, biases)
    token = tuning._TUNING.set(True)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(implementation, "_fixed_projection_config", lambda *args: None)
            patch.setattr(implementation, "_fixed_strategy_device", lambda *args: False)
            patch.setattr(shared_linear, "_fixed_strategy_device", lambda *args: False)
            patch.setattr(implementation, "_GROUP_CONFIGS", {key: (16, 64, 2, 32)})
            expected = implementation._grouped_linear(xs, weights, kinds, biases)
            expected_grads = torch.autograd.grad(expected, (*xs, *weights), upstream)
        actual = implementation._grouped_linear(xs, weights, kinds, biases)
        actual_grads = torch.autograd.grad(actual, (*xs, *weights), upstream)
    finally:
        tuning._TUNING.reset(token)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(actual_grads, expected_grads, strict=True))


@cuda
@pytest.mark.cuda
@pytest.mark.cuda_graph
@pytest.mark.parametrize("channels,ranks", (
    (2048, (96, 96, 256, 64)), (2560, (96, 96, 320, 64)),
    (4096, (128, 128, 480, 96)), (4096, (192, 192, 384, 128)),
))
@pytest.mark.parametrize("stage,count,rows,canonical", (
    ("in", 3, 1, False), ("in", 4, 89, True),
    ("out", 3, 17, True), ("out", 4, 1024, False),
))
def test_fixed_projection_preserves_accumulators_and_graph(
    monkeypatch, tmp_path, channels, ranks, stage, count, rows, canonical,
):
    _check_fixed_projection(monkeypatch, tmp_path, stage, count, rows, canonical, ranks, channels)


def _check_fixed_projection(monkeypatch, tmp_path, stage, count, rows, canonical, ranks4, channels=4096):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.wkv_prepare import tilelang as implementation

    if not tuning._fixed_strategy_device(tuning.device_class(torch.device("cuda"))):
        pytest.skip("fixed grouped strategy is qualified on tested SM89/SM120 devices")
    torch.manual_seed(849)
    ranks = ranks4[:count]
    if stage == "in":
        xs = tuple(torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16) * .02 for _ in ranks)
        weights = tuple(
            (torch.randn(rank, channels, device="cuda", dtype=torch.bfloat16) if canonical else
             torch.randn(channels, rank, device="cuda", dtype=torch.bfloat16).t()) * .02 for rank in ranks
        )
        kinds, biases = (1, 0, 2, 0)[:count], (None,) * count
    else:
        xs = tuple(torch.randn(rows, rank, device="cuda", dtype=torch.bfloat16) * .02 for rank in ranks)
        weights = tuple(
            (torch.randn(channels, rank, device="cuda", dtype=torch.bfloat16) if canonical else
             torch.randn(rank, channels, device="cuda", dtype=torch.bfloat16).t()) * .02 for rank in ranks
        )
        kinds = (0, 2, 0, 0)[:count]
        biases = (None, torch.randn(channels, device="cuda", dtype=torch.bfloat16), None) + (None,) * (count - 3)
    expected, raw = implementation._run_groups(xs, weights, kinds, biases, raw_mode=2, config=(16, 64, 2, 32))
    monkeypatch.setattr(tuning, "_profile_root", lambda: tmp_path)
    monkeypatch.setattr(tuning, "_read_profile", lambda *args: pytest.fail("read a profile"))
    actual, actual_raw = implementation._run_groups(xs, weights, kinds, biases, raw_mode=2)
    assert torch.equal(actual_raw, raw)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))

    def call():
        return implementation._grouped_linear(xs, weights, kinds, biases)

    for mode in ("inspect", "require", "tune"):
        report = tuning.prepare_tuning([call], mode=mode)
        assert report["complete"] and report["counts"]["fixed"] == 1
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    graph.replay()
    assert all(torch.equal(a, b) for a, b in zip(captured, expected, strict=True))

    for weight in weights:
        weight.copy_(torch.randn_like(weight) * .02)
    replacement, replacement_raw = implementation._run_groups(
        xs, weights, kinds, biases, raw_mode=2, config=(16, 64, 2, 32),
    )
    _, actual_raw = implementation._run_groups(xs, weights, kinds, biases, raw_mode=2)
    graph.replay()
    assert torch.equal(actual_raw, replacement_raw)
    assert all(torch.equal(a, b) for a, b in zip(captured, replacement, strict=True))
    assert not tuple(tmp_path.iterdir())


@cuda
@pytest.mark.cuda
@pytest.mark.cuda_graph
@pytest.mark.parametrize("input_major", (False, True))
@pytest.mark.parametrize("count", (3, 4))
@pytest.mark.parametrize("channels,ranks", (
    (2048, (96, 96, 256, 64)), (4096, (192, 192, 384, 128)),
))
def test_fixed_decode_projection_preserves_accumulators_and_graph(
    monkeypatch, tmp_path, input_major, count, channels, ranks,
):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.wkv_prepare import tilelang as implementation

    if not tuning._fixed_strategy_device(tuning.device_class(torch.device("cuda"))):
        pytest.skip("fixed grouped strategy is qualified on tested SM89/SM120 devices")
    torch.manual_seed(716)
    ranks = ranks[:count]
    xs = tuple(torch.randn(1, channels, device="cuda", dtype=torch.bfloat16) * .03 for _ in ranks)
    weights = tuple(
        torch.randn(channels, rank, device="cuda", dtype=torch.bfloat16).t()
        if input_major else torch.randn(rank, channels, device="cuda", dtype=torch.bfloat16)
        for rank in ranks
    )
    kinds, biases = (1, 0, 2, 0)[:count], (None,) * count
    references = {}
    for mode in (0, 1, 2):
        values, raw = implementation._run_groups(
            xs, weights, kinds, biases, raw_mode=mode, config=(16, 64, 2, 32),
        )
        references[mode] = (tuple(value.clone() for value in values), raw.clone())

    monkeypatch.setattr(tuning, "_profile_root", lambda: tmp_path)
    monkeypatch.setattr(tuning, "_read_profile", lambda *args: pytest.fail("read a profile"))
    monkeypatch.setattr(implementation, "_tune_grouped_linear", lambda *args: pytest.fail("tuned a fixed configuration"))
    monkeypatch.setattr(implementation, "_GROUP_CONFIGS", {})

    def call():
        return implementation._grouped_linear(xs, weights, kinds, biases)

    for mode in ("inspect", "require", "tune"):
        report = tuning.prepare_tuning([call], mode=mode)
        assert report["complete"] and report["counts"]["fixed"] == 1
        assert report["records"][0]["config"] == implementation._fixed_projection_config(xs, weights, kinds, biases)
    for mode in (0, 1, 2):
        values, raw = implementation._run_groups(xs, weights, kinds, biases, raw_mode=mode)
        expected, expected_raw = references[mode]
        assert all(torch.equal(value.view(torch.int16), ref.view(torch.int16)) for value, ref in zip(values, expected))
        assert torch.equal(raw, expected_raw)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = call()
    graph.replay()
    assert all(torch.equal(value.view(torch.int16), ref.view(torch.int16)) for value, ref in zip(actual, references[0][0]))
    assert not tuple(tmp_path.iterdir())


@cuda
@pytest.mark.cuda
def test_fixed_grouped_operator_benchmark_uses_its_candidate(monkeypatch):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.wkv_prepare import tilelang as implementation

    if not tuning._fixed_strategy_device(tuning.device_class(torch.device("cuda"))):
        pytest.skip("fixed grouped strategy is qualified on tested SM89/SM120 devices")
    channels, ranks = 2048, (96, 96, 256)
    xs = tuple(torch.randn(1, channels, device="cuda", dtype=torch.bfloat16) for _ in ranks)
    weights = tuple(torch.randn(rank, channels, device="cuda", dtype=torch.bfloat16) for rank in ranks)
    kinds, biases = (1, 0, 2), (None,) * len(ranks)
    fixed, candidate = (16, 16, 2, 512), (16, 32, 2, 256)
    seen = []

    def benchmark():
        key, _ = implementation._group_call(xs, weights, kinds, biases, 0, None)
        seen.append(key[5])
        return implementation._grouped_linear(xs, weights, kinds, biases)

    select_config = tuning.select_config

    def select_short(*args, **kwargs):
        kwargs["repeat"] = 2
        return select_config(*args, **kwargs)

    monkeypatch.setattr(implementation, "_GROUP_CONFIGS", {})
    monkeypatch.setattr(tuning, "projection_configs", lambda *args: [candidate] if not list(args[-1]) else [])
    monkeypatch.setattr(tuning, "projection_finalists", lambda *args: [candidate])
    monkeypatch.setattr(tuning, "select_config", select_short)
    monkeypatch.setattr(tuning, "save_config", lambda *args: None)
    result = implementation._tune_grouped_linear(xs, weights, kinds, biases, benchmark=benchmark)
    assert result["config"] == candidate and seen and set(seen) == {candidate}
    key, _ = implementation._group_call(xs, weights, kinds, biases, 0, None)
    assert key[5] == fixed


@cuda
@pytest.mark.cuda
@pytest.mark.parametrize("channels,ranks,rows", (
    (2048, (96, 96, 256), 1),
    (2048, (96, 96, 256, 64), 128), (2560, (96, 96, 320, 64), 128),
    (4096, (128, 128, 480, 96), 128), (4096, (192, 192, 384), 128),
    (2048, (96, 96, 256), 577), (2048, (96, 96, 256, 64), 897),
    (2560, (96, 96, 320, 64), 801), (2560, (96, 96, 320, 64), 1024),
    (4096, (192, 192, 384, 128), 256),
    (4096, (192, 192, 384, 128), 1024),
))
def test_fixed_decode_projection_preserves_training_gradients(monkeypatch, channels, ranks, rows):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.wkv_prepare import tilelang as implementation

    if not tuning._fixed_strategy_device(tuning.device_class(torch.device("cuda"))):
        pytest.skip("fixed grouped strategy is qualified on tested SM89/SM120 devices")
    torch.manual_seed(717)
    xs = tuple(
        (torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16) * .03).requires_grad_()
        for _ in ranks
    )
    weights = tuple(
        (torch.randn(channels, rank, device="cuda", dtype=torch.bfloat16) * .03).t().detach().requires_grad_()
        for rank in ranks
    )
    kinds, biases = (1, 0, 2, 0)[:len(ranks)], (None,) * len(ranks)
    upstream = tuple(torch.randn(rows, rank, device="cuda", dtype=torch.bfloat16) for rank in ranks)
    key = implementation._group_key(xs, weights, kinds, biases)
    token = tuning._TUNING.set(True)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(implementation, "_fixed_projection_config", lambda *args: None)
            patch.setattr(implementation, "_fixed_strategy_device", lambda *args: False)
            patch.setattr(shared_linear, "_fixed_strategy_device", lambda *args: False)
            patch.setattr(implementation, "_GROUP_CONFIGS", {key: (16, 64, 2, 32)})
            expected = implementation._grouped_linear(xs, weights, kinds, biases)
            expected_grads = torch.autograd.grad(expected, (*xs, *weights), upstream)
        actual = implementation._grouped_linear(xs, weights, kinds, biases)
        actual_grads = torch.autograd.grad(actual, (*xs, *weights), upstream)
    finally:
        tuning._TUNING.reset(token)
    assert all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(actual, expected))
    assert all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(actual_grads, expected_grads))


@cuda
@pytest.mark.cuda
@pytest.mark.parametrize("channels,rows", ((2048, 769), (2560, 1024)))
@pytest.mark.parametrize("canonical", (False, True))
def test_fixed_long_prefill_output_preserves_training_gradients(monkeypatch, channels, rows, canonical):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.wkv_prepare import tilelang as implementation

    if not tuning._fixed_strategy_device(tuning.device_class(torch.device("cuda"))):
        pytest.skip("fixed grouped strategy is qualified on tested SM89/SM120 devices")
    torch.manual_seed(718)
    ranks = (96, 96, 256 if channels == 2048 else 320, 64)
    xs = tuple((torch.randn(rows, rank, device="cuda", dtype=torch.bfloat16) * .03).requires_grad_() for rank in ranks)
    weights = tuple(
        (torch.randn(channels, rank, device="cuda", dtype=torch.bfloat16) if canonical else
         torch.randn(rank, channels, device="cuda", dtype=torch.bfloat16).t()).mul(.03).detach().requires_grad_()
        for rank in ranks
    )
    bias = (torch.randn(channels, device="cuda", dtype=torch.bfloat16) * .03).requires_grad_()
    kinds, biases = (0, 2, 0, 0), (None, bias, None, None)
    upstream = tuple(torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16) for _ in ranks)
    key = implementation._group_key(xs, weights, kinds, biases)
    token = tuning._TUNING.set(True)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(implementation, "_fixed_projection_config", lambda *args: None)
            patch.setattr(implementation, "_fixed_strategy_device", lambda *args: False)
            patch.setattr(shared_linear, "_fixed_strategy_device", lambda *args: False)
            patch.setattr(implementation, "_GROUP_CONFIGS", {key: (16, 64, 2, 32)})
            expected = implementation._grouped_linear(xs, weights, kinds, biases)
            expected_grads = torch.autograd.grad(expected, (*xs, *weights, bias), upstream)
        actual = implementation._grouped_linear(xs, weights, kinds, biases)
        actual_grads = torch.autograd.grad(actual, (*xs, *weights, bias), upstream)
    finally:
        tuning._TUNING.reset(token)
    assert all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(actual, expected, strict=True))
    assert all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(actual_grads, expected_grads, strict=True))


def _linear_reference(x, weight, adapter=(None, None, 1.0, False)):
    result = (x.float() @ weight.float().t()).to(torch.bfloat16)
    a, b, scale, nora = adapter
    if a is not None:
        normalized = a.float()
        if nora:
            normalized = normalized / (normalized.norm(dim=0, keepdim=True) + 1.0e-6)
        features = (x.float() @ normalized.to(torch.bfloat16).float().t()).to(
            torch.bfloat16
        )
        update = features.float() @ b.to(torch.bfloat16).float().t()
        result = (result.float() + update * scale).to(torch.bfloat16)
    return result


def _prepare_reference(
    shifted,
    dense,
    rank_in,
    rank_out,
    vectors,
    v_first,
    head_size=64,
    w0=None,
    adapters=((None, None, 1.0, False),) * 3,
):
    xr, xw, xk, xv, xa, xg = shifted
    v0, kk, a0, ka = vectors
    r, key, value = [
        _linear_reference(x, weight, adapter)
        for x, weight, adapter in zip((xr, xk, xv), dense, adapters, strict=True)
    ]
    low = [
        _linear_reference(x, w) for x, w in zip((xw, xa, xg, xv), rank_in, strict=True)
    ]
    decay = _linear_reference(low[0].float().tanh().to(torch.bfloat16), rank_out[0])
    if w0 is not None:
        decay = (decay.float() + w0.float()).to(torch.bfloat16)
    rate = (
        (_linear_reference(low[1], rank_out[1]).float() + a0.float())
        .sigmoid()
        .to(torch.bfloat16)
    )
    gate = _linear_reference(low[2].float().sigmoid().to(torch.bfloat16), rank_out[2])
    if v_first is None:
        v_first = value
    else:
        vg = (_linear_reference(low[3], rank_out[3]).float() + v0.float()).sigmoid()
        value = (value.float() + (v_first.float() - value.float()) * vg).to(
            torch.bfloat16
        )
    shape = (*key.shape[:-1], key.shape[-1] // head_size, head_size)
    scaled = (key.float() * kk.float()).reshape(shape)
    direction = torch.nn.functional.normalize(scaled, dim=-1, eps=1.0e-12).reshape_as(
        key
    )
    prepared = (key.float() * (1.0 + (rate.float() - 1.0) * ka.float())).to(
        torch.bfloat16
    )
    return (
        r,
        decay,
        prepared,
        value,
        (-direction).to(torch.bfloat16),
        (direction * rate.float()).to(torch.bfloat16),
        gate,
        v_first,
    )


def _inputs(rows, channels, rank, later_layer, *, grad=False):
    def random(*shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.03

    shifted = [random(rows, channels).requires_grad_(grad) for _ in range(6)]
    dense = [random(channels, channels) for _ in range(3)]
    rank_in = [random(rank, channels) for _ in range(4)]
    rank_out = [random(channels, rank) for _ in range(4)]
    vectors = [random(channels) for _ in range(4)]
    first = random(rows, channels) if later_layer else None
    return shifted, dense, rank_in, rank_out, vectors, first


def _runtime_options(rank_in, rank_out):
    return {
        f"{name}{stage}_runtime": weight.t().contiguous()
        for stage, weights in ((1, rank_in), (2, rank_out))
        for name, weight in zip(("w", "a", "g", "v"), weights, strict=True)
    }


@pytest.mark.resource
@pytest.mark.parametrize("arch", (89, 120))
@pytest.mark.parametrize("input_major", (False, True))
@pytest.mark.parametrize("config", ((16, 8, 2, 32), (256, 128, 2, 64)))
def test_grouped_projection_device_compilation_without_gpu(arch, input_major, config):
    from tilelang.contrib.nvcc import get_cuda_version
    from tilelang.env import CUDA_HOME

    from flashrwkv2.tmix.wkv_prepare.tilelang import _compiled_group

    if CUDA_HOME is None or not (Path(CUDA_HOME) / "bin" / "nvcc").is_file():
        pytest.skip("CUDA Toolkit is required for device compilation")
    if arch == 120 and get_cuda_version() < (12, 8):
        pytest.skip("SM120 compilation requires CUDA Toolkit 12.8 or newer")
    channels, outputs = (128, 64, 96, 32), (32, 64, 16, 48)
    strides = tuple((n if input_major else k, 1) for k, n in zip(channels, outputs))
    kernel = _compiled_group(
        channels,
        outputs,
        strides,
        (1, 0, 2, 0),
        (False, True, True, False),
        config,
        input_major,
        2,
        arch,
        4,
    )
    source = kernel.get_kernel_source()
    assert "tl::mma_sync<" in source
    assert "tl::DataType::kFloat32, 16, 8, 16" in source
    assert "warpgroup_reg_" not in source


@cuda
@pytest.mark.cuda
@pytest.mark.memcheck
@pytest.mark.racecheck
@pytest.mark.parametrize("rows", (1, 17))
@pytest.mark.parametrize("count", (3, 4))
@pytest.mark.parametrize("stage", (1, 2))
@pytest.mark.parametrize("layout", ("row_major", "column_major", "mixed"))
def test_grouped_projection_preserves_fp32_bf16_and_vjp(rows, count, stage, layout):
    from flashrwkv2._internal.elementwise import activation
    from flashrwkv2._internal.linear import _aligned, _forward, linear
    from flashrwkv2.tmix.wkv_prepare.tilelang import _grouped_linear, _run_groups

    torch.manual_seed(714)
    ranks = (3, 8, 9, 5)[:count]
    channels = (128,) * count if stage == 1 else ranks
    outputs = ranks if stage == 1 else (128,) * count
    kinds = ((1, 0, 2, 0) if stage == 1 else (0, 2, 0, 0))[:count]

    def random(shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.03

    xs = tuple(
        random((1, rows, width) if rows > 1 else (rows, width)).requires_grad_()
        for width in channels
    )
    weights = []
    for i, (width, inputs) in enumerate(zip(outputs, channels, strict=True)):
        weight = random((width, inputs))
        if layout == "column_major" or (layout == "mixed" and i % 2):
            weight = weight.t().contiguous().t()
        weights.append(weight.requires_grad_())
    weights = tuple(weights)
    biases = tuple(
        random((width,)).requires_grad_() if stage == 2 and i < 2 else None
        for i, width in enumerate(outputs)
    )
    xrefs = tuple(x.detach().clone().requires_grad_() for x in xs)
    wrefs = tuple(w.detach().clone().requires_grad_() for w in weights)
    brefs = tuple(
        None if b is None else b.detach().clone().requires_grad_() for b in biases
    )
    actual = _grouped_linear(xs, weights, kinds, biases)
    expected = []
    for x, w, kind, bias in zip(xrefs, wrefs, kinds, brefs, strict=True):
        projected = linear(x, w)
        expected.append(
            activation(projected, {0: "identity", 1: "tanh", 2: "sigmoid"}[kind], bias)
            if kind or bias is not None
            else projected
        )
    for output, reference in zip(actual, expected, strict=True):
        assert torch.equal(output.view(torch.int16), reference.view(torch.int16))
    with torch.no_grad():
        flat = tuple(
            _aligned(x.reshape(rows, width))
            for x, width in zip(xs, channels, strict=True)
        )
        aligned = tuple(_aligned(w) for w in weights)
        _, raw = _run_groups(flat, aligned, kinds, biases, raw_mode=2)
        position = 0
        for x, w, width in zip(flat, aligned, outputs, strict=True):
            reference = _forward(x, w, None, torch.float32, (16, 64, 2, 32))
            value = raw[position : position + rows * width].view(rows, width)
            assert torch.equal(value.view(torch.int32), reference.view(torch.int32))
            position += rows * width
    variables = (*xs, *weights, *(b for b in biases if b is not None))
    references = (*xrefs, *wrefs, *(b for b in brefs if b is not None))
    upstream = tuple(torch.randn_like(output) for output in actual)
    gradients = torch.autograd.grad(actual, variables, upstream)
    expected_gradients = torch.autograd.grad(expected, references, upstream)
    for gradient, reference in zip(gradients, expected_gradients, strict=True):
        assert torch.equal(gradient.view(torch.int16), reference.view(torch.int16))


def test_wkv_prepare_public_contract():
    parameters = inspect.signature(infer_tmix_wkv_prepare_forward_varlen).parameters
    assert {
        "v_first",
        "receptance_lora_a",
        "key_lora_a",
        "value_lora_a",
    } <= parameters.keys()
    assert parameters["head_size"].default == 64


@pytest.mark.parametrize("bad_input", ("rows", "projection", "lowrank"))
def test_wkv_prepare_rejects_inconsistent_shapes_before_launch(bad_input):
    shifted = [torch.zeros(2, 64, dtype=torch.bfloat16) for _ in range(6)]
    dense = [torch.zeros(64, 64, dtype=torch.bfloat16) for _ in range(3)]
    rank_in = [torch.zeros(8, 64, dtype=torch.bfloat16) for _ in range(4)]
    rank_out = [torch.zeros(64, 8, dtype=torch.bfloat16) for _ in range(4)]
    vectors = [torch.zeros(64, dtype=torch.bfloat16) for _ in range(4)]
    if bad_input == "rows":
        shifted[1] = shifted[1][:1]
        message = "share a nonempty"
    elif bad_input == "projection":
        dense[0] = dense[0][:-1]
        message = "R/K/V weights"
    else:
        rank_out[0] = rank_out[0][:-1]
        message = "w lowrank weights"
    with pytest.raises(ValueError, match=message):
        infer_tmix_wkv_prepare_forward_varlen(
            *shifted,
            *dense,
            *rank_in,
            *rank_out,
            *vectors,
        )


@cuda
@pytest.mark.cuda
@pytest.mark.memcheck
@pytest.mark.parametrize("later_layer", (False, True))
@pytest.mark.parametrize("head_size", (64, 128, 256))
def test_wkv_prepare_first_and_later_layer_reference(later_layer, head_size):
    torch.manual_seed(20260818 + head_size + int(later_layer))
    shifted, dense, rank_in, rank_out, vectors, first = _inputs(
        2, 4096, 32, later_layer
    )
    outputs = infer_tmix_wkv_prepare_forward_varlen(
        *shifted,
        *dense,
        *rank_in,
        *rank_out,
        *vectors,
        v_first=first,
        head_size=head_size,
        **_runtime_options(rank_in, rank_out),
    )
    expected = _prepare_reference(
        shifted, dense, rank_in, rank_out, vectors, first, head_size
    )
    for actual, reference in zip(outputs, expected, strict=True):
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(
            actual.float(), reference.float(), atol=0.10, rtol=0.06
        )


@cuda
@pytest.mark.cuda
def test_wkv_prepare_projection_lora_matches_reference_and_fails_closed():
    torch.manual_seed(20260819)
    shifted, dense, rank_in, rank_out, vectors, _ = _inputs(1, 4096, 32, False)
    a = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16) * 0.02
    b = torch.randn(4096, 8, device="cuda", dtype=torch.bfloat16) * 0.02
    options = {
        "receptance_lora_a": a,
        "receptance_lora_b": b,
        "receptance_lora_scale": 0.5,
    }
    outputs = infer_tmix_wkv_prepare_forward_varlen(
        *shifted,
        *dense,
        *rank_in,
        *rank_out,
        *vectors,
        **options,
    )
    reference = _linear_reference(shifted[0], dense[0], (a, b, 0.5, False))
    torch.testing.assert_close(
        outputs[0].float(), reference.float(), atol=0.08, rtol=0.05
    )
    with pytest.raises(ValueError, match="provided together"):
        infer_tmix_wkv_prepare_forward_varlen(
            *shifted,
            *dense,
            *rank_in,
            *rank_out,
            *vectors,
            **{**options, "receptance_lora_b": None},
        )


@cuda
@pytest.mark.cuda
def test_training_wkv_prepare_nora_independent_gradients():
    from utils import assert_relative_rmse

    torch.manual_seed(89)
    channels, rank = 128, 256
    inputs, weights, rank_in, rank_out, vectors, first = _inputs(
        3, channels, 16, True, grad=True
    )
    first.requires_grad_()
    w0 = torch.randn(channels, device="cuda", dtype=torch.bfloat16) * 0.03
    adapters = [
        (torch.randn(shape, device="cuda") * 0.03).requires_grad_()
        for shape in ((rank, channels), (channels, rank)) * 3
    ]
    refs = [
        tensor.detach().clone().requires_grad_()
        for tensor in (*inputs, first, *adapters)
    ]
    w1, a1, g1, v1 = [weight.t() for weight in rank_in]
    w2, a2, g2, v2 = [weight.t() for weight in rank_out]
    v0, kk, a0, ka = vectors
    output = pretrain_tmix_wkv_prepare_bf16(
        *inputs,
        *weights,
        w0,
        w1,
        w2,
        a0,
        a1,
        a2,
        g1,
        g2,
        kk,
        ka,
        v_first=first,
        v0=v0,
        v1=v1,
        v2=v2,
        receptance_lora_a=adapters[0],
        receptance_lora_b=adapters[1],
        key_lora_a=adapters[2],
        key_lora_b=adapters[3],
        value_lora_a=adapters[4],
        value_lora_b=adapters[5],
        receptance_use_nora=True,
        key_use_nora=True,
        value_use_nora=True,
    )
    expected = _prepare_reference(
        refs[:6],
        weights,
        rank_in,
        rank_out,
        vectors,
        refs[6],
        w0=w0,
        adapters=tuple((refs[7 + 2 * i], refs[8 + 2 * i], 1.0, True) for i in range(3)),
    )
    upstream = [torch.randn_like(tensor) for tensor in output]
    gradients = torch.autograd.grad(output, (*inputs, first, *adapters), upstream)
    reference_gradients = torch.autograd.grad(expected, refs, upstream)
    for actual, reference in zip(output, expected, strict=True):
        assert_relative_rmse(actual, reference)
    for actual, reference in zip(gradients, reference_gradients, strict=True):
        assert_relative_rmse(actual, reference, 0.04)


@cuda
@pytest.mark.cuda
@pytest.mark.parametrize("channels", (768, 4096))
@pytest.mark.parametrize("later_layer", (False, True))
def test_prepare_training_batch_and_raw_decay_bias_are_bitwise_equal(
    channels, later_layer
):
    from flashrwkv2.tmix.wkv7.tilelang import tmix_wkv7_recurrent

    torch.manual_seed(107 + channels + int(later_layer))
    rows = 27
    shifted, dense, rank_in, rank_out, vectors, first = _inputs(
        rows, channels, 32, later_layer, grad=True
    )
    runtime = _runtime_options(rank_in, rank_out)
    w0 = torch.randn(channels, device="cuda", dtype=torch.bfloat16) * 0.1
    with torch.no_grad():
        packed = infer_tmix_wkv_prepare_forward_varlen(
            *shifted,
            *dense,
            *rank_in,
            *rank_out,
            *vectors,
            v_first=first,
        )
        runtime_output = infer_tmix_wkv_prepare_forward_varlen(
            *shifted,
            *dense,
            *((None,) * 8),
            *vectors,
            v_first=first,
            **runtime,
        )
        for a, b in zip(packed, runtime_output, strict=True):
            assert torch.equal(a, b)
        for row in (0, 13, 26):
            single = infer_tmix_wkv_prepare_forward_varlen(
                *[x[row : row + 1] for x in shifted],
                *dense,
                *rank_in,
                *rank_out,
                *vectors,
                v_first=None if first is None else first[row : row + 1],
            )
            for all_rows, one_row in zip(packed, single, strict=True):
                assert torch.equal(all_rows[row : row + 1], one_row)
    v0, kk, a0, ka = vectors
    trained = pretrain_tmix_wkv_prepare_bf16(
        *[x.unsqueeze(0) for x in shifted],
        *dense,
        w0,
        runtime["w1_runtime"],
        runtime["w2_runtime"],
        a0,
        runtime["a1_runtime"],
        runtime["a2_runtime"],
        runtime["g1_runtime"],
        runtime["g2_runtime"],
        kk,
        ka,
        v_first=None if first is None else first.unsqueeze(0),
        v0=v0,
        v1=runtime["v1_runtime"],
        v2=runtime["v2_runtime"],
    )
    for index, (train, infer) in enumerate(zip(trained, packed, strict=True)):
        expected = (
            (infer.float() + w0.float()).to(torch.bfloat16) if index == 1 else infer
        )
        assert torch.equal(train.squeeze(0), expected)
    heads, head_size = channels // 64, 64
    initial = torch.randn(1, heads, head_size, head_size, device="cuda") * 0.02
    cu = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    train_state = tmix_wkv7_recurrent(
        *[x.reshape(rows, heads, head_size) for x in trained[:6]],
        initial,
        cu,
    )
    with torch.no_grad():
        infer_state = tmix_wkv7_recurrent(
            *[x.reshape(rows, heads, head_size) for x in packed[:6]],
            initial,
            cu,
            decay_bias=w0.reshape(heads, head_size),
        )
    for train, infer in zip(train_state, infer_state, strict=True):
        assert torch.equal(train, infer)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fullgraph_grouped_projection_preserves_unused_gradients():
    from flashrwkv2.tmix.wkv_prepare.tilelang import _grouped_linear
    torch.manual_seed(741)
    xs = [torch.randn(3, 64, device="cuda", dtype=torch.bfloat16).requires_grad_() for _ in range(2)]
    weights = [torch.randn(32, 64, device="cuda", dtype=torch.bfloat16).t().contiguous().t().requires_grad_() for _ in range(2)]
    biases = [torch.randn(32, device="cuda", dtype=torch.bfloat16).requires_grad_() for _ in range(2)]
    def model():
        return _grouped_linear(tuple(xs), tuple(weights), (1, 2), tuple(biases))[0]
    eager = model()
    eager.float().sum().backward()
    leaves = (*xs, *weights, *biases)
    expected = [None if value.grad is None else value.grad.clone() for value in leaves]
    for value in leaves:
        value.grad = None
    compiled = torch.compile(model, fullgraph=True, options={"emulate_precision_casts": True})
    actual = compiled()
    actual.float().sum().backward()
    assert torch.equal(actual, eager)
    for value, reference in zip(leaves, expected, strict=True):
        assert (value.grad is None) == (reference is None)
        if reference is not None:
            assert torch.equal(value.grad, reference)


@cuda
@pytest.mark.cuda
@pytest.mark.cuda_graph
@pytest.mark.parametrize("stage", (
    "dense", "lowrank_output", "training_output", "training_output_72b", "training_output_13b_first",
))
def test_fixed_group_projection_preserves_accumulators_and_vjp(monkeypatch, tmp_path, stage):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.wkv_prepare import tilelang as implementation

    if not tuning._fixed_strategy_device(tuning.device_class(torch.device("cuda"))):
        pytest.skip("fixed grouped strategy is qualified on tested devices")
    torch.manual_seed(722)
    channels = 2048
    if stage == "dense":
        ranks, kinds = (channels,) * 3, (0, 0, 0)
        xs = tuple(torch.randn(1, channels, device="cuda", dtype=torch.bfloat16).requires_grad_() for _ in ranks)
        weights = tuple(torch.randn(channels, channels, device="cuda", dtype=torch.bfloat16).requires_grad_() for _ in ranks)
        biases = (None,) * 3
    else:
        if stage == "training_output_72b":
            channels, ranks = 4096, (128, 128, 480, 96)
        elif stage == "training_output_13b_first":
            channels, ranks = 4096, (192, 192, 384)
        else:
            ranks = (96, 96, 256, 64)
        kinds = (0, 2, 0, 0)[:len(ranks)]
        rows = 1 if stage == "lowrank_output" else 128
        xs = tuple(torch.randn(rows, rank, device="cuda", dtype=torch.bfloat16).requires_grad_() for rank in ranks)
        weights = tuple(torch.randn(rank, channels, device="cuda", dtype=torch.bfloat16).t().detach().requires_grad_() for rank in ranks)
        first_bias = None if stage == "lowrank_output" else torch.randn(channels, device="cuda", dtype=torch.bfloat16).requires_grad_()
        biases = (first_bias, torch.randn(channels, device="cuda", dtype=torch.bfloat16).requires_grad_()) + (None,) * (len(ranks) - 2)
    fixed = implementation._fixed_projection_config(xs, weights, kinds, biases)
    assert fixed is not None
    with torch.no_grad():
        reference, raw = implementation._run_groups(xs, weights, kinds, biases, raw_mode=2, config=(16, 64, 2, 32))
        candidate, candidate_raw = implementation._run_groups(xs, weights, kinds, biases, raw_mode=2, config=fixed)
        assert all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(candidate, reference))
        assert torch.equal(candidate_raw.view(torch.int32), raw.view(torch.int32))

    variables = (*xs, *weights, *(bias for bias in biases if bias is not None))
    upstream = tuple(torch.randn_like(value) for value in reference)
    key = implementation._group_key(xs, weights, kinds, biases)
    token = tuning._TUNING.set(True)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(implementation, "_fixed_projection_config", lambda *args: None)
            patch.setattr(implementation, "_fixed_strategy_device", lambda *args: False)
            patch.setattr(shared_linear, "_fixed_strategy_device", lambda *args: False)
            patch.setattr(implementation, "_GROUP_CONFIGS", {key: (16, 64, 2, 32)})
            expected = implementation._grouped_linear(xs, weights, kinds, biases)
            expected_grads = torch.autograd.grad(expected, variables, upstream)
        actual = implementation._grouped_linear(xs, weights, kinds, biases)
        actual_grads = torch.autograd.grad(actual, variables, upstream)
    finally:
        tuning._TUNING.reset(token)
    assert all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(actual, expected))
    assert all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(actual_grads, expected_grads))

    monkeypatch.setattr(tuning, "_profile_root", lambda: tmp_path)
    monkeypatch.setattr(tuning, "_read_profile", lambda *args: pytest.fail("read a profile"))
    monkeypatch.setattr(implementation, "_tune_grouped_linear", lambda *args: pytest.fail("tuned a fixed configuration"))
    monkeypatch.setattr(implementation, "_GROUP_CONFIGS", {})

    def call():
        with torch.no_grad():
            return implementation._grouped_linear(xs, weights, kinds, biases)

    for mode in ("inspect", "require", "tune"):
        report = tuning.prepare_tuning([call], mode=mode)
        assert report["complete"] and report["counts"]["fixed"] == 1
        assert report["records"][0]["config"] == fixed
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    graph.replay()
    assert all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(captured, reference))
    assert not tuple(tmp_path.iterdir())


@cuda
@pytest.mark.cuda
@pytest.mark.cuda_graph
def test_grouped_weight_vjp_reuses_dynamic_reduction(monkeypatch):
    from flashrwkv2._internal import tuning
    from flashrwkv2._internal.linear import _forward
    from flashrwkv2.tmix.wkv_prepare import tilelang as implementation

    if not tuning._fixed_strategy_device(tuning.device_class(torch.device("cuda"))):
        pytest.skip("dynamic grouped VJP is qualified on tested devices")
    torch.manual_seed(723)
    ranks, channels = (96, 96, 256, 64), 2048
    kinds, biases, config = (0,) * 4, (None,) * 4, (64, 64, 1, 64)
    monkeypatch.setattr(implementation, "_GROUP_KERNELS", {})
    for reduction in (17, 32, 64, 65, 89, 128, 1000, 1024):
        xs = tuple(torch.randn(channels, reduction, device="cuda", dtype=torch.bfloat16) * .02 for _ in ranks)
        weights = tuple((torch.randn(reduction, rank, device="cuda", dtype=torch.bfloat16) * .02).t() for rank in ranks)
        values, raw = implementation._run_groups(
            xs, weights, kinds, biases, raw_mode=2, config=config, dynamic_k=True
        )
        offset = 0
        for x, weight, value in zip(xs, weights, values, strict=True):
            reference = _forward(x, weight, None, torch.float32, config=(16, 64, 2, 32))
            size = reference.numel()
            assert torch.equal(raw[offset : offset + size].view_as(reference), reference)
            assert torch.equal(value, reference.bfloat16())
            offset += size
        assert len(implementation._GROUP_KERNELS) <= 2
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured, captured_raw = implementation._run_groups(
                xs, weights, kinds, biases, raw_mode=2, config=config, dynamic_k=True
            )
        graph.replay()
        assert torch.equal(captured_raw, raw)
        assert all(torch.equal(a, b) for a, b in zip(captured, values, strict=True))


@cuda
@pytest.mark.cuda
@pytest.mark.cuda_graph
@pytest.mark.parametrize("stage", ("input", "output"))
def test_grouped_training_vjp_preserves_compile_and_graph(monkeypatch, tmp_path, stage):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.wkv_prepare import tilelang as implementation

    if not tuning._fixed_strategy_device(tuning.device_class(torch.device("cuda"))):
        pytest.skip("grouped VJP is qualified on tested devices")
    torch.manual_seed(724)
    rows, channels, ranks = 128, 2048, (96, 96, 256, 64)
    if stage == "input":
        xs = tuple((torch.randn(rows, channels, device="cuda", dtype=torch.bfloat16) * .02).requires_grad_() for _ in ranks)
        weights = tuple((torch.randn(channels, rank, device="cuda", dtype=torch.bfloat16) * .02).t().detach().requires_grad_() for rank in ranks)
        biases, kinds = (), (1, 0, 2, 0)
    else:
        xs = tuple((torch.randn(rows, rank, device="cuda", dtype=torch.bfloat16) * .02).requires_grad_() for rank in ranks)
        weights = tuple((torch.randn(rank, channels, device="cuda", dtype=torch.bfloat16) * .02).t().detach().requires_grad_() for rank in ranks)
        biases = tuple((torch.randn(channels, device="cuda", dtype=torch.bfloat16) * .02).requires_grad_() for _ in range(2))
        kinds = (0, 2, 0, 0)
    leaves = (*xs, *weights, *biases)
    upstream = tuple(torch.randn(rows, weight.shape[0], device="cuda", dtype=torch.bfloat16) for weight in weights)

    def model(*args):
        packed_biases = (None,) * 4 if stage == "input" else (*args[8:], None, None)
        return implementation._grouped_linear(args[:4], args[4:8], kinds, packed_biases)

    packed_biases = (None,) * 4 if stage == "input" else (*biases, None, None)
    key = implementation._group_key(xs, weights, kinds, packed_biases)
    token = tuning._TUNING.set(True)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(implementation, "_fixed_projection_config", lambda *args: None)
            patch.setattr(implementation, "_fixed_strategy_device", lambda *args: False)
            patch.setattr(shared_linear, "_fixed_strategy_device", lambda *args: False)
            patch.setattr(implementation, "_GROUP_CONFIGS", {key: (16, 64, 2, 32)})
            reference = model(*leaves)
            reference_grads = torch.autograd.grad(reference, leaves, upstream)
    finally:
        tuning._TUNING.reset(token)

    monkeypatch.setattr(tuning, "_profile_root", lambda: tmp_path)
    if stage == "input":
        monkeypatch.setattr(tuning, "_read_profile", lambda *args: None)
    else:
        monkeypatch.setattr(tuning, "_read_profile", lambda *args: pytest.fail("read a profile"))
    monkeypatch.setattr(implementation, "_GROUP_KERNELS", {})
    compiled = torch.compile(model, fullgraph=True, options={"emulate_precision_casts": True})
    actual = compiled(*leaves)
    actual_grads = torch.autograd.grad(actual, leaves, upstream)
    assert all(torch.equal(a, b) for a, b in zip(actual, reference, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(actual_grads, reference_grads, strict=True))
    if stage == "output":
        assert any(len(key) == 12 and key[-2] is True for key in implementation._GROUP_KERNELS)
    else:
        assert any(key[1] == (channels,) * 4 and key[3] == (0,) * 4 and key[7] == 0
                   for key in implementation._GROUP_KERNELS)

    graph_leaves = tuple(value.detach().requires_grad_() for value in leaves)

    def backward():
        values = model(*graph_leaves)
        return torch.autograd.grad(values, graph_leaves, upstream)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        backward()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = backward()
    graph.replay()
    assert all(torch.equal(a, b) for a, b in zip(captured, reference_grads, strict=True))

    compiled_leaves = tuple(value.detach().requires_grad_() for value in leaves)

    def compiled_backward():
        return torch.autograd.grad(compiled(*compiled_leaves), compiled_leaves, upstream)

    with torch.cuda.stream(stream):
        compiled_backward()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    compiled_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(compiled_graph, stream=stream):
        captured_compiled = compiled_backward()
    compiled_graph.replay()
    assert all(torch.equal(a, b) for a, b in zip(captured_compiled, reference_grads, strict=True))

    with monkeypatch.context() as patch:
        patch.setattr(tuning, "_read_profile", lambda *args: None)
        indices = (0, 4) if stage == "input" else (0, 4, 8)
        partial = torch.autograd.grad(model(*leaves)[0], tuple(leaves[i] for i in indices), upstream[0])
        assert all(torch.equal(a, reference_grads[i]) for a, i in zip(partial, indices, strict=True))
        frozen_leaves = (*xs, weights[0].detach(), *weights[1:], *biases)
        frozen_inputs = (*xs, *weights[1:], *biases)
        frozen = torch.autograd.grad(model(*frozen_leaves), frozen_inputs, upstream)
        assert all(torch.equal(a, b) for a, b in zip(frozen, (g for i, g in enumerate(reference_grads) if i != 4), strict=True))


@cuda
@pytest.mark.cuda
@pytest.mark.parametrize("mode", ("frozen_input", "frozen_weight", "frozen_route", "mixed_layout", "unused"))
def test_grouped_vjp_eligibility_preserves_partial_gradients(monkeypatch, mode):
    from flashrwkv2._internal import linear as shared
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.wkv_prepare import tilelang as implementation

    if not tuning._fixed_strategy_device(tuning.device_class(torch.device("cuda"))):
        pytest.skip("grouped VJP is qualified on tested devices")
    torch.manual_seed(725)
    xs = tuple(torch.randn(3, 64, device="cuda").bfloat16().requires_grad_(
        not ((mode == "frozen_input" and i == 1) or (mode == "frozen_route" and i == 0))) for i in range(2))
    weights = tuple((torch.randn(64, 32, device="cuda").bfloat16().t()
                     if mode != "mixed_layout" or i == 0 else
                     torch.randn(32, 64, device="cuda").bfloat16()).requires_grad_(
                         not (mode in ("frozen_weight", "frozen_route") and i == 0)) for i in range(2))
    biases = tuple(torch.randn(32, device="cuda").bfloat16().requires_grad_(
        mode != "frozen_route" or i != 0) for i in range(2))
    leaves = tuple(v for v in (*xs, *weights, *biases) if v.requires_grad)
    def model():
        values = implementation._grouped_linear(xs, weights, (1, 2), biases)
        return values[:1] if mode == "unused" else values
    token = tuning._TUNING.set(True)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(implementation, "_fixed_strategy_device", lambda *args: False)
            patch.setattr(shared, "_fixed_strategy_device", lambda *args: False)
            expected = model()
            gradients = torch.autograd.grad(sum(v.float().sum() for v in expected), leaves, allow_unused=True)
    finally:
        tuning._TUNING.reset(token)
    if mode == "frozen_route":
        activation_kernel = implementation._activation_kernel
        def needed_activation(channels, kind, *args):
            assert kind != "tanh", "computed a derivative for a fully frozen route"
            return activation_kernel(channels, kind, *args)
        monkeypatch.setattr(implementation, "_activation_kernel", needed_activation)
    compiled = torch.compile(model, fullgraph=True, options={"emulate_precision_casts": True})
    for call in (model, compiled):
        actual = call()
        actual_grads = torch.autograd.grad(sum(v.float().sum() for v in actual), leaves, allow_unused=True)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
        for a, b in zip(actual_grads, gradients, strict=True):
            assert (a is None) == (b is None)
            if b is not None:
                assert torch.equal(a, b)
