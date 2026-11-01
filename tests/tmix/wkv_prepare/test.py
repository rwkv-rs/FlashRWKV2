# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

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


def test_fixed_decode_projection_restricts_models_and_layouts(monkeypatch):
    from flashrwkv2.tmix.wkv_prepare import tilelang as implementation

    monkeypatch.setattr(implementation, "device_class", lambda device: ("test", 8, 9, 128))
    ranks = (96, 96, 256)
    xs = tuple(torch.empty(1, 2048, device="meta") for _ in ranks)
    weights = tuple(torch.empty(rank, 2048, device="meta") for rank in ranks)
    kinds, biases = (1, 0, 2), (None,) * 3
    fixed = implementation._fixed_projection_config
    assert fixed(xs, weights, kinds, biases) == (16, 32, 2, 512)
    assert fixed(tuple(torch.empty(2, 2048, device="meta") for _ in ranks), weights, kinds, biases) == (16, 16, 2, 512)
    assert fixed(tuple(torch.empty(513, 2048, device="meta") for _ in ranks), weights, kinds, biases) == (32, 32, 2, 128)
    assert fixed(xs, weights, (0, 0, 0), biases) is None
    assert fixed(xs, weights, kinds, (weights[0][0], None, None)) is None
    mixed_layout = (weights[0].t().contiguous().t(), *weights[1:])
    assert fixed(xs, mixed_layout, kinds, biases) is None
    unknown_ranks = (weights[0], weights[1], torch.empty(128, 2048, device="meta"))
    assert fixed(xs, unknown_ranks, kinds, biases) is None
    output_xs = tuple(torch.empty(1, rank, device="meta") for rank in ranks)
    output_weights = tuple(torch.empty(rank, 2048, device="meta").t() for rank in ranks)
    output_biases = (None, torch.empty(2048, device="meta"), None)
    assert fixed(output_xs, output_weights, (0, 2, 0), output_biases) == (16, 32, 2, 64)
    assert fixed(output_xs, tuple(w.contiguous() for w in output_weights), (0, 2, 0), output_biases) == (16, 32, 2, 64)
    assert fixed(output_xs, output_weights, (0, 2, 0), biases) is None
    dense_weights = tuple(torch.empty(2048, 2048, device="meta") for _ in ranks)
    assert fixed(xs, dense_weights, (0, 0, 0), biases) == (16, 32, 2, 256)

    profiles = (
        (2048, (96, 96, 256, 64)), (2560, (96, 96, 320, 64)),
        (4096, (128, 128, 480, 96)), (4096, (192, 192, 384, 128)),
    )
    for channels, ranks in profiles:
        for rows in ((1025,) if channels < 4096 else (513, 960, 1025)):
            input_xs = tuple(torch.empty(rows, channels, device="meta") for _ in ranks)
            input_weights = tuple(torch.empty(rank, channels, device="meta") for rank in ranks)
            assert fixed(input_xs, input_weights, (1, 0, 2, 0), (None,) * 4) is None
            output_xs = tuple(torch.empty(rows, rank, device="meta") for rank in ranks)
            output_weights = tuple(torch.empty(channels, rank, device="meta") for rank in ranks)
            output_biases = (None, torch.empty(channels, device="meta"), None, None)
            assert fixed(output_xs, output_weights, (0, 2, 0, 0), output_biases) is None
            dense = tuple(torch.empty(channels, channels, device="meta") for _ in range(3))
            assert fixed(input_xs[:3], dense, (0, 0, 0), (None,) * 3) is None

    ranks = (128, 128, 480, 96)
    for rows in (320, 512):
        xs = tuple(torch.empty(rows, rank, device="meta") for rank in ranks)
        weights = tuple(torch.empty(rank, 4096, device="meta").t() for rank in ranks)
        biases = (None, torch.empty(4096, device="meta"), None, None)
        assert fixed(xs, weights, (0, 2, 0, 0), biases) is None


@cuda
@pytest.mark.cuda
@pytest.mark.cuda_graph
@pytest.mark.parametrize("channels,ranks", (
    (2048, (96, 96, 256, 64)), (2560, (96, 96, 320, 64)),
    (4096, (128, 128, 480, 96)), (4096, (192, 192, 384, 128)),
))
@pytest.mark.parametrize("count", (3, 4))
def test_fixed_canonical_decode_output_preserves_accumulators_and_graph(monkeypatch, tmp_path, channels, ranks, count):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.wkv_prepare import tilelang as implementation

    if not tuning._fixed_strategy_device(tuning.device_class(torch.device("cuda"))):
        pytest.skip("fixed grouped strategy is qualified on RTX 4090")
    torch.manual_seed(855)
    ranks = ranks[:count]
    xs = tuple(torch.randn(1, rank, device="cuda", dtype=torch.bfloat16) * .02 for rank in ranks)
    weights = tuple(torch.randn(channels, rank, device="cuda", dtype=torch.bfloat16) * .02 for rank in ranks)
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

    report = tuning.prepare_tuning([call], mode="require")
    assert report["complete"] and report["counts"]["fixed"] == 1
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    graph.replay()
    assert all(torch.equal(a, b) for a, b in zip(captured, expected, strict=True))
    assert not tuple(tmp_path.iterdir())


@cuda
@pytest.mark.cuda
@pytest.mark.cuda_graph
@pytest.mark.parametrize(("channels", "rows"), (
    (2048, 17), (2048, 64), (2048, 65), (2048, 160), (2048, 161),
    (2048, 193), (2048, 256), (2048, 320), (2048, 321),
    (2048, 448), (2048, 449), (2048, 512), (2048, 513), (2048, 577), (2048, 1024),
    (2560, 17), (2560, 64), (2560, 65), (2560, 192), (2560, 193),
    (2560, 256), (2560, 257), (2560, 384), (2560, 385), (2560, 512),
    (2560, 513), (2560, 960), (2560, 1024),
    (4096, 17), (4096, 32), (4096, 33), (4096, 64), (4096, 128), (4096, 192), (4096, 193), (4096, 256),
    (4096, 257), (4096, 320), (4096, 512), (4096, 961), (4096, 1024),
))
def test_fixed_rkv_row_wave_preserves_accumulators_and_graph(monkeypatch, tmp_path, channels, rows):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.wkv_prepare import tilelang as implementation

    if not tuning._fixed_strategy_device(tuning.device_class(torch.device("cuda"))):
        pytest.skip("fixed grouped strategy is qualified on RTX 4090")
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
        pytest.skip("fixed grouped strategy is qualified on RTX 4090")
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
@pytest.mark.parametrize("stage,count,rows", (
    ("in", 3, 2), ("in", 4, 16), ("out", 3, 2), ("out", 4, 16),
    ("in", 3, 17), ("in", 4, 32), ("in", 3, 33),
    ("out", 3, 17), ("out", 4, 32), ("out", 4, 63),
    ("in", 3, 64), ("in", 4, 128), ("in", 3, 320), ("in", 4, 512),
    ("out", 3, 64), ("out", 4, 128), ("out", 3, 320), ("out", 4, 512),
    ("in", 3, 193), ("in", 4, 256), ("in", 3, 961), ("in", 4, 1024),
    ("out", 3, 193), ("out", 4, 256), ("out", 3, 961), ("out", 4, 1024),
))
@pytest.mark.parametrize("canonical", (False, True))
def test_fixed_13b_projection_preserves_accumulators_and_graph(monkeypatch, tmp_path, stage, count, rows, canonical):
    _check_fixed_projection(monkeypatch, tmp_path, stage, count, rows, canonical, (192, 192, 384, 128))


@cuda
@pytest.mark.cuda
@pytest.mark.cuda_graph
@pytest.mark.parametrize("stage,count,rows", (
    ("in", 3, 65), ("in", 3, 80), ("in", 3, 81), ("in", 4, 65), ("in", 4, 96),
    ("in", 3, 97), ("in", 4, 128), ("in", 4, 193), ("in", 3, 256),
    ("out", 3, 65), ("out", 4, 65), ("out", 3, 96), ("out", 4, 96),
    ("out", 3, 97), ("out", 4, 128), ("out", 3, 129), ("out", 4, 129),
    ("out", 3, 160), ("out", 3, 161), ("out", 3, 192), ("out", 4, 192),
    ("out", 3, 193), ("out", 4, 256),
))
@pytest.mark.parametrize("canonical", (False, True))
def test_fixed_72b_projection_preserves_accumulators_and_graph(monkeypatch, tmp_path, stage, count, rows, canonical):
    _check_fixed_projection(monkeypatch, tmp_path, stage, count, rows, canonical, (128, 128, 480, 96))


@cuda
@pytest.mark.cuda
@pytest.mark.cuda_graph
@pytest.mark.parametrize("channels,ranks", (
    (2048, (96, 96, 256, 64)), (2560, (96, 96, 320, 64)),
))
@pytest.mark.parametrize("stage,count,rows", (
    ("in", 3, 17), ("in", 4, 48), ("in", 4, 49), ("in", 3, 64),
    ("in", 4, 65), ("in", 4, 112), ("in", 4, 113), ("in", 3, 144),
    ("in", 3, 145), ("in", 4, 448), ("in", 4, 449), ("in", 4, 512),
    ("out", 3, 17), ("out", 4, 64), ("out", 4, 96), ("out", 4, 97),
    ("out", 3, 320), ("out", 4, 321), ("out", 3, 352), ("out", 3, 353),
    ("out", 4, 512),
    ("in", 3, 513), ("in", 3, 577), ("in", 3, 769), ("in", 4, 673),
    ("in", 4, 769), ("in", 4, 801), ("in", 4, 897), ("in", 4, 1024),
    ("out", 3, 513), ("out", 4, 769), ("out", 4, 1024),
))
@pytest.mark.parametrize("canonical", (False, True))
def test_fixed_small_projection_preserves_accumulators_and_graph(
    monkeypatch, tmp_path, channels, ranks, stage, count, rows, canonical,
):
    _check_fixed_projection(monkeypatch, tmp_path, stage, count, rows, canonical, ranks, channels)


def _check_fixed_projection(monkeypatch, tmp_path, stage, count, rows, canonical, ranks4, channels=4096):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.wkv_prepare import tilelang as implementation

    if not tuning._fixed_strategy_device(tuning.device_class(torch.device("cuda"))):
        pytest.skip("fixed grouped strategy is qualified on RTX 4090")
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
        pytest.skip("fixed grouped strategy is qualified on RTX 4090")
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
        assert report["records"][0]["config"] == (16, 32 if input_major or channels < 4096 else 16, 2, 512)
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
        pytest.skip("fixed grouped strategy is qualified on RTX 4090")
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
    (2048, (96, 96, 256), 577), (2048, (96, 96, 256, 64), 897),
    (2560, (96, 96, 320, 64), 801), (2560, (96, 96, 320, 64), 1024),
    (4096, (192, 192, 384, 128), 256),
    (4096, (192, 192, 384, 128), 1024),
))
def test_fixed_decode_projection_preserves_training_gradients(monkeypatch, channels, ranks, rows):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.wkv_prepare import tilelang as implementation

    if not tuning._fixed_strategy_device(tuning.device_class(torch.device("cuda"))):
        pytest.skip("fixed grouped strategy is qualified on RTX 4090")
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
        pytest.skip("fixed grouped strategy is qualified on RTX 4090")
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
@pytest.mark.parametrize("stage", ("dense", "lowrank_output"))
def test_fixed_decode_remaining_groups_preserve_accumulators_and_vjp(monkeypatch, tmp_path, stage):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.wkv_prepare import tilelang as implementation

    if not tuning._fixed_strategy_device(tuning.device_class(torch.device("cuda"))):
        pytest.skip("fixed grouped strategy is qualified on tested devices")
    torch.manual_seed(722)
    channels = 2048
    if stage == "dense":
        ranks, kinds, fixed = (channels,) * 3, (0, 0, 0), (16, 32, 2, 256)
        xs = tuple(torch.randn(1, channels, device="cuda", dtype=torch.bfloat16).requires_grad_() for _ in ranks)
        weights = tuple(torch.randn(channels, channels, device="cuda", dtype=torch.bfloat16).requires_grad_() for _ in ranks)
        biases = (None,) * 3
    else:
        ranks, kinds, fixed = (96, 96, 256, 64), (0, 2, 0, 0), (16, 32, 2, 64)
        xs = tuple(torch.randn(1, rank, device="cuda", dtype=torch.bfloat16).requires_grad_() for rank in ranks)
        weights = tuple(torch.randn(rank, channels, device="cuda", dtype=torch.bfloat16).t().detach().requires_grad_() for rank in ranks)
        biases = (None, torch.randn(channels, device="cuda", dtype=torch.bfloat16).requires_grad_(), None, None)

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
