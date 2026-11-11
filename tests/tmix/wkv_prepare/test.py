# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flashrwkv2.tmix.wkv_prepare import (
    infer_tmix_wkv_prepare_forward_varlen,
    pretrain_tmix_wkv_prepare_bf16,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@cuda
@pytest.mark.cuda
def test_grouped_operator_benchmark_uses_its_candidate(monkeypatch):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.wkv_prepare import tilelang as implementation

    if not tuning._fixed_strategy_device(tuning.device_class(torch.device("cuda"))):
        pytest.skip("fixed grouped strategy is qualified on tested SM89/SM120 devices")
    channels, ranks = 2048, (96, 96, 256)
    xs = tuple(torch.randn(1, channels, device="cuda", dtype=torch.bfloat16) for _ in ranks)
    weights = tuple(torch.randn(rank, channels, device="cuda", dtype=torch.bfloat16) for rank in ranks)
    kinds, biases = (1, 0, 2), (None,) * len(ranks)
    fixed = implementation._group_call(xs, weights, kinds, biases, 0, None)[0][5]
    candidate = (16, 32, 2, 256)
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


@cuda
@pytest.mark.cuda
@pytest.mark.cuda_graph
@pytest.mark.parametrize(
    "reduction", (17, pytest.param(32, marks=pytest.mark.racecheck), 64, 65, 89, 128, 1000, 1024),
)
def test_grouped_weight_vjp_preserves_accumulators_and_graph(reduction):
    from flashrwkv2._internal import tuning
    from flashrwkv2._internal.linear import _forward
    from flashrwkv2.tmix.wkv_prepare import tilelang as implementation

    if not tuning._fixed_strategy_device(tuning.device_class(torch.device("cuda"))):
        pytest.skip("dynamic grouped VJP is qualified on tested devices")
    torch.manual_seed(723)
    ranks, channels = (96, 96, 256, 64), 2048
    kinds, biases = (0,) * 4, (None,) * 4
    # The 32-element reduction reaches the two-stage pipeline epilogue directly.
    config = (64, 64, 2, 32) if reduction == 32 else (64, 64, 1, 64)
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
