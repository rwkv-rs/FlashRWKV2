# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

import flashrwkv2
from flashrwkv2.tmix.linear import (
    infer_tmix_linear_act_sigmoid_forward_varlen,
    infer_tmix_linear_act_tanh_forward_varlen,
    infer_tmix_linear_attention_c2c_forward_varlen,
    infer_tmix_linear_ffn_key_forward_varlen,
    infer_tmix_linear_forward_varlen,
    infer_tmix_linear_rank_in_forward_varlen,
    infer_tmix_linear_rank_out_forward_varlen,
    infer_tmix_linear_rank_out_sigmoid_forward_varlen,
    infer_tmix_linear_rank_out_tanh_forward_varlen,
    infer_tmix_linear_t_forward_varlen,
    infer_tmix_linear_t_sigmoid_forward_varlen,
    infer_tmix_linear_t_tanh_forward_varlen,
    infer_tmix_linear_t_vres_forward_varlen,
    infer_tmix_lowrank_in_forward_varlen,
    infer_tmix_lowrank_out_forward_varlen,
    infer_tmix_lowrank_vres_forward_varlen,
    infer_tmix_lowrank_wagv_in_forward_varlen,
)

G1H_TUNED_ROWS = (1, 4, 5, 7, 8, 9, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024)


def _weight_layouts(
    runtime: torch.Tensor, layout: str
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    original = runtime.t().contiguous()
    if layout == "original":
        return original, None
    if layout == "runtime":
        return None, runtime
    return original, runtime


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "operator",
    (
        infer_tmix_linear_t_forward_varlen,
        infer_tmix_linear_t_tanh_forward_varlen,
        infer_tmix_linear_t_sigmoid_forward_varlen,
    ),
)
def test_custom_linear_t_rejects_m_beyond_cuda_grid_y(operator) -> None:
    device = torch.device("cuda")
    x = torch.empty(65536, 2, device=device, dtype=torch.float16)
    weight_t = torch.empty(2, 2, device=device, dtype=torch.float16)
    with pytest.raises(ValueError, match=r"linear_t.*65535.*grid\.y.*65536"):
        operator(x, weight_t)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tmix_linear_and_lowrank_families() -> None:
    torch.manual_seed(51)
    device = torch.device("cuda")
    # The canonical Albatross rank kernels intentionally fail closed outside
    # their tuned shape domain: K >= 1024, output C >= 1024, and M <= 4/8.
    rows, channels, rank, output_channels = 4, 1024, 8, 1024
    x = torch.randn(rows, channels, device=device, dtype=torch.float16)
    weight = torch.randn(output_channels, channels, device=device, dtype=torch.float16)
    assert torch.allclose(
        infer_tmix_linear_forward_varlen(x, weight).float(),
        x.float() @ weight.float().t(),
        atol=0.04,
        rtol=0.04,
    )
    transposed = weight.t().contiguous()
    assert torch.allclose(
        infer_tmix_linear_forward_varlen(
            x, transposed, weight_is_transposed=True
        ).float(),
        x.float() @ transposed.float(),
        atol=0.04,
        rtol=0.04,
    )
    attention_linear = infer_tmix_linear_attention_c2c_forward_varlen(x, weight)
    ffn_key_linear = infer_tmix_linear_ffn_key_forward_varlen(x, weight)
    expected_linear = x.float() @ weight.float().t()
    assert torch.allclose(
        attention_linear.float(), expected_linear, atol=0.04, rtol=0.04
    )
    assert torch.allclose(ffn_key_linear.float(), expected_linear, atol=0.04, rtol=0.04)

    rank_rows, rank_k, rank_n = 4, 8, 1024
    rank_x = torch.randn(rank_rows, rank_k, device=device, dtype=torch.float16)
    rank_weight_t = torch.randn(rank_n, rank_k, device=device, dtype=torch.float16)
    rank_weight_t = rank_weight_t.contiguous()
    rank_linear = infer_tmix_linear_t_forward_varlen(rank_x, rank_weight_t)
    assert torch.allclose(
        rank_linear.float(),
        rank_x.float() @ rank_weight_t.float().t(),
        atol=0.04,
        rtol=0.04,
    )
    rank_tanh = infer_tmix_linear_t_tanh_forward_varlen(rank_x, rank_weight_t)
    assert torch.allclose(
        rank_tanh.float(),
        torch.tanh(rank_x.float()) @ rank_weight_t.float().t(),
        atol=0.04,
        rtol=0.04,
    )
    rank_sigmoid = infer_tmix_linear_t_sigmoid_forward_varlen(rank_x, rank_weight_t)
    assert torch.allclose(
        rank_sigmoid.float(),
        torch.sigmoid(rank_x.float()) @ rank_weight_t.float().t(),
        atol=0.04,
        rtol=0.04,
    )
    rank_tanh_input = infer_tmix_linear_act_tanh_forward_varlen(rank_x)
    rank_sigmoid_input = infer_tmix_linear_act_sigmoid_forward_varlen(rank_x)
    assert torch.allclose(
        rank_tanh_input.float(), torch.tanh(rank_x.float()), atol=0.002, rtol=0.002
    )
    assert torch.allclose(
        rank_sigmoid_input.float(),
        torch.sigmoid(rank_x.float()),
        atol=0.002,
        rtol=0.002,
    )
    rank_v = torch.randn(rank_rows, rank_n, device=device, dtype=torch.float16)
    rank_v_first = torch.randn_like(rank_v)
    rank_v0 = torch.randn(rank_n, device=device, dtype=torch.float16)
    rank_vres = infer_tmix_linear_t_vres_forward_varlen(
        rank_x, rank_weight_t, rank_v, rank_v_first, rank_v0
    )
    gate = torch.sigmoid(rank_x.float() @ rank_weight_t.float().t() + rank_v0.float())
    assert torch.allclose(
        rank_vres.float(),
        rank_v.float() + (rank_v_first.float() - rank_v.float()) * gate,
        atol=0.05,
        rtol=0.05,
    )

    x_w, x_a, x_g = [
        torch.randn(rows, channels, device=device, dtype=torch.float16)
        for _ in range(3)
    ]
    x_v = torch.randn(rows, channels, device=device, dtype=torch.float16)
    w1, a1, g1 = [
        torch.randn(rank, channels, device=device, dtype=torch.float16)
        for _ in range(3)
    ]
    v1_t = torch.randn(rank, channels, device=device, dtype=torch.float16)
    lowrank_in = infer_tmix_lowrank_in_forward_varlen(x_w, x_a, x_g, w1, a1, g1)
    for output, source, projection in zip(
        lowrank_in, (x_w, x_a, x_g), (w1, a1, g1), strict=True
    ):
        assert torch.allclose(
            output.float(),
            source.float() @ projection.float().t(),
            atol=0.04,
            rtol=0.04,
        )

    lowrank_wagv_in = infer_tmix_lowrank_wagv_in_forward_varlen(
        x_w, x_a, x_g, x_v, w1, a1, g1, v1_t
    )
    for output, source, projection in zip(
        lowrank_wagv_in,
        (x_w, x_a, x_g, x_v),
        (w1, a1, g1, v1_t),
        strict=True,
    ):
        assert torch.allclose(
            output.float(),
            source.float() @ projection.float().t(),
            atol=0.04,
            rtol=0.04,
        )

    w2, a2, g2 = [
        torch.randn(output_channels, rank, device=device, dtype=torch.float16)
        for _ in range(3)
    ]
    lowrank_out = infer_tmix_lowrank_out_forward_varlen(*lowrank_in, w2, a2, g2)
    assert torch.allclose(
        lowrank_out[0].float(),
        torch.tanh(lowrank_in[0].float()) @ w2.float().t(),
        atol=0.04,
        rtol=0.04,
    )
    assert torch.allclose(
        lowrank_out[1].float(),
        lowrank_in[1].float() @ a2.float().t(),
        atol=0.04,
        rtol=0.04,
    )
    assert torch.allclose(
        lowrank_out[2].float(),
        torch.sigmoid(lowrank_in[2].float()) @ g2.float().t(),
        atol=0.04,
        rtol=0.04,
    )

    v1 = torch.randn(rows, rank, device=device, dtype=torch.float16)
    v2 = torch.randn(output_channels, rank, device=device, dtype=torch.float16)
    v = torch.randn(rows, output_channels, device=device, dtype=torch.float16)
    v_first = torch.randn_like(v)
    v0 = torch.randn(output_channels, device=device, dtype=torch.float16)
    lowrank_vres = infer_tmix_lowrank_vres_forward_varlen(
        *lowrank_in, v1, w2, a2, g2, v2, v, v_first, v0
    )
    expected_v = v.float() + (v_first.float() - v.float()) * torch.sigmoid(
        v1.float() @ v2.float().t() + v0.float()
    )
    assert torch.allclose(lowrank_vres[3].float(), expected_v, atol=0.05, rtol=0.05)

    # These rows are outside the upstream fused linear_t windows.  The
    # caller must therefore select the exact Albatross large-rank linear
    # body, including the standalone activation helpers for rank-out gates.
    large_in = torch.randn(9, channels, device=device, dtype=torch.float16)
    large_in_weight_t = torch.randn(
        rank_n, channels, device=device, dtype=torch.float16
    )
    large_in_output = infer_tmix_linear_rank_in_forward_varlen(
        large_in, weight_t=large_in_weight_t
    )
    assert torch.allclose(
        large_in_output.float(),
        large_in.float() @ large_in_weight_t.float().t(),
        atol=0.04,
        rtol=0.04,
    )

    small_in = torch.randn(7, channels, device=device, dtype=torch.float16)
    small_in_output = infer_tmix_linear_rank_in_forward_varlen(
        small_in, weight_t=large_in_weight_t
    )
    assert torch.allclose(
        small_in_output.float(),
        small_in.float() @ large_in_weight_t.float().t(),
        atol=0.04,
        rtol=0.04,
    )

    large_out = torch.randn(5, rank_k, device=device, dtype=torch.float16)
    large_out_weight_t = torch.randn(
        output_channels, rank_k, device=device, dtype=torch.float16
    )
    large_out_output = infer_tmix_linear_rank_out_forward_varlen(
        large_out, weight_t=large_out_weight_t
    )
    assert torch.allclose(
        large_out_output.float(),
        large_out.float() @ large_out_weight_t.float().t(),
        atol=0.04,
        rtol=0.04,
    )
    large_out_tanh = infer_tmix_linear_rank_out_tanh_forward_varlen(
        large_out, weight_t=large_out_weight_t
    )
    assert torch.allclose(
        large_out_tanh.float(),
        torch.tanh(large_out.float()) @ large_out_weight_t.float().t(),
        atol=0.04,
        rtol=0.04,
    )
    large_out_sigmoid = infer_tmix_linear_rank_out_sigmoid_forward_varlen(
        large_out, weight_t=large_out_weight_t
    )
    assert torch.allclose(
        large_out_sigmoid.float(),
        torch.sigmoid(large_out.float()) @ large_out_weight_t.float().t(),
        atol=0.04,
        rtol=0.04,
    )

    small_out = torch.randn(4, rank_k, device=device, dtype=torch.float16)
    small_out_output = infer_tmix_linear_rank_out_forward_varlen(
        small_out, weight_t=large_out_weight_t
    )
    assert torch.allclose(
        small_out_output.float(),
        small_out.float() @ large_out_weight_t.float().t(),
        atol=0.04,
        rtol=0.04,
    )
    small_out_tanh = infer_tmix_linear_rank_out_tanh_forward_varlen(
        small_out, weight_t=large_out_weight_t
    )
    assert torch.allclose(
        small_out_tanh.float(),
        torch.tanh(small_out.float()) @ large_out_weight_t.float().t(),
        atol=0.04,
        rtol=0.04,
    )
    small_out_sigmoid = infer_tmix_linear_rank_out_sigmoid_forward_varlen(
        small_out, weight_t=large_out_weight_t
    )
    assert torch.allclose(
        small_out_sigmoid.float(),
        torch.sigmoid(small_out.float()) @ large_out_weight_t.float().t(),
        atol=0.04,
        rtol=0.04,
    )

    # Both layouts are supplied for the canonical C=4096 tuned table.  The
    # input table selects the original-layout Lt entry for (rank=128, rows=8),
    # while the output table selects the runtime-layout Lt entry for the same
    # rank/row pair.  This checks the automatic caller dispatch rather than a
    # forced algorithm binding.
    table_in = torch.randn(8, 4096, device=device, dtype=torch.float16)
    table_in_weight = torch.randn(4096, 128, device=device, dtype=torch.float16)
    table_in_weight_t = table_in_weight.t().contiguous()
    table_in_output = infer_tmix_linear_rank_in_forward_varlen(
        table_in, weight=table_in_weight, weight_t=table_in_weight_t
    )
    assert torch.allclose(
        table_in_output.float(),
        table_in.float() @ table_in_weight.float(),
        atol=0.08,
        rtol=0.08,
    )

    table_out = torch.randn(8, 128, device=device, dtype=torch.float16)
    table_out_weight = torch.randn(128, 4096, device=device, dtype=torch.float16)
    table_out_weight_t = table_out_weight.t().contiguous()
    table_out_output = infer_tmix_linear_rank_out_forward_varlen(
        table_out, weight=table_out_weight, weight_t=table_out_weight_t
    )
    assert torch.allclose(
        table_out_output.float(),
        table_out.float() @ table_out_weight.float(),
        atol=0.08,
        rtol=0.08,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("rows", "rank", "channels", "scale"),
    (
        (1, 8, 1024, 0.25),
        (4, 16, 1024, -0.5),
        (8, 64, 1024, 0.0),
        (24, 512, 1024, 0.375),
        (1, 8, 4096, -0.25),
        (4, 16, 4096, 0.5),
        (8, 64, 4096, -0.375),
        (24, 512, 4096, 0.0),
    ),
)
def test_attention_c2c_optional_lora_matches_reference(
    rows: int, rank: int, channels: int, scale: float
) -> None:
    torch.manual_seed(7000 + rows + rank + channels)
    device = torch.device("cuda")
    x = torch.randn(rows, channels, device=device, dtype=torch.float16) * 0.25
    weight = torch.randn(channels, channels, device=device, dtype=torch.float16) * 0.02
    lora_a = torch.randn(rank, channels, device=device, dtype=torch.float16) * 0.02
    lora_b = torch.randn(channels, rank, device=device, dtype=torch.float16) * 0.02
    inputs = (x, weight, lora_a, lora_b)
    snapshots = tuple(tensor.clone() for tensor in inputs)

    actual = infer_tmix_linear_attention_c2c_forward_varlen(
        x, weight, lora_a=lora_a, lora_b=lora_b, lora_scale=scale
    )
    expected = x.float() @ weight.float().t() + scale * (
        (x.float() @ lora_a.float().t()) @ lora_b.float().t()
    )

    assert actual.shape == (rows, channels)
    assert actual.dtype == torch.float16 and actual.is_contiguous()
    torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)
    for tensor, snapshot in zip(inputs, snapshots, strict=True):
        assert torch.equal(tensor, snapshot)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_attention_c2c_optional_lora_preserves_base_path() -> None:
    torch.manual_seed(7100)
    device = torch.device("cuda")
    x = torch.randn(2, 1024, device=device, dtype=torch.float16)
    weight = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    lora_a = torch.randn(8, 1024, device=device, dtype=torch.float16)
    lora_b = torch.randn(1024, 8, device=device, dtype=torch.float16)

    base = infer_tmix_linear_attention_c2c_forward_varlen(x, weight)
    explicit_none = infer_tmix_linear_attention_c2c_forward_varlen(
        x, weight, lora_a=None, lora_b=None, lora_scale=1.0
    )
    zero_scale = infer_tmix_linear_attention_c2c_forward_varlen(
        x, weight, lora_a=lora_a, lora_b=lora_b, lora_scale=0.0
    )
    assert torch.equal(base, explicit_none)
    assert torch.equal(base, zero_scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_attention_c2c_optional_lora_invalid_contracts_fail_closed() -> None:
    device = torch.device("cuda")
    x = torch.randn(2, 1024, device=device, dtype=torch.float16)
    weight = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    lora_a = torch.randn(8, 1024, device=device, dtype=torch.float16)
    lora_b = torch.randn(1024, 8, device=device, dtype=torch.float16)

    with pytest.raises(ValueError, match="provided together"):
        infer_tmix_linear_attention_c2c_forward_varlen(
            x, weight, lora_a=lora_a
        )
    with pytest.raises(ValueError, match="R<=512"):
        infer_tmix_linear_attention_c2c_forward_varlen(
            x,
            weight,
            lora_a=torch.empty(513, 1024, device=device, dtype=torch.float16),
            lora_b=torch.empty(1024, 513, device=device, dtype=torch.float16),
        )
    with pytest.raises(ValueError, match=r"lora_b.*\[N,R\]"):
        infer_tmix_linear_attention_c2c_forward_varlen(
            x,
            weight,
            lora_a=lora_a,
            lora_b=lora_b[:, :-1].contiguous(),
        )
    with pytest.raises(ValueError, match="finite"):
        infer_tmix_linear_attention_c2c_forward_varlen(
            x, weight, lora_a=lora_a, lora_b=lora_b, lora_scale=float("nan")
        )
    with pytest.raises(TypeError, match="finite real"):
        infer_tmix_linear_attention_c2c_forward_varlen(
            x, weight, lora_a=lora_a, lora_b=lora_b, lora_scale=True
        )
    with pytest.raises(TypeError, match="torch.float16"):
        infer_tmix_linear_attention_c2c_forward_varlen(
            x, weight, lora_a=lora_a.float(), lora_b=lora_b
        )
    with pytest.raises(TypeError, match="torch.Tensor"):
        infer_tmix_linear_attention_c2c_forward_varlen(
            x, weight, lora_a="invalid", lora_b=lora_b  # type: ignore[arg-type]
        )

    assert flashrwkv2._C is not None
    with pytest.raises(RuntimeError, match="lora_a.*float16"):
        flashrwkv2._C.tmix_linear_attention_c2c_forward_varlen(
            x, weight, lora_a.float(), lora_b, 1.0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_attention_c2c_optional_lora_cuda_graph_replay() -> None:
    torch.manual_seed(7200)
    device = torch.device("cuda")
    rows, channels, rank = 4, 1024, 16
    static_x = torch.randn(rows, channels, device=device, dtype=torch.float16) * 0.25
    weight = torch.randn(channels, channels, device=device, dtype=torch.float16) * 0.02
    lora_a = torch.randn(rank, channels, device=device, dtype=torch.float16) * 0.02
    lora_b = torch.randn(channels, rank, device=device, dtype=torch.float16) * 0.02
    scale = -0.375

    infer_tmix_linear_attention_c2c_forward_varlen(
        static_x, weight, lora_a=lora_a, lora_b=lora_b, lora_scale=scale
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = infer_tmix_linear_attention_c2c_forward_varlen(
            static_x,
            weight,
            lora_a=lora_a,
            lora_b=lora_b,
            lora_scale=scale,
        )

    replay_x = torch.randn_like(static_x) * 0.25
    static_x.copy_(replay_x)
    graph.replay()
    torch.cuda.synchronize()
    expected = replay_x.float() @ weight.float().t() + scale * (
        (replay_x.float() @ lora_a.float().t()) @ lora_b.float().t()
    )
    torch.testing.assert_close(graph_output.float(), expected, atol=0.02, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rows", G1H_TUNED_ROWS)
@pytest.mark.parametrize("layout", ("original", "runtime", "both"))
@pytest.mark.parametrize("rank", (96, 128, 480))
def test_composite_lowrank_dispatch_covers_g1h_rows(
    rows: int, layout: str, rank: int
) -> None:
    torch.manual_seed(5000 + rows)
    device = torch.device("cuda")
    channels = 4096
    ranks = (rank,) * 4
    sources = [
        torch.randn(rows, channels, device=device, dtype=torch.float16) for _ in ranks
    ]
    rank_in_runtime = [
        torch.randn(channels, rank, device=device, dtype=torch.float16)
        for rank in ranks
    ]
    rank_in_layouts = [_weight_layouts(weight, layout) for weight in rank_in_runtime]

    wag = infer_tmix_lowrank_in_forward_varlen(
        *sources[:3],
        *(weights[0] for weights in rank_in_layouts[:3]),
        w1_runtime=rank_in_layouts[0][1],
        a1_runtime=rank_in_layouts[1][1],
        g1_runtime=rank_in_layouts[2][1],
    )
    wagv = infer_tmix_lowrank_wagv_in_forward_varlen(
        *sources,
        *(weights[0] for weights in rank_in_layouts),
        w1_runtime=rank_in_layouts[0][1],
        a1_runtime=rank_in_layouts[1][1],
        g1_runtime=rank_in_layouts[2][1],
        v1_runtime=rank_in_layouts[3][1],
    )
    for output, source, weight in zip(wagv, sources, rank_in_runtime, strict=True):
        assert (
            output.is_contiguous()
            and output.dtype == torch.float16
            and output.device == sources[0].device
        )
        assert torch.allclose(
            output.float(), source.float() @ weight.float(), atol=0.08, rtol=0.08
        )
    for output, wagv_output in zip(wag, wagv[:3], strict=True):
        assert torch.allclose(output.float(), wagv_output.float(), atol=0.04, rtol=0.04)

    rank_out_runtime = [
        torch.randn(rank, channels, device=device, dtype=torch.float16)
        for rank in ranks
    ]
    rank_out_layouts = [_weight_layouts(weight, layout) for weight in rank_out_runtime]
    output = infer_tmix_lowrank_out_forward_varlen(
        *wag,
        *(weights[0] for weights in rank_out_layouts[:3]),
        w2_runtime=rank_out_layouts[0][1],
        a2_runtime=rank_out_layouts[1][1],
        g2_runtime=rank_out_layouts[2][1],
    )
    expected = (
        torch.tanh(wag[0].float()) @ rank_out_runtime[0].float(),
        wag[1].float() @ rank_out_runtime[1].float(),
        torch.sigmoid(wag[2].float()) @ rank_out_runtime[2].float(),
    )
    for actual, reference in zip(output, expected, strict=True):
        assert actual.is_contiguous() and actual.shape == (rows, channels)
        assert torch.allclose(actual.float(), reference, atol=0.08, rtol=0.08)

    value = torch.randn(rows, channels, device=device, dtype=torch.float16)
    value_first = torch.randn_like(value)
    value_bias = torch.randn(channels, device=device, dtype=torch.float16)
    output_vres = infer_tmix_lowrank_vres_forward_varlen(
        *wagv,
        *(weights[0] for weights in rank_out_layouts),
        value,
        value_first,
        value_bias,
        w2_runtime=rank_out_layouts[0][1],
        a2_runtime=rank_out_layouts[1][1],
        g2_runtime=rank_out_layouts[2][1],
        v2_runtime=rank_out_layouts[3][1],
    )
    value_delta = wagv[3].float() @ rank_out_runtime[3].float()
    expected_value = value.float() + (
        value_first.float() - value.float()
    ) * torch.sigmoid(value_bias.float() + value_delta)
    for actual, reference in zip(output_vres[:3], expected, strict=True):
        assert torch.allclose(actual.float(), reference, atol=0.08, rtol=0.08)
    assert torch.allclose(output_vres[3].float(), expected_value, atol=0.08, rtol=0.08)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_composite_lowrank_fused_rank_limit_fails_closed() -> None:
    device = torch.device("cuda")
    rows, channels, rank = 1, 1024, 513
    sources = [
        torch.randn(rows, channels, device=device, dtype=torch.float16)
        for _ in range(3)
    ]
    original = [
        torch.randn(rank, channels, device=device, dtype=torch.float16)
        for _ in range(3)
    ]
    with pytest.raises(ValueError, match="R<=512"):
        infer_tmix_lowrank_in_forward_varlen(*sources, *original)

    runtime = [weight.t().contiguous() for weight in original]
    with pytest.raises(ValueError, match="R<=512"):
        infer_tmix_lowrank_in_forward_varlen(
            *sources,
            None,
            None,
            None,
            w1_runtime=runtime[0],
            a1_runtime=runtime[1],
            g1_runtime=runtime[2],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_composite_lowrank_inputs_are_immutable_and_invalid_contracts_fail() -> None:
    device = torch.device("cuda")
    rows, channels, rank = 8, 1024, 96
    sources = [
        torch.randn(rows, channels, device=device, dtype=torch.float16)
        for _ in range(4)
    ]
    runtime_in = [
        torch.randn(channels, rank, device=device, dtype=torch.float16)
        for _ in range(4)
    ]
    original_in = [weight.t().contiguous() for weight in runtime_in]
    snapshots = [tensor.clone() for tensor in sources + runtime_in + original_in]

    projected = infer_tmix_lowrank_wagv_in_forward_varlen(
        *sources,
        *original_in,
        w1_runtime=runtime_in[0],
        a1_runtime=runtime_in[1],
        g1_runtime=runtime_in[2],
        v1_runtime=runtime_in[3],
    )
    for tensor, snapshot in zip(
        sources + runtime_in + original_in, snapshots, strict=True
    ):
        assert torch.equal(tensor, snapshot)

    with pytest.raises(TypeError, match="float16"):
        infer_tmix_lowrank_in_forward_varlen(
            sources[0].float(), sources[1], sources[2], *original_in[:3]
        )
    with pytest.raises(ValueError, match="CUDA"):
        infer_tmix_lowrank_in_forward_varlen(
            sources[0].cpu(), sources[1], sources[2], *original_in[:3]
        )
    with pytest.raises(ValueError, match="one of weight"):
        infer_tmix_lowrank_in_forward_varlen(
            *sources[:3], None, original_in[1], original_in[2]
        )
    mismatched_runtime = torch.randn(
        channels, rank + 1, device=device, dtype=torch.float16
    )
    with pytest.raises(ValueError, match="same rank projection"):
        infer_tmix_lowrank_in_forward_varlen(
            *sources[:3],
            *original_in[:3],
            w1_runtime=mismatched_runtime,
        )
    noncontiguous_runtime = runtime_in[0].t()
    with pytest.raises(ValueError, match="contiguous"):
        infer_tmix_lowrank_in_forward_varlen(
            *sources[:3],
            None,
            original_in[1],
            original_in[2],
            w1_runtime=noncontiguous_runtime,
        )

    runtime_out = [
        torch.randn(rank, channels, device=device, dtype=torch.float16)
        for _ in range(4)
    ]
    original_out = [weight.t().contiguous() for weight in runtime_out]
    value = torch.randn(rows, channels, device=device, dtype=torch.float16)
    value_first = torch.randn_like(value)
    value_bias = torch.randn(channels, device=device, dtype=torch.float16)
    immutable = (
        projected
        + tuple(runtime_out)
        + tuple(original_out)
        + (
            value,
            value_first,
            value_bias,
        )
    )
    immutable_snapshots = [tensor.clone() for tensor in immutable]
    infer_tmix_lowrank_vres_forward_varlen(
        *projected,
        *original_out,
        value,
        value_first,
        value_bias,
        w2_runtime=runtime_out[0],
        a2_runtime=runtime_out[1],
        g2_runtime=runtime_out[2],
        v2_runtime=runtime_out[3],
    )
    for tensor, snapshot in zip(immutable, immutable_snapshots, strict=True):
        assert torch.equal(tensor, snapshot)
