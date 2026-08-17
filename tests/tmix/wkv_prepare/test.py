# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect

import pytest
import torch

from flashrwkv2.tmix.wkv_prepare import infer_tmix_wkv_prepare_forward_varlen


def test_wkv_prepare_public_contract() -> None:
    parameters = inspect.signature(infer_tmix_wkv_prepare_forward_varlen).parameters
    assert "v_first" in parameters
    assert "receptance_lora_a" in parameters
    assert "key_lora_a" in parameters
    assert "value_lora_a" in parameters
    assert parameters["head_size"].default == 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.sm120
@pytest.mark.parametrize("later_layer", [False, True])
@pytest.mark.parametrize("head_size", [64, 128, 256])
def test_wkv_prepare_first_and_later_layer_reference(
    later_layer: bool, head_size: int
) -> None:
    torch.manual_seed(20260818 + head_size + int(later_layer))
    rows, channels, rank = 2, 4096, 32
    device = torch.device("cuda")
    shifted = [
        torch.randn(rows, channels, device=device, dtype=torch.float16) * 0.03
        for _ in range(6)
    ]
    dense = [
        torch.randn(channels, channels, device=device, dtype=torch.float16) * 0.01
        for _ in range(3)
    ]
    rank_in = [
        torch.randn(rank, channels, device=device, dtype=torch.float16) * 0.02
        for _ in range(4)
    ]
    rank_out = [
        torch.randn(channels, rank, device=device, dtype=torch.float16) * 0.02
        for _ in range(4)
    ]
    runtime_in = [weight.t().contiguous() for weight in rank_in]
    runtime_out = [weight.t().contiguous() for weight in rank_out]
    vectors = [
        torch.randn(channels, device=device, dtype=torch.float16) * 0.1
        for _ in range(4)
    ]
    v_first = (
        torch.randn(rows, channels, device=device, dtype=torch.float16) * 0.03
        if later_layer
        else None
    )
    outputs = infer_tmix_wkv_prepare_forward_varlen(
        *shifted,
        *dense,
        *rank_in,
        *rank_out,
        *vectors,
        v_first=v_first,
        w1_runtime=runtime_in[0],
        a1_runtime=runtime_in[1],
        g1_runtime=runtime_in[2],
        v1_runtime=runtime_in[3],
        w2_runtime=runtime_out[0],
        a2_runtime=runtime_out[1],
        g2_runtime=runtime_out[2],
        v2_runtime=runtime_out[3],
        head_size=head_size,
        batch_size=1,
        max_seqlen=rows,
    )
    assert len(outputs) == 8
    x_r, x_w, x_k, x_v, x_a, x_g = (value.float() for value in shifted)
    receptance = x_r @ dense[0].float().t()
    key = x_k @ dense[1].float().t()
    value = x_v @ dense[2].float().t()
    lowrank = [
        source @ weight.float()
        for source, weight in zip(
            (x_w, x_a, x_g, x_v), runtime_in, strict=True
        )
    ]
    decay_logits = torch.tanh(lowrank[0]) @ runtime_out[0].float()
    a_delta = lowrank[1] @ runtime_out[1].float()
    gate = torch.sigmoid(lowrank[2]) @ runtime_out[2].float()
    if later_layer:
        value_gate = torch.sigmoid(
            lowrank[3] @ runtime_out[3].float() + vectors[0].float()
        )
        assert v_first is not None
        value = value + (v_first.float() - value) * value_gate
        returned_v_first = v_first.float()
    else:
        returned_v_first = value
    heads = channels // head_size
    scaled_key = (
        key.view(rows, heads, head_size)
        * vectors[1].float().view(1, heads, head_size)
    )
    normalized_key = scaled_key * torch.rsqrt(
        scaled_key.square().sum(dim=-1, keepdim=True).clamp_min(1.0e-24)
    )
    recurrent_gate = torch.sigmoid(
        vectors[2].float().view(1, heads, head_size)
        + a_delta.view(rows, heads, head_size)
    )
    prepared_key = key.view(rows, heads, head_size) * (
        recurrent_gate * vectors[3].float().view(1, heads, head_size)
        + 1.0
        - vectors[3].float().view(1, heads, head_size)
    )
    expected = (
        receptance,
        decay_logits,
        prepared_key.reshape(rows, channels),
        value,
        -normalized_key.reshape(rows, channels),
        (normalized_key * recurrent_gate).reshape(rows, channels),
        gate,
        returned_v_first,
    )
    for output, reference in zip(outputs, expected, strict=True):
        torch.testing.assert_close(output.float(), reference, atol=0.10, rtol=0.06)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.sm120
def test_wkv_prepare_projection_lora_matches_reference_and_fails_closed() -> None:
    torch.manual_seed(20260819)
    rows, channels, rank, lora_rank = 1, 4096, 32, 8
    device = torch.device("cuda")
    shifted = [
        torch.randn(rows, channels, device=device, dtype=torch.float16) * 0.02
        for _ in range(6)
    ]
    dense = [
        torch.randn(channels, channels, device=device, dtype=torch.float16) * 0.01
        for _ in range(3)
    ]
    rank_in = [
        torch.randn(rank, channels, device=device, dtype=torch.float16) * 0.02
        for _ in range(4)
    ]
    rank_out = [
        torch.randn(channels, rank, device=device, dtype=torch.float16) * 0.02
        for _ in range(4)
    ]
    runtime_in = [weight.t().contiguous() for weight in rank_in]
    runtime_out = [weight.t().contiguous() for weight in rank_out]
    vectors = [torch.zeros(channels, device=device, dtype=torch.float16) for _ in range(4)]
    lora_a = torch.randn(
        lora_rank, channels, device=device, dtype=torch.float16
    ) * 0.02
    lora_b = torch.randn(
        channels, lora_rank, device=device, dtype=torch.float16
    ) * 0.02
    scale = 0.5
    kwargs = dict(
        w1_runtime=runtime_in[0],
        a1_runtime=runtime_in[1],
        g1_runtime=runtime_in[2],
        v1_runtime=runtime_in[3],
        w2_runtime=runtime_out[0],
        a2_runtime=runtime_out[1],
        g2_runtime=runtime_out[2],
        v2_runtime=runtime_out[3],
        receptance_lora_a=lora_a,
        receptance_lora_b=lora_b,
        receptance_lora_scale=scale,
    )
    outputs = infer_tmix_wkv_prepare_forward_varlen(
        *shifted, *dense, *rank_in, *rank_out, *vectors, **kwargs
    )
    expected_receptance = (
        shifted[0].float() @ dense[0].float().t()
        + (shifted[0].float() @ lora_a.float().t()) @ lora_b.float().t() * scale
    )
    torch.testing.assert_close(
        outputs[0].float(), expected_receptance, atol=0.08, rtol=0.05
    )
    with pytest.raises(ValueError, match="provided together"):
        infer_tmix_wkv_prepare_forward_varlen(
            *shifted,
            *dense,
            *rank_in,
            *rank_out,
            *vectors,
            **{**kwargs, "receptance_lora_b": None},
        )
