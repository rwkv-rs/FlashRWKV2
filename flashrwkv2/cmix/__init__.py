# SPDX-License-Identifier: MIT
"""ChannelMix with one BF16 forward for training and packed inference."""

import torch


def _check_projection(
    x, key_weight, value_weight, key_a, key_b, value_a, value_b, key_scale, value_scale
):
    from .._internal.linear import check_adapter_scale

    if x.ndim != 2 or x.numel() == 0:
        raise ValueError("CMix projection input must have nonempty shape [rows,C]")
    channels = x.shape[1]
    if (
        key_weight.ndim != 2
        or key_weight.shape[1] != channels
        or key_weight.shape[0] == 0
        or value_weight.shape != (channels, key_weight.shape[0])
    ):
        raise ValueError(
            "CMix projection requires key_weight [F,C] and value_weight [C,F]"
        )
    if any(
        t.dtype != torch.bfloat16 or not t.is_cuda or t.device != x.device
        for t in (x, key_weight, value_weight)
    ):
        raise ValueError(
            "CMix projection inputs and weights must share a CUDA bfloat16 device"
        )
    for name, weight, a, b, scale in (
        ("key", key_weight, key_a, key_b, key_scale),
        ("value", value_weight, value_a, value_b, value_scale),
    ):
        if (a is None) != (b is None):
            raise ValueError(f"CMix {name} adapter A and B must be provided together")
        if a is not None and (
            a.ndim != 2
            or a.shape[0] == 0
            or a.shape[1] != weight.shape[1]
            or b.shape != (weight.shape[0], a.shape[0])
            or any(
                t.dtype not in (torch.bfloat16, torch.float32) or t.device != x.device
                for t in (a, b)
            )
        ):
            raise ValueError(
                f"CMix {name} adapter requires CUDA BF16/FP32 A [rank,in] and B [out,rank]"
            )
        if a is not None:
            check_adapter_scale(scale)


def _project(
    mixed,
    key_weight,
    value_weight,
    key_lora_a=None,
    key_lora_b=None,
    value_lora_a=None,
    value_lora_b=None,
    key_lora_scale=1.0,
    value_lora_scale=1.0,
    key_use_nora=False,
    value_use_nora=False,
):
    from .._internal.elementwise import activation
    from .._internal.linear import adapter_linear

    _check_projection(
        mixed,
        key_weight,
        value_weight,
        key_lora_a,
        key_lora_b,
        value_lora_a,
        value_lora_b,
        key_lora_scale,
        value_lora_scale,
    )
    features = adapter_linear(
        mixed,
        key_weight,
        key_lora_a,
        key_lora_b,
        key_lora_scale,
        key_use_nora,
    )
    return adapter_linear(
        activation(features, "relu2"),
        value_weight,
        value_lora_a,
        value_lora_b,
        value_lora_scale,
        value_use_nora,
    )


def _cmix(
    x,
    initial_shift,
    cu_seqlens,
    x_k,
    key_weight,
    value_weight,
    key_lora_a=None,
    key_lora_b=None,
    value_lora_a=None,
    value_lora_b=None,
    key_lora_scale=1.0,
    value_lora_scale=1.0,
    key_use_nora=False,
    value_use_nora=False,
    *,
    status=None,
):
    from ..tmix.tokenshift.tilelang import tokenshift

    mixed, final_shift = tokenshift(
        x,
        (x_k,),
        cu_seqlens,
        initial_shift,
        _status=status,
    )
    output = _project(
        mixed[0],
        key_weight,
        value_weight,
        key_lora_a,
        key_lora_b,
        value_lora_a,
        value_lora_b,
        key_lora_scale,
        value_lora_scale,
        key_use_nora,
        value_use_nora,
    )
    return output, final_shift


def infer_cmix_forward_varlen(
    x,
    res,
    weight,
    bias,
    x_k,
    key_weight,
    value_weight,
    *,
    shift_state_pool,
    cu_seqlens,
    state_indices,
    max_seqlen=None,
    eps=1.0e-5,
    validated_metadata=None,
    deterministic=False,
    key_lora_a=None,
    key_lora_b=None,
    value_lora_a=None,
    value_lora_b=None,
    key_lora_scale=1.0,
    value_lora_scale=1.0,
    key_use_nora=False,
    value_use_nora=False,
):
    """Apply res/LN, TokenShift, ReLU² and projection to packed rows.

    The inference value weight retains its runtime [F,C] layout. A transpose
    view feeds the same projection as the training [C,F] weight.
    """
    from .._internal.state import gather_state, scatter_state
    from ..post_norm.tilelang import norm
    from ..tmix.tokenshift.tilelang import fused_post_norm_tokenshift
    from ..tmix.wkv7 import (
        _validate_launch_max_seqlen,
        prepare_tmix_wkv7_recurrent_metadata,
    )

    _check_projection(
        x,
        key_weight,
        value_weight.t(),
        key_lora_a,
        key_lora_b,
        value_lora_a,
        value_lora_b,
        key_lora_scale,
        value_lora_scale,
    )
    if (
        shift_state_pool.dtype != torch.bfloat16
        or not shift_state_pool.is_cuda
        or not shift_state_pool.is_contiguous()
        or shift_state_pool.ndim != 2
        or shift_state_pool.shape[1] != x.shape[-1]
        or shift_state_pool.device != x.device
    ):
        raise ValueError("shift_state_pool must be contiguous CUDA bfloat16 [slots,C]")
    ticket = validated_metadata
    if ticket is None:
        ticket = prepare_tmix_wkv7_recurrent_metadata(
            cu_seqlens,
            state_indices,
            total_tokens=x.shape[0],
            state_pool_size=shift_state_pool.shape[0],
            max_seqlen=max_seqlen,
        )
    ticket._check_compatible(
        cu_seqlens,
        state_indices,
        x.shape[0],
        shift_state_pool.shape[0],
        _validate_launch_max_seqlen(max_seqlen),
    )
    cu_seqlens = ticket._query_start_loc_snapshot()
    state_indices = ticket._state_indices_snapshot()
    status = ticket._status()
    if ticket._max_seqlen() == 1 and x.shape[0] == state_indices.numel():
        residual, mixed = fused_post_norm_tokenshift(
            x,
            res,
            weight,
            bias,
            (x_k,),
            shift_state_pool,
            state_indices,
            status,
            eps,
        )
        return residual, _project(
            mixed[0],
            key_weight,
            value_weight.t(),
            key_lora_a,
            key_lora_b,
            value_lora_a,
            value_lora_b,
            key_lora_scale,
            value_lora_scale,
            key_use_nora,
            value_use_nora,
        )
    normalized, residual = norm(x, weight, bias, eps, res, return_res=True)
    initial = gather_state(shift_state_pool, state_indices, status)
    output, final = _cmix(
        normalized,
        initial,
        cu_seqlens,
        x_k,
        key_weight,
        value_weight.t(),
        key_lora_a,
        key_lora_b,
        value_lora_a,
        value_lora_b,
        key_lora_scale,
        value_lora_scale,
        key_use_nora,
        value_use_nora,
        status=status,
    )
    scatter_state(shift_state_pool, final, state_indices, status)
    return residual, output


def pretrain_cmix_bf16(
    x,
    x_k,
    key_weight,
    value_weight,
    *,
    key_lora_a=None,
    key_lora_b=None,
    value_lora_a=None,
    value_lora_b=None,
    key_lora_scale=1.0,
    value_lora_scale=1.0,
    key_use_nora=False,
    value_use_nora=False,
):
    return statetune_cmix_bf16(
        x,
        None,
        x_k,
        key_weight,
        value_weight,
        key_lora_a=key_lora_a,
        key_lora_b=key_lora_b,
        value_lora_a=value_lora_a,
        value_lora_b=value_lora_b,
        key_lora_scale=key_lora_scale,
        value_lora_scale=value_lora_scale,
        key_use_nora=key_use_nora,
        value_use_nora=value_use_nora,
    )[0]


def statetune_cmix_bf16(
    x,
    initial_shift,
    x_k,
    key_weight,
    value_weight,
    *,
    key_lora_a=None,
    key_lora_b=None,
    value_lora_a=None,
    value_lora_b=None,
    key_lora_scale=1.0,
    value_lora_scale=1.0,
    key_use_nora=False,
    value_use_nora=False,
):
    """Use differentiable initial/final TokenShift state for chunked training."""
    if x.ndim != 3 or x.numel() == 0:
        raise ValueError("CMix training input must have non-empty shape [B,T,C]")
    batch, tokens, channels = x.shape
    cu = torch.arange(batch + 1, device=x.device, dtype=torch.int32) * tokens
    output, final = _cmix(
        x.reshape(-1, channels),
        initial_shift,
        cu,
        x_k,
        key_weight,
        value_weight,
        key_lora_a,
        key_lora_b,
        value_lora_a,
        value_lora_b,
        key_lora_scale,
        value_lora_scale,
        key_use_nora,
        value_use_nora,
    )
    return output.view_as(x), final


__all__ = ["infer_cmix_forward_varlen", "pretrain_cmix_bf16", "statetune_cmix_bf16"]
