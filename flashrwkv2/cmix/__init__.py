# SPDX-License-Identifier: MIT
"""ChannelMix with one BF16 forward for training and packed inference."""

import torch


def _check_projection(x, key_weight, value_weight, adapters):
    from .._internal.linear import check_adapter_parameters

    if x.ndim != 2 or x.numel() == 0:
        raise ValueError("CMix projection input must have nonempty shape [rows,C]")
    channels = x.shape[1]
    if (
        key_weight.ndim != 2
        or key_weight.shape[1] != channels
        or key_weight.shape[0] == 0
        or value_weight.shape != (channels, key_weight.shape[0])
    ):
        raise ValueError("CMix projection requires key_weight [F,C] and value_weight [C,F]")
    if any(t.dtype != torch.bfloat16 or t.device != x.device for t in (key_weight, value_weight)):
        raise ValueError("CMix weights must be bfloat16 and share x's device")
    return tuple(
        (a, b, check_adapter_parameters(a, b, scale, weight, x.device), use_nora)
        for weight, (a, b, scale, use_nora) in zip((key_weight, value_weight), adapters, strict=True)
    )


def _project(mixed, key_weight, value_weight, adapters):
    from .._internal.elementwise import activation
    from .._internal.linear import adapter_linear

    key_adapter, value_adapter = adapters
    features = adapter_linear(mixed, key_weight, *key_adapter, _validated=True)
    return adapter_linear(activation(features, "relu2"), value_weight, *value_adapter, _validated=True)


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
    save_normalized: torch.Tensor | None = None,
):
    """Apply res/ln, TokenShift, ReLU² and projection to packed rows.

    The inference value weight retains its runtime [F,C] layout. A transpose
    view feeds the same projection as the training [C,F] weight.
    Projection weights and adapters must use storage separate from ``shift_state_pool``.
    ``save_normalized`` optionally receives the BF16 shift state of each
    active input row.
    """
    from ..tmix.tokenshift import _postnorm_tokenshift_varlen

    value_weight = value_weight.t()
    adapters = _check_projection(
        x, key_weight, value_weight,
        adapters=(
            (key_lora_a, key_lora_b, key_lora_scale, key_use_nora),
            (value_lora_a, value_lora_b, value_lora_scale, value_use_nora),
        ),
    )
    residual, (mixed,) = _postnorm_tokenshift_varlen(
        x, res, weight, bias, (x_k,), shift_state_pool, cu_seqlens, state_indices,
        max_seqlen, eps, validated_metadata, save_normalized=save_normalized,
        save_inputs=(key_weight, value_weight, key_lora_a, key_lora_b, value_lora_a, value_lora_b),
    )
    return residual, _project(mixed, key_weight, value_weight, adapters)


def pretrain_cmix_bf16(
    x,
    x_k,
    key_weight,
    value_weight,
    *,
    initial_shift=None,
    cu_seqlens: torch.Tensor | None = None,
    return_final_state=False,
    key_lora_a=None,
    key_lora_b=None,
    value_lora_a=None,
    value_lora_b=None,
    key_lora_scale=1.0,
    value_lora_scale=1.0,
    key_use_nora=False,
    value_use_nora=False,
):
    """Apply ChannelMix to dense [B,T,C] or packed [N,C] rows with differentiable shift state."""
    from ..tmix.tokenshift.tilelang import tokenshift

    packed = cu_seqlens is not None
    rows = x
    if cu_seqlens is None:
        if x.ndim != 3:
            raise ValueError("CMix training input must have shape [B,T,C]")
        batch, tokens, _ = x.shape
        cu_seqlens = torch.arange(batch + 1, device=x.device, dtype=torch.int32) * tokens
        rows = x.flatten(0, -2)
    adapters = _check_projection(
        rows, key_weight, value_weight,
        adapters=(
            (key_lora_a, key_lora_b, key_lora_scale, key_use_nora),
            (value_lora_a, value_lora_b, value_lora_scale, value_use_nora),
        ),
    )
    (mixed,), final = tokenshift(rows, (x_k,), cu_seqlens, initial_shift, _check_cu=packed)
    output = _project(mixed, key_weight, value_weight, adapters).view_as(x)
    return (output, final) if return_final_state else output


__all__ = ["infer_cmix_forward_varlen", "pretrain_cmix_bf16"]
