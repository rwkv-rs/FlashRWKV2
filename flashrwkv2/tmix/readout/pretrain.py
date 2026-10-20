# SPDX-License-Identifier: MIT

from __future__ import annotations


def pretrain_tmix_readout_bf16(
    x,
    r,
    k,
    v,
    residual_scale,
    weight,
    bias,
    g,
    *,
    head_size=64,
    output_weight=None,
    output_lora_a=None,
    output_lora_b=None,
    output_lora_scale=1.0,
    output_use_nora=False,
):
    """Shared BF16 head LN, recurrent res and output gate with autograd."""
    from ..._internal.linear import adapter_linear
    from .tilelang import readout

    if x.ndim != 3:
        raise ValueError("x must have shape [B,T,C]")
    result = readout(x, r, k, v, residual_scale, weight, bias, g, head_size)
    if output_weight is None:
        if output_lora_a is not None or output_lora_b is not None:
            raise ValueError("output adapters require output_weight")
        return result
    return adapter_linear(
        result, output_weight, output_lora_a, output_lora_b,
        output_lora_scale, output_use_nora,
    )


__all__ = ["pretrain_tmix_readout_bf16"]
