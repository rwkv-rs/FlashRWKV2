# SPDX-License-Identifier: MIT
"""Differentiable TMix readout for dense [B, T, C] training tensors."""

from __future__ import annotations

import torch


def pretrain_tmix_readout_bf16(
    x: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    residual_scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    g: torch.Tensor,
    *,
    head_size: int = 64,
    output_weight: torch.Tensor | None = None,
    output_lora_a: torch.Tensor | None = None,
    output_lora_b: torch.Tensor | None = None,
    output_lora_scale: float = 1.0,
    output_use_nora: bool = False,
) -> torch.Tensor:
    """Differentiable BF16 head LN, RKV residual, output gate, and optional output projection."""
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
        result, output_weight, output_lora_a, output_lora_b, output_lora_scale, output_use_nora,
    )


__all__ = ["pretrain_tmix_readout_bf16"]
