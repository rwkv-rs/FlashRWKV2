# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch


def infer_tmix_readout_forward_varlen(
    wkv_output: torch.Tensor,
    receptance: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    r_k: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    gate: torch.Tensor,
    output_weight: torch.Tensor,
    *,
    output_lora_a: torch.Tensor | None = None,
    output_lora_b: torch.Tensor | None = None,
    output_lora_scale: float = 1.0,
    head_size: int = 64,
    batch_size: int = 1,
    max_seqlen: int | None = None,
    output_use_nora: bool = False,
) -> torch.Tensor:
    """BF16 head LN, RKV res, gate and output projection with FP32 accumulators."""
    from ..._internal.linear import adapter_linear
    from .tilelang import readout

    if wkv_output.ndim != 2:
        raise ValueError("wkv_output must have packed shape [total_tokens,C]")
    prelinear = readout(
        wkv_output, receptance, key, value, r_k, ln_weight, ln_bias, gate, head_size,
    )
    return adapter_linear(
        prelinear, output_weight, output_lora_a, output_lora_b,
        output_lora_scale, output_use_nora,
    )


from .pretrain import pretrain_tmix_readout_bf16

__all__ = ["infer_tmix_readout_forward_varlen", "pretrain_tmix_readout_bf16"]
