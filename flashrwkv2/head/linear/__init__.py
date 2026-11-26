# SPDX-License-Identifier: MIT
"""LM head projection from BF16 hidden states to FP32 logits."""

from __future__ import annotations

import torch

from ..._internal.linear import adapter_linear


def infer_head_linear_all_forward_varlen(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    lora_a: torch.Tensor | None = None,
    lora_b: torch.Tensor | None = None,
    lora_scale: float = 1.0,
    use_nora: bool = False,
) -> torch.Tensor:
    """Project packed BF16 hidden rows ``[N, C]`` to FP32 logits ``[N, V]``."""
    if x.ndim != 2:
        raise ValueError("x must be [rows,C]")
    return adapter_linear(x, weight, lora_a, lora_b, float(lora_scale), use_nora, output_dtype=torch.float32)


def pretrain_head_linear_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    lora_a: torch.Tensor | None = None,
    lora_b: torch.Tensor | None = None,
    lora_scale: float = 1.0,
    use_nora: bool = False,
) -> torch.Tensor:
    """Project BF16 hidden states ``[..., C]`` to FP32 logits ``[..., V]`` with autograd."""
    return adapter_linear(
        x, weight, a=lora_a, b=lora_b, scale=float(lora_scale), use_nora=use_nora, output_dtype=torch.float32,
    )


__all__ = ["infer_head_linear_all_forward_varlen", "pretrain_head_linear_bf16"]
