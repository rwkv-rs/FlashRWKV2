# SPDX-License-Identifier: MIT
"""Shared BF16 residual addition and layer normalization."""

from __future__ import annotations

import torch


def infer_post_norm_output_forward_varlen(
    x: torch.Tensor, res: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, *, eps: float = 1.0e-5,
) -> torch.Tensor:
    """Normalize the rounded BF16 residual sum ``x + res`` with FP32 statistics."""
    from .tilelang import norm

    return norm(x, weight, bias, eps=eps, res=res)


__all__ = ["infer_post_norm_output_forward_varlen"]
