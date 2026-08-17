# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from ...tmix.wkv7 import _extension


def infer_head_linear_all_forward_varlen(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Canonical Albatross ``linear_head(all_logits=True)`` caller path."""

    for name, tensor in (("x", x), ("weight", weight)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.float16 or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA float16")
        if tensor.device != x.device:
            raise ValueError(f"{name} must share x's device")
    if (
        x.ndim != 2
        or x.shape[0] <= 0
        or x.shape[1] <= 0
        or weight.ndim != 2
        or weight.shape[0] <= 0
        or x.shape[1] != weight.shape[1]
    ):
        raise ValueError("x must be [rows,C] and weight must be [vocab,C]")
    return _extension().head_linear_all_forward_varlen(x, weight)


def infer_head_linear_last_forward_varlen(
    x: torch.Tensor, weight: torch.Tensor, *, tokens_count: int
) -> torch.Tensor:
    """Canonical Albatross ``linear_head_last`` caller path."""

    for name, tensor in (("x", x), ("weight", weight)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.float16 or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA float16")
        if tensor.device != x.device:
            raise ValueError(f"{name} must share x's device")
    if (
        x.ndim != 2
        or x.shape[0] <= 0
        or x.shape[1] <= 0
        or weight.ndim != 2
        or weight.shape[0] <= 0
        or x.shape[1] != weight.shape[1]
    ):
        raise ValueError("x must be [batch,C] and weight must be [vocab,C]")
    if not isinstance(tokens_count, int) or tokens_count <= 0:
        raise ValueError("tokens_count must be a positive int")
    return _extension().head_linear_last_forward_varlen(x, weight, tokens_count)


__all__ = [
    "infer_head_linear_all_forward_varlen",
    "infer_head_linear_last_forward_varlen",
]
