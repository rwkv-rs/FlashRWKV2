# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

def _validate(logits: torch.Tensor, targets: torch.Tensor) -> None:
    if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("logits and targets must be tensors")
    if logits.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("logits must have dtype torch.bfloat16 or torch.float32")
    if targets.dtype != torch.int64:
        raise TypeError("targets must have dtype torch.int64")
    if not logits.is_cuda or not targets.is_cuda or not logits.is_contiguous() or not targets.is_contiguous():
        raise ValueError("logits and targets must be contiguous CUDA tensors")
    if logits.ndim < 2 or logits.shape[-1] <= 0:
        raise ValueError("logits must have shape [...,vocab]")
    if targets.numel() != logits.numel() // logits.shape[-1] or targets.device != logits.device:
        raise ValueError("targets must contain one entry per logits row on the same device")
    if logits.numel() == 0:
        raise ValueError("logits must contain at least one row")
    if torch.any((targets < 0) | (targets >= logits.shape[-1])).item():
        raise ValueError("targets must be in [0,vocab)")


def pretrain_l2wrap_ce_bf16(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """FP32 cross entropy and the train_temp L2Wrap gradient."""

    _validate(logits, targets)
    from .tilelang import L2Wrap

    return L2Wrap.apply(logits, targets)


__all__ = ["pretrain_l2wrap_ce_bf16"]
