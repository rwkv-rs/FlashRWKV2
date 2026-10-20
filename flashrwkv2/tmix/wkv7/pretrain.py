# SPDX-License-Identifier: MIT

"""Dense training wrapper around the shared BF16 recurrent forward."""

from __future__ import annotations

import torch


def _validate_inputs(
    tensors: tuple[torch.Tensor, ...], head_size: int
) -> tuple[int, int, int]:
    if head_size not in {64, 128, 256}:
        raise ValueError("head_size must be one of 64, 128, or 256")
    r = tensors[0]
    if not isinstance(r, torch.Tensor):
        raise TypeError("r must be a torch.Tensor")
    if r.ndim != 3 or min(r.shape) <= 0:
        raise ValueError("r must have shape [B,T,C] with nonzero dimensions")
    if r.dtype != torch.bfloat16:
        raise TypeError("r must be bfloat16")
    if not r.is_cuda or not r.is_contiguous():
        raise ValueError("r must be contiguous CUDA")

    for name, tensor in zip(("w", "k", "v", "a", "b"), tensors[1:]):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be bfloat16")
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA")
        if tensor.shape != r.shape:
            raise ValueError(f"{name} must match r shape")
        if tensor.device != r.device:
            raise ValueError(f"{name} must share the r device")

    batch, tokens, channels = r.shape
    if channels % head_size != 0:
        raise ValueError("C must be divisible by head_size")
    return batch, tokens, channels


def pretrain_tmix_wkv7_recurrent_bf16(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    head_size: int = 64,
) -> torch.Tensor:
    """Run BF16 training with the same forward as inference, for any T >= 1."""
    batch, tokens, channels = _validate_inputs((r, w, k, v, a, b), head_size)
    from .tilelang import tmix_wkv7_recurrent

    heads = channels // head_size
    values = tuple(
        tensor.view(batch * tokens, heads, head_size) for tensor in (r, w, k, v, a, b)
    )
    state = torch.zeros(
        (batch, heads, head_size, head_size), dtype=torch.float32, device=r.device
    )
    cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device=r.device) * tokens
    output, _ = tmix_wkv7_recurrent(*values, state, cu_seqlens)
    return output.view(batch, tokens, channels)


__all__ = ["pretrain_tmix_wkv7_recurrent_bf16"]
