# SPDX-License-Identifier: MIT
"""Dense and packed training wrapper around the shared BF16 recurrent forward."""

from __future__ import annotations

import torch

_SUPPORTED_HEAD_DIMS = (64, 128, 256)


def _validate_inputs(tensors: tuple[torch.Tensor, ...], head_size: int) -> tuple[int, int, int]:
    if head_size not in _SUPPORTED_HEAD_DIMS:
        raise ValueError("head_size must be one of 64, 128, or 256")
    r = tensors[0]
    if not isinstance(r, torch.Tensor):
        raise TypeError("r must be a torch.Tensor")
    if r.ndim != 3 or r.shape[0] <= 0 or r.shape[2] <= 0:
        raise ValueError("r must have shape [B,T,C] with positive B and C")
    if r.dtype != torch.bfloat16:
        raise TypeError("r must be bfloat16")
    if not r.is_cuda or not r.is_contiguous():
        raise ValueError("r must be contiguous CUDA")
    for name, tensor in zip(("w", "k", "v", "a", "b"), tensors[1:], strict=True):
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
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    return_final_state: bool = False,
    scale: float = 1.0,
    decay_bias: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Differentiable WKV7 recurrence on dense ``[B,T,C]`` or packed ``[N,H,D]`` with FP32 state."""
    from .tilelang import tmix_wkv7_recurrent

    values = (r, w, k, v, a, b)
    if cu_seqlens is None:
        batch, tokens, channels = _validate_inputs(values, head_size)
        heads = channels // head_size
        values = tuple(t.view(batch * tokens, heads, head_size) for t in values)
        cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device=r.device) * tokens
    else:
        from ..._internal.state import check_cu_seqlens

        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must have shape [B+1] with positive B")
        batch = cu_seqlens.numel() - 1
        _, heads, head_size = r.shape
        check_cu_seqlens(cu_seqlens, r.shape[0])
    if initial_state is None:
        initial_state = r.new_zeros((batch, heads, head_size, head_size), dtype=torch.float32)
    output, final = tmix_wkv7_recurrent(*values, initial_state, cu_seqlens, scale=scale, decay_bias=decay_bias)
    output = output.view_as(r)
    return (output, final) if return_final_state else output


__all__ = ["pretrain_tmix_wkv7_recurrent_bf16"]
