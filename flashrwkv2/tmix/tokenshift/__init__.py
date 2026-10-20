# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from ..wkv7 import (
    _validate_launch_max_seqlen,
    prepare_tmix_wkv7_recurrent_metadata,
)


def infer_tmix_postnorm_tokenshift_forward_varlen(
    x: torch.Tensor,
    res: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
    *,
    shift_state_pool: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    max_seqlen: int | None = None,
    eps: float = 1.0e-5,
    validated_metadata: object | None = None,
) -> tuple[torch.Tensor, ...]:
    """Packed BF16 res, LN and TokenShift with active-slot state updates."""
    from ..._internal.state import gather_state, scatter_state
    from ...post_norm.tilelang import norm
    from .tilelang import fused_post_norm_tokenshift, tokenshift

    if x.ndim != 2:
        raise ValueError("x must have packed shape [total_tokens,C]")
    if (
        shift_state_pool.ndim != 2 or shift_state_pool.shape[1] != x.shape[1]
        or shift_state_pool.dtype != torch.bfloat16 or not shift_state_pool.is_cuda
        or not shift_state_pool.is_contiguous() or shift_state_pool.device != x.device
    ):
        raise ValueError("shift_state_pool must be contiguous CUDA bfloat16 [slots,C]")
    launch_max = _validate_launch_max_seqlen(max_seqlen)
    ticket = validated_metadata
    if ticket is None:
        ticket = prepare_tmix_wkv7_recurrent_metadata(
            cu_seqlens, state_indices, total_tokens=x.shape[0],
            state_pool_size=shift_state_pool.shape[0], max_seqlen=max_seqlen,
        )
    ticket._check_compatible(
        cu_seqlens, state_indices, x.shape[0], shift_state_pool.shape[0], launch_max,
    )
    cu = ticket._query_start_loc_snapshot()
    indices, status = ticket._state_indices_snapshot(), ticket._status()
    if ticket._max_seqlen() == 1 and x.shape[0] == indices.numel():
        total, mixed = fused_post_norm_tokenshift(
            x, res, weight, bias, (x_r, x_w, x_k, x_v, x_a, x_g),
            shift_state_pool, indices, status, eps,
        )
        return (total, *mixed)
    normalized, total = norm(x, weight, bias, eps, res, return_res=True)
    initial = gather_state(shift_state_pool, indices, status)
    mixed, final = tokenshift(
        normalized, (x_r, x_w, x_k, x_v, x_a, x_g), cu, initial, _status=status,
    )
    scatter_state(shift_state_pool, final, indices, status)
    return (total, *mixed)


def _training_tokenshift(x, initial_shift, mixes):
    from .tilelang import tokenshift

    if x.ndim != 3 or x.numel() == 0:
        raise ValueError("x must have non-empty shape [B,T,C]")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous CUDA bfloat16 [B,T,C]")
    batch, length, channels = x.shape
    cu = torch.arange(batch + 1, device=x.device, dtype=torch.int32) * length
    outputs, final = tokenshift(x.view(-1, channels), mixes, cu, initial_shift)
    return tuple(output.view_as(x) for output in outputs) + (final,)


def pretrain_tmix_tokenshift_bf16(
    x: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """BF16 six-way shifted TimeMix with zero initial state."""
    return _training_tokenshift(x, None, (x_r, x_w, x_k, x_v, x_a, x_g))[:-1]


def statetune_tmix_tokenshift_bf16(
    x: torch.Tensor,
    initial_shift: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """BF16 six-way TimeMix with differentiable chunk-boundary shift state."""
    return _training_tokenshift(x, initial_shift, (x_r, x_w, x_k, x_v, x_a, x_g))


__all__ = [
    "infer_tmix_postnorm_tokenshift_forward_varlen",
    "pretrain_tmix_tokenshift_bf16",
    "statetune_tmix_tokenshift_bf16",
]
