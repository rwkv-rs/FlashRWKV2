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
    save_normalized: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Packed BF16 res, ln and TokenShift with active-slot state updates.

    ``save_normalized`` optionally receives the BF16 shift state of each
    active input row, including rows before the end of each sequence.
    """
    from ..._internal.state import gather_state, scatter_state
    from ...post_norm.tilelang import norm
    from .tilelang import _check_save_normalized, fused_post_norm_tokenshift, tokenshift

    if x.ndim != 2:
        raise ValueError("x must have packed shape [total_tokens,C]")
    if (
        shift_state_pool.ndim != 2 or shift_state_pool.shape[1] != x.shape[1]
        or shift_state_pool.dtype != torch.bfloat16 or not shift_state_pool.is_cuda
        or not shift_state_pool.is_contiguous() or shift_state_pool.device != x.device
    ):
        raise ValueError("shift_state_pool must be contiguous CUDA bfloat16 [slots,C]")
    mixes = (x_r, x_w, x_k, x_v, x_a, x_g)
    _check_save_normalized(save_normalized, x, res, weight, bias, *mixes, shift_state_pool)
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
            x, res, weight, bias, mixes, shift_state_pool, indices, status, eps,
            save_normalized=save_normalized,
        )
        return (total, *mixed)
    normalized, total = norm(x, weight, bias, eps, res, return_res=True)
    initial = gather_state(shift_state_pool, indices, status)
    mixed, final = tokenshift(
        normalized, mixes, cu, initial, _status=status, _save_normalized=save_normalized,
    )
    scatter_state(shift_state_pool, final, indices, status)
    return (total, *mixed)


def _validate_packed_training_input(x: torch.Tensor, cu_seqlens: torch.Tensor) -> None:
    from ..._internal.state import check_cuda_condition

    if x.ndim != 2 or x.numel() == 0:
        raise ValueError("x must have non-empty packed shape [N,C]")
    if (
        cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2
        or cu_seqlens.dtype != torch.int32 or not cu_seqlens.is_cuda
        or cu_seqlens.device != x.device or not cu_seqlens.is_contiguous()
    ):
        raise ValueError("cu_seqlens must be contiguous CUDA int32 [B+1] on the x device")
    check_cuda_condition(
        (cu_seqlens[0] == 0) & (cu_seqlens[-1] == x.shape[0])
        & (cu_seqlens[1:] >= cu_seqlens[:-1]).all(),
        "cu_seqlens must be nondecreasing and span [0,N]",
    )


def pretrain_tmix_tokenshift_bf16(
    x: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
    *,
    initial_shift: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    return_final_state: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Mix dense [B,T,C] or packed [N,C] rows with one initial shift per sequence."""
    from .tilelang import tokenshift

    if cu_seqlens is None:
        if x.ndim != 3 or x.numel() == 0:
            raise ValueError("x must have non-empty shape [B,T,C]")
        if not x.is_contiguous():
            raise ValueError("x must be contiguous CUDA bfloat16 [B,T,C]")
        batch, length, _ = x.shape
        cu_seqlens = torch.arange(batch + 1, device=x.device, dtype=torch.int32) * length
    else:
        _validate_packed_training_input(x, cu_seqlens)

    mixes = (x_r, x_w, x_k, x_v, x_a, x_g)
    outputs, final = tokenshift(x.flatten(0, -2), mixes, cu_seqlens, initial_shift)
    mixed = tuple(output.view_as(x) for output in outputs)
    return (*mixed, final) if return_final_state else mixed


__all__ = [
    "infer_tmix_postnorm_tokenshift_forward_varlen",
    "pretrain_tmix_tokenshift_bf16",
]
