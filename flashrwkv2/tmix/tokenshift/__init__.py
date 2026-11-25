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
    if x.ndim != 2:
        raise ValueError("x must have packed shape [total_tokens,C]")
    total, mixed = _postnorm_tokenshift_varlen(
        x, res, weight, bias, (x_r, x_w, x_k, x_v, x_a, x_g),
        shift_state_pool, cu_seqlens, state_indices, max_seqlen, eps, validated_metadata,
        save_normalized=save_normalized,
    )
    return (total, *mixed)


def _postnorm_tokenshift_varlen(
    x, res, weight, bias, mixes, shift_state_pool, cu_seqlens, state_indices,
    max_seqlen, eps, validated_metadata, *, save_normalized=None, save_inputs=(), project=None,
):
    from ..._internal.state import gather_state, scatter_state
    from ...post_norm.tilelang import norm
    from .tilelang import _check_save_normalized, fused_post_norm_tokenshift, tokenshift

    if (
        shift_state_pool.ndim != 2 or shift_state_pool.shape[1] != x.shape[1]
        or shift_state_pool.dtype != torch.bfloat16 or not shift_state_pool.is_cuda
        or not shift_state_pool.is_contiguous() or shift_state_pool.device != x.device
    ):
        raise ValueError("shift_state_pool must be contiguous CUDA bfloat16 [slots,C]")
    _check_save_normalized(
        save_normalized, x, res, weight, bias, *mixes, shift_state_pool, *save_inputs,
    )
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
    fused = ticket._max_seqlen() == 1 and x.shape[0] == indices.numel()
    if fused:
        total, mixed = fused_post_norm_tokenshift(
            x, res, weight, bias, mixes, shift_state_pool, indices, status, eps,
            save_normalized=save_normalized,
        )
    else:
        normalized, total = norm(x, weight, bias, eps, res, return_res=True)
        initial = gather_state(shift_state_pool, indices, status)
        mixed, final = tokenshift(
            normalized, mixes, cu, initial, _status=status, _save_normalized=save_normalized,
        )
    output = mixed if project is None else project(mixed)
    if not fused:
        scatter_state(shift_state_pool, final, indices, status)
    return total, output


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

    packed = cu_seqlens is not None
    rows = x
    if cu_seqlens is None:
        if x.ndim != 3:
            raise ValueError("x must have shape [B,T,C]")
        batch, length, channels = x.shape
        cu_seqlens = torch.arange(batch + 1, device=x.device, dtype=torch.int32) * length
        rows = x.view(batch * length, channels)

    mixes = (x_r, x_w, x_k, x_v, x_a, x_g)
    outputs, final = tokenshift(rows, mixes, cu_seqlens, initial_shift, _check_cu=packed)
    mixed = tuple(output.view_as(x) for output in outputs)
    return (*mixed, final) if return_final_state else mixed


__all__ = [
    "infer_tmix_postnorm_tokenshift_forward_varlen",
    "pretrain_tmix_tokenshift_bf16",
]
