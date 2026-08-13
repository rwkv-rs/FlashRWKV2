# SPDX-License-Identifier: MIT

from __future__ import annotations

import math

import torch

from . import _extension


def _check_tensor(tensor, name, dtype=None):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous CUDA")
    if dtype is not None and tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")


def _validate_cu_seqlens(
    cu_seqlens: torch.Tensor, *, total_tokens: int, device: torch.device
) -> None:
    if cu_seqlens.dtype != torch.int32:
        raise TypeError("cu_seqlens must have dtype torch.int32")
    if not cu_seqlens.is_cuda or not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous CUDA int32")
    if cu_seqlens.device != device:
        raise ValueError("cu_seqlens must share the token device")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must have shape [B+1]")
    if int(cu_seqlens[0].item()) != 0:
        raise ValueError("cu_seqlens must start at zero")
    if int(cu_seqlens[-1].item()) != total_tokens:
        raise ValueError("cu_seqlens must end at total_tokens")
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if bool((lengths <= 0).any().item()):
        raise ValueError("each RL/Infctx sequence must be non-empty")


def _make_chunk_metadata(cu_seqlens: torch.Tensor, chunk_size: int):
    # This is scheduler metadata preparation.  It happens once per request
    # batch, outside the recurrent launch; no token padding is introduced.
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).detach().cpu().tolist()
    sequence_offsets = [0]
    starts: list[int] = []
    ends: list[int] = []
    token_start = 0
    for length in lengths:
        if length <= 0:
            raise ValueError("each RL/Infctx sequence must be non-empty")
        token_end = token_start + int(length)
        for start in range(token_start, token_end, chunk_size):
            starts.append(start)
            ends.append(min(start + chunk_size, token_end))
        sequence_offsets.append(len(starts))
        token_start = token_end
    device = cu_seqlens.device
    return (
        torch.tensor(sequence_offsets, device=device, dtype=torch.int32),
        torch.tensor(starts, device=device, dtype=torch.int32),
        torch.tensor(ends, device=device, dtype=torch.int32),
    )


def rl_infctx_chunk_fp32io16(
    r,
    decay_logits,
    k,
    v,
    a,
    b,
    *,
    state_pool=None,
    cu_seqlens,
    state_indices=None,
    chunk_size: int = 16,
    strategy: str = "recompute",
    scale: float = 1.0,
    decay_bias=None,
):
    """Run the mechanically migrated packed RL/Infctx chunk family.

    This is the original forward-only workload: ``materialized`` and
    ``recompute`` select the retained CUDA strategy bodies, while the
    backward-replay stage is exposed separately by the native module.  RL is
    not routed through the train_temp recurrent operator.
    """

    if chunk_size not in {16, 32, 64}:
        raise ValueError("chunk_size must be one of 16, 32, or 64")
    if strategy not in {"materialized", "recompute"}:
        raise ValueError("strategy must be 'materialized' or 'recompute'")
    if not math.isfinite(float(scale)):
        raise ValueError("scale must be finite")
    tensors = {"r": r, "decay_logits": decay_logits, "k": k, "v": v, "a": a, "b": b}
    for name, tensor in tensors.items():
        _check_tensor(tensor, name)
        if tensor.dtype not in {torch.float16, torch.bfloat16}:
            raise TypeError(f"{name} must be float16 or bfloat16")
    if r.ndim != 3 or r.shape[0] <= 0 or r.shape[1] <= 0 or r.shape[2] not in {64, 128, 256}:
        raise ValueError("RL/Infctx token tensors must have shape [N,H,D]")
    if any(tensor.shape != r.shape or tensor.device != r.device for tensor in tensors.values()):
        raise ValueError("RL/Infctx token tensors must match r and share a device")
    if decay_bias is not None:
        _check_tensor(decay_bias, "decay_bias")
        if decay_bias.dtype != r.dtype or decay_bias.device != r.device:
            raise TypeError("decay_bias must match token dtype and device")
        if decay_bias.shape not in {
            (r.shape[1] * r.shape[2],),
            (r.shape[1], r.shape[2]),
        }:
            raise ValueError("decay_bias must have shape [H*D] or [H,D]")
    _check_tensor(cu_seqlens, "cu_seqlens", torch.int32)
    _validate_cu_seqlens(
        cu_seqlens, total_tokens=r.shape[0], device=r.device
    )
    batch = cu_seqlens.numel() - 1
    if state_indices is None:
        state_indices = torch.arange(batch, device=r.device, dtype=torch.int32)
    _check_tensor(state_indices, "state_indices", torch.int32)
    if state_indices.ndim != 1 or state_indices.numel() != batch:
        raise ValueError("state_indices must have shape [B]")
    if state_indices.device != r.device:
        raise ValueError("state_indices must share the token device")
    indices_cpu = state_indices.detach().cpu().tolist()
    if len(set(indices_cpu)) != len(indices_cpu):
        raise ValueError("state_indices must not contain duplicates")
    if state_pool is None:
        state_pool = torch.zeros(batch, r.shape[1], r.shape[2], r.shape[2], device=r.device, dtype=torch.float32)
    _check_tensor(state_pool, "state_pool", torch.float32)
    if state_pool.ndim != 4 or state_pool.shape[1:] != (r.shape[1], r.shape[2], r.shape[2]):
        raise ValueError("state_pool must have shape [slots,H,D,D]")
    if max(indices_cpu, default=-1) >= state_pool.shape[0] or min(indices_cpu, default=0) < 0:
        raise ValueError("state_indices must be within state_pool")
    chunk_metadata = _make_chunk_metadata(cu_seqlens, chunk_size)
    working = state_pool.index_select(0, state_indices.to(torch.int64))
    output, final_working = _extension().rl_infctx_chunk_fp32io16_forward(
        *chunk_metadata,
        working,
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        0 if strategy == "materialized" else 1,
        float(scale),
        decay_bias,
    )
    final_pool = state_pool.clone().index_copy(0, state_indices.to(torch.int64), final_working)
    return output, final_pool


def rl_infctx_chunk_fp32io16_factor_recompute(*args, **kwargs):
    """Explicit recompute entry for the RL/Infctx operator family."""

    kwargs["strategy"] = "recompute"
    return rl_infctx_chunk_fp32io16(*args, **kwargs)


__all__ = ["rl_infctx_chunk_fp32io16", "rl_infctx_chunk_fp32io16_factor_recompute"]
