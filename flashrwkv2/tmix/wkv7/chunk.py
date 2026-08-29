# SPDX-License-Identifier: MIT

from __future__ import annotations

import math

import torch

from . import (
    _check_metadata_inputs,
    _extension,
    _metadata_launch_args,
    _resolve_max_seqlen,
    prepare_tmix_wkv7_recurrent_metadata,
)


def infer_tmix_wkv7_chunk_bf16_forward_varlen(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    chunk_size: int = 16,
    max_seqlen: int | None = None,
    scale: float = 1.0,
    decay_bias: torch.Tensor | None = None,
    validated_metadata: object | None = None,
) -> torch.Tensor:
    """Run the RWKV-7 BF16 K1/K2 chunk family on packed requests.

    K1 and K2 remain one public operator with an internal workspace boundary.
    The state pool is updated in place at ``state_indices``; no fixed-length
    padding or request-by-request launch is introduced by this wrapper.
    """

    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError("chunk_size must be an integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not math.isfinite(float(scale)):
        raise ValueError("scale must be finite")
    tensors = {
        "r": r,
        "decay_logits": decay_logits,
        "k": k,
        "v": v,
        "a": a,
        "b": b,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.bfloat16 or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA bfloat16")
        if tensor.device != r.device:
            raise ValueError(f"{name} must share r's device")
        if tensor.shape != r.shape:
            raise ValueError(f"{name} must have the same shape as r")
    if r.ndim != 3 or r.shape[0] <= 0 or r.shape[1] <= 0 or r.shape[2] != 64:
        raise ValueError("packed chunk inputs must have shape [total_tokens,H,64]")
    if (
        not isinstance(state_pool, torch.Tensor)
        or state_pool.dtype != torch.bfloat16
        or not state_pool.is_cuda
        or not state_pool.is_contiguous()
        or state_pool.ndim != 4
        or state_pool.shape[1:] != (r.shape[1], 64, 64)
        or state_pool.device != r.device
    ):
        raise ValueError("state_pool must be contiguous CUDA bfloat16 [slots,H,64,64]")
    _check_metadata_inputs(cu_seqlens, state_indices)
    if (
        cu_seqlens.dtype != torch.int32
        or state_indices.dtype != torch.int32
        or not cu_seqlens.is_cuda
        or not state_indices.is_cuda
        or not cu_seqlens.is_contiguous()
        or not state_indices.is_contiguous()
        or cu_seqlens.device != r.device
        or state_indices.device != r.device
    ):
        raise ValueError("chunk metadata must be contiguous CUDA int32")
    if validated_metadata is not None and max_seqlen is None:
        launch_max_seqlen = int(validated_metadata._max_seqlen())
    else:
        launch_max_seqlen = _resolve_max_seqlen(cu_seqlens, max_seqlen)
    ticket = (
        validated_metadata
        if validated_metadata is not None
        else prepare_tmix_wkv7_recurrent_metadata(
            cu_seqlens,
            state_indices,
            total_tokens=r.shape[0],
            state_pool_size=state_pool.shape[0],
            max_seqlen=launch_max_seqlen,
        )
    )
    if decay_bias is not None and (
        decay_bias.dtype != torch.bfloat16
        or not decay_bias.is_cuda
        or not decay_bias.is_contiguous()
        or decay_bias.device != r.device
        or decay_bias.shape not in {(r.shape[1], 64), (r.shape[1] * 64,)}
    ):
        raise ValueError(
            "decay_bias must be contiguous CUDA bfloat16 [H,64] or [H*64]"
        )
    launch_metadata = _metadata_launch_args(
        ticket,
        cu_seqlens,
        state_indices,
        r.shape[0],
        state_pool.shape[0],
        launch_max_seqlen,
    )
    output, _ = _extension().infer_tmix_wkv7_chunk_bf16_forward_varlen(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        state_pool,
        launch_metadata[0],
        launch_metadata[1],
        launch_metadata[2],
        int(chunk_size),
        int(launch_max_seqlen),
        float(scale),
        decay_bias,
    )
    return output


__all__ = ["infer_tmix_wkv7_chunk_bf16_forward_varlen"]
