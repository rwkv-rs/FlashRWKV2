# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch


def _extension():
    import flashrwkv2

    extension = getattr(flashrwkv2, "_C", None)
    if extension is None:
        raise RuntimeError(
            "FlashRWKV2 CUDA extension is not built; build flashrwkv2._C before "
            "using the accelerated recurrent operator"
        )
    return extension


def _validate_packed_inputs(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    names = ("r", "decay_logits", "k", "v", "a", "b")
    for name, tensor in zip(names, tensors):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 3:
            raise ValueError(f"{name} must have packed shape [total_tokens,H,D]")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    return tensors


def _validate_state(state: torch.Tensor) -> None:
    if not isinstance(state, torch.Tensor):
        raise TypeError("state_pool must be a torch.Tensor")
    if not state.is_contiguous():
        raise ValueError("state_pool must be contiguous")
    if state.dtype != torch.float32:
        raise TypeError("state_pool must have dtype float32")


def _validate_fp16_state(state: torch.Tensor) -> None:
    if not isinstance(state, torch.Tensor):
        raise TypeError("state_pool must be a torch.Tensor")
    if not state.is_contiguous():
        raise ValueError("state_pool must be contiguous")
    if state.dtype != torch.float16:
        raise TypeError("FP16-state state_pool must have dtype float16")


def _validate_fp16_elapsed_state(
    elapsed_state_pool: torch.Tensor, state_pool: torch.Tensor
) -> None:
    if not isinstance(elapsed_state_pool, torch.Tensor):
        raise TypeError("elapsed_state_pool must be a torch.Tensor")
    if elapsed_state_pool.dtype != torch.int32:
        raise TypeError("elapsed_state_pool must have dtype torch.int32")
    if not elapsed_state_pool.is_cuda or not elapsed_state_pool.is_contiguous():
        raise ValueError("elapsed_state_pool must be contiguous CUDA int32")
    if elapsed_state_pool.device != state_pool.device:
        raise ValueError("elapsed_state_pool must share state_pool's device")
    if elapsed_state_pool.ndim != 1 or elapsed_state_pool.shape[0] != state_pool.shape[0]:
        raise ValueError("elapsed_state_pool must have shape [state_pool_slots]")


def _check_metadata_inputs(
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
) -> None:
    if not isinstance(cu_seqlens, torch.Tensor):
        raise TypeError("cu_seqlens must be a torch.Tensor")
    if cu_seqlens.ndim != 1:
        raise ValueError("cu_seqlens must be one-dimensional")
    if not isinstance(state_indices, torch.Tensor):
        raise TypeError("state_indices must be a torch.Tensor")


def _resolve_max_seqlen(
    cu_seqlens: torch.Tensor,
    max_seqlen: int | None,
) -> int:
    if max_seqlen is not None:
        value = int(max_seqlen)
        if value <= 0:
            raise ValueError("max_seqlen must be positive")
        return value
    if cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must contain at least one sequence")
    return int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())


def prepare_tmix_wkv7_recurrent_metadata(
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    state_pool_size: int,
    total_tokens: int | None = None,
    max_seqlen: int | None = None,
    token_capacity: int | None = None,
    sequence_capacity: int | None = None,
    max_seqlen_capacity: int | None = None,
    num_active_tokens: torch.Tensor | None = None,
    num_active_sequences: torch.Tensor | None = None,
) -> object:
    """Create a static ticket or a live capacity ticket.

    The legacy ``total_tokens`` form snapshots and validates fixed metadata.
    The capacity form keeps metadata and the two scalar active counts live;
    use it with zero active counts for same-stream pre-capture warmup, then
    call it again with the same buffers inside capture and reuse that ticket
    for every stateful operator in the graph.
    """

    _check_metadata_inputs(cu_seqlens, state_indices)
    graph_values = (
        token_capacity,
        sequence_capacity,
        max_seqlen_capacity,
        num_active_tokens,
        num_active_sequences,
    )
    graph_mode = any(value is not None for value in graph_values)
    if graph_mode:
        if total_tokens is not None or max_seqlen is not None:
            raise ValueError(
                "graph metadata uses capacity arguments, not total_tokens/max_seqlen"
            )
        if any(value is None for value in graph_values):
            raise ValueError("all graph metadata capacity and active-count arguments are required")
        assert token_capacity is not None
        assert sequence_capacity is not None
        assert max_seqlen_capacity is not None
        assert num_active_tokens is not None
        assert num_active_sequences is not None
        for name, value in (
            ("token_capacity", token_capacity),
            ("sequence_capacity", sequence_capacity),
            ("state_pool_size", state_pool_size),
            ("max_seqlen_capacity", max_seqlen_capacity),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("num_active_tokens", num_active_tokens),
            ("num_active_sequences", num_active_sequences),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.int32
                or not value.is_cuda
                or not value.is_contiguous()
                or value.numel() != 1
                or value.device != cu_seqlens.device
            ):
                raise ValueError(
                    f"{name} must be a one-element contiguous CUDA int32 tensor"
                )
        return _extension().prepare_tmix_wkv7_recurrent_graph_metadata(
            cu_seqlens,
            state_indices,
            num_active_tokens,
            num_active_sequences,
            token_capacity,
            sequence_capacity,
            state_pool_size,
            max_seqlen_capacity,
        )

    if total_tokens is None:
        raise ValueError("total_tokens is required for static recurrent metadata")
    return _extension().prepare_tmix_wkv7_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens,
        state_pool_size,
        -1 if max_seqlen is None else int(max_seqlen),
    )


def _run(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float,
    decay_bias: torch.Tensor | None,
    max_seqlen: int | None,
    validated_metadata: object | None,
) -> torch.Tensor:
    packed = _validate_packed_inputs(r, decay_logits, k, v, a, b)
    _validate_state(state)
    _check_metadata_inputs(cu_seqlens, state_indices)
    ticket = (
        validated_metadata
        if validated_metadata is not None
        else prepare_tmix_wkv7_recurrent_metadata(
            cu_seqlens,
            state_indices,
            total_tokens=r.shape[0],
            state_pool_size=state.shape[0],
            max_seqlen=max_seqlen,
        )
    )
    output = torch.empty_like(packed[3])
    _extension().tmix_wkv7_recurrent_fp32_from_decay_logits(
        cu_seqlens,
        state_indices,
        state,
        *packed,
        output,
        float(scale),
        decay_bias,
        ticket,
        -1 if max_seqlen is None else int(max_seqlen),
    )
    return output


def infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
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
    scale: float = 1.0,
    decay_bias: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    validated_metadata: object | None = None,
) -> torch.Tensor:
    """Run packed raw-decay RWKV-7 recurrence in-place on ``state_pool``."""

    return _run(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        state=state_pool,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        scale=scale,
        decay_bias=decay_bias,
        max_seqlen=max_seqlen,
        validated_metadata=validated_metadata,
    )


def infer_tmix_wkv7_recurrent_fp16_forward_varlen(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    elapsed_state_pool: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float = 1.0,
    decay_bias: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    validated_metadata: object | None = None,
) -> torch.Tensor:
    """Run Albatross-family packed recurrence with an FP16 state pool."""

    packed = _validate_packed_inputs(r, decay_logits, k, v, a, b)
    _validate_fp16_state(state_pool)
    _validate_fp16_elapsed_state(elapsed_state_pool, state_pool)
    _check_metadata_inputs(cu_seqlens, state_indices)
    if max_seqlen is None:
        # Inference of max_seqlen belongs to metadata preparation.  Do not
        # synchronously inspect the CUDA offsets on every FP16 launch.
        launch_max_seqlen = -1
    else:
        if (
            not isinstance(max_seqlen, int)
            or isinstance(max_seqlen, bool)
            or max_seqlen <= 0
        ):
            raise ValueError("max_seqlen must be a positive integer")
        launch_max_seqlen = int(max_seqlen)
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
    output = torch.empty_like(packed[3])
    _extension().tmix_wkv7_recurrent_fp16_from_decay_logits(
        cu_seqlens,
        state_indices,
        elapsed_state_pool,
        state_pool,
        *packed,
        output,
        float(scale),
        decay_bias,
        ticket,
        launch_max_seqlen,
    )
    return output


__all__ = [
    "infer_tmix_wkv7_recurrent_fp32io16_forward_varlen",
    "infer_tmix_wkv7_recurrent_fp16_forward_varlen",
    "prepare_tmix_wkv7_recurrent_metadata",
]

from .pretrain import pretrain_tmix_wkv7_recurrent_bf16
from .chunk import infer_tmix_wkv7_chunk_bf16_forward_varlen
from .rl_infctx import (
    rl_infctx_tmix_wkv7_chunk_fp32io16,
    rl_infctx_tmix_wkv7_chunk_fp32io16_factor_recompute,
)

__all__.extend(
    [
        "pretrain_tmix_wkv7_recurrent_bf16",
        "infer_tmix_wkv7_chunk_bf16_forward_varlen",
        "rl_infctx_tmix_wkv7_chunk_fp32io16",
        "rl_infctx_tmix_wkv7_chunk_fp32io16_factor_recompute",
    ]
)
