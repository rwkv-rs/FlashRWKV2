# SPDX-License-Identifier: MIT
"""BF16 WKV7: default FP32 training parity and optional FP16 inference state."""

from __future__ import annotations

import torch


def _extension():
    import flashrwkv2

    return flashrwkv2._extension()


def _validate_state(state, dtypes=(torch.float32,)):
    if (
        not isinstance(state, torch.Tensor) or state.dtype not in dtypes
        or not state.is_cuda or not state.is_contiguous()
        or state.ndim != 4 or state.shape[0] <= 0 or state.shape[1] <= 0
        or state.shape[2] not in (64, 128, 256) or state.shape[3] != state.shape[2]
    ):
        dtype_names = "/".join(str(dtype).removeprefix("torch.") for dtype in dtypes)
        raise ValueError(f"state must be contiguous CUDA {dtype_names} [slots,H,D,D], D=64/128/256")


class _TmixWkv7RecurrentState:
    """Owned or externally bound state pool, including prefix-cache COW."""

    __slots__ = ("_state_pool",)

    def __init__(self, state_pool):
        _validate_state(state_pool, (torch.float32, torch.float16))
        self._state_pool = state_pool

    @property
    def memory_layout(self):
        total = self._state_pool.numel() * self._state_pool.element_size()
        per_slot = total // self._state_pool.shape[0]
        return {
            "base_bytes_per_slot": per_slot, "private_bytes_per_slot": 0,
            "bytes_per_slot": per_slot, "fixed_workspace_nbytes": 0,
            "total_nbytes": total,
        }

    def _components(self):
        return (self._state_pool,)

    def _indices(self, indices):
        if (
            indices.dtype != torch.int32 or indices.ndim != 1
            or indices.device != self._state_pool.device or not indices.is_contiguous()
        ):
            raise ValueError("state indices must be contiguous CUDA int32 [B] on the state device")
        if torch.any((indices < 0) | (indices >= self._state_pool.shape[0])).item():
            raise ValueError("state indices must be within the state pool")
        return indices.long()

    def clone(self):
        return type(self)(self._state_pool.clone())

    def clone_slots(self, state_indices):
        return type(self)(self._state_pool.index_select(0, self._indices(state_indices)))

    def copy_slots_(self, source, source_indices, destination_indices):
        if (
            source._state_pool.shape[1:] != self._state_pool.shape[1:]
            or source._state_pool.device != self._state_pool.device
            or source._state_pool.dtype != self._state_pool.dtype
        ):
            raise ValueError("source and destination state geometry, dtype and device must match")
        src, dst = source._indices(source_indices), self._indices(destination_indices)
        if src.numel() != dst.numel():
            raise ValueError("source and destination slot counts must match")
        if dst.unique().numel() != dst.numel():
            raise ValueError("destination state indices must be unique")
        values = source._state_pool.index_select(0, src)
        self._state_pool.index_copy_(0, dst, values)
        return self

    def reset_slots_(self, state_indices):
        self._state_pool.index_fill_(0, self._indices(state_indices), 0)
        return self

    def materialize_slots_(self, state_indices):
        """Recurrent state is materialized after every call."""
        self._indices(state_indices)
        return self

    def copy_(self, source):
        if (
            source._state_pool.shape != self._state_pool.shape
            or source._state_pool.device != self._state_pool.device
            or source._state_pool.dtype != self._state_pool.dtype
        ):
            raise ValueError("source and destination state shapes, dtype and device must match")
        self._state_pool.copy_(source._state_pool)
        return self

    def zero_(self):
        self._state_pool.zero_()
        return self


def _prepare_state(
    state_pool_size, channels, *, sequence_capacity, head_size=64, device=None,
    dtype,
):
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
           for value in (state_pool_size, channels, sequence_capacity, head_size)):
        raise ValueError("state dimensions and sequence_capacity must be positive integers")
    if head_size not in (64, 128, 256) or channels % head_size:
        raise ValueError("head_size must be 64, 128, or 256 and divide channels")
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    elif isinstance(device, int):
        device = torch.device("cuda", device)
    return _TmixWkv7RecurrentState(torch.zeros(
        state_pool_size, channels // head_size, head_size, head_size,
        device=device, dtype=dtype,
    ))


def prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(state):
    _validate_state(state)
    return _TmixWkv7RecurrentState(state)


def prepare_tmix_wkv7_recurrent_fp32io16_state(
    state_pool_size, channels, *, sequence_capacity, head_size=64, device=None,
):
    return _prepare_state(
        state_pool_size, channels, sequence_capacity=sequence_capacity,
        head_size=head_size, device=device, dtype=torch.float32,
    )


def prepare_tmix_wkv7_recurrent_fp16_state(
    state_pool_size, channels, *, sequence_capacity, head_size=64, device=None,
):
    """Allocate optional FP16 inference state; training keeps FP32 state."""
    return _prepare_state(
        state_pool_size, channels, sequence_capacity=sequence_capacity,
        head_size=head_size, device=device, dtype=torch.float16,
    )


def prepare_tmix_wkv7_recurrent_fp16_state_from_tensor(state):
    _validate_state(state, (torch.float16,))
    return _TmixWkv7RecurrentState(state)


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



def _validate_launch_max_seqlen(max_seqlen):
    if max_seqlen is None:
        return -1
    if not isinstance(max_seqlen, int) or isinstance(max_seqlen, bool) or max_seqlen <= 0:
        raise ValueError("max_seqlen must be a positive integer")
    return max_seqlen


def _infer_forward_varlen(
    r, decay_logits, k, v, a, b, *,
    state, cu_seqlens, state_indices, scale=1.0, decay_bias=None,
    max_seqlen=None, validated_metadata=None,
    state_dtype,
):
    """Launch indexed recurrence with the requested state storage precision."""
    from .tilelang import _infer_recurrent

    if not isinstance(state, _TmixWkv7RecurrentState):
        raise TypeError("state must be a prepared WKV7 recurrent state handle")
    if state._state_pool.dtype != state_dtype:
        raise ValueError(f"state must be {state_dtype}")
    ticket = validated_metadata
    if ticket is None:
        ticket = prepare_tmix_wkv7_recurrent_metadata(
            cu_seqlens, state_indices, total_tokens=r.shape[0],
            state_pool_size=state._state_pool.shape[0], max_seqlen=max_seqlen,
        )
    ticket._check_compatible(
        cu_seqlens, state_indices, r.shape[0], state._state_pool.shape[0],
        _validate_launch_max_seqlen(max_seqlen),
    )
    cu_seqlens = ticket._query_start_loc_snapshot()
    state_indices = ticket._state_indices_snapshot()
    status = ticket._status()
    return _infer_recurrent(
        r, decay_logits, k, v, a, b, state._state_pool, cu_seqlens,
        state_indices, status, scale=scale, decay_bias=decay_bias,
    )


@torch.no_grad()
def infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
    r, decay_logits, k, v, a, b, *,
    state, cu_seqlens, state_indices, scale=1.0, decay_bias=None,
    max_seqlen=None, validated_metadata=None,
):
    """Match training with BF16 token IO, FP32 recurrence and FP32 state."""
    return _infer_forward_varlen(
        r, decay_logits, k, v, a, b, state=state, cu_seqlens=cu_seqlens,
        state_indices=state_indices, scale=scale, decay_bias=decay_bias,
        max_seqlen=max_seqlen, validated_metadata=validated_metadata,
        state_dtype=torch.float32,
    )


@torch.no_grad()
def infer_tmix_wkv7_recurrent_fp16_forward_varlen(
    r, decay_logits, k, v, a, b, *,
    state, cu_seqlens, state_indices, scale=1.0, decay_bias=None,
    max_seqlen=None, validated_metadata=None,
):
    """Optional inference: BF16 token IO, FP32 arithmetic, FP16 state per token.

    This mode does not promise bitwise equality with the FP32 training state.
    """
    return _infer_forward_varlen(
        r, decay_logits, k, v, a, b, state=state, cu_seqlens=cu_seqlens,
        state_indices=state_indices, scale=scale, decay_bias=decay_bias,
        max_seqlen=max_seqlen, validated_metadata=validated_metadata,
        state_dtype=torch.float16,
    )


def rl_infctx_tmix_wkv7_recurrent_bf16(
    r, decay_logits, k, v, a, b, *,
    state_pool, cu_seqlens, state_indices=None, scale=1.0, decay_bias=None,
):
    """Differentiable packed recurrence and final FP32 state for RL/Infctx."""
    from .tilelang import tmix_wkv7_recurrent

    _validate_state(state_pool)
    if (
        cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2
        or cu_seqlens.dtype != torch.int32
        or cu_seqlens.device != state_pool.device or not cu_seqlens.is_contiguous()
    ):
        raise ValueError("cu_seqlens must be contiguous CUDA int32 [B+1]")
    invalid_lengths = (
        (cu_seqlens[0] != 0)
        | (cu_seqlens[-1] != r.shape[0])
        | (cu_seqlens[1:] < cu_seqlens[:-1]).any()
    )
    if bool(invalid_lengths.item()):
        raise ValueError("cu_seqlens must be nondecreasing and span [0,N]")
    if state_indices is None:
        return tmix_wkv7_recurrent(
            r, decay_logits, k, v, a, b, state_pool, cu_seqlens,
            scale=scale, decay_bias=decay_bias,
        )
    if (
        state_indices.ndim != 1
        or state_indices.shape != (cu_seqlens.numel() - 1,)
        or state_indices.dtype != torch.int32
        or state_indices.device != state_pool.device
        or not state_indices.is_contiguous()
    ):
        raise ValueError("state_indices must be contiguous CUDA int32 [B]")
    if bool(((state_indices < 0) | (state_indices >= state_pool.shape[0])).any().item()):
        raise ValueError("state_indices must be within state_pool")
    if state_indices.unique().numel() != state_indices.numel():
        raise ValueError("state_indices must be unique")
    indices = state_indices.long()
    output, final = tmix_wkv7_recurrent(
        r, decay_logits, k, v, a, b, state_pool.index_select(0, indices), cu_seqlens,
        scale=scale, decay_bias=decay_bias,
    )
    return output, state_pool.index_copy(0, indices, final)


from .pretrain import pretrain_tmix_wkv7_recurrent_bf16

__all__ = [
    "infer_tmix_wkv7_recurrent_fp16_forward_varlen",
    "infer_tmix_wkv7_recurrent_fp32io16_forward_varlen",
    "prepare_tmix_wkv7_recurrent_fp32io16_state",
    "prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor",
    "prepare_tmix_wkv7_recurrent_fp16_state",
    "prepare_tmix_wkv7_recurrent_fp16_state_from_tensor",
    "prepare_tmix_wkv7_recurrent_metadata",
    "pretrain_tmix_wkv7_recurrent_bf16",
    "rl_infctx_tmix_wkv7_recurrent_bf16",
]
