# SPDX-License-Identifier: MIT
"""BF16 WKV7: default FP32 training parity and optional FP16 inference state."""

from __future__ import annotations

import torch

_SUPPORTED_HEAD_DIMS = (64, 128, 256)
_FP16_SERVING_HEADS = (40, 64)
_DELTALOG_STANDARD_FIELDS = 5
_DELTALOG_COMPACT_FIELDS = 6
_COMPACT_DELTALOG_DEVICE = (8, 9, 128)  # RTX 4090 SM89, 128 SMs


def _extension():
    import flashrwkv2

    return flashrwkv2._extension()


def _validate_state(state: torch.Tensor, dtypes: tuple[torch.dtype, ...] = (torch.float32,)) -> None:
    if (
        not isinstance(state, torch.Tensor)
        or state.dtype not in dtypes
        or not state.is_cuda
        or not state.is_contiguous()
        or state.ndim != 4
        or state.shape[0] <= 0
        or state.shape[1] <= 0
        or state.shape[2] not in _SUPPORTED_HEAD_DIMS
        or state.shape[3] != state.shape[2]
    ):
        dtype_names = "/".join(str(dtype).removeprefix("torch.") for dtype in dtypes)
        raise ValueError(f"state must be contiguous CUDA {dtype_names} [slots,H,D,D], D=64/128/256")


def _require_positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class _TmixWkv7RecurrentState:
    """Managed WKV7 state pool (FP32 or FP16+DeltaLog), supporting prefix-cache COW."""

    __slots__ = ("_delta", "_state_pool")

    def __init__(self, state_pool: torch.Tensor) -> None:
        _validate_state(state_pool, (torch.float32, torch.float16))
        self._state_pool = state_pool
        slots, heads, dim, _ = state_pool.shape
        self._delta: tuple[torch.Tensor, torch.Tensor] | None = None
        if state_pool.dtype == torch.float16 and dim == 64 and heads in _FP16_SERVING_HEADS:
            props = torch.cuda.get_device_properties(state_pool.device)
            compact = heads == 64 and (props.major, props.minor, props.multi_processor_count) == _COMPACT_DELTALOG_DEVICE
            fields = _DELTALOG_COMPACT_FIELDS if compact else _DELTALOG_STANDARD_FIELDS
            self._delta = (
                state_pool.new_empty((slots, heads, fields, dim), dtype=torch.float32),
                state_pool.new_zeros((slots, heads), dtype=torch.int32),
            )

    @property
    def memory_layout(self) -> dict[str, int]:
        """Per-slot and total byte accounting (base pool + private DeltaLog)."""
        slots = self._state_pool.shape[0]
        base = self._state_pool.nbytes // slots
        private = sum(t.nbytes for t in self._delta) // slots if self._delta is not None else 0
        return {
            "base_bytes_per_slot": base,
            "private_bytes_per_slot": private,
            "bytes_per_slot": base + private,
            "fixed_workspace_nbytes": 0,
            "total_nbytes": (base + private) * slots,
        }

    def _components(self) -> tuple[torch.Tensor]:
        return (self._state_pool,)

    def _check_index_tensor(self, indices: torch.Tensor) -> None:
        if (
            indices.dtype != torch.int32
            or indices.ndim != 1
            or indices.device != self._state_pool.device
            or not indices.is_contiguous()
        ):
            raise ValueError("state indices must be contiguous CUDA int32 [B] on the state device")

    def _indices(self, indices: torch.Tensor) -> torch.Tensor:
        from ..._internal.state import check_cuda_condition

        self._check_index_tensor(indices)
        check_cuda_condition(
            ((indices >= 0) & (indices < self._state_pool.shape[0])).all(),
            "state indices must be within the state pool",
        )
        return indices

    def _drain(self, indices: torch.Tensor, status: torch.Tensor | None = None) -> None:
        if self._delta is not None and indices.numel():
            from .tilelang import _materialize_deltalog

            _materialize_deltalog(self._state_pool.shape[1], self._delta[0].shape[2], status is not None)(
                self._state_pool, *self._delta, indices, status if status is not None else indices[:0],
            )

    def _drain_all(self) -> None:
        if self._delta is not None:
            self._drain(torch.arange(self._state_pool.shape[0], device=self._state_pool.device, dtype=torch.int32))

    def clone(self) -> _TmixWkv7RecurrentState:
        self._drain_all()
        return type(self)(self._state_pool.clone())

    def clone_slots(self, state_indices: torch.Tensor) -> _TmixWkv7RecurrentState:
        indices = self._indices(state_indices)
        self._drain(indices)
        return type(self)(self._state_pool.index_select(0, indices))

    def copy_slots_(
        self, source: _TmixWkv7RecurrentState, source_indices: torch.Tensor, destination_indices: torch.Tensor,
        *, validated_metadata: object | None = None,
    ) -> _TmixWkv7RecurrentState:
        """Copy selected slots, or just the active prefix of a live metadata ticket."""
        if (
            source._state_pool.shape[1:] != self._state_pool.shape[1:]
            or source._state_pool.device != self._state_pool.device
            or source._state_pool.dtype != self._state_pool.dtype
        ):
            raise ValueError("source and destination state geometry, dtype and device must match")
        if validated_metadata is not None:
            self._copy_live_slots(source, source_indices, destination_indices, validated_metadata)
            return self
        src, dst = source._indices(source_indices), self._indices(destination_indices)
        if src.numel() != dst.numel():
            raise ValueError("source and destination slot counts must match")
        from ..._internal.state import check_cuda_condition

        ordered = dst.sort().values
        check_cuda_condition((ordered[1:] != ordered[:-1]).all(), "destination state indices must be unique")
        source._drain(src)
        values = source._state_pool.index_select(0, src)
        dst_long = dst.long()
        self._state_pool.index_copy_(0, dst_long, values)
        if self._delta is not None:
            self._delta[1].index_fill_(0, dst_long, 0)
        return self

    def _copy_live_slots(self, source, src, dst, ticket):
        from ..._internal.state import gather_state
        from .tilelang import _copy_recurrent_slots

        if not hasattr(ticket, "_is_graph") or not ticket._is_graph():
            raise ValueError("copy_slots_ requires a live validated_metadata ticket")
        source._check_index_tensor(src)
        self._check_index_tensor(dst)
        status = ticket._status()
        batch = ticket._state_indices_snapshot().numel()
        pool, source_pool = self._state_pool, source._state_pool
        if src.numel() != batch or dst.numel() != batch or status.device != pool.device:
            raise ValueError("slot indices must match the live metadata sequence capacity and device")

        active = (status[0] == 0) & (torch.arange(batch, device=pool.device, dtype=torch.int32) < status[2])
        ordered = torch.where(active, dst, pool.shape[0]).sort().values
        valid = (
            (~active | ((src >= 0) & (src < source_pool.shape[0]) & (dst >= 0) & (dst < pool.shape[0]))).all()
            & (~active[1:] | (ordered[1:] != ordered[:-1])).all()
        )
        torch._assert_async(valid, "active state indices must be in range with unique destinations")
        status = torch.where(valid, status, 0)
        source._drain(src, status)

        # Stage overlapping pools before writing so all reads see the original slots.
        src_ptr, dst_ptr = source_pool.data_ptr(), pool.data_ptr()
        overlap = src_ptr < dst_ptr + pool.nbytes and dst_ptr < src_ptr + source_pool.nbytes
        if overlap:
            source_pool = gather_state(source_pool, src, status)
        phase = self._delta[1] if self._delta is not None else status[:0].view(0, pool.shape[1])
        flat = pool.flatten(1)
        _copy_recurrent_slots(
            flat.shape[1], pool.shape[1], str(pool.dtype).removeprefix("torch."), overlap, self._delta is not None,
        )(source_pool.flatten(1), flat, src, dst, status, phase)

    def reset_slots_(self, state_indices: torch.Tensor) -> _TmixWkv7RecurrentState:
        indices = self._indices(state_indices).long()
        self._state_pool.index_fill_(0, indices, 0)
        if self._delta is not None:
            self._delta[1].index_fill_(0, indices, 0)
        return self

    def materialize_slots_(self, state_indices: torch.Tensor) -> _TmixWkv7RecurrentState:
        """Flush pending DeltaLog updates for ``state_indices`` into the bound tensor."""
        self._drain(self._indices(state_indices))
        return self

    def copy_(self, source: _TmixWkv7RecurrentState) -> _TmixWkv7RecurrentState:
        if (
            source._state_pool.shape != self._state_pool.shape
            or source._state_pool.device != self._state_pool.device
            or source._state_pool.dtype != self._state_pool.dtype
        ):
            raise ValueError("source and destination state shapes, dtype and device must match")
        source._drain_all()
        self._state_pool.copy_(source._state_pool)
        if self._delta is not None:
            self._delta[1].zero_()
        return self

    def zero_(self) -> _TmixWkv7RecurrentState:
        self._state_pool.zero_()
        if self._delta is not None:
            self._delta[1].zero_()
        return self


def prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(state: torch.Tensor) -> _TmixWkv7RecurrentState:
    """Bind a contiguous CUDA FP32 tensor ``[slots, H, D, D]`` as a WKV7 state handle."""
    _validate_state(state)
    return _TmixWkv7RecurrentState(state)


def prepare_tmix_wkv7_recurrent_fp16_state_from_tensor(state: torch.Tensor) -> _TmixWkv7RecurrentState:
    """Bind a contiguous CUDA FP16 tensor ``[slots, H, D, D]``; call ``materialize_slots_`` before direct reads."""
    _validate_state(state, (torch.float16,))
    return _TmixWkv7RecurrentState(state)


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
    """Create a static eager ticket (``total_tokens``) or a live CUDA Graph capacity ticket."""
    if not isinstance(cu_seqlens, torch.Tensor):
        raise TypeError("cu_seqlens must be a torch.Tensor")
    if cu_seqlens.ndim != 1:
        raise ValueError("cu_seqlens must be one-dimensional")
    if not isinstance(state_indices, torch.Tensor):
        raise TypeError("state_indices must be a torch.Tensor")
    graph_values = (token_capacity, sequence_capacity, max_seqlen_capacity, num_active_tokens, num_active_sequences)
    if any(value is not None for value in graph_values):
        if total_tokens is not None or max_seqlen is not None:
            raise ValueError("graph metadata uses capacity arguments, not total_tokens/max_seqlen")
        if any(value is None for value in graph_values):
            raise ValueError("all graph metadata capacity and active-count arguments are required")
        tok_cap = _require_positive_int("token_capacity", token_capacity)
        seq_cap = _require_positive_int("sequence_capacity", sequence_capacity)
        pool_size = _require_positive_int("state_pool_size", state_pool_size)
        max_cap = _require_positive_int("max_seqlen_capacity", max_seqlen_capacity)
        for name, value in (("num_active_tokens", num_active_tokens), ("num_active_sequences", num_active_sequences)):
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.int32
                or not value.is_cuda
                or not value.is_contiguous()
                or value.numel() != 1
                or value.device != cu_seqlens.device
            ):
                raise ValueError(f"{name} must be a one-element contiguous CUDA int32 tensor")
        return _extension().prepare_tmix_wkv7_recurrent_graph_metadata(
            cu_seqlens, state_indices, num_active_tokens, num_active_sequences, tok_cap, seq_cap, pool_size, max_cap,
        )
    if total_tokens is None:
        raise ValueError("total_tokens is required for static recurrent metadata")
    return _extension().prepare_tmix_wkv7_recurrent_metadata(
        cu_seqlens, state_indices, total_tokens, state_pool_size, -1 if max_seqlen is None else int(max_seqlen),
    )


def _validate_launch_max_seqlen(max_seqlen: int | None) -> int:
    return -1 if max_seqlen is None else _require_positive_int("max_seqlen", max_seqlen)


def _infer_forward_varlen(
    r, decay_logits, k, v, a, b, *,
    state, cu_seqlens, state_indices, scale=1.0, decay_bias=None,
    max_seqlen=None, validated_metadata=None, state_dtype, save, boundaries, checkpoint_interval,
):
    """Launch indexed recurrence with the requested state storage precision."""
    from .tilelang import _infer_recurrent

    if not isinstance(state, _TmixWkv7RecurrentState):
        raise TypeError("state must be a prepared WKV7 recurrent state handle")
    if state._state_pool.dtype != state_dtype:
        raise ValueError(f"state must be {state_dtype}")
    geometry = decay_logits if r is None else r
    ticket = validated_metadata
    if ticket is None:
        ticket = prepare_tmix_wkv7_recurrent_metadata(
            cu_seqlens, state_indices, total_tokens=geometry.shape[0],
            state_pool_size=state._state_pool.shape[0], max_seqlen=max_seqlen,
        )
    ticket._check_compatible(
        cu_seqlens, state_indices, geometry.shape[0], state._state_pool.shape[0], _validate_launch_max_seqlen(max_seqlen),
    )
    cu_seqlens, state_indices, status = (
        ticket._query_start_loc_snapshot(), ticket._state_indices_snapshot(), ticket._status(),
    )
    batch = state_indices.numel()
    # The device chooses append or writeback independently for each physical slot.
    delta = state._delta if state._state_pool.shape[0] >= 16 else None
    append = not save and delta is not None and batch >= 16
    if append and geometry.shape[1] == 40 and batch > 64:
        append = torch.cuda.get_device_capability(geometry.device) == (12, 0)
    return _infer_recurrent(
        r, decay_logits, k, v, a, b, state._state_pool, cu_seqlens, state_indices, status,
        scale=scale, decay_bias=decay_bias, row_order=ticket._row_order(),
        delta_buffers=delta, append_log=append, save=save, boundaries=boundaries,
        checkpoint_interval=checkpoint_interval,
    )


@torch.no_grad()
def infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
    r: torch.Tensor | None,
    decay_logits: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
    *,
    state: _TmixWkv7RecurrentState, cu_seqlens: torch.Tensor, state_indices: torch.Tensor,
    scale: float = 1.0, decay_bias: torch.Tensor | None = None, max_seqlen: int | None = None,
    validated_metadata: object | None = None, save: bool = False, boundaries: torch.Tensor | None = None,
    checkpoint_interval: int = 16,
) -> torch.Tensor | tuple[torch.Tensor | None, torch.Tensor] | None:
    """Update FP32 WKV7 state in place. Pass ``r=None`` to skip the token output."""
    return _infer_forward_varlen(
        r, decay_logits, k, v, a, b, state=state, cu_seqlens=cu_seqlens, state_indices=state_indices,
        scale=scale, decay_bias=decay_bias, max_seqlen=max_seqlen, validated_metadata=validated_metadata,
        state_dtype=torch.float32, save=save, boundaries=boundaries, checkpoint_interval=checkpoint_interval,
    )


@torch.no_grad()
def infer_tmix_wkv7_recurrent_fp16_forward_varlen(
    r: torch.Tensor | None,
    decay_logits: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
    *,
    state: _TmixWkv7RecurrentState, cu_seqlens: torch.Tensor, state_indices: torch.Tensor,
    scale: float = 1.0, decay_bias: torch.Tensor | None = None, max_seqlen: int | None = None,
    validated_metadata: object | None = None, save: bool = False, boundaries: torch.Tensor | None = None,
    checkpoint_interval: int = 16,
) -> torch.Tensor | tuple[torch.Tensor | None, torch.Tensor] | None:
    """Update FP16 WKV7 state with per-token rounding. Pass ``r=None`` to skip the token output."""
    return _infer_forward_varlen(
        r, decay_logits, k, v, a, b, state=state, cu_seqlens=cu_seqlens, state_indices=state_indices,
        scale=scale, decay_bias=decay_bias, max_seqlen=max_seqlen, validated_metadata=validated_metadata,
        state_dtype=torch.float16, save=save, boundaries=boundaries, checkpoint_interval=checkpoint_interval,
    )


def rl_infctx_tmix_wkv7_recurrent_bf16(
    r: torch.Tensor, decay_logits: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
    *,
    state_pool: torch.Tensor, cu_seqlens: torch.Tensor, state_indices: torch.Tensor | None = None,
    scale: float = 1.0, decay_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable packed WKV7 over a slot-indexed FP32 ``state_pool``; unselected slots stay unchanged."""
    from ..._internal.state import check_cuda_condition
    from .tilelang import tmix_wkv7_recurrent

    _validate_state(state_pool)
    if (
        cu_seqlens.ndim != 1
        or cu_seqlens.numel() < 2
        or cu_seqlens.dtype != torch.int32
        or cu_seqlens.device != state_pool.device
        or not cu_seqlens.is_contiguous()
    ):
        raise ValueError("cu_seqlens must be contiguous CUDA int32 [B+1]")
    invalid_lengths = (
        (cu_seqlens[0] != 0) | (cu_seqlens[-1] != r.shape[0]) | (cu_seqlens[1:] < cu_seqlens[:-1]).any()
    )
    check_cuda_condition(~invalid_lengths, "cu_seqlens must be nondecreasing and span [0,N]")
    if state_indices is None:
        return tmix_wkv7_recurrent(
            r, decay_logits, k, v, a, b, state_pool, cu_seqlens, scale=scale, decay_bias=decay_bias,
        )
    if (
        state_indices.ndim != 1
        or state_indices.shape != (cu_seqlens.numel() - 1,)
        or state_indices.dtype != torch.int32
        or state_indices.device != state_pool.device
        or not state_indices.is_contiguous()
    ):
        raise ValueError("state_indices must be contiguous CUDA int32 [B]")
    check_cuda_condition(
        ((state_indices >= 0) & (state_indices < state_pool.shape[0])).all(),
        "state_indices must be within state_pool",
    )
    ordered = state_indices.sort().values
    check_cuda_condition((ordered[1:] != ordered[:-1]).all(), "state_indices must be unique")
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
    "prepare_tmix_wkv7_recurrent_fp16_state_from_tensor",
    "prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor",
    "prepare_tmix_wkv7_recurrent_metadata",
    "pretrain_tmix_wkv7_recurrent_bf16",
    "rl_infctx_tmix_wkv7_recurrent_bf16",
]
