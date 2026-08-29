# SPDX-License-Identifier: MIT

from __future__ import annotations

from itertools import pairwise

import torch

_DELTALOG_POLICY_SOURCE_REVISION = (
    "3465da5070beceb4bab9e07b03abee1642a0bdf8"
)

# Exact ordinary DeltaLog gates from Albatross faster3a_2607.  APW-only
# entries are model-graph policies and intentionally do not belong here.
_DELTALOG_TUNED_M = {
    (768, 16): 2,
    (768, 32): 3,
    (768, 64): 3,
    (768, 128): 3,
    (768, 256): 3,
    (768, 512): 3,
    (1024, 16): 2,
    (1024, 32): 3,
    (1024, 64): 3,
    (1024, 256): 3,
    (1024, 512): 3,
    (2048, 8): 2,
    (2048, 16): 3,
    (2048, 32): 3,
    (2048, 64): 3,
    (2048, 256): 3,
    (2048, 512): 4,
    (2560, 8): 2,
    (2560, 16): 3,
    (2560, 32): 3,
    (2560, 64): 3,
    (2560, 256): 3,
    (2560, 512): 4,
    (4096, 8): 2,
    (4096, 16): 3,
    (4096, 32): 3,
    (4096, 64): 3,
    (4096, 128): 3,
    (4096, 256): 3,
    (4096, 512): 4,
}

# The ordinary FlashRWKV2 FP16-state launcher is faster than DeltaLog for
# these upstream candidates on PRO6000.  Keep the exact Albatross table above
# as the source policy, then fail closed to the ordinary launcher at the
# locally unprofitable points.  FP32IO16 benefits at every upstream point.
_DELTALOG_FP16_UNPROFITABLE_SHAPES = frozenset(
    {
        (768, 64),
        (768, 128),
        (1024, 64),
        (2048, 32),
        (2048, 64),
        (2560, 32),
        (4096, 32),
    }
)


def _select_deltalog_merge_interval(
    channels: int,
    sequence_capacity: int,
    head_size: int,
    capability: tuple[int, int],
    numerical_mode: str,
) -> int:
    if capability[0] != 12 or head_size != 64:
        return 0
    shape = (channels, sequence_capacity)
    if (
        numerical_mode == "fp16"
        and shape in _DELTALOG_FP16_UNPROFITABLE_SHAPES
    ):
        return 0
    return _DELTALOG_TUNED_M.get(shape, 0)


def _extension():
    import flashrwkv2

    extension = getattr(flashrwkv2, "_C", None)
    if extension is None:
        raise RuntimeError(
            "FlashRWKV2 CUDA extension is not built; build flashrwkv2._C before "
            "using the accelerated recurrent operator"
        )
    return torch.ops.flashrwkv2


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
    if not state.is_cuda or not state.is_contiguous():
        raise ValueError("state_pool must be contiguous CUDA")
    if state.dtype != torch.float32:
        raise TypeError("state_pool must have dtype float32")
    if (
        state.ndim != 4
        or state.shape[0] <= 0
        or state.shape[1] <= 0
        or state.shape[2] <= 0
        or state.shape[2] != state.shape[3]
    ):
        raise ValueError("state_pool must have shape [state_pool_slots,H,D,D]")


def _validate_fp16_state(state: torch.Tensor) -> None:
    if not isinstance(state, torch.Tensor):
        raise TypeError("state_pool must be a torch.Tensor")
    if not state.is_cuda or not state.is_contiguous():
        raise ValueError("state_pool must be contiguous CUDA")
    if state.dtype != torch.float16:
        raise TypeError("FP16-state state_pool must have dtype float16")
    if (
        state.ndim != 4
        or state.shape[0] <= 0
        or state.shape[1] <= 0
        or state.shape[2] <= 0
        or state.shape[2] != state.shape[3]
    ):
        raise ValueError("state_pool must have shape [state_pool_slots,H,D,D]")


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


def _validate_fp16_deltalog_state_bundle(
    state_pool: torch.Tensor,
    elapsed_state_pool: torch.Tensor,
    deltalog_phase_pool: torch.Tensor,
    deltalog_pool: torch.Tensor,
) -> None:
    _validate_fp16_state(state_pool)
    _validate_fp16_elapsed_state(elapsed_state_pool, state_pool)
    if not isinstance(deltalog_phase_pool, torch.Tensor):
        raise TypeError("deltalog_phase_pool must be a torch.Tensor")
    if deltalog_phase_pool.dtype != torch.int32:
        raise TypeError("deltalog_phase_pool must have dtype torch.int32")
    if not deltalog_phase_pool.is_cuda or not deltalog_phase_pool.is_contiguous():
        raise ValueError("deltalog_phase_pool must be contiguous CUDA int32")
    if deltalog_phase_pool.device != state_pool.device:
        raise ValueError("deltalog_phase_pool must share state_pool's device")
    if (
        deltalog_phase_pool.ndim != 1
        or deltalog_phase_pool.shape[0] != state_pool.shape[0]
    ):
        raise ValueError("deltalog_phase_pool must have shape [state_pool_slots]")
    if not isinstance(deltalog_pool, torch.Tensor):
        raise TypeError("deltalog_pool must be a torch.Tensor")
    if deltalog_pool.dtype != torch.float16:
        raise TypeError("deltalog_pool must have dtype torch.float16")
    if not deltalog_pool.is_cuda or not deltalog_pool.is_contiguous():
        raise ValueError("deltalog_pool must be contiguous CUDA float16")
    if deltalog_pool.device != state_pool.device:
        raise ValueError("deltalog_pool must share state_pool's device")
    if deltalog_pool.ndim != 5 or tuple(deltalog_pool.shape[1:]) != (
        5,
        state_pool.shape[0],
        state_pool.shape[1],
        64,
    ):
        raise ValueError(
            "deltalog_pool must have shape [M-1,5,state_pool_slots,H,64]"
        )
    if deltalog_pool.shape[0] + 1 not in {2, 3, 4, 6, 8}:
        raise ValueError("DeltaLog M must be one of {2,3,4,6,8}")


def _validate_fp32io16_deltalog_state_bundle(
    state_pool: torch.Tensor,
    deltalog_phase_pool: torch.Tensor,
    deltalog_pool: torch.Tensor,
) -> None:
    _validate_state(state_pool)
    if not isinstance(deltalog_phase_pool, torch.Tensor):
        raise TypeError("deltalog_phase_pool must be a torch.Tensor")
    if deltalog_phase_pool.dtype != torch.int32:
        raise TypeError("deltalog_phase_pool must have dtype torch.int32")
    if not deltalog_phase_pool.is_cuda or not deltalog_phase_pool.is_contiguous():
        raise ValueError("deltalog_phase_pool must be contiguous CUDA int32")
    if deltalog_phase_pool.device != state_pool.device:
        raise ValueError("deltalog_phase_pool must share state_pool's device")
    if (
        deltalog_phase_pool.ndim != 1
        or deltalog_phase_pool.shape[0] != state_pool.shape[0]
    ):
        raise ValueError("deltalog_phase_pool must have shape [state_pool_slots]")
    if not isinstance(deltalog_pool, torch.Tensor):
        raise TypeError("deltalog_pool must be a torch.Tensor")
    if deltalog_pool.dtype != torch.float32:
        raise TypeError("deltalog_pool must have dtype torch.float32")
    if not deltalog_pool.is_cuda or not deltalog_pool.is_contiguous():
        raise ValueError("deltalog_pool must be contiguous CUDA float32")
    if deltalog_pool.device != state_pool.device:
        raise ValueError("deltalog_pool must share state_pool's device")
    if deltalog_pool.ndim != 5 or tuple(deltalog_pool.shape[1:]) != (
        5,
        state_pool.shape[0],
        state_pool.shape[1],
        64,
    ):
        raise ValueError(
            "deltalog_pool must have shape [M-1,5,state_pool_slots,H,64]"
        )
    if deltalog_pool.shape[0] + 1 not in {2, 3, 4, 6, 8}:
        raise ValueError("DeltaLog M must be one of {2,3,4,6,8}")


class _TmixWkv7RecurrentState:
    __slots__ = (
        "_deltalog_phase_pool",
        "_deltalog_pool",
        "_deltalog_status",
        "_elapsed_state_pool",
        "_merge_interval",
        "_sequence_capacity",
        "_state_pool",
    )

    def __init__(
        self,
        state_pool: torch.Tensor,
        elapsed_state_pool: torch.Tensor | None,
        sequence_capacity: int | None,
        merge_interval: int,
    ) -> None:
        self._state_pool = state_pool
        self._elapsed_state_pool = elapsed_state_pool
        self._sequence_capacity = sequence_capacity
        self._merge_interval = merge_interval
        if merge_interval:
            self._deltalog_phase_pool = torch.zeros(
                state_pool.shape[0],
                dtype=torch.int32,
                device=state_pool.device,
            )
            self._deltalog_pool = torch.zeros(
                (
                    merge_interval - 1,
                    5,
                    state_pool.shape[0],
                    state_pool.shape[1],
                    64,
                ),
                dtype=state_pool.dtype,
                device=state_pool.device,
            )
            self._deltalog_status = torch.empty(
                1 + 2 * sequence_capacity,
                dtype=torch.int32,
                device=state_pool.device,
            )
        else:
            self._deltalog_phase_pool = None
            self._deltalog_pool = None
            self._deltalog_status = None

    def _components(self) -> tuple[torch.Tensor, ...]:
        components = (self._state_pool,)
        if self._elapsed_state_pool is not None:
            components = (*components, self._elapsed_state_pool)
        if self._deltalog_phase_pool is None or self._deltalog_pool is None:
            return components
        return (*components, self._deltalog_phase_pool, self._deltalog_pool)

    @property
    def memory_layout(self) -> dict[str, int]:
        """Return the exact scalable and fixed bytes owned by this handle."""

        slots = self._state_pool.shape[0]
        base_nbytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self._state_pool, self._elapsed_state_pool)
            if tensor is not None
        )
        private_slot_nbytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self._deltalog_phase_pool, self._deltalog_pool)
            if tensor is not None
        )
        fixed_workspace_nbytes = (
            0
            if self._deltalog_status is None
            else self._deltalog_status.numel()
            * self._deltalog_status.element_size()
        )
        return {
            "base_bytes_per_slot": base_nbytes // slots,
            "private_bytes_per_slot": private_slot_nbytes // slots,
            "bytes_per_slot": (base_nbytes + private_slot_nbytes) // slots,
            "fixed_workspace_nbytes": fixed_workspace_nbytes,
            "total_nbytes": (
                base_nbytes + private_slot_nbytes + fixed_workspace_nbytes
            ),
        }

    def _check_compatible_slots(
        self, source: _TmixWkv7RecurrentState
    ) -> None:
        if not isinstance(source, _TmixWkv7RecurrentState):
            raise TypeError("source must be a WKV7 recurrent state handle")
        if (
            self._merge_interval != source._merge_interval
            or self._state_pool.dtype != source._state_pool.dtype
            or self._state_pool.device != source._state_pool.device
            or self._state_pool.shape[1:] != source._state_pool.shape[1:]
            or (self._elapsed_state_pool is None)
            != (source._elapsed_state_pool is None)
        ):
            raise ValueError("WKV7 recurrent state handles have incompatible slots")

    def _slot_indices(
        self,
        state_indices: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        if not isinstance(state_indices, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if (
            not state_indices.is_cuda
            or not state_indices.is_contiguous()
            or state_indices.dtype != torch.int32
            or state_indices.ndim != 1
            or state_indices.device != self._state_pool.device
        ):
            raise ValueError(
                f"{name} must be one-dimensional contiguous CUDA int32 on "
                "the state device"
            )
        return state_indices

    def clone_slots(
        self, state_indices: torch.Tensor
    ) -> _TmixWkv7RecurrentState:
        """Clone selected complete slots into a compact checkpoint handle."""

        indices = self._slot_indices(
            state_indices, name="state_indices"
        ).long()
        if indices.numel() == 0:
            raise ValueError("state_indices must select at least one slot")
        cloned = object.__new__(type(self))
        cloned._sequence_capacity = self._sequence_capacity
        cloned._merge_interval = self._merge_interval
        cloned._state_pool = self._state_pool.index_select(0, indices)
        cloned._elapsed_state_pool = (
            None
            if self._elapsed_state_pool is None
            else self._elapsed_state_pool.index_select(0, indices)
        )
        cloned._deltalog_phase_pool = (
            None
            if self._deltalog_phase_pool is None
            else self._deltalog_phase_pool.index_select(0, indices)
        )
        cloned._deltalog_pool = (
            None
            if self._deltalog_pool is None
            else self._deltalog_pool.index_select(2, indices)
        )
        cloned._deltalog_status = (
            None
            if self._deltalog_status is None
            else torch.empty_like(self._deltalog_status)
        )
        return cloned

    def copy_slots_(
        self,
        source: _TmixWkv7RecurrentState,
        source_indices: torch.Tensor,
        destination_indices: torch.Tensor,
    ) -> _TmixWkv7RecurrentState:
        """Copy selected complete slots with simultaneous COW semantics."""

        self._check_compatible_slots(source)
        source_slots = source._slot_indices(
            source_indices, name="source_indices"
        ).long()
        destination_slots = self._slot_indices(
            destination_indices, name="destination_indices"
        ).long()
        if source_slots.numel() != destination_slots.numel():
            raise ValueError(
                "source_indices and destination_indices must have equal length"
            )
        if source_slots.numel() == 0:
            return self
        if torch.unique(destination_slots).numel() != destination_slots.numel():
            raise ValueError("destination_indices must be unique")

        # Gather every source component before the first write.  This preserves
        # prefix-cache COW semantics even when source and destination overlap.
        source_values = [source._state_pool.index_select(0, source_slots)]
        destination_components: list[tuple[torch.Tensor, int]] = [
            (self._state_pool, 0)
        ]
        if self._elapsed_state_pool is not None:
            assert source._elapsed_state_pool is not None
            source_values.append(
                source._elapsed_state_pool.index_select(0, source_slots)
            )
            destination_components.append((self._elapsed_state_pool, 0))
        if self._deltalog_phase_pool is not None:
            assert source._deltalog_phase_pool is not None
            assert self._deltalog_pool is not None
            assert source._deltalog_pool is not None
            source_values.extend(
                (
                    source._deltalog_phase_pool.index_select(0, source_slots),
                    source._deltalog_pool.index_select(2, source_slots),
                )
            )
            destination_components.extend(
                (
                    (self._deltalog_phase_pool, 0),
                    (self._deltalog_pool, 2),
                )
            )
        for (destination, dimension), value in zip(
            destination_components, source_values, strict=True
        ):
            destination.index_copy_(dimension, destination_slots, value)
        return self

    def reset_slots_(
        self, state_indices: torch.Tensor
    ) -> _TmixWkv7RecurrentState:
        """Reset selected complete slots without scanning the state pool."""

        indices = self._slot_indices(
            state_indices, name="state_indices"
        ).long()
        if indices.numel() == 0:
            return self
        self._state_pool.index_fill_(0, indices, 0)
        if self._elapsed_state_pool is not None:
            self._elapsed_state_pool.index_fill_(0, indices, 0)
        if self._deltalog_phase_pool is not None:
            assert self._deltalog_pool is not None
            self._deltalog_phase_pool.index_fill_(0, indices, 0)
            self._deltalog_pool.index_fill_(2, indices, 0)
        return self

    def materialize_slots_(
        self, state_indices: torch.Tensor
    ) -> _TmixWkv7RecurrentState:
        """Merge pending private DeltaLog entries into selected base slots."""

        self._materialize_slots(state_indices, None)
        return self

    def _materialize_slots(
        self,
        state_indices: torch.Tensor,
        metadata_status: torch.Tensor | None,
    ) -> None:
        indices = self._slot_indices(state_indices, name="state_indices")
        if indices.numel() == 0 or self._deltalog_phase_pool is None:
            return
        assert self._deltalog_pool is not None
        assert self._deltalog_status is not None
        extension = _extension()
        if self._state_pool.dtype == torch.float16:
            extension.tmix_wkv7_recurrent_deltalog_fp16_materialize_slots(
                state_indices,
                self._deltalog_phase_pool,
                self._state_pool,
                self._deltalog_pool,
                self._deltalog_status,
                metadata_status,
            )
        else:
            extension.tmix_wkv7_recurrent_deltalog_fp32io16_materialize_slots(
                state_indices,
                self._deltalog_phase_pool,
                self._state_pool,
                self._deltalog_pool,
                self._deltalog_status,
                metadata_status,
            )

    def clone(self) -> _TmixWkv7RecurrentState:
        cloned = object.__new__(type(self))
        cloned._sequence_capacity = self._sequence_capacity
        cloned._merge_interval = self._merge_interval
        cloned._state_pool = self._state_pool.clone()
        cloned._elapsed_state_pool = (
            None
            if self._elapsed_state_pool is None
            else self._elapsed_state_pool.clone()
        )
        cloned._deltalog_phase_pool = (
            None
            if self._deltalog_phase_pool is None
            else self._deltalog_phase_pool.clone()
        )
        cloned._deltalog_pool = (
            None if self._deltalog_pool is None else self._deltalog_pool.clone()
        )
        cloned._deltalog_status = (
            None
            if self._deltalog_status is None
            else torch.empty_like(self._deltalog_status)
        )
        return cloned

    def copy_(
        self, source: _TmixWkv7RecurrentState
    ) -> _TmixWkv7RecurrentState:
        self._check_compatible_slots(source)
        if (
            self._sequence_capacity != source._sequence_capacity
        ):
            raise ValueError("WKV7 recurrent state handles have incompatible policies")
        destination_components = self._components()
        source_components = source._components()
        if tuple(tensor.shape for tensor in destination_components) != tuple(
            tensor.shape for tensor in source_components
        ):
            raise ValueError("WKV7 recurrent state handles have incompatible shapes")
        for destination, value in zip(
            destination_components, source_components, strict=True
        ):
            destination.copy_(value)
        return self

    def zero_(self) -> _TmixWkv7RecurrentState:
        for tensor in self._components():
            tensor.zero_()
        if self._deltalog_status is not None:
            self._deltalog_status.zero_()
        return self


def _prepare_recurrent_state(
    state_pool_size: int,
    channels: int,
    sequence_capacity: int,
    head_size: int,
    device: torch.device | str | int | None,
    state_dtype: torch.dtype,
) -> object:
    for name, value in (
        ("state_pool_size", state_pool_size),
        ("channels", channels),
        ("sequence_capacity", sequence_capacity),
        ("head_size", head_size),
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    if channels % head_size != 0:
        raise ValueError("channels must be divisible by head_size")
    if device is None:
        resolved_device = torch.device("cuda", torch.cuda.current_device())
    elif isinstance(device, int):
        resolved_device = torch.device("cuda", device)
    else:
        resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise ValueError("device must identify a CUDA device")

    numerical_mode = "fp16" if state_dtype == torch.float16 else "fp32io16"
    merge_interval = _select_deltalog_merge_interval(
        channels,
        sequence_capacity,
        head_size,
        tuple(torch.cuda.get_device_capability(resolved_device)),
        numerical_mode,
    )
    state_pool = torch.zeros(
        state_pool_size,
        channels // head_size,
        head_size,
        head_size,
        dtype=state_dtype,
        device=resolved_device,
    )
    elapsed_state_pool = (
        torch.zeros(
            state_pool_size,
            dtype=torch.int32,
            device=resolved_device,
        )
        if state_dtype == torch.float16
        else None
    )
    return _TmixWkv7RecurrentState(
        state_pool,
        elapsed_state_pool,
        sequence_capacity,
        merge_interval,
    )


def prepare_tmix_wkv7_recurrent_fp16_state(
    state_pool_size: int,
    channels: int,
    *,
    sequence_capacity: int,
    head_size: int = 64,
    device: torch.device | str | int | None = None,
) -> object:
    """Allocate one complete opaque FP16 WKV7 state package."""

    return _prepare_recurrent_state(
        state_pool_size,
        channels,
        sequence_capacity,
        head_size,
        device,
        torch.float16,
    )


def prepare_tmix_wkv7_recurrent_fp32io16_state(
    state_pool_size: int,
    channels: int,
    *,
    sequence_capacity: int,
    head_size: int = 64,
    device: torch.device | str | int | None = None,
) -> object:
    """Allocate one complete opaque FP32IO16 WKV7 state package."""

    return _prepare_recurrent_state(
        state_pool_size,
        channels,
        sequence_capacity,
        head_size,
        device,
        torch.float32,
    )


def prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(
    state: torch.Tensor,
) -> object:
    """Bind an opaque materialized FP32IO16 handle to a caller-owned tensor."""

    _validate_state(state)
    return _TmixWkv7RecurrentState(
        state,
        elapsed_state_pool=None,
        sequence_capacity=None,
        merge_interval=0,
    )


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
    if state_indices.ndim != 1:
        raise ValueError("state_indices must be one-dimensional")
    for name, tensor in (("cu_seqlens", cu_seqlens), ("state_indices", state_indices)):
        if tensor.dtype != torch.int32 or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be a contiguous CUDA int32 tensor")
    if cu_seqlens.device != state_indices.device:
        raise ValueError("cu_seqlens and state_indices must share a CUDA device")


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


def _tensor_version(tensor: torch.Tensor) -> int | None:
    try:
        return tensor._version
    except RuntimeError:
        return None


class _RecurrentMetadataTicket:
    def __init__(
        self,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc_snapshot: torch.Tensor,
        state_indices_snapshot: torch.Tensor,
        status: torch.Tensor,
        token_predecessor: torch.Tensor,
        workspace: torch.Tensor,
        num_active_tokens: torch.Tensor | None,
        num_active_sequences: torch.Tensor | None,
        total_tokens: int,
        state_pool_size: int,
        max_seqlen: int,
        graph_mode: bool,
    ) -> None:
        self._query_start_loc = query_start_loc
        self._state_indices = state_indices
        self.__query_start_loc_snapshot = query_start_loc_snapshot
        self.__state_indices_snapshot = state_indices_snapshot
        self.__status = status
        self.__token_predecessor = token_predecessor
        self._workspace = workspace
        self.__num_active_tokens = num_active_tokens
        self.__num_active_sequences = num_active_sequences
        self._query_start_loc_version = _tensor_version(query_start_loc)
        self._state_indices_version = _tensor_version(state_indices)
        self._query_start_loc_data = query_start_loc.data_ptr()
        self._state_indices_data = state_indices.data_ptr()
        self._query_start_loc_shape = tuple(query_start_loc.shape)
        self._state_indices_shape = tuple(state_indices.shape)
        self._query_start_loc_stride = query_start_loc.stride()
        self._state_indices_stride = state_indices.stride()
        self._total_tokens = total_tokens
        self._state_pool_size = state_pool_size
        self.__max_seqlen = max_seqlen
        self.__graph_mode = graph_mode
        self._device = query_start_loc.device
        self._stream = torch.cuda.current_stream(self._device).cuda_stream

    def _check_compatible(
        self,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        total_tokens: int,
        state_pool_size: int,
        max_seqlen: int,
    ) -> None:
        if query_start_loc is not self._query_start_loc:
            raise RuntimeError("validated_metadata query_start_loc identity mismatch")
        if state_indices is not self._state_indices:
            raise RuntimeError("validated_metadata state_indices identity mismatch")
        if (
            query_start_loc.data_ptr() != self._query_start_loc_data
            or state_indices.data_ptr() != self._state_indices_data
        ):
            raise RuntimeError("validated_metadata metadata data_ptr mismatch")
        if (
            tuple(query_start_loc.shape) != self._query_start_loc_shape
            or tuple(state_indices.shape) != self._state_indices_shape
            or query_start_loc.stride() != self._query_start_loc_stride
            or state_indices.stride() != self._state_indices_stride
        ):
            raise RuntimeError("validated_metadata metadata shape or stride mismatch")
        if not self.__graph_mode:
            for tensor, expected, name in (
                (query_start_loc, self._query_start_loc_version, "query_start_loc"),
                (state_indices, self._state_indices_version, "state_indices"),
            ):
                if expected is not None and _tensor_version(tensor) != expected:
                    raise RuntimeError(f"validated_metadata {name} version mismatch")
        if query_start_loc.device != self._device or state_indices.device != self._device:
            raise RuntimeError("validated_metadata device mismatch")
        if total_tokens != self._total_tokens:
            raise RuntimeError("validated_metadata total_tokens mismatch")
        if state_pool_size != self._state_pool_size:
            raise RuntimeError("validated_metadata state_pool_size mismatch")
        if max_seqlen > 0 and max_seqlen != self.__max_seqlen:
            raise RuntimeError("validated_metadata max_seqlen mismatch")
        if torch.cuda.current_stream(self._device).cuda_stream != self._stream:
            raise RuntimeError(
                "validated_metadata stream mismatch; prepare and consume the ticket "
                "on the same CUDA stream"
            )

    def _query_start_loc_snapshot(self) -> torch.Tensor:
        return self.__query_start_loc_snapshot

    def _state_indices_snapshot(self) -> torch.Tensor:
        return self.__state_indices_snapshot

    def _status(self) -> torch.Tensor:
        return self.__status

    def _active_status(self) -> torch.Tensor:
        return self.__status

    def _token_predecessor(self) -> torch.Tensor:
        return self.__token_predecessor

    def _max_seqlen(self) -> int:
        return self.__max_seqlen

    def _is_graph(self) -> bool:
        return self.__graph_mode

    def _num_active_tokens(self) -> torch.Tensor | None:
        return self.__num_active_tokens

    def _num_active_sequences(self) -> torch.Tensor | None:
        return self.__num_active_sequences


def _metadata_launch_args(
    ticket: object,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    total_tokens: int,
    state_pool_size: int,
    max_seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if not isinstance(ticket, _RecurrentMetadataTicket):
        raise TypeError("validated_metadata must come from prepare_tmix_wkv7_recurrent_metadata")
    ticket._check_compatible(
        query_start_loc, state_indices, total_tokens, state_pool_size, max_seqlen
    )
    return (
        ticket._query_start_loc_snapshot(),
        ticket._state_indices_snapshot(),
        ticket._status(),
        ticket._max_seqlen(),
    )


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
        prepared = _extension().prepare_tmix_wkv7_recurrent_graph_metadata(
            cu_seqlens,
            state_indices,
            num_active_tokens,
            num_active_sequences,
            token_capacity,
            sequence_capacity,
            state_pool_size,
            max_seqlen_capacity,
        )
        return _RecurrentMetadataTicket(
            cu_seqlens,
            state_indices,
            *prepared,
            num_active_tokens,
            num_active_sequences,
            token_capacity,
            state_pool_size,
            max_seqlen_capacity,
            True,
        )

    if total_tokens is None:
        raise ValueError("total_tokens is required for static recurrent metadata")
    offsets = cu_seqlens.to(device="cpu", copy=True).tolist()
    slots = state_indices.to(device="cpu", copy=True).tolist()
    if len(offsets) != len(slots) + 1 or not slots:
        raise ValueError("cu_seqlens must have shape [B+1] and state_indices shape [B]")
    if offsets[0] != 0 or offsets[-1] != total_tokens:
        raise ValueError("cu_seqlens endpoints must be 0 and total_tokens")
    if any(start < 0 or end <= start for start, end in pairwise(offsets)):
        raise ValueError(
            "cu_seqlens must be strictly increasing with non-empty sequences"
        )
    if len(set(slots)) != len(slots) or any(
        slot < 0 or slot >= state_pool_size for slot in slots
    ):
        raise ValueError("state_indices must be unique and within the state pool")
    inferred_max_seqlen = max(end - start for start, end in pairwise(offsets))
    resolved_max_seqlen = (
        inferred_max_seqlen if max_seqlen is None else int(max_seqlen)
    )
    if resolved_max_seqlen != inferred_max_seqlen:
        raise ValueError("max_seqlen must equal the largest packed sequence length")
    prepared = _extension().prepare_tmix_wkv7_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens,
        state_pool_size,
    )
    return _RecurrentMetadataTicket(
        cu_seqlens,
        state_indices,
        *prepared,
        None,
        None,
        total_tokens,
        state_pool_size,
        resolved_max_seqlen,
        False,
    )


def _run_fp32io16(
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
    launch_metadata = _metadata_launch_args(
        ticket,
        cu_seqlens,
        state_indices,
        r.shape[0],
        state.shape[0],
        -1 if max_seqlen is None else int(max_seqlen),
    )
    _extension().tmix_wkv7_recurrent_fp32_from_decay_logits(
        launch_metadata[0],
        launch_metadata[1],
        state,
        *packed,
        output,
        float(scale),
        decay_bias,
        launch_metadata[2],
        launch_metadata[3],
    )
    return output


def _materialize_before_ordinary(
    state: _TmixWkv7RecurrentState,
    *,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    total_tokens: int,
    max_seqlen: int | None,
    validated_metadata: object | None,
) -> object | None:
    if state._merge_interval == 0:
        return validated_metadata
    launch_max_seqlen = _validate_launch_max_seqlen(max_seqlen)
    ticket = (
        validated_metadata
        if validated_metadata is not None
        else prepare_tmix_wkv7_recurrent_metadata(
            cu_seqlens,
            state_indices,
            total_tokens=total_tokens,
            state_pool_size=state._state_pool.shape[0],
            max_seqlen=max_seqlen,
        )
    )
    ticket._check_compatible(
        cu_seqlens,
        state_indices,
        total_tokens,
        state._state_pool.shape[0],
        launch_max_seqlen,
    )
    state._materialize_slots(
        ticket._state_indices_snapshot(),
        ticket._status(),
    )
    return ticket


def infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    state: object,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float = 1.0,
    decay_bias: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    validated_metadata: object | None = None,
) -> torch.Tensor:
    """Run the provider-selected FP32IO16 WKV7 implementation."""

    if (
        not isinstance(state, _TmixWkv7RecurrentState)
        or state._state_pool.dtype != torch.float32
        or state._elapsed_state_pool is not None
    ):
        raise TypeError(
            "state must come from "
            "prepare_tmix_wkv7_recurrent_fp32io16_state or "
            "prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor"
        )
    packed = _validate_packed_inputs(r, decay_logits, k, v, a, b)
    _check_metadata_inputs(cu_seqlens, state_indices)
    launch_max_seqlen = _dispatch_max_seqlen(max_seqlen, validated_metadata)
    use_deltalog = (
        state._merge_interval != 0
        and r.dtype == torch.float16
        and r.shape[0] == state._sequence_capacity
        and state_indices.numel() == state._sequence_capacity
        and launch_max_seqlen in {-1, 1}
    )
    if use_deltalog:
        assert state._deltalog_phase_pool is not None
        assert state._deltalog_pool is not None
        assert state._deltalog_status is not None
        return _run_deltalog_fp32io16(
            *packed,
            state_pool=state._state_pool,
            deltalog_phase_pool=state._deltalog_phase_pool,
            deltalog_pool=state._deltalog_pool,
            deltalog_status=state._deltalog_status,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            scale=scale,
            decay_bias=decay_bias,
            validated_metadata=validated_metadata,
        )
    ordinary_metadata = _materialize_before_ordinary(
        state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        total_tokens=r.shape[0],
        max_seqlen=max_seqlen,
        validated_metadata=validated_metadata,
    )
    return _run_fp32io16(
        *packed,
        state=state._state_pool,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        scale=scale,
        decay_bias=decay_bias,
        max_seqlen=max_seqlen,
        validated_metadata=ordinary_metadata,
    )


def _validate_launch_max_seqlen(max_seqlen: int | None) -> int:
    if max_seqlen is None:
        return -1
    if (
        not isinstance(max_seqlen, int)
        or isinstance(max_seqlen, bool)
        or max_seqlen <= 0
    ):
        raise ValueError("max_seqlen must be a positive integer")
    return int(max_seqlen)


def _dispatch_max_seqlen(
    max_seqlen: int | None,
    validated_metadata: object | None,
) -> int:
    launch_max_seqlen = _validate_launch_max_seqlen(max_seqlen)
    if validated_metadata is not None:
        ticket_max_seqlen = int(validated_metadata._max_seqlen())
        if launch_max_seqlen < 0:
            return ticket_max_seqlen
        if launch_max_seqlen != ticket_max_seqlen:
            return 0
    return launch_max_seqlen


def infer_tmix_wkv7_recurrent_fp16_forward_varlen(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    state: object,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float = 1.0,
    decay_bias: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    validated_metadata: object | None = None,
) -> torch.Tensor:
    """Run the provider-selected FP16 WKV7 implementation."""

    if (
        not isinstance(state, _TmixWkv7RecurrentState)
        or state._state_pool.dtype != torch.float16
        or state._elapsed_state_pool is None
    ):
        raise TypeError(
            "state must come from prepare_tmix_wkv7_recurrent_fp16_state"
        )

    packed = _validate_packed_inputs(r, decay_logits, k, v, a, b)
    if any(tensor.dtype != torch.float16 for tensor in packed):
        raise TypeError("FP16-state token tensors must have dtype float16")
    _check_metadata_inputs(cu_seqlens, state_indices)
    launch_max_seqlen = _dispatch_max_seqlen(max_seqlen, validated_metadata)
    use_deltalog = (
        state._merge_interval != 0
        and r.shape[0] == state._sequence_capacity
        and state_indices.numel() == state._sequence_capacity
        and launch_max_seqlen in {-1, 1}
    )
    if use_deltalog:
        assert state._deltalog_phase_pool is not None
        assert state._deltalog_pool is not None
        assert state._deltalog_status is not None
        return _run_deltalog_fp16(
            *packed,
            state_pool=state._state_pool,
            elapsed_state_pool=state._elapsed_state_pool,
            deltalog_phase_pool=state._deltalog_phase_pool,
            deltalog_pool=state._deltalog_pool,
            deltalog_status=state._deltalog_status,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            scale=scale,
            decay_bias=decay_bias,
            validated_metadata=validated_metadata,
        )
    ordinary_metadata = _materialize_before_ordinary(
        state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        total_tokens=r.shape[0],
        max_seqlen=max_seqlen,
        validated_metadata=validated_metadata,
    )
    return _run_fp16(
        *packed,
        state_pool=state._state_pool,
        elapsed_state_pool=state._elapsed_state_pool,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        scale=scale,
        decay_bias=decay_bias,
        max_seqlen=max_seqlen,
        validated_metadata=ordinary_metadata,
    )


def _run_fp16(
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
    """Run the private ordinary FP16 implementation."""

    packed = _validate_packed_inputs(r, decay_logits, k, v, a, b)
    _validate_fp16_state(state_pool)
    _validate_fp16_elapsed_state(elapsed_state_pool, state_pool)
    _check_metadata_inputs(cu_seqlens, state_indices)
    launch_max_seqlen = _validate_launch_max_seqlen(max_seqlen)
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
    launch_metadata = _metadata_launch_args(
        ticket,
        cu_seqlens,
        state_indices,
        r.shape[0],
        state_pool.shape[0],
        launch_max_seqlen,
    )
    _extension().tmix_wkv7_recurrent_fp16_from_decay_logits(
        launch_metadata[0],
        launch_metadata[1],
        elapsed_state_pool,
        state_pool,
        *packed,
        output,
        float(scale),
        decay_bias,
        launch_metadata[2],
        launch_metadata[3],
    )
    return output


def _run_deltalog_fp16(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    elapsed_state_pool: torch.Tensor,
    deltalog_phase_pool: torch.Tensor,
    deltalog_pool: torch.Tensor,
    deltalog_status: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float = 1.0,
    decay_bias: torch.Tensor | None = None,
    validated_metadata: object | None = None,
) -> torch.Tensor:
    """Run the private slot-native DeltaLog implementation."""

    packed = _validate_packed_inputs(r, decay_logits, k, v, a, b)
    _validate_fp16_deltalog_state_bundle(
        state_pool,
        elapsed_state_pool,
        deltalog_phase_pool,
        deltalog_pool,
    )
    _check_metadata_inputs(cu_seqlens, state_indices)
    ticket = (
        validated_metadata
        if validated_metadata is not None
        else prepare_tmix_wkv7_recurrent_metadata(
            cu_seqlens,
            state_indices,
            total_tokens=r.shape[0],
            state_pool_size=state_pool.shape[0],
        )
    )
    output = torch.empty_like(packed[3])
    launch_metadata = _metadata_launch_args(
        ticket, cu_seqlens, state_indices, r.shape[0], state_pool.shape[0], -1
    )
    _extension().tmix_wkv7_recurrent_deltalog_fp16_from_decay_logits(
        launch_metadata[0],
        launch_metadata[1],
        elapsed_state_pool,
        deltalog_phase_pool,
        state_pool,
        deltalog_pool,
        *packed,
        output,
        float(scale),
        decay_bias,
        launch_metadata[2],
        deltalog_status,
    )
    return output


def _run_deltalog_fp32io16(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    deltalog_phase_pool: torch.Tensor,
    deltalog_pool: torch.Tensor,
    deltalog_status: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float = 1.0,
    decay_bias: torch.Tensor | None = None,
    validated_metadata: object | None = None,
) -> torch.Tensor:
    """Run the private FP32IO16 DeltaLog implementation."""

    packed = _validate_packed_inputs(r, decay_logits, k, v, a, b)
    _validate_fp32io16_deltalog_state_bundle(
        state_pool,
        deltalog_phase_pool,
        deltalog_pool,
    )
    _check_metadata_inputs(cu_seqlens, state_indices)
    ticket = (
        validated_metadata
        if validated_metadata is not None
        else prepare_tmix_wkv7_recurrent_metadata(
            cu_seqlens,
            state_indices,
            total_tokens=r.shape[0],
            state_pool_size=state_pool.shape[0],
        )
    )
    output = torch.empty_like(packed[3])
    launch_metadata = _metadata_launch_args(
        ticket, cu_seqlens, state_indices, r.shape[0], state_pool.shape[0], -1
    )
    _extension().tmix_wkv7_recurrent_deltalog_fp32io16_from_decay_logits(
        launch_metadata[0],
        launch_metadata[1],
        deltalog_phase_pool,
        state_pool,
        deltalog_pool,
        *packed,
        output,
        float(scale),
        decay_bias,
        launch_metadata[2],
        deltalog_status,
    )
    return output


from .chunk import infer_tmix_wkv7_chunk_bf16_forward_varlen
from .pretrain import pretrain_tmix_wkv7_recurrent_bf16
from .rl_infctx import (
    rl_infctx_tmix_wkv7_chunk_fp32io16,
    rl_infctx_tmix_wkv7_chunk_fp32io16_factor_recompute,
)

__all__ = [
    "infer_tmix_wkv7_chunk_bf16_forward_varlen",
    "infer_tmix_wkv7_recurrent_fp16_forward_varlen",
    "infer_tmix_wkv7_recurrent_fp32io16_forward_varlen",
    "prepare_tmix_wkv7_recurrent_fp16_state",
    "prepare_tmix_wkv7_recurrent_fp32io16_state",
    "prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor",
    "prepare_tmix_wkv7_recurrent_metadata",
    "pretrain_tmix_wkv7_recurrent_bf16",
    "rl_infctx_tmix_wkv7_chunk_fp32io16",
    "rl_infctx_tmix_wkv7_chunk_fp32io16_factor_recompute",
]
