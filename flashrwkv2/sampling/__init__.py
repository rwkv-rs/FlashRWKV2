# SPDX-License-Identifier: MIT

from __future__ import annotations

from numbers import Real

import torch


def _extension():
    import flashrwkv2

    extension = getattr(flashrwkv2, "_C", None)
    if extension is None:
        raise RuntimeError(
            "FlashRWKV2 CUDA extension is not built; build flashrwkv2._C before "
            "using the sampling operators"
        )
    return extension


def _scalar(value: object, name: str) -> int | float | None:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, not bool")
    if isinstance(value, Real):
        return value
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a scalar or torch.Tensor")
    return None


def _parameter_vector(
    value: float | torch.Tensor,
    *,
    name: str,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, not bool")
    if isinstance(value, Real):
        return torch.full((batch_size,), value, dtype=dtype, device=device)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a scalar or torch.Tensor")
    if value.shape != (batch_size,):
        raise ValueError(f"{name} must have shape [{batch_size}]")
    if value.device != device or not value.is_cuda:
        raise ValueError(f"{name} must be on {device}")
    if value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return value


def _validate_hot_inputs(
    logits: torch.Tensor,
    states: torch.Tensor,
    slot_indices: torch.Tensor,
    penalties: torch.Tensor | None = None,
) -> None:
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if logits.dtype != torch.float32 or not logits.is_cuda or not logits.is_contiguous():
        raise ValueError("logits must be contiguous CUDA float32")
    if logits.ndim != 2 or logits.shape[0] <= 0:
        raise ValueError("logits must have compact request shape [B,V]")
    vocab_size = logits.shape[1]
    if vocab_size <= 0 or vocab_size > 1_048_576 or vocab_size % 4:
        raise ValueError(
            "vocabulary size must be positive, divisible by 4, and at most 1048576"
        )
    if not isinstance(states, torch.Tensor):
        raise TypeError("states must be a torch.Tensor")
    if states.dtype != torch.int8 or not states.is_cuda or not states.is_contiguous():
        raise ValueError("states must be contiguous CUDA int8")
    if states.ndim != 2 or states.shape[0] <= 0:
        raise ValueError("states must come from setup_sampling_states")
    if (
        not isinstance(slot_indices, torch.Tensor)
        or slot_indices.dtype != torch.int32
        or not slot_indices.is_cuda
        or not slot_indices.is_contiguous()
        or slot_indices.shape != (logits.shape[0],)
    ):
        raise ValueError("slot_indices must be contiguous CUDA int32 with shape [B]")
    if states.device != logits.device or slot_indices.device != logits.device:
        raise ValueError("states and slot_indices must share logits' CUDA device")
    if penalties is not None and (
        penalties.dtype != torch.float32
        or not penalties.is_cuda
        or not penalties.is_contiguous()
        or penalties.shape != (states.shape[0], vocab_size)
        or penalties.device != logits.device
    ):
        raise ValueError("penalties must be contiguous CUDA float32 [num_slots,V]")


def _validate_activity(
    logits: torch.Tensor,
    sample_capacity: int | None,
    num_active_samples: torch.Tensor | None,
) -> tuple[int, torch.Tensor | None]:
    if sample_capacity is None and num_active_samples is None:
        return -1, None
    if sample_capacity is None or num_active_samples is None:
        raise ValueError(
            "sample_capacity and num_active_samples must be provided together"
        )
    if (
        not isinstance(sample_capacity, int)
        or isinstance(sample_capacity, bool)
        or sample_capacity <= 0
        or sample_capacity != logits.shape[0]
    ):
        raise ValueError("sample_capacity must equal the positive logits row capacity")
    if (
        not isinstance(num_active_samples, torch.Tensor)
        or num_active_samples.dtype != torch.int32
        or not num_active_samples.is_cuda
        or not num_active_samples.is_contiguous()
        or num_active_samples.numel() != 1
        or num_active_samples.device != logits.device
    ):
        raise ValueError(
            "num_active_samples must be a one-element contiguous CUDA int32 tensor"
        )
    return sample_capacity, num_active_samples


def setup_sampling_states(seed: int, num_slots: int) -> torch.Tensor:
    """Create an explicit persistent Philox state pool for request slots."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an int")
    if not isinstance(num_slots, int) or isinstance(num_slots, bool) or num_slots <= 0:
        raise ValueError("num_slots must be a positive int")
    return _extension().setup_sampling_states(seed, num_slots)


def infer_sampling_temperature_topk_topp_forward_varlen(
    logits: torch.Tensor,
    states: torch.Tensor,
    slot_indices: torch.Tensor,
    *,
    temperature: float | torch.Tensor = 1.0,
    top_k: int | torch.Tensor = -1,
    top_p: float | torch.Tensor = 1.0,
    sample_capacity: int | None = None,
    num_active_samples: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample one compact logits row per output-producing request.

    ``slot_indices`` is validated on device. With ``sample_capacity`` and a
    live ``num_active_samples`` scalar, inactive rows emit ``-1`` without
    reading logits or advancing their request RNG.
    """

    _validate_hot_inputs(logits, states, slot_indices)
    launch_capacity, active_samples = _validate_activity(
        logits, sample_capacity, num_active_samples
    )
    scalar_temperature = _scalar(temperature, "temperature")
    scalar_top_k = _scalar(top_k, "top_k")
    scalar_top_p = _scalar(top_p, "top_p")
    if None not in (scalar_temperature, scalar_top_k, scalar_top_p):
        return _extension().sampling_temperature_topk_topp_forward_varlen(
            logits,
            states,
            slot_indices,
            float(scalar_temperature),
            int(scalar_top_k),
            float(scalar_top_p),
            launch_capacity,
            active_samples,
        )
    batch_size = logits.shape[0]
    return _extension().sampling_temperature_topk_topp_forward_varlen(
        logits,
        states,
        slot_indices,
        _parameter_vector(
            temperature,
            name="temperature",
            batch_size=batch_size,
            dtype=torch.float32,
            device=logits.device,
        ),
        _parameter_vector(
            top_k,
            name="top_k",
            batch_size=batch_size,
            dtype=torch.int32,
            device=logits.device,
        ),
        _parameter_vector(
            top_p,
            name="top_p",
            batch_size=batch_size,
            dtype=torch.float32,
            device=logits.device,
        ),
        launch_capacity,
        active_samples,
    )


def infer_sampling_six_parameter_forward_varlen(
    logits: torch.Tensor,
    penalties: torch.Tensor,
    states: torch.Tensor,
    slot_indices: torch.Tensor,
    *,
    presence_penalty: float | torch.Tensor = 0.0,
    frequency_penalty: float | torch.Tensor = 0.0,
    penalty_decay: float | torch.Tensor = 0.996,
    temperature: float | torch.Tensor = 1.0,
    top_k: int | torch.Tensor = -1,
    top_p: float | torch.Tensor = 1.0,
    sample_capacity: int | None = None,
    num_active_samples: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample with six controls and update per-slot additive penalties.

    ``frequency_penalty`` is the additive increment called
    ``repetition_penalty`` by Rapid-Sampling; it is not a multiplicative
    repetition penalty. ``slot_indices`` must contain unique in-range slots.
    """

    _validate_hot_inputs(logits, states, slot_indices, penalties)
    launch_capacity, active_samples = _validate_activity(
        logits, sample_capacity, num_active_samples
    )
    values = (
        presence_penalty,
        frequency_penalty,
        penalty_decay,
        temperature,
        top_k,
        top_p,
    )
    names = (
        "presence_penalty",
        "frequency_penalty",
        "penalty_decay",
        "temperature",
        "top_k",
        "top_p",
    )
    scalars = tuple(_scalar(value, name) for value, name in zip(values, names, strict=True))
    if all(value is not None for value in scalars):
        return _extension().sampling_six_parameter_forward_varlen(
            logits,
            penalties,
            states,
            slot_indices,
            float(scalars[0]),
            float(scalars[1]),
            float(scalars[2]),
            float(scalars[3]),
            int(scalars[4]),
            float(scalars[5]),
            launch_capacity,
            active_samples,
        )

    batch_size = logits.shape[0]
    vectors = tuple(
        _parameter_vector(
            value,
            name=name,
            batch_size=batch_size,
            dtype=torch.int32 if name == "top_k" else torch.float32,
            device=logits.device,
        )
        for value, name in zip(values, names, strict=True)
    )
    return _extension().sampling_six_parameter_forward_varlen(
        logits,
        penalties,
        states,
        slot_indices,
        *vectors,
        launch_capacity,
        active_samples,
    )


__all__ = [
    "infer_sampling_six_parameter_forward_varlen",
    "infer_sampling_temperature_topk_topp_forward_varlen",
    "setup_sampling_states",
]
