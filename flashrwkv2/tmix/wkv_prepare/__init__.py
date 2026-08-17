# SPDX-License-Identifier: MIT

from __future__ import annotations

from numbers import Real

import torch

from ..wkv7 import _extension


def _check_packed(reference: torch.Tensor, tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != torch.float16:
        raise TypeError(f"{name} must have dtype torch.float16")
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous CUDA float16")
    if tensor.device != reference.device or tensor.shape != reference.shape:
        raise ValueError(f"{name} must match x_r's packed shape and device")


def _check_lora(
    source: torch.Tensor,
    output_features: int,
    a: torch.Tensor | None,
    b: torch.Tensor | None,
    scale: float,
    name: str,
) -> float:
    if (a is None) != (b is None):
        raise ValueError(f"{name}_lora_a and {name}_lora_b must be provided together")
    if isinstance(scale, bool) or not isinstance(scale, Real):
        raise TypeError(f"{name}_lora_scale must be a real number")
    normalized = float(scale)
    if a is None or b is None:
        return normalized
    for suffix, tensor in (("a", a), ("b", b)):
        if tensor.dtype != torch.float16 or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name}_lora_{suffix} must be contiguous CUDA float16")
        if tensor.device != source.device or tensor.ndim != 2:
            raise ValueError(f"{name}_lora_{suffix} has an invalid device or rank")
    rank = a.shape[0]
    if rank > 512 or a.shape[1] != source.shape[1] or b.shape != (output_features, rank):
        raise ValueError(f"{name} LoRA shapes are incompatible with the projection")
    return normalized


def infer_tmix_wkv_prepare_forward_varlen(
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
    receptance_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    w1: torch.Tensor | None,
    a1: torch.Tensor | None,
    g1: torch.Tensor | None,
    v1: torch.Tensor | None,
    w2: torch.Tensor | None,
    a2: torch.Tensor | None,
    g2: torch.Tensor | None,
    v2: torch.Tensor | None,
    v0: torch.Tensor,
    k_k: torch.Tensor,
    a0: torch.Tensor,
    k_a: torch.Tensor,
    *,
    v_first: torch.Tensor | None = None,
    w1_runtime: torch.Tensor | None = None,
    a1_runtime: torch.Tensor | None = None,
    g1_runtime: torch.Tensor | None = None,
    v1_runtime: torch.Tensor | None = None,
    w2_runtime: torch.Tensor | None = None,
    a2_runtime: torch.Tensor | None = None,
    g2_runtime: torch.Tensor | None = None,
    v2_runtime: torch.Tensor | None = None,
    receptance_lora_a: torch.Tensor | None = None,
    receptance_lora_b: torch.Tensor | None = None,
    receptance_lora_scale: float = 1.0,
    key_lora_a: torch.Tensor | None = None,
    key_lora_b: torch.Tensor | None = None,
    key_lora_scale: float = 1.0,
    value_lora_a: torch.Tensor | None = None,
    value_lora_b: torch.Tensor | None = None,
    value_lora_scale: float = 1.0,
    head_size: int = 64,
    batch_size: int = 1,
    max_seqlen: int | None = None,
) -> tuple[torch.Tensor, ...]:
    """Prepare every recurrent input and the readout gate for one TMix layer."""

    if x_r.ndim != 2 or x_r.shape[0] <= 0 or x_r.shape[1] <= 0:
        raise ValueError("x_r must have packed shape [total_tokens,C]")
    _check_packed(x_r, x_r, "x_r")
    for name, tensor in (("x_w", x_w), ("x_k", x_k), ("x_v", x_v), ("x_a", x_a), ("x_g", x_g)):
        _check_packed(x_r, tensor, name)
    if head_size not in {64, 128, 256} or x_r.shape[1] % head_size:
        raise ValueError("head_size must be 64, 128, or 256 and divide C")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if max_seqlen is None:
        max_seqlen = x_r.shape[0]
    if not isinstance(max_seqlen, int) or isinstance(max_seqlen, bool) or max_seqlen <= 0:
        raise ValueError("max_seqlen must be a positive integer")
    projection_specs = (
        ("receptance", x_r, receptance_weight, receptance_lora_a, receptance_lora_b, receptance_lora_scale),
        ("key", x_k, key_weight, key_lora_a, key_lora_b, key_lora_scale),
        ("value", x_v, value_weight, value_lora_a, value_lora_b, value_lora_scale),
    )
    scales: list[float] = []
    for name, source, weight, lora_a, lora_b, scale in projection_specs:
        if weight.dtype != torch.float16 or not weight.is_cuda or not weight.is_contiguous():
            raise ValueError(f"{name}_weight must be contiguous CUDA float16")
        if weight.device != x_r.device or weight.shape != (x_r.shape[1], x_r.shape[1]):
            raise ValueError(f"{name}_weight must have shape [C,C] on x_r's device")
        scales.append(_check_lora(source, x_r.shape[1], lora_a, lora_b, scale, name))
    if v_first is not None:
        _check_packed(x_r, v_first, "v_first")
    return tuple(
        _extension().tmix_wkv_prepare_forward_varlen(
            x_r, x_w, x_k, x_v, x_a, x_g,
            receptance_weight, key_weight, value_weight,
            w1, a1, g1, v1, w2, a2, g2, v2, v0, k_k, a0, k_a,
            v_first,
            w1_runtime, a1_runtime, g1_runtime, v1_runtime,
            w2_runtime, a2_runtime, g2_runtime, v2_runtime,
            receptance_lora_a, receptance_lora_b, scales[0],
            key_lora_a, key_lora_b, scales[1],
            value_lora_a, value_lora_b, scales[2],
            int(head_size), int(batch_size), int(max_seqlen),
        )
    )


__all__ = ["infer_tmix_wkv_prepare_forward_varlen"]
