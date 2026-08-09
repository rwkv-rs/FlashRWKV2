# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from numbers import Real

import torch

from ..wkv7 import _extension

_MAX_GRID_DIM_YZ = 65535


def _check_rows(tensor: torch.Tensor, name: str, *, rows: int | None = None) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != torch.float16:
        raise TypeError(f"{name} must have dtype torch.float16")
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"{name} must be CUDA and contiguous")
    if tensor.ndim != 2 or tensor.shape[0] <= 0:
        raise ValueError(f"{name} must have packed shape [total_tokens,features]")
    if rows is not None and tensor.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} packed rows")


def infer_tmix_linear_forward_varlen(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    weight_is_transposed: bool = False,
) -> torch.Tensor:
    """Packed Albatross ordinary TMix linear caller path."""

    _check_rows(x, "x")
    _check_rows(weight, "weight")
    if weight.device != x.device:
        raise ValueError("weight must share x's device")
    if weight_is_transposed:
        if weight.shape[0] != x.shape[1]:
            raise ValueError("transposed weight first dimension must match x")
    elif weight.shape[1] != x.shape[1]:
        raise ValueError("weight second dimension must match x")
    return _extension().tmix_linear_forward_varlen(
        x, weight, bool(weight_is_transposed)
    )


def infer_tmix_linear_attention_c2c_forward_varlen(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    lora_a: torch.Tensor | None = None,
    lora_b: torch.Tensor | None = None,
    lora_scale: float = 1.0,
) -> torch.Tensor:
    """Packed Albatross attention C2C projection with optional vanilla LoRA.

    When both LoRA tensors are supplied, computes
    ``x @ weight.T + lora_scale * (x @ lora_a.T) @ lora_b.T``.  ``lora_a``
    uses PEFT layout ``[rank,K]`` and ``lora_b`` uses ``[N,rank]``.
    """

    _check_rows(x, "x")
    _check_rows(weight, "weight")
    if weight.device != x.device or weight.shape[1] != x.shape[1]:
        raise ValueError("weight must have shape [N,K] and share x's device")
    if (lora_a is None) != (lora_b is None):
        raise ValueError("lora_a and lora_b must be provided together")
    if isinstance(lora_scale, bool) or not isinstance(lora_scale, Real):
        raise TypeError("lora_scale must be a finite real number")
    normalized_scale = float(lora_scale)
    if (
        not math.isfinite(normalized_scale)
        or abs(normalized_scale) > 3.4028234663852886e38
    ):
        raise ValueError("lora_scale must be finite and representable as float32")
    if lora_a is not None and lora_b is not None:
        _check_rows(lora_a, "lora_a")
        _check_rows(lora_b, "lora_b")
        rank = lora_a.shape[0]
        if lora_a.device != x.device or lora_a.shape[1] != x.shape[1]:
            raise ValueError("lora_a must have shape [R,K] and share x's device")
        if rank > 512:
            raise ValueError("LoRA projection requires R<=512")
        if lora_b.device != x.device or lora_b.shape != (weight.shape[0], rank):
            raise ValueError("lora_b must have shape [N,R] and share x's device")

    return _extension().tmix_linear_attention_c2c_forward_varlen(
        x, weight, lora_a, lora_b, normalized_scale
    )


def infer_tmix_linear_ffn_key_forward_varlen(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Packed Albatross FFN key original-layout linear caller path."""

    _check_rows(x, "x")
    _check_rows(weight, "weight")
    if weight.device != x.device or weight.shape[1] != x.shape[1]:
        raise ValueError("weight must have shape [N,C] and share x's device")
    return _extension().tmix_linear_ffn_key_forward_varlen(x, weight)


def _check_linear_t(x: torch.Tensor, weight_t: torch.Tensor) -> None:
    _check_rows(x, "x")
    _check_rows(weight_t, "weight_t")
    if weight_t.device != x.device or weight_t.shape[1] != x.shape[1]:
        raise ValueError("weight_t must have shape [N,K]")
    if x.shape[0] > _MAX_GRID_DIM_YZ:
        raise ValueError(
            f"tmix linear_t supports at most {_MAX_GRID_DIM_YZ} packed rows because "
            f"M maps to CUDA grid.y; got M={x.shape[0]}"
        )


def infer_tmix_linear_t_forward_varlen(
    x: torch.Tensor, weight_t: torch.Tensor
) -> torch.Tensor:
    """Packed Albatross ``linear_t_f16`` caller path."""

    _check_linear_t(x, weight_t)
    return _extension().tmix_linear_t_forward_varlen(x, weight_t)


def infer_tmix_linear_t_tanh_forward_varlen(
    x: torch.Tensor, weight_t: torch.Tensor
) -> torch.Tensor:
    """Packed Albatross ``linear_t_act_f16`` tanh caller path."""

    _check_linear_t(x, weight_t)
    return _extension().tmix_linear_t_tanh_forward_varlen(x, weight_t)


def infer_tmix_linear_t_sigmoid_forward_varlen(
    x: torch.Tensor, weight_t: torch.Tensor
) -> torch.Tensor:
    """Packed Albatross ``linear_t_act_f16`` sigmoid caller path."""

    _check_linear_t(x, weight_t)
    return _extension().tmix_linear_t_sigmoid_forward_varlen(x, weight_t)


def infer_tmix_linear_act_tanh_forward_varlen(x: torch.Tensor) -> torch.Tensor:
    """Packed Albatross ``act_tanh`` caller helper for large-rank paths."""

    _check_rows(x, "x")
    if x.numel() % 2:
        raise ValueError("x must contain an even number of elements")
    return _extension().tmix_linear_act_tanh_forward_varlen(x)


def infer_tmix_linear_act_sigmoid_forward_varlen(x: torch.Tensor) -> torch.Tensor:
    """Packed Albatross ``act_sigmoid`` caller helper for large-rank paths."""

    _check_rows(x, "x")
    if x.numel() % 2:
        raise ValueError("x must contain an even number of elements")
    return _extension().tmix_linear_act_sigmoid_forward_varlen(x)


def infer_tmix_linear_t_vres_forward_varlen(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    v: torch.Tensor,
    v_first: torch.Tensor,
    v0: torch.Tensor,
) -> torch.Tensor:
    """Packed Albatross ``linear_t_vres_f16`` caller path."""

    _check_linear_t(x, weight_t)
    for name, tensor in (("v", v), ("v_first", v_first)):
        _check_rows(tensor, name, rows=x.shape[0])
        if tensor.device != x.device or tensor.shape[1] != weight_t.shape[0]:
            raise ValueError(f"{name} must have shape [total_tokens,N]")
    if (
        v0.dtype != torch.float16
        or not v0.is_cuda
        or not v0.is_contiguous()
        or v0.device != x.device
        or v0.shape != (weight_t.shape[0],)
    ):
        raise ValueError("v0 must be contiguous CUDA float16 with shape [N]")
    return _extension().tmix_linear_t_vres_forward_varlen(x, weight_t, v, v_first, v0)


def _check_rank_source(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    weight_t: torch.Tensor | None,
    *,
    input_projection: bool,
    enforce_lowrank_limit: bool = False,
) -> None:
    if weight is None and weight_t is None:
        raise ValueError("one of weight or weight_t must be provided")
    if weight is not None:
        if (
            weight.dtype != torch.float16
            or not weight.is_cuda
            or not weight.is_contiguous()
        ):
            raise ValueError("weight must be contiguous CUDA float16")
        if (
            weight.device != x.device
            or weight.ndim != 2
            or weight.shape[0] != x.shape[1]
        ):
            raise ValueError(
                "weight must have shape [input,rank] for rank-in or [rank,output] for rank-out"
            )
    if weight_t is not None:
        if (
            weight_t.dtype != torch.float16
            or not weight_t.is_cuda
            or not weight_t.is_contiguous()
        ):
            raise ValueError("weight_t must be contiguous CUDA float16")
        if (
            weight_t.device != x.device
            or weight_t.ndim != 2
            or weight_t.shape[1] != x.shape[1]
        ):
            raise ValueError(
                "weight_t must have shape [rank,input] for rank-in or [output,rank] for rank-out"
            )
    rank = (
        (weight.shape[1] if input_projection else weight.shape[0])
        if weight is not None
        else (weight_t.shape[0] if input_projection else weight_t.shape[1])
    )
    if enforce_lowrank_limit and rank > 512:
        raise ValueError("low-rank projection requires R<=512")
    if (
        weight is not None
        and weight_t is not None
        and weight.shape[1] != weight_t.shape[0]
    ):
        raise ValueError("weight and weight_t must describe the same rank projection")


def infer_tmix_linear_rank_in_forward_varlen(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    weight_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Canonical Albatross ``linear_rank_in`` caller dispatch.

    ``weight`` is the runtime [C,rank] layout and ``weight_t`` is the
    original [rank,C] layout.  The upstream ``linear_t_f16`` family is used
    for the M<=7 window; larger packed batches use the exact Albatross
    ``linear_f16``/``linear_f16_orig`` body and the canonical C=4096 table.
    """

    _check_rows(x, "x")
    _check_rank_source(x, weight, weight_t, input_projection=True)
    if weight_t is not None and x.shape[0] <= 7:
        return infer_tmix_linear_t_forward_varlen(x, weight_t)
    return _extension().tmix_linear_rank_in_dispatch_forward_varlen(x, weight, weight_t)


def infer_tmix_linear_rank_out_forward_varlen(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    weight_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Canonical Albatross ``linear_rank_out`` caller dispatch."""

    _check_rows(x, "x")
    _check_rank_source(x, weight, weight_t, input_projection=False)
    output_channels = weight_t.shape[0] if weight_t is not None else weight.shape[1]
    if weight_t is not None and output_channels >= 1024 and x.shape[0] <= 4:
        return infer_tmix_linear_t_forward_varlen(x, weight_t)
    return _extension().tmix_linear_rank_out_dispatch_forward_varlen(
        x, weight, weight_t
    )


def infer_tmix_linear_rank_out_tanh_forward_varlen(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    weight_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Canonical Albatross ``linear_rank_out_act(..., tanh)`` dispatch."""

    _check_rows(x, "x")
    _check_rank_source(x, weight, weight_t, input_projection=False)
    output_channels = weight_t.shape[0] if weight_t is not None else weight.shape[1]
    if weight_t is not None and output_channels >= 1024 and x.shape[0] <= 4:
        return infer_tmix_linear_t_tanh_forward_varlen(x, weight_t)
    return infer_tmix_linear_rank_out_forward_varlen(
        infer_tmix_linear_act_tanh_forward_varlen(x), weight, weight_t
    )


def infer_tmix_linear_rank_out_sigmoid_forward_varlen(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    weight_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Canonical Albatross ``linear_rank_out_act(..., sigmoid)`` dispatch."""

    _check_rows(x, "x")
    _check_rank_source(x, weight, weight_t, input_projection=False)
    output_channels = weight_t.shape[0] if weight_t is not None else weight.shape[1]
    if weight_t is not None and output_channels >= 1024 and x.shape[0] <= 4:
        return infer_tmix_linear_t_sigmoid_forward_varlen(x, weight_t)
    return infer_tmix_linear_rank_out_forward_varlen(
        infer_tmix_linear_act_sigmoid_forward_varlen(x), weight, weight_t
    )


def infer_tmix_lowrank_in_forward_varlen(
    x_w: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
    w1: torch.Tensor | None,
    a1: torch.Tensor | None,
    g1: torch.Tensor | None,
    *,
    w1_runtime: torch.Tensor | None = None,
    a1_runtime: torch.Tensor | None = None,
    g1_runtime: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Canonical W/A/G rank-in dispatch over packed rows.

    The positional weights keep the existing original layout ``[rank,C]``.
    Optional runtime weights use ``[C,rank]``.  Callers should prepare both
    layouts once when they want every Albatross tuned-table winner available.
    """

    rows = x_w.shape[0]
    for name, tensor in (("x_w", x_w), ("x_a", x_a), ("x_g", x_g)):
        _check_rows(tensor, name, rows=rows)
        if tensor.device != x_w.device:
            raise ValueError(f"{name} must share x_w's device")
    for source, original, runtime in (
        (x_w, w1, w1_runtime),
        (x_a, a1, a1_runtime),
        (x_g, g1, g1_runtime),
    ):
        _check_rank_source(
            source,
            runtime,
            original,
            input_projection=True,
            enforce_lowrank_limit=True,
        )
    return tuple(
        _extension().tmix_lowrank_in_forward_varlen(
            x_w, x_a, x_g, w1, a1, g1, w1_runtime, a1_runtime, g1_runtime
        )
    )


def infer_tmix_lowrank_wagv_in_forward_varlen(
    x_w: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
    x_v: torch.Tensor,
    w1: torch.Tensor | None,
    a1: torch.Tensor | None,
    g1: torch.Tensor | None,
    v1: torch.Tensor | None,
    *,
    w1_runtime: torch.Tensor | None = None,
    a1_runtime: torch.Tensor | None = None,
    g1_runtime: torch.Tensor | None = None,
    v1_runtime: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Canonical W/A/G/V rank-in dispatch over packed rows."""

    rows = x_w.shape[0]
    for name, tensor in (
        ("x_w", x_w),
        ("x_a", x_a),
        ("x_g", x_g),
        ("x_v", x_v),
    ):
        _check_rows(tensor, name, rows=rows)
        if tensor.device != x_w.device or tensor.shape[1] != x_w.shape[1]:
            raise ValueError(f"{name} must match x_w's packed shape and device")
    for source, original, runtime in (
        (x_w, w1, w1_runtime),
        (x_a, a1, a1_runtime),
        (x_g, g1, g1_runtime),
        (x_v, v1, v1_runtime),
    ):
        _check_rank_source(
            source,
            runtime,
            original,
            input_projection=True,
            enforce_lowrank_limit=True,
        )
    return tuple(
        _extension().tmix_lowrank_wagv_in_forward_varlen(
            x_w,
            x_a,
            x_g,
            x_v,
            w1,
            a1,
            g1,
            v1,
            w1_runtime,
            a1_runtime,
            g1_runtime,
            v1_runtime,
        )
    )


def infer_tmix_lowrank_out_forward_varlen(
    w1: torch.Tensor,
    a1: torch.Tensor,
    g1: torch.Tensor,
    w2: torch.Tensor | None,
    a2: torch.Tensor | None,
    g2: torch.Tensor | None,
    *,
    w2_runtime: torch.Tensor | None = None,
    a2_runtime: torch.Tensor | None = None,
    g2_runtime: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Canonical W/A/G rank-out dispatch over packed rows."""

    rows = w1.shape[0]
    for name, tensor in (("w1", w1), ("a1", a1), ("g1", g1)):
        _check_rows(tensor, name, rows=rows)
    for source, original, runtime in (
        (w1, w2, w2_runtime),
        (a1, a2, a2_runtime),
        (g1, g2, g2_runtime),
    ):
        _check_rank_source(
            source,
            runtime,
            original,
            input_projection=False,
            enforce_lowrank_limit=True,
        )
    return tuple(
        _extension().tmix_lowrank_out_forward_varlen(
            w1, a1, g1, w2, a2, g2, w2_runtime, a2_runtime, g2_runtime
        )
    )


def infer_tmix_lowrank_vres_forward_varlen(
    w1: torch.Tensor,
    a1: torch.Tensor,
    g1: torch.Tensor,
    v1: torch.Tensor,
    w2: torch.Tensor | None,
    a2: torch.Tensor | None,
    g2: torch.Tensor | None,
    v2: torch.Tensor | None,
    v: torch.Tensor,
    v_first: torch.Tensor,
    v0: torch.Tensor,
    *,
    w2_runtime: torch.Tensor | None = None,
    a2_runtime: torch.Tensor | None = None,
    g2_runtime: torch.Tensor | None = None,
    v2_runtime: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = w1.shape[0]
    for name, tensor in (("w1", w1), ("a1", a1), ("g1", g1), ("v1", v1)):
        _check_rows(tensor, name, rows=rows)
    for source, original, runtime in (
        (w1, w2, w2_runtime),
        (a1, a2, a2_runtime),
        (g1, g2, g2_runtime),
        (v1, v2, v2_runtime),
    ):
        _check_rank_source(
            source,
            runtime,
            original,
            input_projection=False,
            enforce_lowrank_limit=True,
        )
    if w2 is not None:
        output_channels = w2.shape[0]
    elif w2_runtime is not None:
        output_channels = w2_runtime.shape[1]
    else:
        raise AssertionError("rank-out validation accepted no W projection weight")
    for name, tensor in (("v", v), ("v_first", v_first)):
        _check_rows(tensor, name, rows=rows)
        if tensor.device != w1.device or tensor.shape[1] != output_channels:
            raise ValueError(f"{name} must have shape [total_tokens,C]")
    if v0.dtype != torch.float16 or not v0.is_cuda or not v0.is_contiguous():
        raise ValueError("v0 must be contiguous CUDA float16")
    if v0.ndim != 1 or v0.shape[0] != v.shape[1] or v0.device != w1.device:
        raise ValueError("v0 must have shape [C]")
    return tuple(
        _extension().tmix_lowrank_vres_forward_varlen(
            w1,
            a1,
            g1,
            v1,
            w2,
            a2,
            g2,
            v2,
            w2_runtime,
            a2_runtime,
            g2_runtime,
            v2_runtime,
            v,
            v_first,
            v0,
        )
    )


__all__ = [
    "infer_tmix_linear_act_sigmoid_forward_varlen",
    "infer_tmix_linear_act_tanh_forward_varlen",
    "infer_tmix_linear_attention_c2c_forward_varlen",
    "infer_tmix_linear_ffn_key_forward_varlen",
    "infer_tmix_linear_forward_varlen",
    "infer_tmix_linear_rank_in_forward_varlen",
    "infer_tmix_linear_rank_out_forward_varlen",
    "infer_tmix_linear_rank_out_sigmoid_forward_varlen",
    "infer_tmix_linear_rank_out_tanh_forward_varlen",
    "infer_tmix_linear_t_forward_varlen",
    "infer_tmix_linear_t_sigmoid_forward_varlen",
    "infer_tmix_linear_t_tanh_forward_varlen",
    "infer_tmix_linear_t_vres_forward_varlen",
    "infer_tmix_lowrank_in_forward_varlen",
    "infer_tmix_lowrank_out_forward_varlen",
    "infer_tmix_lowrank_vres_forward_varlen",
    "infer_tmix_lowrank_wagv_in_forward_varlen",
]
