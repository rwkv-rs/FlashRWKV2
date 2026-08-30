# SPDX-License-Identifier: MIT

from __future__ import annotations

import math

import torch

from ..wkv7 import (
    _check_metadata_inputs,
    _extension,
    _metadata_launch_args,
    _resolve_max_seqlen,
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
) -> tuple[torch.Tensor, ...]:
    """Run the packed Res, LN and TokenShift fusion island."""

    tensors = (x, res, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g)
    names = (
        "x",
        "res",
        "weight",
        "bias",
        "x_r",
        "x_w",
        "x_k",
        "x_v",
        "x_a",
        "x_g",
    )
    for name, tensor in zip(names, tensors, strict=True):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.float16:
            raise TypeError(f"{name} must have dtype torch.float16")
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be CUDA and contiguous")
        if tensor.device != x.device:
            raise ValueError(f"{name} must share x's device")
    if x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] <= 0 or x.shape[1] % 2:
        raise ValueError("x must have packed shape [total_tokens,C] with even C")
    if res.shape != x.shape:
        raise ValueError("res must have the same shape as x")
    if any(tensor.shape != (x.shape[1],) for tensor in tensors[2:]):
        raise ValueError("LN and TokenShift parameters must have shape [C]")
    if not isinstance(shift_state_pool, torch.Tensor):
        raise TypeError("shift_state_pool must be a torch.Tensor")
    if (
        shift_state_pool.dtype != torch.float16
        or not shift_state_pool.is_cuda
        or not shift_state_pool.is_contiguous()
        or shift_state_pool.ndim != 2
        or shift_state_pool.shape[1] != x.shape[1]
        or shift_state_pool.device != x.device
    ):
        raise ValueError(
            "shift_state_pool must be contiguous CUDA float16 [slots,C]"
        )
    if (
        not isinstance(eps, (float, int))
        or isinstance(eps, bool)
        or not math.isfinite(float(eps))
    ):
        raise ValueError("eps must be finite")
    if float(eps) <= 0.0:
        raise ValueError("eps must be positive")
    _check_metadata_inputs(cu_seqlens, state_indices)
    if state_indices.ndim != 1 or state_indices.numel() == 0:
        raise ValueError("state_indices must have shape [B]")
    if validated_metadata is None:
        launch_max_seqlen = _resolve_max_seqlen(cu_seqlens, max_seqlen)
        ticket = prepare_tmix_wkv7_recurrent_metadata(
            cu_seqlens,
            state_indices,
            total_tokens=x.shape[0],
            state_pool_size=shift_state_pool.shape[0],
            max_seqlen=launch_max_seqlen,
        )
    else:
        if max_seqlen is None:
            launch_max_seqlen = -1
        elif (
            not isinstance(max_seqlen, int)
            or isinstance(max_seqlen, bool)
            or max_seqlen <= 0
        ):
            raise ValueError("max_seqlen must be a positive integer")
        else:
            launch_max_seqlen = int(max_seqlen)
        ticket = validated_metadata
    launch_metadata = _metadata_launch_args(
        ticket,
        cu_seqlens,
        state_indices,
        x.shape[0],
        shift_state_pool.shape[0],
        launch_max_seqlen,
    )
    token_predecessor = ticket._token_predecessor()
    return tuple(
        _extension().tmix_postnorm_tokenshift_forward_varlen(
            x,
            res,
            shift_state_pool,
            weight,
            bias,
            x_r,
            x_w,
            x_k,
            x_v,
            x_a,
            x_g,
            *launch_metadata,
            token_predecessor,
            float(eps),
        )
    )


class _PretrainTokenShift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_r, x_w, x_k, x_v, x_a, x_g):
        output = _extension().pretrain_tmix_tokenshift_forward(
            x, x_r, x_w, x_k, x_v, x_a, x_g
        )
        ctx.save_for_backward(x, x_r, x_w, x_k, x_v, x_a, x_g)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grad_outputs):
        if any(gradient is None for gradient in grad_outputs):
            return (None,) * 7
        return tuple(
            _extension().pretrain_tmix_tokenshift_backward(
                *(gradient.contiguous() for gradient in grad_outputs),
                *ctx.saved_tensors,
            )
        )


class _StatetuneTokenShift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, initial_shift, x_r, x_w, x_k, x_v, x_a, x_g):
        outputs = tuple(
            _extension().statetune_tmix_tokenshift_forward(
                x, initial_shift, x_r, x_w, x_k, x_v, x_a, x_g
            )
        )
        ctx.save_for_backward(x, initial_shift, x_r, x_w, x_k, x_v, x_a, x_g)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        return tuple(
            _extension().statetune_tmix_tokenshift_backward(
                *(gradient.contiguous() for gradient in grad_outputs),
                *ctx.saved_tensors,
            )
        )


def pretrain_tmix_tokenshift_bf16(
    x: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Train-temp BF16 six-way shifted TimeMix preparation."""

    tensors = (x, x_r, x_w, x_k, x_v, x_a, x_g)
    if (
        not isinstance(x, torch.Tensor)
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
    ):
        raise ValueError("x must be contiguous CUDA bfloat16 [B,T,C]")
    if x.ndim != 3 or x.numel() == 0:
        raise ValueError("x must have shape [B,T,C]")
    for name, tensor in zip(
        ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"), tensors[1:], strict=True
    ):
        if (
            tensor.dtype != torch.bfloat16
            or not tensor.is_cuda
            or not tensor.is_contiguous()
            or tensor.shape != (x.shape[-1],)
            or tensor.device != x.device
        ):
            raise ValueError(f"{name} must be contiguous CUDA bfloat16 [C]")
    return _PretrainTokenShift.apply(*tensors)


def statetune_tmix_tokenshift_bf16(
    x: torch.Tensor,
    initial_shift: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """BF16 six-way TimeMix with differentiable chunk-boundary shift state."""

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.dtype != torch.bfloat16:
        raise TypeError("x must have dtype torch.bfloat16")
    if not x.is_cuda or not x.is_contiguous():
        raise ValueError("x must be contiguous CUDA bfloat16 [B,T,C]")
    if x.data_ptr() % 4:
        raise ValueError("x must be 4-byte aligned for BF16 vec2 access")
    if x.ndim != 3 or any(size <= 0 for size in x.shape):
        raise ValueError("x must have non-empty shape [B,T,C]")
    if x.shape[2] % 2:
        raise ValueError("x channel dimension C must be divisible by 2")
    tensors = (initial_shift, x_r, x_w, x_k, x_v, x_a, x_g)
    names = ("initial_shift", "x_r", "x_w", "x_k", "x_v", "x_a", "x_g")
    expected_shapes = ((x.shape[0], x.shape[2]),) + ((x.shape[2],),) * 6
    for name, tensor, shape in zip(names, tensors, expected_shapes, strict=True):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype torch.bfloat16")
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA bfloat16")
        if tensor.data_ptr() % 4:
            raise ValueError(f"{name} must be 4-byte aligned for BF16 vec2 access")
        if tensor.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if tensor.device != x.device:
            raise ValueError(f"{name} must share x's device")
    return _StatetuneTokenShift.apply(x, *tensors)


__all__ = [
    "infer_tmix_postnorm_tokenshift_forward_varlen",
    "pretrain_tmix_tokenshift_bf16",
    "statetune_tmix_tokenshift_bf16",
]
