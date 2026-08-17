# SPDX-License-Identifier: MIT

from __future__ import annotations

import math

import torch

from ..tmix.wkv7 import (
    _check_metadata_inputs,
    _extension,
    _resolve_max_seqlen,
    prepare_tmix_wkv7_recurrent_metadata,
)


def infer_cmix_forward_varlen(
    x: torch.Tensor,
    res: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    x_k: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    *,
    shift_state_pool: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    max_seqlen: int | None = None,
    eps: float = 1.0e-5,
    validated_metadata: object | None = None,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the complete packed ChannelMix inference island."""

    tensors = (x, res, weight, bias, x_k, key_weight, value_weight)
    names = ("x", "res", "weight", "bias", "x_k", "key_weight", "value_weight")
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
    if any(tensor.shape != (x.shape[1],) for tensor in (weight, bias, x_k)):
        raise ValueError("weight, bias, and x_k must have shape [C]")
    if key_weight.ndim != 2 or key_weight.shape[1] != x.shape[1]:
        raise ValueError("key_weight must have shape [F,C]")
    if value_weight.shape != (key_weight.shape[0], x.shape[1]):
        raise ValueError("value_weight must have runtime shape [F,C]")
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
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic must be a bool")
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
    return tuple(
        _extension().cmix_forward_varlen(
            x,
            res,
            shift_state_pool,
            weight,
            bias,
            x_k,
            key_weight,
            value_weight,
            cu_seqlens,
            state_indices,
            launch_max_seqlen,
            float(eps),
            ticket,
            deterministic,
        )
    )


class _PretrainCmix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_k, key_weight, value_weight):
        output, mixed, preact = _extension().pretrain_cmix_forward(
            x, x_k, key_weight, value_weight
        )
        ctx.save_for_backward(x, x_k, key_weight, value_weight, mixed, preact)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None
        gradients = _extension().pretrain_cmix_backward(
            grad_output.contiguous(), *ctx.saved_tensors
        )
        return tuple(gradients)


class _StatetuneCmix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, initial_shift, x_k, key_weight, value_weight):
        output, next_shift, mixed, activation = _extension().statetune_cmix_forward(
            x, initial_shift, x_k, key_weight, value_weight
        )
        ctx.save_for_backward(
            x, initial_shift, x_k, key_weight, value_weight, mixed, activation
        )
        return output, next_shift

    @staticmethod
    def backward(ctx, grad_output, grad_next_shift):
        return tuple(
            _extension().statetune_cmix_backward(
                grad_output.contiguous(),
                grad_next_shift.contiguous(),
                *ctx.saved_tensors,
            )
        )


def pretrain_cmix_bf16(
    x: torch.Tensor,
    x_k: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
) -> torch.Tensor:
    """Train-temp BF16 CMix forward/backward family."""

    tensors = (x, x_k, key_weight, value_weight)
    for name, tensor in zip(
        ("x", "x_k", "key_weight", "value_weight"), tensors, strict=True
    ):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.bfloat16
            or not tensor.is_cuda
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous CUDA bfloat16")
        if tensor.device != x.device:
            raise ValueError(f"{name} must share x's device")
    if x.ndim != 3 or x.numel() == 0 or x_k.shape != (x.shape[-1],):
        raise ValueError("invalid CMix input shapes")
    if key_weight.shape != (4 * x.shape[-1], x.shape[-1]) or value_weight.shape != (
        x.shape[-1],
        4 * x.shape[-1],
    ):
        raise ValueError("CMix weights must have shapes [4C,C] and [C,4C]")
    return _PretrainCmix.apply(x, x_k, key_weight, value_weight)


def statetune_cmix_bf16(
    x: torch.Tensor,
    initial_shift: torch.Tensor,
    x_k: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 ChannelMix with differentiable chunk-boundary shift state."""

    tensors = (x, initial_shift, x_k, key_weight, value_weight)
    names = ("x", "initial_shift", "x_k", "key_weight", "value_weight")
    for name, tensor in zip(names, tensors, strict=True):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype torch.bfloat16")
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA bfloat16")
        if tensor.data_ptr() % 4:
            raise ValueError(f"{name} must be 4-byte aligned for BF16 vec2 access")
        if tensor.device != x.device:
            raise ValueError(f"{name} must share x's device")
    if x.ndim != 3 or any(size <= 0 for size in x.shape):
        raise ValueError("x must have non-empty shape [B,T,C]")
    b, _, c = x.shape
    if c % 2:
        raise ValueError("x channel dimension C must be divisible by 2")
    if initial_shift.shape != (b, c):
        raise ValueError("initial_shift must have shape [B,C]")
    if x_k.shape != (c,):
        raise ValueError("x_k must have shape [C]")
    if key_weight.shape != (4 * c, c):
        raise ValueError("key_weight must have shape [4C,C]")
    if value_weight.shape != (c, 4 * c):
        raise ValueError("value_weight must have shape [C,4C]")
    return _StatetuneCmix.apply(*tensors)


__all__ = [
    "infer_cmix_forward_varlen",
    "pretrain_cmix_bf16",
    "statetune_cmix_bf16",
]
