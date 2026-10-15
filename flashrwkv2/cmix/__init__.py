# SPDX-License-Identifier: MIT

from __future__ import annotations

import math

import torch

from ..tmix.wkv7 import (
    _check_metadata_inputs,
    _extension,
    _validate_launch_max_seqlen,
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
        raise ValueError("shift_state_pool must be contiguous CUDA float16 [slots,C]")
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
    launch_max_seqlen = _validate_launch_max_seqlen(max_seqlen)
    ticket = (
        validated_metadata
        if validated_metadata is not None
        else prepare_tmix_wkv7_recurrent_metadata(
            cu_seqlens,
            state_indices,
            total_tokens=x.shape[0],
            state_pool_size=shift_state_pool.shape[0],
            max_seqlen=max_seqlen,
        )
    )
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


class _Cmix(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        initial_shift,
        x_k,
        key_weight,
        value_weight,
        key_lora_a,
        key_lora_b,
        value_lora_a,
        value_lora_b,
        key_lora_scale,
        value_lora_scale,
        key_use_nora,
        value_use_nora,
    ):
        ctx.stateful = initial_shift is not None
        ctx.options = (key_lora_scale, value_lora_scale, key_use_nora, value_use_nora)
        adapters = (key_lora_a, key_lora_b, value_lora_a, value_lora_b)
        if ctx.stateful:
            output, next_shift, mixed, activation, *adapter_state = (
                _extension().statetune_cmix_forward(
                    x,
                    initial_shift,
                    x_k,
                    key_weight,
                    value_weight,
                    *adapters,
                    *ctx.options,
                )
            )
        else:
            output, mixed, activation, *adapter_state = (
                _extension().pretrain_cmix_forward(
                    x, x_k, key_weight, value_weight, *adapters, *ctx.options
                )
            )
        ctx.save_for_backward(
            x,
            initial_shift,
            x_k,
            key_weight,
            value_weight,
            *adapters,
            mixed,
            activation,
            *adapter_state,
        )
        return (output, next_shift) if ctx.stateful else output

    @staticmethod
    def backward(ctx, grad_output, grad_next_shift=None):
        x, initial, x_k, key, value, *saved = ctx.saved_tensors
        adapters, mixed, activation, adapter_state = (
            saved[:4],
            saved[4],
            saved[5],
            saved[6:],
        )
        if grad_output is None:
            grad_output = torch.zeros_like(x)
        args = (x, x_k, key, value, mixed, activation)
        tail = (*adapters, *ctx.options, adapter_state, list(ctx.needs_input_grad[:9]))
        if ctx.stateful:
            if grad_next_shift is None:
                grad_next_shift = torch.zeros_like(initial)
            gradients = _extension().statetune_cmix_backward(
                grad_output.contiguous(),
                grad_next_shift.contiguous(),
                x,
                initial,
                x_k,
                key,
                value,
                mixed,
                activation,
                *tail,
            )
        else:
            gradients = list(
                _extension().pretrain_cmix_backward(
                    grad_output.contiguous(), *args, *tail
                )
            )
            gradients.insert(1, None)
        return (
            tuple(
                gradient if needed else None
                for gradient, needed in zip(
                    gradients, ctx.needs_input_grad[:9], strict=True
                )
            )
            + (None,) * 4
        )


def pretrain_cmix_bf16(
    x: torch.Tensor,
    x_k: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    *,
    key_lora_a: torch.Tensor | None = None,
    key_lora_b: torch.Tensor | None = None,
    value_lora_a: torch.Tensor | None = None,
    value_lora_b: torch.Tensor | None = None,
    key_lora_scale: float = 1.0,
    value_lora_scale: float = 1.0,
    key_use_nora: bool = False,
    value_use_nora: bool = False,
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
    if x.shape[-1] % 2:
        raise ValueError("x channel dimension C must be divisible by 2")
    if key_weight.shape != (4 * x.shape[-1], x.shape[-1]) or value_weight.shape != (
        x.shape[-1],
        4 * x.shape[-1],
    ):
        raise ValueError("CMix weights must have shapes [4C,C] and [C,4C]")
    return _Cmix.apply(
        x,
        None,
        x_k,
        key_weight,
        value_weight,
        key_lora_a,
        key_lora_b,
        value_lora_a,
        value_lora_b,
        float(key_lora_scale),
        float(value_lora_scale),
        key_use_nora,
        value_use_nora,
    )


def statetune_cmix_bf16(
    x: torch.Tensor,
    initial_shift: torch.Tensor,
    x_k: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    *,
    key_lora_a: torch.Tensor | None = None,
    key_lora_b: torch.Tensor | None = None,
    value_lora_a: torch.Tensor | None = None,
    value_lora_b: torch.Tensor | None = None,
    key_lora_scale: float = 1.0,
    value_lora_scale: float = 1.0,
    key_use_nora: bool = False,
    value_use_nora: bool = False,
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
    return _Cmix.apply(
        *tensors,
        key_lora_a,
        key_lora_b,
        value_lora_a,
        value_lora_b,
        float(key_lora_scale),
        float(value_lora_scale),
        key_use_nora,
        value_use_nora,
    )


__all__ = [
    "infer_cmix_forward_varlen",
    "pretrain_cmix_bf16",
    "statetune_cmix_bf16",
]
