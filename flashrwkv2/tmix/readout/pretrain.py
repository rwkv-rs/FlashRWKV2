# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from . import _extension


def _check(
    tensors: dict[str, torch.Tensor], x: torch.Tensor, heads: int, head_size: int
) -> None:
    if head_size not in {64, 128, 256}:
        raise ValueError("head_size must be one of 64, 128, or 256")
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if (
            tensor.dtype != torch.bfloat16
            or not tensor.is_cuda
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous CUDA bfloat16")
    for name in ("r", "k", "v", "g"):
        if tensors[name].shape != x.shape:
            raise ValueError(f"{name} must match x")
    if x.ndim != 3 or x.numel() == 0 or x.shape[-1] % head_size:
        raise ValueError(
            "x must have non-empty shape [B,T,C], C divisible by head_size"
        )
    if tensors["residual_scale"].shape != (heads, head_size):
        raise ValueError("residual_scale must have shape [C/head_size,head_size]")
    if (
        tensors["weight"].shape != (x.shape[-1],)
        or tensors["bias"].shape != tensors["weight"].shape
    ):
        raise ValueError("weight and bias must have shape [C]")
    if any(tensor.device != x.device for tensor in tensors.values()):
        raise ValueError("lnx tensors must share a device")


class _Readout(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        r,
        k,
        v,
        res_scale,
        weight,
        bias,
        g,
        output_weight,
        output_lora_a,
        output_lora_b,
        head_size,
        output_lora_scale,
        output_use_nora,
    ):
        output, mean, rstd, *projection_state = (
            _extension().pretrain_tmix_readout_forward(
                x,
                r,
                k,
                v,
                res_scale,
                weight,
                bias,
                g,
                head_size,
                output_weight,
                output_lora_a,
                output_lora_b,
                output_lora_scale,
                output_use_nora,
            )
        )
        ctx.save_for_backward(
            x,
            r,
            k,
            v,
            res_scale,
            weight,
            bias,
            g,
            output_weight,
            output_lora_a,
            output_lora_b,
            mean,
            rstd,
            *projection_state,
        )
        ctx.options = (head_size, output_lora_scale, output_use_nora)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return (None,) * 14
        tensors = ctx.saved_tensors
        head_size, scale, nora = ctx.options
        gradients = _extension().pretrain_tmix_readout_backward(
            grad_output.contiguous(),
            *tensors[:8],
            *tensors[11:13],
            head_size,
            *tensors[8:11],
            scale,
            nora,
            *tensors[13:],
            list(ctx.needs_input_grad[:11]),
        )
        return (
            tuple(
                gradient if needed else None
                for gradient, needed in zip(
                    gradients, ctx.needs_input_grad[:11], strict=True
                )
            )
            + (None,) * 3
        )


def pretrain_tmix_readout_bf16(
    x,
    r,
    k,
    v,
    residual_scale,
    weight,
    bias,
    g,
    *,
    head_size=64,
    output_weight=None,
    output_lora_a=None,
    output_lora_b=None,
    output_lora_scale=1.0,
    output_use_nora=False,
):
    """Train-temp head-wise LN, recurrent residual and output gate."""

    tensors = {
        "x": x,
        "r": r,
        "k": k,
        "v": v,
        "residual_scale": residual_scale,
        "weight": weight,
        "bias": bias,
        "g": g,
    }
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    _check(tensors, x, x.shape[-1] // head_size if x.ndim >= 3 else 0, head_size)
    return _Readout.apply(
        x,
        r,
        k,
        v,
        residual_scale,
        weight,
        bias,
        g,
        output_weight,
        output_lora_a,
        output_lora_b,
        head_size,
        float(output_lora_scale),
        output_use_nora,
    )


__all__ = ["pretrain_tmix_readout_bf16"]
