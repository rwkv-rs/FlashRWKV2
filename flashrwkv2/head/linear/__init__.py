# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from ...tmix.wkv7 import _extension


def infer_head_linear_all_forward_varlen(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Canonical Albatross ``linear_head(all_logits=True)`` caller path."""

    for name, tensor in (("x", x), ("weight", weight)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if (
            tensor.dtype != torch.float16
            or not tensor.is_cuda
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous CUDA float16")
        if tensor.device != x.device:
            raise ValueError(f"{name} must share x's device")
    if (
        x.ndim != 2
        or x.shape[0] <= 0
        or x.shape[1] <= 0
        or weight.ndim != 2
        or weight.shape[0] <= 0
        or x.shape[1] != weight.shape[1]
    ):
        raise ValueError("x must be [rows,C] and weight must be [vocab,C]")
    return _extension().head_linear_all_forward_varlen(x, weight)


def infer_head_linear_last_forward_varlen(
    x: torch.Tensor, weight: torch.Tensor, *, tokens_count: int
) -> torch.Tensor:
    """Canonical Albatross ``linear_head_last`` caller path."""

    for name, tensor in (("x", x), ("weight", weight)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if (
            tensor.dtype != torch.float16
            or not tensor.is_cuda
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous CUDA float16")
        if tensor.device != x.device:
            raise ValueError(f"{name} must share x's device")
    if (
        x.ndim != 2
        or x.shape[0] <= 0
        or x.shape[1] <= 0
        or weight.ndim != 2
        or weight.shape[0] <= 0
        or x.shape[1] != weight.shape[1]
    ):
        raise ValueError("x must be [batch,C] and weight must be [vocab,C]")
    if not isinstance(tokens_count, int) or tokens_count <= 0:
        raise ValueError("tokens_count must be a positive int")
    return _extension().head_linear_last_forward_varlen(x, weight, tokens_count)


class _HeadLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, lora_a, lora_b, lora_scale, use_nora):
        output, features, normalized_a = _extension().pretrain_head_linear_forward(
            x, weight, lora_a, lora_b, lora_scale, use_nora
        )
        ctx.save_for_backward(x, weight, lora_a, lora_b, features, normalized_a)
        ctx.options = (lora_scale, use_nora)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return (None,) * 6
        gradients = _extension().pretrain_head_linear_backward(
            grad_output.contiguous(),
            *ctx.saved_tensors,
            *ctx.options,
            list(ctx.needs_input_grad[:4]),
        )
        return tuple(gradients) + (None, None)


def pretrain_head_linear_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    lora_a: torch.Tensor | None = None,
    lora_b: torch.Tensor | None = None,
    lora_scale: float = 1.0,
    use_nora: bool = False,
) -> torch.Tensor:
    """Project selected hidden rows to logits with differentiable LoRA/NoRA."""
    return _HeadLinear.apply(x, weight, lora_a, lora_b, float(lora_scale), use_nora)


__all__ = [
    "pretrain_head_linear_bf16",
    "infer_head_linear_all_forward_varlen",
    "infer_head_linear_last_forward_varlen",
]
