# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from ...tmix.wkv7 import _extension


def _check(hidden, weight, targets, chunk_rows: int) -> None:
    for name, tensor in (("hidden", hidden), ("weight", weight)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if (
            tensor.dtype != torch.bfloat16
            or not tensor.is_cuda
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous CUDA bfloat16")
    if not isinstance(targets, torch.Tensor):
        raise TypeError("targets must be a torch.Tensor")
    if (
        targets.dtype != torch.int64
        or not targets.is_cuda
        or not targets.is_contiguous()
    ):
        raise ValueError("targets must be contiguous CUDA int64")
    if hidden.ndim != 3 or hidden.numel() == 0:
        raise ValueError("hidden must have shape [B,T,C]")
    if weight.shape != (65536, hidden.shape[-1]):
        raise ValueError("weight must have shape [65536,C]")
    if targets.numel() != hidden.shape[0] * hidden.shape[1]:
        raise ValueError("targets must contain one token per hidden row")
    if hidden.device != weight.device or hidden.device != targets.device:
        raise ValueError("head loss tensors must share a device")
    if torch.any((targets < 0) | (targets >= 65536)).item():
        raise ValueError("targets must be in [0,65536)")
    if (
        not isinstance(chunk_rows, int)
        or isinstance(chunk_rows, bool)
        or chunk_rows <= 0
    ):
        raise ValueError("chunk_rows must be a positive integer")


class _HeadL2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, hidden, weight, targets, lora_a, lora_b, lora_scale, use_nora, chunk_rows
    ):
        loss, *gradients = _extension().pretrain_head_l2wrap_ce_forward(
            hidden,
            weight,
            targets,
            int(chunk_rows),
            lora_a,
            lora_b,
            lora_scale,
            use_nora,
            [ctx.needs_input_grad[i] for i in (0, 1, 3, 4)],
        )
        ctx.save_for_backward(*gradients)
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        if grad_loss is None:
            return (None,) * 8
        gradients = [
            None if grad is None else grad * grad_loss.to(grad.dtype)
            for grad in ctx.saved_tensors
        ]
        return (
            gradients[0],
            gradients[1],
            None,
            gradients[2],
            gradients[3],
            None,
            None,
            None,
        )


def pretrain_head_l2wrap_ce_bf16(
    hidden,
    weight,
    targets,
    *,
    chunk_rows: int = 4096,
    lora_a=None,
    lora_b=None,
    lora_scale=1.0,
    use_nora=False,
):
    """Memory-bounded output-head CE with L2Wrap and optional LoRA/NoRA."""
    _check(hidden, weight, targets, chunk_rows)
    return _HeadL2Wrap.apply(
        hidden, weight, targets, lora_a, lora_b, float(lora_scale), use_nora, chunk_rows
    )


__all__ = ["pretrain_head_l2wrap_ce_bf16"]
