# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from ..linear import pretrain_head_linear_bf16


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
    def forward(ctx, hidden, weight, targets, a, b, scale, use_nora, chunk_rows):
        from ..._internal.logprob import logprob_stats

        rows = hidden.view(-1, hidden.shape[-1])
        flat_targets = targets.view(-1)
        logprobs = torch.empty(rows.shape[0], dtype=torch.float32, device=hidden.device)
        for start in range(0, rows.shape[0], chunk_rows):
            stop = min(start + chunk_rows, rows.shape[0])
            logits = pretrain_head_linear_bf16(
                rows[start:stop], weight, lora_a=a, lora_b=b,
                lora_scale=scale, use_nora=use_nora,
            )
            logprobs[start:stop] = logprob_stats(logits, flat_targets[start:stop])[0]
            del logits
        ctx.save_for_backward(hidden, weight, targets, a, b)
        ctx.options = (scale, use_nora, chunk_rows)
        return -logprobs.mean()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss):
        from ..._internal.linear import (
            linear, linear_input_vjp, linear_weight_vjp_fp32,
            nora_backward, nora_forward,
        )
        from ..._internal.logprob import logprob_stats
        from ...loss.l2wrap_ce.tilelang import _backward as loss_backward

        hidden, weight, targets, a, b = ctx.saved_tensors
        scale, use_nora, chunk_rows = ctx.options
        need_x, need_w, _, need_a, need_b = ctx.needs_input_grad[:5]
        rows, flat_targets = hidden.view(-1, hidden.shape[-1]), targets.view(-1)
        dx = torch.empty_like(rows) if need_x else None
        dw = torch.zeros_like(weight, dtype=torch.float32) if need_w else None
        da = torch.zeros_like(a, dtype=torch.float32) if need_a else None
        db = torch.zeros_like(b, dtype=torch.float32) if need_b else None
        a_compute = b_compute = inverse = column_norm = None
        if a is not None:
            if use_nora:
                a_compute, inverse, column_norm = nora_forward(a)
            else:
                a_compute = a.to(torch.bfloat16)
            b_compute = b.to(torch.bfloat16)
        for start in range(0, rows.shape[0], chunk_rows):
            stop = min(start + chunk_rows, rows.shape[0])
            x, target = rows[start:stop], flat_targets[start:stop]
            # Reuse the exact forward weights; normalize once for the whole VJP.
            logits = pretrain_head_linear_bf16(
                x, weight, lora_a=a_compute, lora_b=b_compute, lora_scale=scale,
            )
            _, logsum, maximum, argmax = logprob_stats(logits, target)
            dy = loss_backward(weight.shape[0], "float32")(
                logits, target, logsum, maximum, argmax,
                grad_loss.float() * ((stop - start) / rows.shape[0]),
            )
            del logits
            if need_x:
                dx[start:stop] = linear_input_vjp(dy, weight)
            if need_w:
                linear_weight_vjp_fp32(dy, x, out=dw)
            if a is not None and (need_x or need_a or need_b):
                dy.mul_(scale)
                if need_b:
                    features = linear(x, a_compute)
                    linear_weight_vjp_fp32(dy, features, out=db)
                if need_x or need_a:
                    dfeatures = linear_input_vjp(dy, b_compute)
                    if need_x:
                        # The base and adapter input gradients each cross a BF16 edge.
                        dx[start:stop].add_(linear_input_vjp(dfeatures, a_compute))
                    if need_a:
                        linear_weight_vjp_fp32(dfeatures, x, out=da)
            del dy
        if need_a:
            da = (
                nora_backward(da, a, inverse, column_norm)
                if use_nora else da.to(torch.bfloat16).to(a.dtype)
            )
        return (
            None if dx is None else dx.view_as(hidden),
            None if dw is None else dw.to(weight.dtype),
            None,
            da,
            None if db is None else db.to(torch.bfloat16).to(b.dtype),
            None, None, None,
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
    """Memory-bounded head CE; parameter VJPs accumulate across chunks in FP32."""
    _check(hidden, weight, targets, chunk_rows)
    return _HeadL2Wrap.apply(
        hidden, weight, targets, lora_a, lora_b,
        float(lora_scale), bool(use_nora), chunk_rows,
    )


__all__ = ["pretrain_head_l2wrap_ce_bf16"]
