# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from ..linear import pretrain_head_linear_bf16
from ..._internal.tuning import _AutogradContext


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
    from ..._internal.state import check_cuda_condition
    check_cuda_condition(((targets >= 0) & (targets < 65536)).all(), "targets must be in [0,65536)")
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


@torch.library.custom_op("flashrwkv2::head_l2wrap_ce_forward", mutates_args=())
def _compiled_head_loss(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
                        a: torch.Tensor | None, b: torch.Tensor | None, scale: float,
                        use_nora: bool, chunk_rows: int) -> torch.Tensor:
    return _HeadL2Wrap.forward(_AutogradContext(), hidden, weight, targets, a, b, scale, use_nora, chunk_rows)


@_compiled_head_loss.register_fake
def _compiled_head_loss_fake(hidden, weight, targets, a, b, scale, use_nora, chunk_rows):
    return hidden.new_empty((), dtype=torch.float32)


def _compiled_head_loss_context(ctx, inputs, output):
    hidden, weight, targets, a, b, scale, use_nora, chunk_rows = inputs
    ctx.save_for_backward(hidden, weight, targets, a, b)
    ctx.options = (scale, use_nora, chunk_rows)


@torch.library.custom_op(
    "flashrwkv2::head_l2wrap_ce_backward", mutates_args=(),
    schema="(Tensor dy, Tensor hidden, Tensor weight, Tensor targets, Tensor? a, Tensor? b, float scale, bool use_nora, int chunk_rows, bool[] needed) -> Tensor?[]",
)
def _compiled_head_loss_vjp(dy, hidden, weight, targets, a, b, scale, use_nora, chunk_rows, needed):
    ctx = _AutogradContext(saved_tensors=(hidden, weight, targets, a, b), options=(scale, use_nora, chunk_rows),
                           needs_input_grad=(*needed, False, False, False))
    return list(_HeadL2Wrap.backward(ctx, dy)[:5])


@_compiled_head_loss_vjp.register_fake
def _compiled_head_loss_vjp_fake(dy, hidden, weight, targets, a, b, scale, use_nora, chunk_rows, needed):
    return [torch.empty_like(hidden) if needed[0] else None,
            torch.empty_like(weight) if needed[1] else None, None,
            (a.new_empty(a.shape) if use_nora else torch.empty_like(a)) if needed[3] else None,
            torch.empty_like(b) if needed[4] else None]


def _compiled_head_loss_backward(ctx, dy):
    return (*_compiled_head_loss_vjp(dy, *ctx.saved_tensors, *ctx.options, list(ctx.needs_input_grad[:5])), None, None, None)


_compiled_head_loss.register_autograd(_compiled_head_loss_backward, setup_context=_compiled_head_loss_context)


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
    operation = _compiled_head_loss if torch.compiler.is_compiling() else _HeadL2Wrap.apply
    return operation(
        hidden, weight, targets, lora_a, lora_b,
        float(lora_scale), bool(use_nora), chunk_rows,
    )


__all__ = ["pretrain_head_l2wrap_ce_bf16"]
