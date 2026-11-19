# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from ..linear import pretrain_head_linear_bf16
from ..._internal.execution import _AutogradContext


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
    if weight.ndim != 2 or weight.shape[0] == 0 or weight.shape[1] != hidden.shape[-1]:
        raise ValueError("weight must have shape [V,C]")
    if targets.numel() != hidden.shape[0] * hidden.shape[1]:
        raise ValueError("targets must contain one token per hidden row")
    if hidden.device != weight.device or hidden.device != targets.device:
        raise ValueError("head loss tensors must share a device")
    from ..._internal.state import check_cuda_condition
    check_cuda_condition(((targets == -100) | ((targets >= 0) & (targets < weight.shape[0]))).all(), "targets must be -100 or in [0,V)")
    if (
        not isinstance(chunk_rows, int)
        or isinstance(chunk_rows, bool)
        or chunk_rows <= 0
    ):
        raise ValueError("chunk_rows must be a positive integer")


# A multiple of the 1024-column reduction tile keeps vocabulary statistics
# independent of the projection working set.
_VOCAB_TILE = 16384


def _vocab_logits(x, weight, a, b, scale, use_nora):
    if (a is None) != (b is None) or (b is not None and b.shape[0] != weight.shape[0]):
        raise ValueError("adapter A and B must cover the full head vocabulary")
    for offset in range(0, weight.shape[0], _VOCAB_TILE):
        band = slice(offset, offset + _VOCAB_TILE)
        logits = pretrain_head_linear_bf16(
            x, weight[band], lora_a=a, lora_b=None if b is None else b[band],
            lora_scale=scale, use_nora=use_nora,
        )
        yield offset, logits


class _HeadCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, a, b, scale, use_nora, chunk_rows, backward_chunk_rows):
        from .tilelang import finish_stats, partial_stats

        rows = hidden.view(-1, hidden.shape[-1])
        flat_targets = targets.view(-1)
        loss, logsum, maximum = (hidden.new_empty(rows.shape[0], dtype=torch.float32) for _ in range(3))
        for start in range(0, rows.shape[0], chunk_rows):
            band = slice(start, start + chunk_rows)
            target = flat_targets[band]
            partials = [
                partial_stats(logits.shape[1])(logits, target, offset)
                for offset, logits in _vocab_logits(rows[band], weight, a, b, scale, use_nora)
            ]
            high, denominator, selected = (torch.cat(parts, dim=1) for parts in zip(*partials))
            loss[band], logsum[band], maximum[band] = finish_stats(weight.shape[0])(
                selected, target, high, denominator,
            )
        ctx.save_for_backward(hidden, weight, targets, a, b, logsum, maximum)
        ctx.options = (scale, use_nora, backward_chunk_rows)
        return loss.view_as(targets)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss):
        from ..._internal.linear import (
            linear, linear_input_vjp, linear_weight_vjp_fp32,
            nora_backward, nora_forward,
        )
        from .tilelang import ce_gradient

        hidden, weight, targets, a, b, logsum, maximum = ctx.saved_tensors
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
            # Keep the VJP over the full vocabulary in its original reduction order.
            dy = x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.float32)
            upstream = grad_loss.reshape(-1)[start:stop].contiguous()
            for offset, logits in _vocab_logits(x, weight, a_compute, b_compute, scale, False):
                width = logits.shape[1]
                dy[:, offset:offset + width] = ce_gradient(width)(
                    logits, target, logsum[start:stop], maximum[start:stop], upstream, offset,
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
            None, None, None, None,
        )


@torch.library.custom_op("flashrwkv2::head_ce_forward", mutates_args=())
def _compiled_head_loss(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
                        a: torch.Tensor | None, b: torch.Tensor | None, scale: float,
                        use_nora: bool, chunk_rows: int, backward_chunk_rows: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ctx = _AutogradContext()
    loss = _HeadCE.forward(
        ctx, hidden, weight, targets, a, b, scale, use_nora, chunk_rows, backward_chunk_rows,
    )
    return loss, *ctx.saved_tensors[-2:]


@_compiled_head_loss.register_fake
def _compiled_head_loss_fake(hidden, weight, targets, a, b, scale, use_nora, chunk_rows, backward_chunk_rows):
    return (hidden.new_empty(targets.shape, dtype=torch.float32),
            hidden.new_empty(targets.numel(), dtype=torch.float32), hidden.new_empty(targets.numel(), dtype=torch.float32))


def _compiled_head_loss_context(ctx, inputs, output):
    hidden, weight, targets, a, b, scale, use_nora, _, backward_chunk_rows = inputs
    ctx.save_for_backward(hidden, weight, targets, a, b, *output[1:])
    ctx.mark_non_differentiable(*output[1:])
    ctx.options = (scale, use_nora, backward_chunk_rows)


@torch.library.custom_op(
    "flashrwkv2::head_ce_backward", mutates_args=(),
    schema="(Tensor dy, Tensor hidden, Tensor weight, Tensor targets, Tensor? a, Tensor? b, Tensor logsum, Tensor maximum, float scale, bool use_nora, int chunk_rows, bool[] needed) -> Tensor?[]",
)
def _compiled_head_loss_vjp(dy, hidden, weight, targets, a, b, logsum, maximum, scale, use_nora, chunk_rows, needed):
    ctx = _AutogradContext(saved_tensors=(hidden, weight, targets, a, b, logsum, maximum), options=(scale, use_nora, chunk_rows),
                           needs_input_grad=(*needed, False, False, False))
    return list(_HeadCE.backward(ctx, dy)[:5])


@_compiled_head_loss_vjp.register_fake
def _compiled_head_loss_vjp_fake(dy, hidden, weight, targets, a, b, logsum, maximum, scale, use_nora, chunk_rows, needed):
    return [torch.empty_like(hidden) if needed[0] else None,
            torch.empty_like(weight) if needed[1] else None, None,
            (a.new_empty(a.shape) if use_nora else torch.empty_like(a)) if needed[3] else None,
            torch.empty_like(b) if needed[4] else None]


def _compiled_head_loss_backward(ctx, dy, unused_logsum, unused_maximum):
    return (*_compiled_head_loss_vjp(dy, *ctx.saved_tensors, *ctx.options, list(ctx.needs_input_grad[:5])), None, None, None, None)


_compiled_head_loss.register_autograd(_compiled_head_loss_backward, setup_context=_compiled_head_loss_context)


def pretrain_head_ce_bf16(
    hidden,
    weight,
    targets,
    *,
    chunk_rows: int = 128,
    backward_chunk_rows: int | None = None,
    lora_a=None,
    lora_b=None,
    lora_scale=1.0,
    use_nora=False,
):
    """Return one FP32 cross entropy per target and zero for ignored targets (-100).

    ``chunk_rows`` bounds the forward logits. ``backward_chunk_rows`` independently
    bounds recomputation and VJPs and defaults to ``chunk_rows``.
    """
    _check(hidden, weight, targets, chunk_rows)
    if backward_chunk_rows is None:
        backward_chunk_rows = chunk_rows
    elif type(backward_chunk_rows) is not int or backward_chunk_rows <= 0:
        raise ValueError("backward_chunk_rows must be a positive integer")
    args = (hidden, weight, targets, lora_a, lora_b, float(lora_scale), bool(use_nora), chunk_rows, backward_chunk_rows)
    if torch.compiler.is_compiling():
        return _compiled_head_loss(*args)[0]
    return _HeadCE.apply(*args)


__all__ = ["pretrain_head_ce_bf16"]
