# SPDX-License-Identifier: MIT
"""Chunked LM head cross-entropy loss with FP32 accumulation and optional LoRA/NoRA."""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from ..._internal.execution import _AutogradContext
from ..._internal.linear import chunked_head_backward, validate_head_loss_inputs
from ..linear import pretrain_head_linear_bf16

# A multiple of the 1024-column reduction tile keeps vocabulary statistics
# independent of the projection working set.
_VOCAB_TILE = 16384


def _vocab_logits(x, weight, a, b, scale, use_nora):
    if (a is None) != (b is None) or (b is not None and b.shape[0] != weight.shape[0]):
        raise ValueError("adapter A and B must cover the full head vocabulary")
    for offset in range(0, weight.shape[0], _VOCAB_TILE):
        band = slice(offset, offset + _VOCAB_TILE)
        yield offset, pretrain_head_linear_bf16(
            x, weight[band], lora_a=a, lora_b=None if b is None else b[band],
            lora_scale=scale, use_nora=use_nora,
        )


class _HeadCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, a, b, scale, use_nora, chunk_rows, backward_chunk_rows):
        from .tilelang import finish_stats, partial_stats

        rows, flat_targets = hidden.view(-1, hidden.shape[-1]), targets.view(-1)
        loss, logsum, maximum = (hidden.new_empty(rows.shape[0], dtype=torch.float32) for _ in range(3))
        for start in range(0, rows.shape[0], chunk_rows):
            band = slice(start, start + chunk_rows)
            target = flat_targets[band]
            partials = [
                partial_stats(logits.shape[1])(logits, target, offset)
                for offset, logits in _vocab_logits(rows[band], weight, a, b, scale, use_nora)
            ]
            high, denominator, selected = (torch.cat(parts, dim=1) for parts in zip(*partials, strict=True))
            loss[band], logsum[band], maximum[band] = finish_stats(weight.shape[0])(
                selected, target, high, denominator,
            )
        ctx.save_for_backward(hidden, weight, targets, a, b, logsum, maximum)
        ctx.options = (scale, use_nora, backward_chunk_rows)
        return loss.view_as(targets)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss):
        from .tilelang import ce_gradient

        hidden, weight, targets, a, b, logsum, maximum = ctx.saved_tensors
        scale, use_nora, chunk_rows = ctx.options
        flat_targets, flat_grad = targets.view(-1), grad_loss.reshape(-1)

        def compute_dy(x, start, stop, a_compute, b_compute):
            # Keep the VJP over the full vocabulary in its original reduction order.
            dy = x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.float32)
            target, upstream = flat_targets[start:stop], flat_grad[start:stop].contiguous()
            for offset, logits in _vocab_logits(x, weight, a_compute, b_compute, scale, False):
                width = logits.shape[1]
                dy[:, offset : offset + width] = ce_gradient(width)(
                    logits, target, logsum[start:stop], maximum[start:stop], upstream, offset,
                )
                del logits
            return dy

        grads = chunked_head_backward(
            hidden, weight, a, b, scale, use_nora, chunk_rows, ctx.needs_input_grad, compute_dy,
        )
        return (*grads, None, None, None, None)


@torch.library.custom_op("flashrwkv2::head_ce_forward", mutates_args=())
def _compiled_head_loss(
    hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
    a: torch.Tensor | None, b: torch.Tensor | None, scale: float,
    use_nora: bool, chunk_rows: int, backward_chunk_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ctx = _AutogradContext()
    loss = _HeadCE.forward(ctx, hidden, weight, targets, a, b, scale, use_nora, chunk_rows, backward_chunk_rows)
    return loss, *ctx.saved_tensors[-2:]


@_compiled_head_loss.register_fake
def _compiled_head_loss_fake(hidden, weight, targets, a, b, scale, use_nora, chunk_rows, backward_chunk_rows):
    return (
        hidden.new_empty(targets.shape, dtype=torch.float32),
        hidden.new_empty(targets.numel(), dtype=torch.float32),
        hidden.new_empty(targets.numel(), dtype=torch.float32),
    )


def _compiled_head_loss_context(ctx, inputs, output):
    hidden, weight, targets, a, b, scale, use_nora, _, backward_chunk_rows = inputs
    ctx.save_for_backward(hidden, weight, targets, a, b, *output[1:])
    ctx.mark_non_differentiable(*output[1:])
    ctx.options = (scale, use_nora, backward_chunk_rows)


@torch.library.custom_op(
    "flashrwkv2::head_ce_backward",
    mutates_args=(),
    schema="(Tensor dy, Tensor hidden, Tensor weight, Tensor targets, Tensor? a, Tensor? b, Tensor logsum, Tensor maximum, float scale, bool use_nora, int chunk_rows, bool[] needed) -> Tensor?[]",
)
def _compiled_head_loss_vjp(dy, hidden, weight, targets, a, b, logsum, maximum, scale, use_nora, chunk_rows, needed):
    ctx = _AutogradContext(
        saved_tensors=(hidden, weight, targets, a, b, logsum, maximum),
        options=(scale, use_nora, chunk_rows),
        needs_input_grad=(*needed, False, False, False),
    )
    return list(_HeadCE.backward(ctx, dy)[:5])


@_compiled_head_loss_vjp.register_fake
def _compiled_head_loss_vjp_fake(dy, hidden, weight, targets, a, b, logsum, maximum, scale, use_nora, chunk_rows, needed):
    return [
        torch.empty_like(hidden) if needed[0] else None,
        torch.empty_like(weight) if needed[1] else None,
        None,
        (a.new_empty(a.shape) if use_nora else torch.empty_like(a)) if needed[3] else None,
        torch.empty_like(b) if needed[4] else None,
    ]


def _compiled_head_loss_backward(ctx, dy, unused_logsum, unused_maximum):
    return (*_compiled_head_loss_vjp(dy, *ctx.saved_tensors, *ctx.options, list(ctx.needs_input_grad[:5])), None, None, None, None)


_compiled_head_loss.register_autograd(_compiled_head_loss_backward, setup_context=_compiled_head_loss_context)


def pretrain_head_ce_bf16(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_rows: int = 128,
    backward_chunk_rows: int | None = None,
    lora_a: torch.Tensor | None = None,
    lora_b: torch.Tensor | None = None,
    lora_scale: float = 1.0,
    use_nora: bool = False,
) -> torch.Tensor:
    """Chunked LM-head FP32 cross-entropy (``[B,T,C]`` + ``[V,C]`` -> ``targets.shape``; ``-100`` ignored)."""
    validate_head_loss_inputs(hidden, weight, targets, chunk_rows, allow_ignore=True)
    if backward_chunk_rows is None:
        backward_chunk_rows = chunk_rows
    elif type(backward_chunk_rows) is not int or backward_chunk_rows <= 0:
        raise ValueError("backward_chunk_rows must be a positive integer")
    args = (hidden, weight, targets, lora_a, lora_b, float(lora_scale), bool(use_nora), chunk_rows, backward_chunk_rows)
    return _compiled_head_loss(*args)[0] if torch.compiler.is_compiling() else _HeadCE.apply(*args)


__all__ = ["pretrain_head_ce_bf16"]
