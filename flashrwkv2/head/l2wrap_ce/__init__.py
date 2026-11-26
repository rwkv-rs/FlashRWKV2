# SPDX-License-Identifier: MIT
"""Chunked LM head cross-entropy with train_temp L2Wrap regularization."""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from ..._internal.execution import _AutogradContext
from ..._internal.linear import chunked_head_backward, validate_head_loss_inputs
from ..linear import pretrain_head_linear_bf16

_L2WRAP_VOCAB_SIZE = 65536


class _HeadL2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, a, b, scale, use_nora, chunk_rows):
        from ..._internal.logprob import logprob_stats

        rows, flat_targets = hidden.view(-1, hidden.shape[-1]), targets.view(-1)
        logprobs = torch.empty(rows.shape[0], dtype=torch.float32, device=hidden.device)
        for start in range(0, rows.shape[0], chunk_rows):
            stop = min(start + chunk_rows, rows.shape[0])
            logits = pretrain_head_linear_bf16(
                rows[start:stop], weight, lora_a=a, lora_b=b, lora_scale=scale, use_nora=use_nora,
            )
            logprobs[start:stop] = logprob_stats(logits, flat_targets[start:stop])[0]
            del logits
        ctx.save_for_backward(hidden, weight, targets, a, b)
        ctx.options = (scale, use_nora, chunk_rows)
        return -logprobs.mean()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss):
        from ..._internal.logprob import logprob_stats
        from ...loss.l2wrap_ce.tilelang import _backward as loss_backward

        hidden, weight, targets, a, b = ctx.saved_tensors
        scale, use_nora, chunk_rows = ctx.options
        total_rows, flat_targets = hidden.shape[0] * hidden.shape[1], targets.view(-1)

        def compute_dy(x, start, stop, a_compute, b_compute):
            # Reuse the exact forward weights; normalize once for the whole VJP.
            logits = pretrain_head_linear_bf16(x, weight, lora_a=a_compute, lora_b=b_compute, lora_scale=scale)
            target = flat_targets[start:stop]
            _, logsum, maximum, argmax = logprob_stats(logits, target)
            return loss_backward(weight.shape[0], "float32")(
                logits, target, logsum, maximum, argmax, grad_loss.float() * ((stop - start) / total_rows),
            )

        grads = chunked_head_backward(
            hidden, weight, a, b, scale, use_nora, chunk_rows, ctx.needs_input_grad, compute_dy,
        )
        return (*grads, None, None, None)


@torch.library.custom_op("flashrwkv2::head_l2wrap_ce_forward", mutates_args=())
def _compiled_head_loss(
    hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
    a: torch.Tensor | None, b: torch.Tensor | None, scale: float,
    use_nora: bool, chunk_rows: int,
) -> torch.Tensor:
    return _HeadL2Wrap.forward(_AutogradContext(), hidden, weight, targets, a, b, scale, use_nora, chunk_rows)


@_compiled_head_loss.register_fake
def _compiled_head_loss_fake(hidden, weight, targets, a, b, scale, use_nora, chunk_rows):
    return hidden.new_empty((), dtype=torch.float32)


def _compiled_head_loss_context(ctx, inputs, output):
    hidden, weight, targets, a, b, scale, use_nora, chunk_rows = inputs
    ctx.save_for_backward(hidden, weight, targets, a, b)
    ctx.options = (scale, use_nora, chunk_rows)


@torch.library.custom_op(
    "flashrwkv2::head_l2wrap_ce_backward",
    mutates_args=(),
    schema="(Tensor dy, Tensor hidden, Tensor weight, Tensor targets, Tensor? a, Tensor? b, float scale, bool use_nora, int chunk_rows, bool[] needed) -> Tensor?[]",
)
def _compiled_head_loss_vjp(dy, hidden, weight, targets, a, b, scale, use_nora, chunk_rows, needed):
    ctx = _AutogradContext(
        saved_tensors=(hidden, weight, targets, a, b),
        options=(scale, use_nora, chunk_rows),
        needs_input_grad=(*needed, False, False, False),
    )
    return list(_HeadL2Wrap.backward(ctx, dy)[:5])


@_compiled_head_loss_vjp.register_fake
def _compiled_head_loss_vjp_fake(dy, hidden, weight, targets, a, b, scale, use_nora, chunk_rows, needed):
    return [
        torch.empty_like(hidden) if needed[0] else None,
        torch.empty_like(weight) if needed[1] else None,
        None,
        (a.new_empty(a.shape) if use_nora else torch.empty_like(a)) if needed[3] else None,
        torch.empty_like(b) if needed[4] else None,
    ]


def _compiled_head_loss_backward(ctx, dy):
    return (*_compiled_head_loss_vjp(dy, *ctx.saved_tensors, *ctx.options, list(ctx.needs_input_grad[:5])), None, None, None)


_compiled_head_loss.register_autograd(_compiled_head_loss_backward, setup_context=_compiled_head_loss_context)


def pretrain_head_l2wrap_ce_bf16(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_rows: int = 4096,
    lora_a: torch.Tensor | None = None,
    lora_b: torch.Tensor | None = None,
    lora_scale: float = 1.0,
    use_nora: bool = False,
) -> torch.Tensor:
    """Chunked LM-head FP32 mean CE with train_temp L2Wrap gradient (``V=65536``)."""
    validate_head_loss_inputs(hidden, weight, targets, chunk_rows, fixed_vocab=_L2WRAP_VOCAB_SIZE)
    operation = _compiled_head_loss if torch.compiler.is_compiling() else _HeadL2Wrap.apply
    return operation(hidden, weight, targets, lora_a, lora_b, float(lora_scale), bool(use_nora), chunk_rows)


__all__ = ["pretrain_head_l2wrap_ce_bf16"]
