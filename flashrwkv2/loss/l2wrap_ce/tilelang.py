# SPDX-License-Identifier: MIT
"""RWKV L2Wrap gradient with the shared FP32 vocabulary reduction."""

import tilelang
import tilelang.language as T
import torch

from ..._internal.logprob import logprob_stats
from ..._internal.tuning import _AutogradContext


@tilelang.jit(pass_configs={"tl.enable_fast_math": False}, out_idx=[6])
def _backward(vocab: int, dtype: str):
    rows = T.dynamic("rows")

    @T.prim_func
    def kernel(
        logits: T.Tensor((rows, vocab), dtype),
        targets: T.Tensor((rows,), T.int64),
        logsum: T.Tensor((rows,), T.float32),
        maximum: T.Tensor((rows,), T.float32),
        argmax: T.Tensor((rows,), T.int32),
        grad: T.Tensor((), T.float32),
        result: T.Tensor((rows, vocab), dtype),
    ):
        with T.Kernel(rows, T.ceildiv(vocab, 256), threads=128) as (row, part):
            for i in T.Parallel(256):
                col = part * 256 + i
                if col < vocab:
                    value = T.exp(
                        (T.cast(logits[row, col], T.float32) - maximum[row]) - logsum[row]
                    )
                    value -= T.cast(col == targets[row], T.float32)
                    value += T.if_then_else(col == argmax[row], 1.0e-4 * maximum[row], 0.0)
                    result[row, col] = value * grad[()] / T.cast(rows, T.float32)

    return kernel


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        flat = logits.reshape(-1, logits.shape[-1])
        logprobs, logsum, maximum, argmax = logprob_stats(flat.float(), targets.reshape(-1))
        ctx.save_for_backward(flat, targets, logsum, maximum, argmax)
        ctx.shape = logits.shape
        return -logprobs.mean()

    @staticmethod
    def backward(ctx, grad):
        logits, targets, logsum, maximum, argmax = ctx.saved_tensors
        dtype = "bfloat16" if logits.dtype == torch.bfloat16 else "float32"
        dx = _backward(logits.shape[1], dtype)(
            logits, targets.reshape(-1), logsum, maximum, argmax, grad.float().contiguous()
        )
        return dx.reshape(ctx.shape), None


@torch.library.custom_op("flashrwkv2::loss_l2wrap_ce_forward", mutates_args=())
def compiled_l2wrap(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ctx = _AutogradContext()
    loss = L2Wrap.forward(ctx, logits, targets)
    return loss, *ctx.saved_tensors[2:]


@compiled_l2wrap.register_fake
def _compiled_l2wrap_fake(logits, targets):
    return logits.new_empty((), dtype=torch.float32), logits.new_empty(targets.numel(), dtype=torch.float32), logits.new_empty(targets.numel(), dtype=torch.float32), targets.new_empty(targets.numel(), dtype=torch.int32)


def _compiled_l2wrap_context(ctx, inputs, output):
    logits, targets = inputs
    ctx.save_for_backward(logits.reshape(-1, logits.shape[-1]), targets, *output[1:])
    ctx.shape = logits.shape
    ctx.mark_non_differentiable(*output[1:])


@torch.library.custom_op("flashrwkv2::loss_l2wrap_ce_backward", mutates_args=())
def _compiled_l2wrap_vjp(dy: torch.Tensor, saved: list[torch.Tensor], shape: list[int]) -> torch.Tensor:
    return L2Wrap.backward(_AutogradContext(saved_tensors=saved, shape=shape), dy)[0]


@_compiled_l2wrap_vjp.register_fake
def _compiled_l2wrap_vjp_fake(dy, saved, shape):
    return saved[0].new_empty(shape)


def _compiled_l2wrap_backward(ctx, dy, unused_logsum, unused_maximum, unused_argmax):
    return _compiled_l2wrap_vjp(dy, list(ctx.saved_tensors), list(ctx.shape)), None


compiled_l2wrap.register_autograd(_compiled_l2wrap_backward, setup_context=_compiled_l2wrap_context)
