# SPDX-License-Identifier: MIT
"""RWKV L2Wrap gradient with the shared FP32 vocabulary reduction."""

import tilelang
import tilelang.language as T
import torch

from ..._internal.logprob import logprob_stats


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
