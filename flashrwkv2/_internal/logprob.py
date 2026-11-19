# SPDX-License-Identifier: MIT
"""Vocabulary statistics for L2Wrap with deterministic argmax tie-breaking."""

import tilelang
import tilelang.language as T


@tilelang.jit(pass_configs={"tl.enable_fast_math": False}, out_idx=[1, 2, 3])
def _partials(vocab: int):
    rows = T.dynamic("rows")
    tile = 1024
    parts = T.ceildiv(vocab, tile)

    @T.prim_func
    def kernel(
        logits: T.Tensor((rows, vocab), T.float32),
        maximum: T.Tensor((rows, parts), T.float32),
        denominator: T.Tensor((rows, parts), T.float32),
        argmax: T.Tensor((rows, parts), T.int32),
    ):
        with T.Kernel(rows, parts, threads=128) as (row, part):
            values = T.alloc_fragment((tile,), T.float32)
            exponentials = T.alloc_fragment((tile,), T.float32)
            high = T.alloc_fragment((1,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            indices = T.alloc_fragment((tile,), T.int32)
            first = T.alloc_fragment((1,), T.int32)
            for i in T.Parallel(tile):
                col = part * tile + i
                values[i] = T.if_then_else(col < vocab, logits[row, col], -T.infinity(T.float32))
            T.reduce_max(values, high, dim=0)
            for i in T.Parallel(tile):
                exponentials[i] = T.if_then_else(
                    high[0] == -T.infinity(T.float32), 0.0,
                    T.exp(values[i] - high[0]),
                )
                indices[i] = T.if_then_else(values[i] == high[0], part * tile + i, vocab)
            T.reduce_sum(exponentials, total, dim=0)
            T.reduce_min(indices, first, dim=0)
            maximum[row, part] = high[0]
            denominator[row, part] = total[0]
            argmax[row, part] = first[0]

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": False}, out_idx=[5, 6, 7, 8])
def _finish(vocab: int):
    rows = T.dynamic("rows")
    parts = (vocab + 1023) // 1024
    block = tilelang.next_power_of_2(parts)

    @T.prim_func
    def kernel(
        logits: T.Tensor((rows, vocab), T.float32),
        targets: T.Tensor((rows,), T.int64),
        maximum: T.Tensor((rows, parts), T.float32),
        denominator: T.Tensor((rows, parts), T.float32),
        argmax: T.Tensor((rows, parts), T.int32),
        output: T.Tensor((rows,), T.float32),
        logsum: T.Tensor((rows,), T.float32),
        max_values: T.Tensor((rows,), T.float32),
        max_indices: T.Tensor((rows,), T.int32),
    ):
        with T.Kernel(rows, threads=32) as row:
            maxima = T.alloc_fragment((block,), T.float32)
            sums = T.alloc_fragment((block,), T.float32)
            high = T.alloc_fragment((1,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            indices = T.alloc_fragment((block,), T.int32)
            first = T.alloc_fragment((1,), T.int32)
            for i in T.Parallel(block):
                maxima[i] = T.if_then_else(i < parts, maximum[row, i], -T.infinity(T.float32))
            T.reduce_max(maxima, high, dim=0)
            for i in T.Parallel(block):
                sums[i] = T.if_then_else(
                    i < parts and denominator[row, i] > 0.0,
                    denominator[row, i] * T.exp(maxima[i] - high[0]), 0.0,
                )
                indices[i] = T.if_then_else(
                    i < parts and maxima[i] == high[0], argmax[row, i], vocab
                )
            T.reduce_sum(sums, total, dim=0)
            T.reduce_min(indices, first, dim=0)
            logsum[row] = T.log(total[0])
            max_values[row] = high[0]
            max_indices[row] = first[0]
            if 0 <= targets[row] and targets[row] < vocab:
                output[row] = (logits[row, targets[row]] - high[0]) - T.log(total[0])
            else:
                output[row] = T.float32(float("nan"))

    return kernel


def logprob_stats(logits, targets):
    maximum, denominator, argmax = _partials(logits.shape[1])(logits)
    return _finish(logits.shape[1])(logits, targets, maximum, denominator, argmax)
