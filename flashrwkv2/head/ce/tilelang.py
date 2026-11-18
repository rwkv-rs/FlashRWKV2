# SPDX-License-Identifier: MIT
"""Selected FP32 logprobs with a vocabulary reduction independent of batch size."""

import tilelang
import tilelang.language as T


@tilelang.jit(pass_configs={"tl.enable_fast_math": False}, out_idx=[3, 4, 5])
def partial_stats(vocab: int):
    rows = T.dynamic("rows")
    tile = 1024
    parts = T.ceildiv(vocab, tile)

    @T.prim_func
    def kernel(
        logits: T.Tensor((rows, vocab), T.float32),
        targets: T.Tensor((rows,), T.int64),
        offset: T.int32,
        maximum: T.Tensor((rows, parts), T.float32),
        denominator: T.Tensor((rows, parts), T.float32),
        selected: T.Tensor((rows, parts), T.float32),
    ):
        with T.Kernel(rows, parts, threads=128) as (row, part):
            values = T.alloc_fragment((tile,), T.float32)
            exponentials = T.alloc_fragment((tile,), T.float32)
            high = T.alloc_fragment((1,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            for i in T.Parallel(tile):
                col = part * tile + i
                values[i] = T.if_then_else(col < vocab, logits[row, col], -T.infinity(T.float32))
            T.reduce_max(values, high, dim=0)
            for i in T.Parallel(tile):
                exponentials[i] = T.if_then_else(
                    high[0] == -T.infinity(T.float32), 0.0,
                    T.exp(values[i] - high[0]),
                )
            T.reduce_sum(exponentials, total, dim=0)
            maximum[row, part] = high[0]
            denominator[row, part] = total[0]
            local = targets[row] - offset
            if part * tile <= local and local < T.min((part + 1) * tile, vocab):
                selected[row, part] = logits[row, local]
            else:
                selected[row, part] = -T.infinity(T.float32)

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": False}, out_idx=[4, 5, 6])
def finish_stats(vocab: int):
    rows = T.dynamic("rows")
    parts = (vocab + 1023) // 1024
    block = tilelang.next_power_of_2(parts)

    @T.prim_func
    def kernel(
        selected: T.Tensor((rows, parts), T.float32),
        targets: T.Tensor((rows,), T.int64),
        maximum: T.Tensor((rows, parts), T.float32),
        denominator: T.Tensor((rows, parts), T.float32),
        output: T.Tensor((rows,), T.float32),
        logsum: T.Tensor((rows,), T.float32),
        max_values: T.Tensor((rows,), T.float32),
    ):
        with T.Kernel(rows, threads=32) as row:
            maxima = T.alloc_fragment((block,), T.float32)
            sums = T.alloc_fragment((block,), T.float32)
            high = T.alloc_fragment((1,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            for i in T.Parallel(block):
                maxima[i] = T.if_then_else(i < parts, maximum[row, i], -T.infinity(T.float32))
            T.reduce_max(maxima, high, dim=0)
            for i in T.Parallel(block):
                sums[i] = T.if_then_else(
                    i < parts and denominator[row, i] > 0.0,
                    denominator[row, i] * T.exp(maxima[i] - high[0]), 0.0,
                )
            T.reduce_sum(sums, total, dim=0)
            logsum[row] = T.log(total[0])
            max_values[row] = high[0]
            for i in T.Parallel(block):
                maxima[i] = T.if_then_else(i < parts, selected[row, i], -T.infinity(T.float32))
            chosen = T.alloc_fragment((1,), T.float32)
            T.reduce_max(maxima, chosen, dim=0)
            if targets[row] != -100:
                output[row] = T.log(total[0]) - (chosen[0] - high[0])
            else:
                output[row] = T.float32(0)

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": False}, out_idx=[6])
def ce_gradient(vocab: int):
    rows = T.dynamic("rows")

    @T.prim_func
    def kernel(
        logits: T.Tensor((rows, vocab), T.float32),
        targets: T.Tensor((rows,), T.int64),
        logsum: T.Tensor((rows,), T.float32),
        maximum: T.Tensor((rows,), T.float32),
        grad: T.Tensor((rows,), T.float32),
        offset: T.int32,
        result: T.Tensor((rows, vocab), T.float32),
    ):
        with T.Kernel(rows, T.ceildiv(vocab, 256), threads=128) as (row, part):
            for i in T.Parallel(256):
                col = part * 256 + i
                if col < vocab:
                    result[row, col] = T.if_then_else(targets[row] == -100, 0.0, grad[row] * (
                        T.exp((logits[row, col] - maximum[row]) - logsum[row])
                        - T.cast(col + offset == targets[row], T.float32)
                    ))

    return kernel
