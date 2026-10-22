# SPDX-License-Identifier: MIT
"""Shared BF16 ln forward; every row uses the same FP32 reduction tree."""

import math

import torch
import tilelang
import tilelang.language as T

_ROW_CONFIGS = {}


def _config_key(x, has_res):
    return (x.device.index, x.shape, has_res)


def _row_layout(channels, rows_per_block):
    block = max(64, tilelang.next_power_of_2(channels))
    threads = min(256, max(32, block // 8))
    layout = T.Fragment(
        (rows_per_block, block),
        forward_thread_fn=lambda row, c: row * threads + (c // 2) % threads,
        forward_index_fn=lambda row, c: c // (2 * threads) * 2 + c % 2,
    )
    return block, threads, layout


@T.macro
def _normalize(values, centered, square, mean, variance, weight, bias, output, channels, eps):
    """One row-local FP32 algorithm, also used by fused PostNorm+TokenShift."""
    T.reduce_sum(values, mean, dim=1)
    for r, c in T.Parallel(values.shape[0], values.shape[1]):
        centered[r, c] = T.if_then_else(c < channels, values[r, c] - mean[r] / channels, 0)
        square[r, c] = centered[r, c] * centered[r, c]
    T.reduce_sum(square, variance, dim=1)
    for r, c in T.Parallel(values.shape[0], values.shape[1]):
        if c < channels:
            output[r, c] = (
                centered[r, c] * T.rsqrt(variance[r] / channels + eps)
                * T.cast(weight[c], "float32") + T.cast(bias[c], "float32")
            )


@tilelang.jit(pass_configs={"tl.enable_fast_math": False}, out_idx=[5, 6, 7], compile_flags=["--fmad=false"])
def _forward(
    channels: int, eps: float, has_res: bool, save_stats: bool = True,
    store_res: bool = True, rows_per_block: int = 1,
):
    rows = T.dynamic("rows")
    block, row_threads, layout = _row_layout(channels, rows_per_block)

    @T.prim_func
    def kernel(
        X: T.Tensor((rows, channels), "bfloat16"),
        res: T.Tensor((rows, channels), "bfloat16"),
        Weight: T.Tensor((channels,), "bfloat16"),
        Bias: T.Tensor((channels,), "bfloat16"),
        Sum: T.Tensor((rows if store_res else 0, channels), "bfloat16"),
        Y: T.Tensor((rows, channels), "bfloat16"),
        Mean: T.Tensor((rows if save_stats else 0,), "float32"),
        Rstd: T.Tensor((rows if save_stats else 0,), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows, rows_per_block), threads=row_threads * rows_per_block) as tile:
            values = T.alloc_fragment((rows_per_block, block), "float32")
            centered = T.alloc_fragment((rows_per_block, block), "float32")
            square = T.alloc_fragment((rows_per_block, block), "float32")
            mean = T.alloc_fragment((rows_per_block,), "float32")
            variance = T.alloc_fragment((rows_per_block,), "float32")
            normalized = T.alloc_fragment((rows_per_block, block), "bfloat16")
            T.annotate_layout({
                values: layout, centered: layout, square: layout, normalized: layout,
            })
            for r, c in T.Parallel(rows_per_block, block):
                row = tile * rows_per_block + r
                values[r, c] = 0
                if row < rows and c < channels:
                    if has_res:
                        values[r, c] = T.cast(
                            T.cast(X[row, c], "float32") + T.cast(res[row, c], "float32"),
                            "bfloat16",
                        )
                        if store_res:
                            Sum[row, c] = values[r, c]
                    else:
                        values[r, c] = X[row, c]
            _normalize(values, centered, square, mean, variance, Weight, Bias, normalized, channels, eps)
            if save_stats:
                for r in T.Parallel(rows_per_block):
                    row = tile * rows_per_block + r
                    if row < rows:
                        Mean[row] = mean[r] / channels
                        Rstd[row] = T.rsqrt(variance[r] / channels + eps)
            for r, c in T.Parallel(rows_per_block, block):
                row = tile * rows_per_block + r
                if row < rows and c < channels:
                    Y[row, c] = normalized[r, c]

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": False}, out_idx=[6, 7, 8], compile_flags=["--fmad=false"])
def _backward(
    channels: int, has_dsum: bool, rows_per_block: int = 1,
    need_weight: bool = True, need_bias: bool = True,
):
    rows = T.dynamic("rows")
    block = max(32, tilelang.next_power_of_2(channels))
    threads = min(256, max(32, block // 8))
    layout = T.Fragment(
        (block,), forward_thread_fn=lambda c: c % threads,
        forward_index_fn=lambda c: c // threads,
    )

    @T.prim_func
    def kernel(
        DY: T.Tensor((rows, channels), "bfloat16"),
        DSum: T.Tensor((rows, channels), "bfloat16"),
        Weight: T.Tensor((channels,), "bfloat16"),
        Sum: T.Tensor((rows, channels), "bfloat16"),
        Mean: T.Tensor((rows,), "float32"),
        Rstd: T.Tensor((rows,), "float32"),
        DX: T.Tensor((rows, channels), "bfloat16"),
        DW: T.Tensor((T.ceildiv(rows, rows_per_block) if need_weight else 0, channels), "float32"),
        DB: T.Tensor((T.ceildiv(rows, rows_per_block) if need_bias else 0, channels), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows, rows_per_block), threads=threads) as tile:
            xhat = T.alloc_fragment((block,), "float32")
            weighted = T.alloc_fragment((block,), "float32")
            product = T.alloc_fragment((block,), "float32")
            total = T.alloc_fragment((1,), "float32")
            dot = T.alloc_fragment((1,), "float32")
            dw = T.alloc_fragment((block,), "float32")
            db = T.alloc_fragment((block,), "float32")
            T.annotate_layout({
                xhat: layout, weighted: layout, product: layout, dw: layout, db: layout,
            })
            T.clear(dw)
            T.clear(db)
            for offset in T.serial(rows_per_block):
                row = tile * rows_per_block + offset
                if row < rows:
                    for c in T.Parallel(block):
                        xhat[c] = 0
                        weighted[c] = 0
                        if c < channels:
                            xhat[c] = (T.cast(Sum[row, c], "float32") - Mean[row]) * Rstd[row]
                            weighted[c] = T.cast(DY[row, c], "float32") * T.cast(Weight[c], "float32")
                        product[c] = xhat[c] * weighted[c]
                    T.reduce_sum(weighted, total, dim=0)
                    T.reduce_sum(product, dot, dim=0)
                    for c in T.Parallel(block):
                        if c < channels:
                            if has_dsum:
                                DX[row, c] = (
                                    (weighted[c] - (total[0] + xhat[c] * dot[0]) / channels)
                                    * Rstd[row] + T.cast(DSum[row, c], "float32")
                                )
                            else:
                                DX[row, c] = (
                                    weighted[c] - (total[0] + xhat[c] * dot[0]) / channels
                                ) * Rstd[row]
                            if need_weight:
                                dw[c] += T.cast(DY[row, c], "float32") * xhat[c]
                            if need_bias:
                                db[c] += T.cast(DY[row, c], "float32")
            for c in T.Parallel(block):
                if c < channels:
                    if need_weight:
                        DW[tile, c] = dw[c]
                    if need_bias:
                        DB[tile, c] = db[c]

    return kernel


class _Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, res, eps):
        total = torch.empty_like(x) if res is not None else x
        rows_per_block = _ROW_CONFIGS.get(_config_key(x, res is not None), 1)
        y, mean, rstd = _forward(x.shape[-1], eps, res is not None, rows_per_block=rows_per_block)(
            x, res if res is not None else x, weight, bias, total
        )
        ctx.save_for_backward(total, weight, mean, rstd)
        ctx.has_res = res is not None
        ctx.set_materialize_grads(False)
        return y, total

    @staticmethod
    def backward(ctx, dy, dsum):
        total, weight, mean, rstd = ctx.saved_tensors
        if dy is None:
            return dsum, None, None, dsum if ctx.has_res else None, None
        dy = dy.contiguous()
        from .._internal.elementwise import sum_partials

        group = min(32, tilelang.next_power_of_2(max(1, total.shape[0] // 128)))
        need_weight, need_bias = ctx.needs_input_grad[1:3]
        dx, dw, db = _backward(total.shape[-1], dsum is not None, group, need_weight, need_bias)(
            dy, dsum.contiguous() if dsum is not None else dy, weight, total, mean, rstd
        )
        return (
            dx, sum_partials(dw) if need_weight else None, sum_partials(db) if need_bias else None,
            dx if ctx.has_res else None, None,
        )


def _check_inputs(x, weight, bias, res, eps):
    tensors = (x, weight, bias) if res is None else (x, weight, bias, res)
    if any(t.dtype != torch.bfloat16 or not t.is_cuda for t in tensors):
        raise ValueError("norm inputs must be CUDA bfloat16 tensors")
    if x.ndim < 2 or x.numel() == 0 or weight.shape != (x.shape[-1],) or bias.shape != weight.shape:
        raise ValueError("norm expects nonempty [..., C] inputs and [C] affine parameters")
    if any(t.device != x.device or not t.is_contiguous() for t in tensors):
        raise ValueError("norm inputs must be contiguous and share a device")
    if x.shape[-1] % 2 == 0 and any(t.data_ptr() % 4 for t in tensors):
        raise ValueError("norm inputs must be 4-byte aligned")
    if res is not None and res.shape != x.shape:
        raise ValueError("res must have the same shape as x")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    return tensors


def norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    res: torch.Tensor | None = None,
    *,
    return_res: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """ln over the final dimension, with an optional BF16-rounded res sum.

    Forward is identical with and without autograd. Casts have PyTorch's usual
    identity gradient; ln statistics and all arithmetic are FP32. ``return_res``
    exposes the same rounded sum used to compute the statistics.
    """
    tensors = _check_inputs(x, weight, bias, res, eps)
    flat = x.view(-1, x.shape[-1])
    flat_res = None if res is None else res.view_as(flat)
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        y, total = _Norm.apply(flat, weight, bias, flat_res, float(eps))
    else:
        store_res = return_res and res is not None
        total = torch.empty_like(flat) if store_res else flat
        buffer = total if store_res else flat.new_empty((0, x.shape[-1]))
        rows_per_block = _ROW_CONFIGS.get(_config_key(flat, res is not None), 1)
        y, _, _ = _forward(x.shape[-1], float(eps), res is not None, False, store_res, rows_per_block)(
            flat, flat_res if flat_res is not None else flat, weight, bias, buffer
        )
    return (y.view_as(x), total.view_as(x)) if return_res else y.view_as(x)


@torch.no_grad()
def tune_norm(x, weight, bias, eps=1e-5, res=None):
    """Select row grouping while preserving the canonical per-row FP32 tree."""
    from .._internal.tuning import select_config

    norm(x, weight, bias, eps, res)
    flat = x.view(-1, x.shape[-1])
    other = flat if res is None else res.view_as(flat)
    summed = torch.empty_like(flat) if res is not None else flat
    buffer = flat.new_empty((0, flat.shape[-1]))

    def run(group, values=flat, residual=other):
        return _forward(flat.shape[-1], eps, res is not None, False, False, group)(
            values, residual, weight, bias, buffer
        )[0]

    def check(group, values=flat, residual=other):
        total = torch.empty_like(values) if res is not None else values
        training = _forward(flat.shape[-1], eps, res is not None, rows_per_block=group)(
            values, residual, weight, bias, total
        )
        outputs = (*training, run(group, values, residual))
        if res is not None:
            stored = torch.empty_like(values)
            production = _forward(flat.shape[-1], eps, True, False, True, group)(
                values, residual, weight, bias, stored
            )[0]
            outputs += (production, stored)
        return outputs

    training = _forward(flat.shape[-1], eps, res is not None)(flat, other, weight, bias, summed)
    # Keep FP32 statistics in the check and compare both production stores
    # directly with the canonical training result.
    reference = (*training, training[0])
    if res is not None:
        reference += (training[0], summed.clone())
    for row in sorted({0, flat.shape[0] // 2, flat.shape[0] - 1}):
        single = check(1, flat[row:row + 1], other[row:row + 1])
        if not all(torch.equal(full[row:row + 1], one) for full, one in zip(reference, single, strict=True)):
            raise RuntimeError("canonical norm failed batch invariance")

    best, measurements = select_config((1, 2, 4), run, check=check, reference=reference)
    _ROW_CONFIGS[_config_key(flat, res is not None)] = best
    return measurements
