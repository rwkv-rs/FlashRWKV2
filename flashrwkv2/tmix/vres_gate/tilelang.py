# SPDX-License-Identifier: MIT
"""Shared value-residual gate, evaluated in FP32 with BF16 output."""

import tilelang
import tilelang.language as T
import torch

from ..._internal.elementwise import sum_partials


@tilelang.jit(pass_configs={"tl.enable_fast_math": False})
def _forward_kernel(channels: int):
    rows = T.symbolic("rows")

    @T.prim_func
    def kernel(
        value: T.Tensor((rows, channels), T.bfloat16),
        first: T.Tensor((rows, channels), T.bfloat16),
        bias: T.Tensor((channels,), T.bfloat16),
        logits: T.Tensor((rows, channels), T.bfloat16),
        output: T.Tensor((rows, channels), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(rows * channels, 256), threads=128) as block:
            for i in T.Parallel(256):
                index = block * 256 + i
                if index < rows * channels:
                    row, col = index // channels, index % channels
                    v = T.cast(value[row, col], T.float32)
                    vf = T.cast(first[row, col], T.float32)
                    gate = T.sigmoid(
                        T.cast(bias[col], T.float32)
                        + T.cast(logits[row, col], T.float32)
                    )
                    output[row, col] = v + (vf - v) * gate

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": False})
def _backward_kernel(channels: int, need_bias: bool):
    rows = T.symbolic("rows")
    block_r, block_c = 32, min(128, 1 << (channels - 1).bit_length())
    partial_shape = (T.ceildiv(rows, block_r), channels) if need_bias else (0,)

    @T.prim_func
    def kernel(
        value: T.Tensor((rows, channels), T.bfloat16),
        first: T.Tensor((rows, channels), T.bfloat16),
        bias: T.Tensor((channels,), T.bfloat16),
        logits: T.Tensor((rows, channels), T.bfloat16),
        dy: T.Tensor((rows, channels), T.bfloat16),
        dvalue: T.Tensor((rows, channels), T.bfloat16),
        dfirst: T.Tensor((rows, channels), T.bfloat16),
        dlogits: T.Tensor((rows, channels), T.bfloat16),
        partial: T.Tensor(partial_shape, T.float32),
    ):
        with T.Kernel(
            T.ceildiv(channels, block_c), T.ceildiv(rows, block_r), threads=128
        ) as (bc, br):
            gradients = T.alloc_fragment((block_r, block_c), T.float32)
            column_sum = T.alloc_fragment((block_c,), T.float32)
            for i, j in T.Parallel(block_r, block_c):
                row, col = br * block_r + i, bc * block_c + j
                if row < rows and col < channels:
                    v = T.cast(value[row, col], T.float32)
                    vf = T.cast(first[row, col], T.float32)
                    g = T.cast(dy[row, col], T.float32)
                    gate = T.sigmoid(
                        T.cast(bias[col], T.float32)
                        + T.cast(logits[row, col], T.float32)
                    )
                    gf = g * gate
                    gl = g * (vf - v) * gate * (1.0 - gate)
                    dvalue[row, col] = g - gf
                    dfirst[row, col] = gf
                    dlogits[row, col] = gl
                    gradients[i, j] = gl
                else:
                    gradients[i, j] = 0.0
            if need_bias:
                T.reduce_sum(gradients, column_sum, dim=0)
                for j in T.Parallel(block_c):
                    if bc * block_c + j < channels:
                        partial[br, bc * block_c + j] = column_sum[j]

    return kernel


class _VRes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, first, bias, logits):
        rows = value.reshape(-1, value.shape[-1])
        output = torch.empty_like(rows)
        _forward_kernel(rows.shape[1])(
            rows, first.view_as(rows), bias, logits.view_as(rows), output
        )
        ctx.save_for_backward(value, first, bias, logits)
        return output.view_as(value)

    @staticmethod
    def backward(ctx, grad):
        value, first, bias, logits = ctx.saved_tensors
        rows = value.reshape(-1, value.shape[-1])
        outputs = tuple(torch.empty_like(rows) for _ in range(3))
        need_bias = ctx.needs_input_grad[2]
        partial = torch.empty(
            ((rows.shape[0] + 31) // 32, rows.shape[1]) if need_bias else (0,),
            device=value.device,
            dtype=torch.float32,
        )
        _backward_kernel(rows.shape[1], need_bias)(
            rows,
            first.view_as(rows),
            bias,
            logits.view_as(rows),
            grad.contiguous().view_as(rows),
            *outputs,
            partial,
        )
        dv, df, dl = (output.view_as(value) for output in outputs)
        return dv, df, sum_partials(partial) if need_bias else None, dl


def vres(value, first, bias, logits):
    tensors = (value, first, bias, logits)
    if any(
        t.dtype != torch.bfloat16
        or not t.is_cuda
        or not t.is_contiguous()
        or t.device != value.device
        for t in tensors
    ):
        raise ValueError("VRes inputs must be contiguous CUDA bfloat16 on one device")
    if value.ndim < 2 or value.numel() == 0:
        raise ValueError("VRes value must have nonempty shape [...,C]")
    if (
        first.shape != value.shape
        or logits.shape != value.shape
        or bias.shape != (value.shape[-1],)
    ):
        raise ValueError("VRes requires value/first/logits [...,C] and bias [C]")
    return _VRes.apply(value, first, bias, logits)
