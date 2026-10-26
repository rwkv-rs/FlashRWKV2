# SPDX-License-Identifier: MIT
"""Shared BF16 activations and FP32 reductions for parameter gradients."""

import tilelang
import tilelang.language as T
import torch

from .tuning import _AutogradContext


@tilelang.jit(pass_configs={"tl.enable_fast_math": False})
def _activation_kernel(
    channels: int,
    kind: str,
    has_bias: bool,
    backward: bool,
    parameter_grad: bool = True,
):
    rows = T.symbolic("rows")
    block_c = min(128 if backward else 256, 1 << (channels - 1).bit_length())
    block_r = 32 if backward else 256 // block_c
    bias_shape = (channels,) if has_bias else (0,)
    reduce_bias = backward and has_bias and parameter_grad
    partial_shape = (T.ceildiv(rows, block_r), channels) if reduce_bias else (0,)

    @T.prim_func
    def kernel(
        x: T.Tensor((rows, channels), T.bfloat16),
        bias: T.Tensor(bias_shape, T.bfloat16),
        dy: T.Tensor((rows, channels), T.bfloat16),
        out: T.Tensor((rows, channels), T.bfloat16),
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
                    value = T.cast(x[row, col], T.float32)
                    if has_bias:
                        value += T.cast(bias[col], T.float32)
                    if kind == "sigmoid":
                        y = T.sigmoid(value)
                        derivative = y * (1.0 - y)
                    elif kind == "tanh":
                        y = T.tanh(value)
                        derivative = 1.0 - y * y
                    elif kind == "identity":
                        y = value
                        derivative = 1.0
                    else:
                        positive = T.max(value, 0.0)
                        y = positive * positive
                        derivative = 2.0 * positive
                    if backward:
                        grad = T.cast(dy[row, col], T.float32) * derivative
                        out[row, col] = grad
                        gradients[i, j] = grad
                    else:
                        out[row, col] = y
                elif backward:
                    gradients[i, j] = 0.0
            if reduce_bias:
                T.reduce_sum(gradients, column_sum, dim=0)
                for j in T.Parallel(block_c):
                    if bc * block_c + j < channels:
                        partial[br, bc * block_c + j] = column_sum[j]

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": False})
def _sum_partials_kernel(channels: int):
    blocks = T.symbolic("blocks")

    @T.prim_func
    def kernel(
        partial: T.Tensor((blocks, channels), T.float32),
        output: T.Tensor((channels,), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(channels, 128), threads=128) as bc:
            total = T.alloc_fragment((128,), T.float32)
            T.clear(total)
            for block in T.serial(blocks):
                for j in T.Parallel(128):
                    if bc * 128 + j < channels:
                        total[j] += partial[block, bc * 128 + j]
            for j in T.Parallel(128):
                if bc * 128 + j < channels:
                    output[bc * 128 + j] = total[j]

    return kernel


def sum_partials(partial):
    """Finish a parameter VJP from fixed-order FP32 block partials."""
    output = torch.empty(partial.shape[1], dtype=torch.bfloat16, device=partial.device)
    _sum_partials_kernel(partial.shape[1])(partial, output)
    return output


class _Activation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias, kind):
        rows = x.reshape(-1, x.shape[-1])
        output = torch.empty_like(rows)
        placeholder = x.new_empty(0)
        _activation_kernel(rows.shape[1], kind, bias is not None, False)(
            rows,
            placeholder if bias is None else bias,
            rows,
            output,
            x.new_empty(0, dtype=torch.float32),
        )
        ctx.save_for_backward(x, bias)
        ctx.kind = kind
        return output.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        x, bias = ctx.saved_tensors
        rows = x.reshape(-1, x.shape[-1])
        dx = torch.empty_like(rows)
        need_bias = ctx.needs_input_grad[1]
        partial = torch.empty(
            ((rows.shape[0] + 31) // 32, rows.shape[1]) if need_bias else (0,),
            device=x.device,
            dtype=torch.float32,
        )
        _activation_kernel(rows.shape[1], ctx.kind, bias is not None, True, need_bias)(
            rows,
            x.new_empty(0) if bias is None else bias,
            grad.contiguous().view_as(rows),
            dx,
            partial,
        )
        return dx.view_as(x), sum_partials(partial) if need_bias else None, None


@torch.library.custom_op("flashrwkv2::_activation_forward", mutates_args=())
def _compiled_activation(x: torch.Tensor, bias: torch.Tensor | None, kind: str) -> torch.Tensor:
    return _Activation.forward(_AutogradContext(), x, bias, kind)


@_compiled_activation.register_fake
def _compiled_activation_fake(x, bias, kind):
    return torch.empty_like(x)


def _compiled_activation_context(ctx, inputs, output):
    x, bias, ctx.kind = inputs
    ctx.save_for_backward(x, bias)


@torch.library.custom_op(
    "flashrwkv2::_activation_backward", mutates_args=(),
    schema="(Tensor dy, Tensor x, Tensor? bias, str kind, bool need_bias) -> (Tensor, Tensor?)",
)
def _compiled_activation_vjp(dy, x, bias, kind, need_bias):
    ctx = _AutogradContext(saved_tensors=(x, bias), kind=kind,
                           needs_input_grad=(True, need_bias, False))
    return _Activation.backward(ctx, dy)[:2]


@_compiled_activation_vjp.register_fake
def _compiled_activation_vjp_fake(dy, x, bias, kind, need_bias):
    return torch.empty_like(x), torch.empty_like(bias) if need_bias else None


def _compiled_activation_backward(ctx, dy):
    return (*_compiled_activation_vjp(dy, *ctx.saved_tensors, ctx.kind, ctx.needs_input_grad[1]), None)


_compiled_activation.register_autograd(_compiled_activation_backward, setup_context=_compiled_activation_context)


def activation(x, kind, bias=None):
    """Apply sigmoid, tanh, ReLU², or identity+bias in FP32; return BF16."""
    if kind not in {"sigmoid", "tanh", "relu2", "identity"}:
        raise ValueError("activation must be sigmoid, tanh, relu2, or identity")
    if x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("activation input must be contiguous CUDA bfloat16")
    if x.ndim < 2 or x.numel() == 0:
        raise ValueError("activation input must have nonempty shape [...,C]")
    if bias is not None and (
        bias.shape != (x.shape[-1],)
        or bias.dtype != x.dtype
        or bias.device != x.device
        or not bias.is_contiguous()
    ):
        raise ValueError(
            "activation bias must be contiguous [C] with the input dtype/device"
        )
    operation = _compiled_activation if torch.compiler.is_compiling() else _Activation.apply
    return operation(x, bias, kind)
