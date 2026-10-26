# SPDX-License-Identifier: MIT
"""RWKV-7 key normalization and recurrent directions, with FP32 arithmetic."""

import tilelang
import tilelang.language as T
import torch

from ..._internal.elementwise import sum_partials
from ..._internal.tuning import _AutogradContext


@tilelang.jit(pass_configs={"tl.enable_fast_math": False})
def _prepare_kernel(channels: int, head_size: int):
    rows = T.symbolic("rows")
    heads = channels // head_size

    @T.prim_func
    def kernel(
        key: T.Tensor((rows, channels), T.bfloat16),
        key_scale: T.Tensor((channels,), T.bfloat16),
        rate: T.Tensor((rows, channels), T.bfloat16),
        rate_scale: T.Tensor((channels,), T.bfloat16),
        out: T.Tensor((rows, channels), T.bfloat16),
        a: T.Tensor((rows, channels), T.bfloat16),
        b: T.Tensor((rows, channels), T.bfloat16),
        inverse: T.Tensor((rows, heads), T.float32),
    ):
        with T.Kernel(rows, heads, threads=32) as (row, head):
            scaled = T.alloc_fragment((head_size,), T.float32)
            squares = T.alloc_fragment((head_size,), T.float32)
            norm = T.alloc_fragment((1,), T.float32)
            for i in T.Parallel(head_size):
                col = head * head_size + i
                scaled[i] = T.cast(key[row, col], T.float32) * T.cast(
                    key_scale[col], T.float32
                )
                squares[i] = scaled[i] * scaled[i]
            T.reduce_sum(squares, norm, dim=0)
            norm[0] = 1.0 / T.max(T.sqrt(norm[0]), 1.0e-12)
            inverse[row, head] = norm[0]
            for i in T.Parallel(head_size):
                col = head * head_size + i
                kv = T.cast(key[row, col], T.float32)
                av = T.cast(rate[row, col], T.float32)
                ka = T.cast(rate_scale[col], T.float32)
                direction = scaled[i] * norm[0]
                out[row, col] = kv * (1.0 + (av - 1.0) * ka)
                a[row, col] = -direction
                b[row, col] = direction * av

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": False})
def _backward_kernel(channels: int, head_size: int, need_ks: bool, need_ka: bool):
    rows = T.symbolic("rows")
    heads = channels // head_size
    blocks = T.ceildiv(rows, 32)

    @T.prim_func
    def kernel(
        key: T.Tensor((rows, channels), T.bfloat16),
        key_scale: T.Tensor((channels,), T.bfloat16),
        rate: T.Tensor((rows, channels), T.bfloat16),
        rate_scale: T.Tensor((channels,), T.bfloat16),
        inverse: T.Tensor((rows, heads), T.float32),
        gkey: T.Tensor((rows, channels), T.bfloat16),
        ga: T.Tensor((rows, channels), T.bfloat16),
        gb: T.Tensor((rows, channels), T.bfloat16),
        dkey: T.Tensor((rows, channels), T.bfloat16),
        drate: T.Tensor((rows, channels), T.bfloat16),
        partial_ks: T.Tensor((blocks, channels) if need_ks else (0,), T.float32),
        partial_ka: T.Tensor((blocks, channels) if need_ka else (0,), T.float32),
    ):
        with T.Kernel(blocks, heads, threads=32) as (br, head):
            direction = T.alloc_fragment((head_size,), T.float32)
            gdir = T.alloc_fragment((head_size,), T.float32)
            products = T.alloc_fragment((head_size,), T.float32)
            dot = T.alloc_fragment((1,), T.float32)
            dks = T.alloc_fragment((head_size,), T.float32)
            dka = T.alloc_fragment((head_size,), T.float32)
            T.clear(dks)
            T.clear(dka)
            for r in T.serial(32):
                row = br * 32 + r
                if row < rows:
                    inv = inverse[row, head]
                    for i in T.Parallel(head_size):
                        col = head * head_size + i
                        direction[i] = (
                            T.cast(key[row, col], T.float32)
                            * T.cast(key_scale[col], T.float32)
                            * inv
                        )
                        gdir[i] = -T.cast(ga[row, col], T.float32) + T.cast(
                            gb[row, col], T.float32
                        ) * T.cast(rate[row, col], T.float32)
                        products[i] = direction[i] * gdir[i]
                    T.reduce_sum(products, dot, dim=0)
                    for i in T.Parallel(head_size):
                        col = head * head_size + i
                        k = T.cast(key[row, col], T.float32)
                        ks = T.cast(key_scale[col], T.float32)
                        av = T.cast(rate[row, col], T.float32)
                        ka = T.cast(rate_scale[col], T.float32)
                        gk = T.cast(gkey[row, col], T.float32)
                        projection = T.if_then_else(
                            inv < 1.0e12, direction[i] * dot[0], 0.0
                        )
                        gz = (gdir[i] - projection) * inv
                        dkey[row, col] = gk * (1.0 + (av - 1.0) * ka) + gz * ks
                        drate[row, col] = (
                            gk * k * ka + T.cast(gb[row, col], T.float32) * direction[i]
                        )
                        dks[i] += gz * k
                        dka[i] += gk * k * (av - 1.0)
            for i in T.Parallel(head_size):
                col = head * head_size + i
                if need_ks:
                    partial_ks[br, col] = dks[i]
                if need_ka:
                    partial_ka[br, col] = dka[i]

    return kernel


class _Prepare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, key, key_scale, rate, rate_scale, head_size):
        rows = key.reshape(-1, key.shape[-1])
        outputs = tuple(torch.empty_like(rows) for _ in range(3))
        inverse = torch.empty(
            (rows.shape[0], rows.shape[1] // head_size),
            device=key.device,
            dtype=torch.float32,
        )
        _prepare_kernel(rows.shape[1], head_size)(
            rows,
            key_scale,
            rate.view_as(rows),
            rate_scale,
            *outputs,
            inverse,
        )
        ctx.save_for_backward(key, key_scale, rate, rate_scale, inverse)
        ctx.head_size = head_size
        return tuple(output.view_as(key) for output in outputs)

    @staticmethod
    def backward(ctx, grad_key, grad_a, grad_b):
        key, key_scale, rate, rate_scale, inverse = ctx.saved_tensors
        rows = key.reshape(-1, key.shape[-1])
        dk, da = torch.empty_like(rows), torch.empty_like(rows)
        need_ks, need_ka = ctx.needs_input_grad[1], ctx.needs_input_grad[3]
        partials = [
            torch.empty(
                ((rows.shape[0] + 31) // 32, rows.shape[1]) if needed else (0,),
                device=key.device,
                dtype=torch.float32,
            )
            for needed in (need_ks, need_ka)
        ]
        _backward_kernel(rows.shape[1], ctx.head_size, need_ks, need_ka)(
            rows,
            key_scale,
            rate.view_as(rows),
            rate_scale,
            inverse,
            grad_key.contiguous().view_as(rows),
            grad_a.contiguous().view_as(rows),
            grad_b.contiguous().view_as(rows),
            dk,
            da,
            *partials,
        )
        return (
            dk.view_as(key),
            sum_partials(partials[0]) if need_ks else None,
            da.view_as(rate),
            sum_partials(partials[1]) if need_ka else None,
            None,
        )


@torch.library.custom_op("flashrwkv2::tmix_kk_pre_forward", mutates_args=())
def _compiled_prepare(key: torch.Tensor, key_scale: torch.Tensor, rate: torch.Tensor,
                      rate_scale: torch.Tensor, head_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ctx = _AutogradContext()
    outputs = _Prepare.forward(ctx, key, key_scale, rate, rate_scale, head_size)
    return *outputs, ctx.saved_tensors[-1]


@_compiled_prepare.register_fake
def _compiled_prepare_fake(key, key_scale, rate, rate_scale, head_size):
    return *(torch.empty_like(key) for _ in range(3)), key.new_empty((key.numel() // key.shape[-1], key.shape[-1] // head_size), dtype=torch.float32)


def _compiled_prepare_context(ctx, inputs, output):
    key, key_scale, rate, rate_scale, ctx.head_size = inputs
    ctx.save_for_backward(key, key_scale, rate, rate_scale, output[3])
    ctx.mark_non_differentiable(output[3])


@torch.library.custom_op(
    "flashrwkv2::tmix_kk_pre_backward", mutates_args=(),
    schema="(Tensor gkey, Tensor ga, Tensor gb, Tensor[] saved, int head_size, bool[] needed) -> Tensor?[]",
)
def _compiled_prepare_vjp(gkey, ga, gb, saved, head_size, needed):
    ctx = _AutogradContext(saved_tensors=saved, head_size=head_size, needs_input_grad=(*needed, False))
    return list(_Prepare.backward(ctx, gkey, ga, gb)[:4])


@_compiled_prepare_vjp.register_fake
def _compiled_prepare_vjp_fake(gkey, ga, gb, saved, head_size, needed):
    return [None if i in (1, 3) and not needed[i] else torch.empty_like(t) for i, t in enumerate(saved[:4])]


def _compiled_prepare_backward(ctx, gkey, ga, gb, unused_inverse):
    return (*_compiled_prepare_vjp(gkey, ga, gb, list(ctx.saved_tensors), ctx.head_size, list(ctx.needs_input_grad[:4])), None)


_compiled_prepare.register_autograd(_compiled_prepare_backward, setup_context=_compiled_prepare_context)


def prepare(key, key_scale, rate, rate_scale, head_size=64):
    tensors = (key, key_scale, rate, rate_scale)
    if any(
        t.dtype != torch.bfloat16
        or not t.is_cuda
        or not t.is_contiguous()
        or t.device != key.device
        for t in tensors
    ):
        raise ValueError(
            "KK preparation inputs must be contiguous CUDA bfloat16 on one device"
        )
    if key.ndim < 2 or key.numel() == 0:
        raise ValueError("KK preparation key must have nonempty shape [...,C]")
    if head_size not in (64, 128, 256):
        raise ValueError("head_size must be 64, 128, or 256")
    if key.shape[-1] % head_size:
        raise ValueError("C must be divisible by head_size")
    if (
        rate.shape != key.shape
        or key_scale.shape != (key.shape[-1],)
        or rate_scale.shape != key_scale.shape
    ):
        raise ValueError("KK preparation requires key/rate [...,C] and scales [C]")
    if torch.compiler.is_compiling():
        return _compiled_prepare(key, key_scale, rate, rate_scale, head_size)[:3]
    return _Prepare.apply(key, key_scale, rate, rate_scale, head_size)
