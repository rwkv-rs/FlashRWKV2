# SPDX-License-Identifier: MIT
"""Packed TokenShift with identical BF16 arithmetic at every chunk boundary."""

from collections.abc import Sequence

import tilelang
import tilelang.language as T
import torch

from ..._internal.tuning import _AutogradContext, device_class, tuning_operation

from ...post_norm.tilelang import _check_inputs as _check_norm
from ...post_norm.tilelang import _normalize, _row_layout

_BLOCK_CONFIGS = {}
_FUSED_CONFIGS = {}


def _config_key(x, mixes, batch):
    return (device_class(x.device), x.shape, len(mixes), batch)


def _pack_mixes(x, mixes, *, pack=True):
    if isinstance(mixes, torch.Tensor) and pack:
        return mixes
    mixes = tuple(mixes)
    if not mixes or any(t.shape != (x.shape[-1],) for t in mixes):
        raise ValueError("TokenShift parameters must have shape [C]")
    if any(t.device != x.device or t.dtype != torch.bfloat16 or not t.is_contiguous() for t in mixes):
        raise ValueError("TokenShift parameters must be contiguous bfloat16 and share x's device")
    if not pack:
        if x.shape[-1] % 2 == 0 and any(mix.data_ptr() % 4 for mix in mixes):
            mixes = tuple(torch.stack(mixes).unbind(0))
        return mixes
    return mixes[0].unsqueeze(0) if len(mixes) == 1 else torch.stack(mixes)


@tilelang.jit(pass_configs={"tl.enable_fast_math": False}, out_idx=[13, 14], compile_flags=["--fmad=false"])
def _fused_forward(channels: int, count: int, eps: float, rows_per_block: int = 1):
    rows, slots = T.dynamic("rows, slots")
    block, row_threads, layout = _row_layout(channels, rows_per_block)

    @T.prim_func
    def kernel(
        X: T.Tensor((rows, channels), "bfloat16"),
        res: T.Tensor((rows, channels), "bfloat16"),
        Weight: T.Tensor((channels,), "bfloat16"),
        Bias: T.Tensor((channels,), "bfloat16"),
        Mix0: T.Tensor((channels,), "bfloat16"),
        Mix1: T.Tensor((channels,), "bfloat16"),
        Mix2: T.Tensor((channels,), "bfloat16"),
        Mix3: T.Tensor((channels,), "bfloat16"),
        Mix4: T.Tensor((channels,), "bfloat16"),
        Mix5: T.Tensor((channels,), "bfloat16"),
        Pool: T.Tensor((slots, channels), "bfloat16"),
        Indices: T.Tensor((rows,), "int32"),
        Status: T.Tensor((3,), "int32"),
        Sum: T.Tensor((rows, channels), "bfloat16"),
        Y: T.Tensor((count, rows, channels), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows, rows_per_block), threads=row_threads * rows_per_block) as tile:
            values = T.alloc_fragment((rows_per_block, block), "float32")
            centered = T.alloc_fragment((rows_per_block, block), "float32")
            square = T.alloc_fragment((rows_per_block, block), "float32")
            mean = T.alloc_fragment((rows_per_block,), "float32")
            variance = T.alloc_fragment((rows_per_block,), "float32")
            normalized = T.alloc_fragment((rows_per_block, block), "bfloat16")
            previous = T.alloc_fragment((rows_per_block, block), "bfloat16")
            T.annotate_layout({
                values: layout, centered: layout, square: layout,
                normalized: layout, previous: layout,
            })
            for r, c in T.Parallel(rows_per_block, block):
                row = tile * rows_per_block + r
                values[r, c] = 0
                if row < rows and c < channels and Status[0] == 0 and row < Status[2]:
                    values[r, c] = T.cast(
                        T.cast(X[row, c], "float32") + T.cast(res[row, c], "float32"), "bfloat16",
                    )
                    Sum[row, c] = values[r, c]
                    previous[r, c] = Pool[Indices[row], c]
                elif row < rows and c < channels:
                    Sum[row, c] = T.float32(float("nan"))
            _normalize(values, centered, square, mean, variance, Weight, Bias, normalized, channels, eps)
            for m in T.unroll(count):
                for r, c in T.Parallel(rows_per_block, block):
                    row = tile * rows_per_block + r
                    if row < rows and c < channels and Status[0] == 0 and row < Status[2]:
                        current = T.cast(normalized[r, c], "float32")
                        factor = T.cast(T.if_then_else(
                            m == 0, Mix0[c], T.if_then_else(
                                m == 1, Mix1[c], T.if_then_else(
                                    m == 2, Mix2[c], T.if_then_else(
                                        m == 3, Mix3[c], T.if_then_else(
                                            m == 4, Mix4[c], Mix5[c]
                                        )
                                    )
                                )
                            )
                        ), "float32")
                        Y[m, row, c] = current + (
                            T.cast(previous[r, c], "float32") - current
                        ) * factor
                    elif row < rows and c < channels:
                        Y[m, row, c] = T.float32(float("nan"))
            for r, c in T.Parallel(rows_per_block, block):
                row = tile * rows_per_block + r
                if row < rows and c < channels and Status[0] == 0 and row < Status[2]:
                    Pool[Indices[row], c] = normalized[r, c]

    return kernel


@torch.no_grad()
def fused_post_norm_tokenshift(x, res, weight, bias, mixes, pool, indices, status, eps=1e-5):
    """Fuse T=1 ln and indexed TokenShift for TMix's six or CMix's one mix."""
    _check_norm(x, weight, bias, res, eps)
    mixes = _pack_mixes(x, mixes, pack=False)
    if (
        x.ndim != 2 or indices.shape != (x.shape[0],)
    ):
        raise ValueError("fused TokenShift expects one packed row per sequence and mixes [K,C]")
    if (
        pool.ndim != 2 or pool.shape[1] != x.shape[1] or pool.dtype != torch.bfloat16
        or pool.device != x.device or not pool.is_contiguous()
    ):
        raise ValueError("shift state must be contiguous CUDA bfloat16 [slots,C]")
    if x.shape[1] % 2 == 0 and pool.data_ptr() % 4:
        raise ValueError("fused TokenShift state and parameters must be 4-byte aligned")
    from ..._internal.tuning import get_config
    group = get_config(_FUSED_CONFIGS, "tmix/tokenshift/fused", _config_key(x, mixes, x.shape[0]), 1, tune=lambda: tune_fused_tokenshift(x, res, weight, bias, mixes, pool, indices, status, eps))
    # CMix reuses its pointer for the unused TMix parameter arguments.
    mix_args = mixes + (mixes[-1],) * (6 - len(mixes))
    total, output = _fused_forward(x.shape[1], len(mixes), float(eps), group)(
        x, res, weight, bias, *mix_args, pool, indices, status
    )
    return total, tuple(output.unbind(0))


@tilelang.jit(pass_configs={"tl.enable_fast_math": False}, out_idx=[5, 6], compile_flags=["--fmad=false"])
def _forward(
    channels: int, count: int, batch: int, index_dtype: str,
    block: int = 256, use_status: bool = False,
):
    rows = T.dynamic("rows")
    threads = min(128, block // 2)
    layout = T.Fragment(
        (block,),
        forward_thread_fn=lambda c: (c // 2) % threads,
        forward_index_fn=lambda c: c // (2 * threads) * 2 + c % 2,
    )

    @T.prim_func
    def kernel(
        X: T.Tensor((rows, channels), "bfloat16"),
        Mix: T.Tensor((count, channels), "bfloat16"),
        Cu: T.Tensor((batch + 1,), index_dtype),
        Initial: T.Tensor((batch, channels), "bfloat16"),
        Status: T.Tensor((3 if use_status else 0,), index_dtype),
        Y: T.Tensor((count, rows, channels), "bfloat16"),
        Final: T.Tensor((batch, channels), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(channels, block), T.max(rows, batch), threads=threads) as (bc, row):
            lo = T.alloc_local((1,), "int32")
            hi = T.alloc_local((1,), "int32")
            current = T.alloc_fragment((block,), "float32")
            previous = T.alloc_fragment((block,), "float32")
            final = T.alloc_fragment((block,), "bfloat16")
            T.annotate_layout({current: layout, previous: layout, final: layout})
            if row < rows and (not use_status or (Status[0] == 0 and row < Status[1])):
                lo[0], hi[0] = 0, T.if_then_else(use_status, Status[2], batch)
                for _ in T.unroll(batch.bit_length()):
                    if lo[0] < hi[0]:
                        middle = (lo[0] + hi[0]) // 2
                        if Cu[middle + 1] <= row:
                            lo[0] = middle + 1
                        else:
                            hi[0] = middle
                for c in T.Parallel(block):
                    channel = bc * block + c
                    if channel < channels:
                        current[c] = X[row, channel]
                        previous[c] = T.if_then_else(
                            row == Cu[lo[0]], Initial[lo[0], channel], X[row - 1, channel]
                        )
                for m in T.unroll(count):
                    for c in T.Parallel(block):
                        channel = bc * block + c
                        if channel < channels:
                            Y[m, row, channel] = current[c] + (
                                previous[c] - current[c]
                            ) * T.cast(Mix[m, channel], "float32")
            if row < batch and (not use_status or (Status[0] == 0 and row < Status[2])):
                for c in T.Parallel(block):
                    channel = bc * block + c
                    if channel < channels:
                        final[c] = T.if_then_else(
                            Cu[row + 1] > Cu[row], X[Cu[row + 1] - 1, channel], Initial[row, channel]
                        )
                for c in T.Parallel(block):
                    channel = bc * block + c
                    if channel < channels:
                        Final[row, channel] = final[c]

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": False}, out_idx=[6, 7, 8], compile_flags=["--fmad=false"])
def _backward(
    channels: int, count: int, batch: int, index_dtype: str, block: int = 256,
    rows_per_block: int = 1, need_mix: bool = True,
):
    rows = T.dynamic("rows")
    threads = min(128, block // 2)
    layout = T.Fragment(
        (block,),
        forward_thread_fn=lambda c: (c // 2) % threads,
        forward_index_fn=lambda c: c // (2 * threads) * 2 + c % 2,
    )

    partial_layout = T.Fragment(
        (count, block),
        forward_thread_fn=lambda m, c: (c // 2) % threads,
        forward_index_fn=lambda m, c: m * (block // threads) + c // (2 * threads) * 2 + c % 2,
    )

    @T.prim_func
    def kernel(
        DY: T.Tensor((count, rows, channels), "bfloat16"),
        DFinal: T.Tensor((batch, channels), "bfloat16"),
        X: T.Tensor((rows, channels), "bfloat16"),
        Mix: T.Tensor((count, channels), "bfloat16"),
        Cu: T.Tensor((batch + 1,), index_dtype),
        Initial: T.Tensor((batch, channels), "bfloat16"),
        DX: T.Tensor((rows, channels), "bfloat16"),
        DMix: T.Tensor((T.ceildiv(rows, rows_per_block) if need_mix else 0, count, channels), "float32"),
        DInitial: T.Tensor((batch, channels), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(channels, block), T.max(T.ceildiv(rows, rows_per_block), batch), threads=threads) as (bc, tile):
            lo = T.alloc_local((1,), "int32")
            hi = T.alloc_local((1,), "int32")
            total = T.alloc_fragment((block,), "float32")
            partial = T.alloc_fragment((count, block), "float32")
            T.annotate_layout({total: layout, partial: partial_layout})
            T.clear(partial)
            for offset in T.serial(rows_per_block):
                row = tile * rows_per_block + offset
                if row < rows:
                    lo[0], hi[0] = 0, batch
                    for _ in T.unroll(batch.bit_length()):
                        if lo[0] < hi[0]:
                            middle = (lo[0] + hi[0]) // 2
                            if Cu[middle + 1] <= row:
                                lo[0] = middle + 1
                            else:
                                hi[0] = middle
                    T.clear(total)
                    for m in T.unroll(count):
                        for c in T.Parallel(block):
                            channel = bc * block + c
                            if channel < channels:
                                coeff = T.cast(Mix[m, channel], "float32")
                                gradient = T.cast(DY[m, row, channel], "float32")
                                previous = T.cast(T.if_then_else(
                                    row == Cu[lo[0]], Initial[lo[0], channel], X[row - 1, channel]
                                ), "float32")
                                total[c] += gradient * (1 - coeff)
                                if row + 1 < Cu[lo[0] + 1]:
                                    total[c] += T.cast(DY[m, row + 1, channel], "float32") * coeff
                                if need_mix:
                                    partial[m, c] += gradient * (
                                        previous - T.cast(X[row, channel], "float32")
                                    )
                    for c in T.Parallel(block):
                        channel = bc * block + c
                        if channel < channels:
                            if row + 1 == Cu[lo[0] + 1]:
                                total[c] += T.cast(DFinal[lo[0], channel], "float32")
                            DX[row, channel] = total[c]
            if need_mix and tile < T.ceildiv(rows, rows_per_block):
                for m, c in T.Parallel(count, block):
                    channel = bc * block + c
                    if channel < channels:
                        DMix[tile, m, channel] = partial[m, c]
            if tile < batch:
                T.clear(total)
                if Cu[tile + 1] > Cu[tile]:
                    for m in T.unroll(count):
                        for c in T.Parallel(block):
                            channel = bc * block + c
                            if channel < channels:
                                total[c] += (
                                    T.cast(DY[m, Cu[tile], channel], "float32")
                                    * T.cast(Mix[m, channel], "float32")
                                )
                else:
                    for c in T.Parallel(block):
                        channel = bc * block + c
                        if channel < channels:
                            total[c] = DFinal[tile, channel]
                for c in T.Parallel(block):
                    channel = bc * block + c
                    if channel < channels:
                        DInitial[tile, channel] = total[c]

    return kernel


class _TokenShift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mixes, cu_seqlens, initial):
        if x.shape[-1] % 2 == 0 and any(t.data_ptr() % 4 for t in (x, mixes, initial)):
            raise ValueError("tokenshift inputs must be 4-byte aligned")
        from ..._internal.tuning import get_config
        block = get_config(_BLOCK_CONFIGS, "tmix/tokenshift", _config_key(x, mixes, initial.shape[0]), 256, tune=lambda: tune_tokenshift(x, mixes, cu_seqlens, initial))
        options = (x.shape[-1], mixes.shape[0], initial.shape[0], str(cu_seqlens.dtype).split(".")[-1], block)
        y, final = _forward(*options)(x, mixes, cu_seqlens, initial, cu_seqlens[:0])
        ctx.save_for_backward(x, mixes, cu_seqlens, initial)
        ctx.options = options
        return y, final

    @staticmethod
    def backward(ctx, dy, dfinal):
        x, mixes, cu, initial = ctx.saved_tensors
        from ..._internal.elementwise import sum_partials

        group = min(32, tilelang.next_power_of_2(max(1, x.shape[0] // 128)))
        need_mix = ctx.needs_input_grad[1]
        dx, dm, di = _backward(*ctx.options, group, need_mix)(
            dy.contiguous(), dfinal.contiguous(), x, mixes, cu, initial
        )
        gradient = sum_partials(dm.flatten(1)).view_as(mixes) if need_mix else None
        return dx, gradient, None, di


@torch.library.custom_op("flashrwkv2::tmix_tokenshift_forward", mutates_args=())
def _compiled_tokenshift(
    x: torch.Tensor, mixes: torch.Tensor, cu: torch.Tensor, initial: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _TokenShift.forward(_AutogradContext(), x, mixes, cu, initial)


@_compiled_tokenshift.register_fake
def _compiled_tokenshift_fake(x, mixes, cu, initial):
    return x.new_empty((mixes.shape[0], *x.shape)), torch.empty_like(initial)


def _compiled_tokenshift_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


@torch.library.custom_op(
    "flashrwkv2::tmix_tokenshift_backward", mutates_args=(),
    schema="(Tensor dy, Tensor dfinal, Tensor x, Tensor mixes, Tensor cu, Tensor initial, bool need_mix) -> (Tensor, Tensor?, Tensor)",
)
def _compiled_tokenshift_vjp(
    dy: torch.Tensor, dfinal: torch.Tensor, x: torch.Tensor, mixes: torch.Tensor,
    cu: torch.Tensor, initial: torch.Tensor, need_mix: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    from ..._internal.tuning import get_config
    block = get_config(_BLOCK_CONFIGS, "tmix/tokenshift", _config_key(x, mixes, initial.shape[0]), 256,
                       tune=lambda: tune_tokenshift(x, mixes, cu, initial))
    ctx = _AutogradContext(saved_tensors=(x, mixes, cu, initial),
                          options=(x.shape[-1], mixes.shape[0], initial.shape[0], str(cu.dtype).split(".")[-1], block),
                          needs_input_grad=(True, need_mix, False, True))
    dx, dm, _, di = _TokenShift.backward(ctx, dy, dfinal)
    return dx, dm, di


@_compiled_tokenshift_vjp.register_fake
def _compiled_tokenshift_vjp_fake(dy, dfinal, x, mixes, cu, initial, need_mix):
    return torch.empty_like(x), torch.empty_like(mixes) if need_mix else None, torch.empty_like(initial)


def _compiled_tokenshift_backward(ctx, dy, dfinal):
    dx, dm, di = _compiled_tokenshift_vjp(dy, dfinal, *ctx.saved_tensors, ctx.needs_input_grad[1])
    return dx, dm, None, di


_compiled_tokenshift.register_autograd(_compiled_tokenshift_backward, setup_context=_compiled_tokenshift_context)


def tokenshift(
    x: torch.Tensor,
    mixes: torch.Tensor | Sequence[torch.Tensor],
    cu_seqlens: torch.Tensor,
    initial_shift: torch.Tensor | None = None,
    *,
    _status: torch.Tensor | None = None,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Mix packed rows with their predecessor, preserving differentiable state.

    ``mixes`` is [K,C] or K vectors; stacking vectors can be avoided by callers
    storing packed parameters. Empty sequences preserve their initial shift.
    Metadata must describe a packed input: nondecreasing offsets from 0 to T.
    """
    mixes = _pack_mixes(x, mixes)
    if x.ndim != 2 or x.shape[-1] == 0 or mixes.ndim != 2 or mixes.shape[-1] != x.shape[-1] or mixes.shape[0] == 0:
        raise ValueError("tokenshift expects x [T,C] and mixes [K,C]")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2 or cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must be an int32 or int64 [B+1] tensor")
    batch, channels = cu_seqlens.numel() - 1, x.shape[-1]
    if initial_shift is None:
        initial_shift = x.new_zeros(batch, channels)
    tensors = (x, mixes, initial_shift)
    if any(t.dtype != torch.bfloat16 or not t.is_cuda for t in tensors):
        raise ValueError("tokenshift inputs must be CUDA bfloat16")
    if initial_shift.shape != (batch, channels):
        raise ValueError("initial_shift must have shape [B,C]")
    if any(t.device != x.device or not t.is_contiguous() for t in (*tensors, cu_seqlens)):
        raise ValueError("tokenshift inputs and metadata must be contiguous and share a device")
    if _status is not None and channels % 2 == 0 and any(t.data_ptr() % 4 for t in tensors):
        raise ValueError("tokenshift inputs must be 4-byte aligned")
    if _status is None:
        operation = _compiled_tokenshift if torch.compiler.is_compiling() else _TokenShift.apply
        y, final = operation(x, mixes, cu_seqlens, initial_shift)
    else:
        from ..._internal.tuning import get_config
        block = get_config(_BLOCK_CONFIGS, "tmix/tokenshift", _config_key(x, mixes, batch), 256, tune=lambda: tune_tokenshift(x, mixes, cu_seqlens, initial_shift, _status=_status))
        y, final = _forward(channels, mixes.shape[0], batch, str(cu_seqlens.dtype).split(".")[-1], block, True)(
            x, mixes, cu_seqlens, initial_shift, _status
        )
    return tuple(y.unbind(0)), final


@torch.no_grad()
@tuning_operation
def tune_tokenshift(x, mixes, cu_seqlens, initial_shift=None, *, _status=None):
    """Check functional/status variants and time the requested production path.

    Tuning requires valid, nonempty active metadata before Graph capture.
    A live status is read once; inactive metadata tails are excluded from the
    functional reference, and only defined active status outputs are compared.
    """
    from ..._internal.tuning import save_config, select_config

    mixes = _pack_mixes(x, mixes)
    batch, channels = cu_seqlens.numel() - 1, x.shape[-1]
    if initial_shift is None:
        initial_shift = x.new_zeros((batch, channels))
    if _status is None:
        active_rows, active_batch = x.shape[0], batch
        status = cu_seqlens.new_tensor([0, active_rows, active_batch])
        valid = True
    else:
        status = _status
        error, active_rows, active_batch = status.tolist()
        valid = error == 0
    if not valid or not 0 < active_rows <= x.shape[0] or not 0 < active_batch <= batch:
        raise ValueError("TokenShift tuning requires valid nonempty active tokens and sequences")
    tokenshift(x, mixes, cu_seqlens, initial_shift, _status=status)
    options = (channels, mixes.shape[0], batch, str(cu_seqlens.dtype).split(".")[-1])
    functional_cu = cu_seqlens
    if active_batch < batch:
        functional_cu = cu_seqlens.clone()
        functional_cu[active_batch + 1:] = active_rows

    def functional(block):
        return _forward(*options, block)(
            x[:active_rows], mixes, functional_cu, initial_shift, cu_seqlens[:0]
        )

    def production(block):
        return _forward(*options, block, True)(
            x, mixes, cu_seqlens, initial_shift, status
        )

    def check(block):
        output, final = production(block)
        return (*functional(block), output[:, :active_rows], final[:active_batch])

    canonical = functional(256)
    reference = (*canonical, canonical[0], canonical[1][:active_batch])
    run = functional if _status is None else production
    best, measurements = select_config((256, 128, 512), run, check=check, reference=reference)
    _BLOCK_CONFIGS[_config_key(x, mixes, batch)] = best
    save_config("tmix/tokenshift", _config_key(x, mixes, batch), best)
    return measurements


@torch.no_grad()
@tuning_operation
def tune_fused_tokenshift(x, res, weight, bias, mixes, pool, indices, status, eps=1e-5):
    """Tune T=1 row grouping using private state, checking outputs and final slots."""
    from ..._internal.tuning import save_config, select_config

    mixes = _pack_mixes(x, mixes, pack=False)
    scratch = pool.clone()
    fused_post_norm_tokenshift(x, res, weight, bias, mixes, scratch, indices, status, eps)
    mix_args = mixes + (mixes[-1],) * (6 - len(mixes))

    def run(group):
        return _fused_forward(x.shape[1], len(mixes), eps, group)(
            x, res, weight, bias, *mix_args, scratch, indices, status
        )

    def check(group):
        scratch.copy_(pool)
        total, output = run(group)
        return total, output, scratch.clone()

    best, measurements = select_config((1, 2, 4), run, check=check)
    _FUSED_CONFIGS[_config_key(x, mixes, x.shape[0])] = best
    save_config("tmix/tokenshift/fused", _config_key(x, mixes, x.shape[0]), best)
    return measurements
