# SPDX-License-Identifier: MIT
"""Head ln, RKV res and output gate with a shared BF16 forward."""

import math

import torch
import tilelang
import tilelang.language as T

_ROW_CONFIGS = {}


def _config_key(x, head_size):
    return (x.device.index, x.shape, x.dtype, head_size)


@tilelang.jit(pass_configs={"tl.enable_fast_math": False}, out_idx=[8, 9], compile_flags=["--fmad=false"])
def _forward(
    channels: int, head_size: int, eps: float, x_dtype: str,
    save_stats: bool = True, rows_per_block: int = 1,
):
    rows = T.dynamic("rows")
    heads = channels // head_size
    layout = T.Fragment(
        (rows_per_block, head_size),
        forward_thread_fn=lambda r, d: r * 32 + (d // 2) % 32,
        forward_index_fn=lambda r, d: d // 64 * 2 + d % 2,
    )

    @T.prim_func
    def kernel(
        X: T.Tensor((rows, channels), x_dtype),
        R: T.Tensor((rows, channels), "bfloat16"),
        K: T.Tensor((rows, channels), "bfloat16"),
        V: T.Tensor((rows, channels), "bfloat16"),
        RK: T.Tensor((channels,), "bfloat16"),
        Weight: T.Tensor((channels,), "bfloat16"),
        Bias: T.Tensor((channels,), "bfloat16"),
        Gate: T.Tensor((rows, channels), "bfloat16"),
        Y: T.Tensor((rows, channels), "bfloat16"),
        Stats: T.Tensor((rows if save_stats else 0, heads, 3), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows * heads, rows_per_block), threads=32 * rows_per_block) as tile:
            x = T.alloc_fragment((rows_per_block, head_size), "float32")
            centered = T.alloc_fragment((rows_per_block, head_size), "float32")
            square = T.alloc_fragment((rows_per_block, head_size), "float32")
            rkv = T.alloc_fragment((rows_per_block, head_size), "float32")
            mean = T.alloc_fragment((rows_per_block,), "float32")
            variance = T.alloc_fragment((rows_per_block,), "float32")
            scale = T.alloc_fragment((rows_per_block,), "float32")
            T.annotate_layout({x: layout, centered: layout, square: layout, rkv: layout})
            for group, d in T.Parallel(rows_per_block, head_size):
                index = tile * rows_per_block + group
                row, c = index // heads, index % heads * head_size + d
                x[group, d], rkv[group, d] = 0, 0
                if index < rows * heads:
                    x[group, d] = X[row, c]
                    rkv[group, d] = (
                        T.cast(R[row, c], "float32") * T.cast(K[row, c], "float32")
                        * T.cast(RK[c], "float32")
                    )
            T.reduce_sum(x, mean, dim=1)
            T.reduce_sum(rkv, scale, dim=1)
            for group, d in T.Parallel(rows_per_block, head_size):
                centered[group, d] = x[group, d] - mean[group] / head_size
                square[group, d] = centered[group, d] * centered[group, d]
            T.reduce_sum(square, variance, dim=1)
            if save_stats:
                for group in T.Parallel(rows_per_block):
                    index = tile * rows_per_block + group
                    if index < rows * heads:
                        Stats[index // heads, index % heads, 0] = mean[group] / head_size
                        Stats[index // heads, index % heads, 1] = T.rsqrt(variance[group] / head_size + eps)
                        Stats[index // heads, index % heads, 2] = scale[group]
            for group, d in T.Parallel(rows_per_block, head_size):
                index = tile * rows_per_block + group
                row, c = index // heads, index % heads * head_size + d
                if index < rows * heads:
                    Y[row, c] = (
                        centered[group, d] * T.rsqrt(variance[group] / head_size + eps) * T.cast(Weight[c], "float32")
                        + T.cast(Bias[c], "float32") + scale[group] * T.cast(V[row, c], "float32")
                    ) * T.cast(Gate[row, c], "float32")

    return kernel


@tilelang.jit(pass_configs={"tl.enable_fast_math": False}, out_idx=list(range(10, 18)), compile_flags=["--fmad=false"])
def _backward(
    channels: int, head_size: int, x_dtype: str, rows_per_block: int = 1,
    parameter_grads: tuple = (True, True, True),
):
    rows = T.dynamic("rows")
    heads = channels // head_size
    layout = T.Fragment(
        (head_size,), forward_thread_fn=lambda d: (d // 2) % 32,
        forward_index_fn=lambda d: d // 64 * 2 + d % 2,
    )

    @T.prim_func
    def kernel(
        DY: T.Tensor((rows, channels), "bfloat16"),
        X: T.Tensor((rows, channels), x_dtype),
        R: T.Tensor((rows, channels), "bfloat16"),
        K: T.Tensor((rows, channels), "bfloat16"),
        V: T.Tensor((rows, channels), "bfloat16"),
        RK: T.Tensor((channels,), "bfloat16"),
        Weight: T.Tensor((channels,), "bfloat16"),
        Bias: T.Tensor((channels,), "bfloat16"),
        Gate: T.Tensor((rows, channels), "bfloat16"),
        Stats: T.Tensor((rows, heads, 3), "float32"),
        DX: T.Tensor((rows, channels), x_dtype),
        DR: T.Tensor((rows, channels), "bfloat16"),
        DK: T.Tensor((rows, channels), "bfloat16"),
        DV: T.Tensor((rows, channels), "bfloat16"),
        DRK: T.Tensor((T.ceildiv(rows, rows_per_block) if parameter_grads[0] else 0, channels), "float32"),
        DW: T.Tensor((T.ceildiv(rows, rows_per_block) if parameter_grads[1] else 0, channels), "float32"),
        DB: T.Tensor((T.ceildiv(rows, rows_per_block) if parameter_grads[2] else 0, channels), "float32"),
        DGate: T.Tensor((rows, channels), "bfloat16"),
    ):
        with T.Kernel(heads, T.ceildiv(rows, rows_per_block), threads=32) as (head, tile):
            xhat = T.alloc_fragment((head_size,), "float32")
            dy = T.alloc_fragment((head_size,), "float32")
            weighted = T.alloc_fragment((head_size,), "float32")
            product = T.alloc_fragment((head_size,), "float32")
            res_product = T.alloc_fragment((head_size,), "float32")
            total = T.alloc_fragment((1,), "float32")
            dot = T.alloc_fragment((1,), "float32")
            dscale = T.alloc_fragment((1,), "float32")
            drk = T.alloc_fragment((head_size,), "float32")
            dw = T.alloc_fragment((head_size,), "float32")
            db = T.alloc_fragment((head_size,), "float32")
            T.annotate_layout({
                xhat: layout, dy: layout, weighted: layout, product: layout,
                res_product: layout, drk: layout, dw: layout, db: layout,
            })
            T.clear(drk)
            T.clear(dw)
            T.clear(db)
            for offset in T.serial(rows_per_block):
                row = tile * rows_per_block + offset
                if row < rows:
                    for d in T.Parallel(head_size):
                        c = head * head_size + d
                        xhat[d] = (T.cast(X[row, c], "float32") - Stats[row, head, 0]) * Stats[row, head, 1]
                        dy[d] = T.cast(DY[row, c], "float32") * T.cast(Gate[row, c], "float32")
                        weighted[d] = dy[d] * T.cast(Weight[c], "float32")
                        product[d] = weighted[d] * xhat[d]
                        res_product[d] = dy[d] * T.cast(V[row, c], "float32")
                    T.reduce_sum(weighted, total, dim=0)
                    T.reduce_sum(product, dot, dim=0)
                    T.reduce_sum(res_product, dscale, dim=0)
                    for d in T.Parallel(head_size):
                        c = head * head_size + d
                        DX[row, c] = (
                            weighted[d] - (total[0] + xhat[d] * dot[0]) / head_size
                        ) * Stats[row, head, 1]
                        DR[row, c] = dscale[0] * T.cast(K[row, c], "float32") * T.cast(RK[c], "float32")
                        DK[row, c] = dscale[0] * T.cast(R[row, c], "float32") * T.cast(RK[c], "float32")
                        DV[row, c] = dy[d] * Stats[row, head, 2]
                        if parameter_grads[0]:
                            drk[d] += dscale[0] * T.cast(R[row, c], "float32") * T.cast(K[row, c], "float32")
                        if parameter_grads[1]:
                            dw[d] += dy[d] * xhat[d]
                        if parameter_grads[2]:
                            db[d] += dy[d]
                        DGate[row, c] = T.cast(DY[row, c], "float32") * (
                            xhat[d] * T.cast(Weight[c], "float32") + T.cast(Bias[c], "float32")
                            + Stats[row, head, 2] * T.cast(V[row, c], "float32")
                        )
            for d in T.Parallel(head_size):
                c = head * head_size + d
                if parameter_grads[0]:
                    DRK[tile, c] = drk[d]
                if parameter_grads[1]:
                    DW[tile, c] = dw[d]
                if parameter_grads[2]:
                    DB[tile, c] = db[d]

    return kernel


class _Readout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, r, k, v, r_k, weight, bias, gate, head_size, eps):
        options = (x.shape[-1], head_size, str(x.dtype).split(".")[-1])
        group = _ROW_CONFIGS.get(_config_key(x, head_size), 1)
        y, stats = _forward(options[0], head_size, eps, options[2], rows_per_block=group)(
            x, r, k, v, r_k, weight, bias, gate
        )
        ctx.save_for_backward(x, r, k, v, r_k, weight, bias, gate, stats)
        ctx.options = options
        return y

    @staticmethod
    def backward(ctx, dy):
        from ..._internal.elementwise import sum_partials

        group = min(32, tilelang.next_power_of_2(max(1, dy.shape[0] // 128)))
        needed = ctx.needs_input_grad[4:7]
        gradients = list(_backward(*ctx.options, group, needed)(dy.contiguous(), *ctx.saved_tensors))
        for i, need in zip((4, 5, 6), needed):
            gradients[i] = sum_partials(gradients[i]) if need else None
        return (*gradients, None, None)


def readout(
    x: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r_k: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    gate: torch.Tensor,
    head_size: int = 64,
    eps: float = 64e-5,
) -> torch.Tensor:
    """Compute BF16 preprojection readout with FP32 head ln and RKV res."""
    if head_size not in (64, 128, 256) or x.ndim < 2 or x.numel() == 0 or x.shape[-1] % head_size:
        raise ValueError("readout expects nonempty [..., C] with head_size 64, 128, or 256")
    rows = (r, k, v, gate)
    affine = (r_k, ln_weight, ln_bias)
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("readout x must be CUDA bfloat16 or float32")
    if any(t.dtype != torch.bfloat16 or not t.is_cuda for t in (*rows, *affine)):
        raise ValueError("readout parameters must be CUDA bfloat16")
    if any(t.shape != x.shape for t in rows) or any(t.numel() != x.shape[-1] for t in affine):
        raise ValueError("readout rows must match x and affine parameters must have C elements")
    if any(not t.is_contiguous() or t.device != x.device for t in (x, *rows, *affine)):
        raise ValueError("readout inputs must be contiguous and share a device")
    if x.data_ptr() % (2 * x.element_size()) or any(t.data_ptr() % 4 for t in (*rows, *affine)):
        raise ValueError("readout inputs must be aligned for two-element vector access")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    c = x.shape[-1]
    args = (
        x.view(-1, c), r.view(-1, c), k.view(-1, c), v.view(-1, c),
        r_k.view(c), ln_weight.view(c), ln_bias.view(c), gate.view(-1, c),
    )
    if torch.is_grad_enabled() and any(t.requires_grad for t in args):
        output = _Readout.apply(*args, head_size, float(eps))
    else:
        group = _ROW_CONFIGS.get(_config_key(args[0], head_size), 1)
        output, _ = _forward(c, head_size, float(eps), str(x.dtype).split(".")[-1], False, group)(*args)
    return output.view_as(x)


@torch.no_grad()
def tune_readout(x, r, k, v, r_k, weight, bias, gate, head_size=64, eps=64e-5):
    """Measure grouped heads; each head retains one warp and its reduction tree."""
    from ..._internal.tuning import select_config

    readout(x, r, k, v, r_k, weight, bias, gate, head_size, eps)
    c = x.shape[-1]
    args = (
        x.view(-1, c), r.view(-1, c), k.view(-1, c), v.view(-1, c),
        r_k.view(c), weight.view(c), bias.view(c), gate.view(-1, c),
    )
    dtype = str(x.dtype).split(".")[-1]

    def run(group, inputs=args):
        return _forward(c, head_size, eps, dtype, False, group)(*inputs)[0]

    def check(group, inputs=args):
        training = _forward(c, head_size, eps, dtype, rows_per_block=group)(*inputs)
        return (*training, run(group, inputs))

    training = _forward(c, head_size, eps, dtype)(*args)
    # FP32 ln/RKV statistics remain part of candidate acceptance; the actual
    # no-statistics production output must also equal the training result.
    reference = (*training, training[0])
    for row in sorted({0, args[0].shape[0] // 2, args[0].shape[0] - 1}):
        single_args = tuple(t[row:row + 1] if t.ndim == 2 else t for t in args)
        single = check(1, single_args)
        if not all(torch.equal(full[row:row + 1], one) for full, one in zip(reference, single, strict=True)):
            raise RuntimeError("canonical readout failed batch invariance")

    best, measurements = select_config((1, 2, 4, 8, 16), run, check=check, reference=reference)
    _ROW_CONFIGS[_config_key(args[0], head_size)] = best
    return measurements
