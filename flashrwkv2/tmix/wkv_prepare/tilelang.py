# SPDX-License-Identifier: MIT
"""One WKV preparation graph for packed inference and differentiable training."""

import tilelang
import tilelang.language as T
import torch
from torch.autograd.function import once_differentiable

from ..._internal.tuning import device_class, tuning_operation

from ..._internal.elementwise import _activation_kernel, activation, sum_partials
from ..._internal.linear import _aligned, _forward, adapter_linear, linear_input_vjp
from ..kk_pre.tilelang import prepare as prepare_key
from ..vres_gate.tilelang import vres


def prepare(
    x_r,
    x_w,
    x_k,
    x_v,
    x_a,
    x_g,
    receptance_weight,
    key_weight,
    value_weight,
    w1,
    w2,
    a1,
    a2,
    g1,
    g2,
    *,
    k_k,
    a0,
    k_a,
    w0=None,
    v_first=None,
    v0=None,
    v1=None,
    v2=None,
    head_size=64,
    adapters=((None, None, 1.0, False),) * 3,
):
    """All matrix weights use [out,in]; activations round to BF16 at each edge."""
    if (
        x_r.ndim < 2
        or x_r.numel() == 0
        or any(x.shape != x_r.shape for x in (x_w, x_k, x_v, x_a, x_g))
    ):
        raise ValueError("WKV preparation inputs must share a nonempty [...,C] shape")
    channels = x_r.shape[-1]
    if any(
        weight.shape != (channels, channels)
        for weight in (receptance_weight, key_weight, value_weight)
    ):
        raise ValueError("WKV preparation R/K/V weights must have shape [C,C]")
    lowrank_pairs = (("w", w1, w2), ("a", a1, a2), ("g", g1, g2))
    if v_first is not None:
        if v0 is None or v1 is None or v2 is None:
            raise ValueError("v0, v1, and v2 are required with v_first")
        lowrank_pairs += (("v", v1, v2),)
    for name, first, second in lowrank_pairs:
        if (
            first.ndim != 2
            or first.shape[0] == 0
            or first.shape[1] != channels
            or second.shape != (channels, first.shape[0])
        ):
            raise ValueError(f"{name} lowrank weights must have shapes [R,C] and [C,R]")
    dense_inputs = (x_r, x_k, x_v)
    dense_weights = (receptance_weight, key_weight, value_weight)
    projected = _grouped_linear(
        dense_inputs, dense_weights, (0, 0, 0), (None, None, None)
    )
    r, key, value = (
        adapter_linear(x, weight, a, b, scale, nora, base_result=base)
        for x, weight, (a, b, scale, nora), base in zip(
            dense_inputs, dense_weights, adapters, projected, strict=True
        )
    )
    inputs, first_weights = (x_w, x_a, x_g), (w1, a1, g1)
    second_weights = (w2, a2, g2)
    kinds, biases = (1, 0, 2), (w0, a0, None)
    if v_first is not None:
        inputs += (x_v,)
        first_weights += (v1,)
        second_weights += (v2,)
        kinds += (0,)
        biases += (None,)
    reduced = _grouped_linear(inputs, first_weights, kinds, (None,) * len(inputs))
    values = _grouped_linear(
        reduced,
        second_weights,
        (0, 2, 0) + ((0,) if v_first is not None else ()),
        biases,
    )
    decay, rate, gate = values[:3]
    if v_first is None:
        v_first = value
    else:
        value = vres(value, v_first, v0, values[3])
    key, recurrent_a, recurrent_b = prepare_key(key, k_k, rate, k_a, head_size)
    return r, decay, key, value, recurrent_a, recurrent_b, gate, v_first


@T.macro
def _store_group(
    acc,
    bias,
    Y,
    Raw,
    rows,
    outputs,
    prefix,
    block_m,
    block_n,
    mb,
    nb,
    kind,
    has_bias,
    raw_mode,
    raw_prefix,
):
    for i, j in T.Parallel(block_m, block_n):
        row = T.cast(mb, "int64") * block_m + i
        col = nb * block_n + j
        if row < rows and col < outputs:
            rounded = T.cast(T.cast(acc[i, j], T.bfloat16), T.float32)
            value = T.if_then_else(
                has_bias, rounded + T.cast(bias[col], T.float32), rounded
            )
            result = T.if_then_else(
                kind == 1,
                T.tanh(value),
                T.if_then_else(kind == 2, T.sigmoid(value), value),
            )
            Y[prefix * rows + row * outputs + col] = result
            if raw_mode == 2:
                Raw[raw_prefix * rows + row * outputs + col] = acc[i, j]
            elif raw_mode == 1 and kind != 0:
                Raw[raw_prefix * rows + row * outputs + col] = T.cast(
                    acc[i, j], T.bfloat16
                )


def _group_program(
    channels,
    outputs,
    strides,
    kinds,
    biases,
    config,
    input_major,
    raw_mode,
    arch,
    count,
):
    rows = T.dynamic("rows", dtype="int64")
    bm, bn, stages, copy_k = config
    total = sum(outputs[:count])
    raw_sizes = tuple(
        outputs[i] if raw_mode == 2 or (raw_mode == 1 and kinds[i] != 0) else 0
        for i in range(4)
    )
    raw_total = sum(raw_sizes[:count])
    raw_dtype = "float32" if raw_mode == 2 else "bfloat16"
    raw_offsets = (
        0,
        raw_sizes[0],
        raw_sizes[0] + raw_sizes[1],
        raw_sizes[0] + raw_sizes[1] + raw_sizes[2],
    )
    tiles = tuple((n + bn - 1) // bn for n in outputs)
    offsets = (
        0,
        outputs[0],
        outputs[0] + outputs[1],
        outputs[0] + outputs[1] + outputs[2],
    )

    @T.prim_func
    def kernel(
        X0: T.Tensor((rows, channels[0]), T.bfloat16),
        X1: T.Tensor((rows, channels[1]), T.bfloat16),
        X2: T.Tensor((rows, channels[2]), T.bfloat16),
        X3: T.Tensor((rows, channels[3]), T.bfloat16),
        W0: T.StridedTensor(
            (channels[0], outputs[0]) if input_major else (outputs[0], channels[0]),
            strides[0],
            T.bfloat16,
        ),
        W1: T.StridedTensor(
            (channels[1], outputs[1]) if input_major else (outputs[1], channels[1]),
            strides[1],
            T.bfloat16,
        ),
        W2: T.StridedTensor(
            (channels[2], outputs[2]) if input_major else (outputs[2], channels[2]),
            strides[2],
            T.bfloat16,
        ),
        W3: T.StridedTensor(
            (channels[3], outputs[3]) if input_major else (outputs[3], channels[3]),
            strides[3],
            T.bfloat16,
        ),
        B0: T.Tensor((outputs[0] if biases[0] else 0,), T.bfloat16),
        B1: T.Tensor((outputs[1] if biases[1] else 0,), T.bfloat16),
        B2: T.Tensor((outputs[2] if biases[2] else 0,), T.bfloat16),
        B3: T.Tensor((outputs[3] if biases[3] else 0,), T.bfloat16),
        Y: T.Tensor((rows * total,), T.bfloat16),
        Raw: T.Tensor((rows * raw_total,), raw_dtype),
    ):
        with T.Kernel(
            sum(tiles[:count]),
            T.ceildiv(rows, bm),
            threads=min(256, max(128, bm), bn * 4),
        ) as (block, mb):
            if arch // 10 == 12:
                T.no_set_max_nreg()
            group = T.if_then_else(
                block < tiles[0],
                0,
                T.if_then_else(
                    block < tiles[0] + tiles[1],
                    1,
                    T.if_then_else(block < tiles[0] + tiles[1] + tiles[2], 2, 3),
                ),
            )
            nb = block - T.if_then_else(
                group == 0,
                0,
                T.if_then_else(
                    group == 1,
                    tiles[0],
                    T.if_then_else(
                        group == 2, tiles[0] + tiles[1], tiles[0] + tiles[1] + tiles[2]
                    ),
                ),
            )
            active_channels = T.if_then_else(
                group == 0,
                channels[0],
                T.if_then_else(
                    group == 1,
                    channels[1],
                    T.if_then_else(group == 2, channels[2], channels[3]),
                ),
            )
            xs = T.alloc_shared((bm, copy_k), T.bfloat16)
            ws = T.alloc_shared(
                (copy_k, bn) if input_major else (bn, copy_k), T.bfloat16
            )
            acc = T.alloc_fragment((bm, bn), T.float32)
            T.clear(acc)
            for k in T.Pipelined(
                T.ceildiv(max(channels[:count]), copy_k), num_stages=stages
            ):
                if group == 0:
                    T.copy(X0[mb * bm, k * copy_k], xs)
                    if input_major:
                        T.copy(W0[k * copy_k, nb * bn], ws)
                    else:
                        T.copy(W0[nb * bn, k * copy_k], ws)
                elif group == 1:
                    T.copy(X1[mb * bm, k * copy_k], xs)
                    if input_major:
                        T.copy(W1[k * copy_k, nb * bn], ws)
                    else:
                        T.copy(W1[nb * bn, k * copy_k], ws)
                elif group == 2:
                    T.copy(X2[mb * bm, k * copy_k], xs)
                    if input_major:
                        T.copy(W2[k * copy_k, nb * bn], ws)
                    else:
                        T.copy(W2[nb * bn, k * copy_k], ws)
                else:
                    T.copy(X3[mb * bm, k * copy_k], xs)
                    if input_major:
                        T.copy(W3[k * copy_k, nb * bn], ws)
                    else:
                        T.copy(W3[nb * bn, k * copy_k], ws)
                for inner in T.unroll(copy_k // 32):
                    if k * copy_k + inner * 32 < active_channels:
                        if input_major:
                            T.gemm(
                                xs[:, inner * 32 : (inner + 1) * 32],
                                ws[inner * 32 : (inner + 1) * 32, :],
                                acc,
                                policy=T.GemmWarpPolicy.FullCol,
                            )
                        else:
                            T.gemm(
                                xs[:, inner * 32 : (inner + 1) * 32],
                                ws[:, inner * 32 : (inner + 1) * 32],
                                acc,
                                transpose_B=True,
                                policy=T.GemmWarpPolicy.FullCol,
                            )
            if group == 0:
                _store_group(
                    acc,
                    B0,
                    Y,
                    Raw,
                    rows,
                    outputs[0],
                    offsets[0],
                    bm,
                    bn,
                    mb,
                    nb,
                    kinds[0],
                    biases[0],
                    raw_mode,
                    raw_offsets[0],
                )
            elif group == 1:
                _store_group(
                    acc,
                    B1,
                    Y,
                    Raw,
                    rows,
                    outputs[1],
                    offsets[1],
                    bm,
                    bn,
                    mb,
                    nb,
                    kinds[1],
                    biases[1],
                    raw_mode,
                    raw_offsets[1],
                )
            elif group == 2:
                _store_group(
                    acc,
                    B2,
                    Y,
                    Raw,
                    rows,
                    outputs[2],
                    offsets[2],
                    bm,
                    bn,
                    mb,
                    nb,
                    kinds[2],
                    biases[2],
                    raw_mode,
                    raw_offsets[2],
                )
            else:
                _store_group(
                    acc,
                    B3,
                    Y,
                    Raw,
                    rows,
                    outputs[3],
                    offsets[3],
                    bm,
                    bn,
                    mb,
                    nb,
                    kinds[3],
                    biases[3],
                    raw_mode,
                    raw_offsets[3],
                )

    return kernel


_GROUP_KERNELS = {}
_GROUP_CONFIGS = {}


def _group_key(xs, weights, kinds, biases):
    return (
        device_class(xs[0].device),
        xs[0].shape[0],
        tuple(x.shape[1] for x in xs),
        tuple(w.shape for w in weights),
        tuple(w.stride() for w in weights),
        kinds,
        tuple(bias is not None for bias in biases),
    )


def _group_spec(
    channels,
    outputs,
    strides,
    kinds,
    biases,
    config,
    input_major,
    raw_mode,
    arch,
    count,
):
    flags = {"tl.disable_wgmma": True, "tl.enable_fast_math": False}
    if arch // 10 == 12:
        flags["tl.disable_warp_specialized"] = True
    if any(
        c % 2 or (n if input_major else c) % 2 or stride[0] % 2 or stride[1] != 1
        for c, n, stride in zip(channels, outputs, strides, strict=True)
    ):
        flags.update({"tl.enable_async_copy": False, "tirx.use_async_copy": False})
    return (
        _group_program(
            channels,
            outputs,
            strides,
            kinds,
            biases,
            config,
            input_major,
            raw_mode,
            arch,
            count,
        ),
        {
            "out_idx": [12, 13],
            "target": {"kind": "cuda", "arch": f"sm_{arch}"},
            "pass_configs": flags,
        },
    )


def _compiled_group(*key):
    if key not in _GROUP_KERNELS:
        program, options = _group_spec(*key)
        _GROUP_KERNELS[key] = tilelang.compile(program, **options)
    return _GROUP_KERNELS[key]


def _group_call(xs, weights, kinds, biases, raw_mode, config):
    count = len(xs)
    if config is None:
        from ..._internal.tuning import get_config

        config = get_config(_GROUP_CONFIGS, "tmix/wkv_prepare", _group_key(xs, weights, kinds, biases), tune=lambda: _tune_grouped_linear(xs, weights, kinds, biases))
        if config is None:
            narrow = max(w.shape[0] for w in weights) <= 128
            copy_k = next(
                (
                    span
                    for span in ((512, 128, 64) if narrow else (128, 64))
                    if all(x.shape[1] % span == 0 for x in xs)
                ),
                32,
            )
            config = (16, 8 if narrow else 64, 2, copy_k)
    input_major = all(w.stride(0) == 1 for w in weights)
    physical = tuple(w.t() if input_major else w for w in weights)
    channels = tuple(x.shape[1] for x in xs)
    outputs = tuple(w.shape[0] for w in weights)
    strides = tuple(w.stride() for w in physical)
    flags = tuple(bias is not None for bias in biases)
    pad = 4 - count
    channels += channels[-1:] * pad
    outputs += outputs[-1:] * pad
    strides += strides[-1:] * pad
    padded_kinds = kinds + kinds[-1:] * pad
    flags += flags[-1:] * pad
    major, minor = torch.cuda.get_device_capability(xs[0].device)
    key = (
        channels,
        outputs,
        strides,
        padded_kinds,
        flags,
        config,
        input_major,
        raw_mode,
        major * 10 + minor,
        count,
    )
    empty = xs[0].new_empty(0)
    bias_args = tuple(empty if bias is None else bias for bias in biases)
    args = (
        *(xs + xs[-1:] * pad),
        *(physical + physical[-1:] * pad),
        *(bias_args + bias_args[-1:] * pad),
    )
    return key, args


def _run_groups(xs, weights, kinds, biases, raw_mode=0, config=None):
    key, args = _group_call(xs, weights, kinds, biases, raw_mode, config)
    result, raw = _compiled_group(*key)(*args)
    rows = xs[0].shape[0]
    values, start = [], 0
    for weight in weights:
        width = weight.shape[0]
        values.append(result[start : start + rows * width].view(rows, width))
        start += rows * width
    return tuple(values), raw


class _GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kinds, *args):
        count = len(kinds)
        xs, weights, biases = args[:count], args[count : 2 * count], args[2 * count :]
        values, raw = _run_groups(xs, weights, kinds, biases, raw_mode=1)
        ctx.save_for_backward(*xs, *weights, *biases, raw)
        ctx.kinds = kinds
        ctx.set_materialize_grads(False)
        return values

    @staticmethod
    @once_differentiable
    def backward(ctx, *gradients):
        count = len(ctx.kinds)
        saved = ctx.saved_tensors
        xs, weights, biases = (
            saved[:count],
            saved[count : 2 * count],
            saved[2 * count : 3 * count],
        )
        raw = saved[-1]
        dxs, dws, dbs = [None] * count, [None] * count, [None] * count
        position = 0
        for i, (x, weight, bias, kind, gradient) in enumerate(
            zip(xs, weights, biases, ctx.kinds, gradients, strict=True)
        ):
            width = weight.shape[0]
            if kind:
                activation_input = raw[position : position + x.shape[0] * width].view(
                    x.shape[0], width
                )
                position += x.shape[0] * width
            if gradient is None:
                continue
            dy = _aligned(gradient.contiguous())
            need_bias = ctx.needs_input_grad[1 + 2 * count + i]
            if kind or bias is not None:
                # Identity's derivative does not read the saved activation.
                if not kind:
                    activation_input = dy
                activation_input = _aligned(activation_input)
                partial = torch.empty(
                    ((x.shape[0] + 31) // 32, width) if need_bias else (0,),
                    dtype=torch.float32,
                    device=x.device,
                )
                dlinear = torch.empty_like(dy)
                _activation_kernel(
                    width,
                    {0: "identity", 1: "tanh", 2: "sigmoid"}[kind],
                    bias is not None,
                    True,
                    need_bias,
                )(
                    activation_input,
                    x.new_empty(0) if bias is None else bias,
                    dy,
                    dlinear,
                    partial,
                )
                dy = dlinear
                if need_bias:
                    dbs[i] = sum_partials(partial)
            if ctx.needs_input_grad[1 + i]:
                dxs[i] = linear_input_vjp(dy, weight)
            if ctx.needs_input_grad[1 + count + i]:
                dws[i] = _forward(dy.t().contiguous(), x.t(), None, weight.dtype)
        return None, *dxs, *dws, *dbs


def _group_inputs(xs, weights, biases):
    device = xs[0].device
    if any(not t.is_cuda or t.device != device for t in (*xs, *weights)):
        raise ValueError("linear operands must share a CUDA device")
    if any(t.dtype != torch.bfloat16 for t in (*xs, *weights)):
        raise ValueError("linear inputs and weights must be bfloat16")
    if any(
        bias is not None
        and (
            bias.shape != (weight.shape[0],)
            or bias.dtype != torch.bfloat16
            or bias.device != device
            or not bias.is_contiguous()
        )
        for bias, weight in zip(biases, weights, strict=True)
    ):
        raise ValueError(
            "activation bias must be contiguous [C] with the input dtype/device"
        )
    xs = tuple(_aligned(x.reshape(-1, x.shape[-1]).contiguous()) for x in xs)
    weights = tuple(_aligned(weight) for weight in weights)
    biases = tuple(None if bias is None else _aligned(bias) for bias in biases)
    return xs, weights, biases


def _grouped_linear(xs, weights, kinds, biases):
    shape = xs[0].shape[:-1]
    xs, weights, biases = _group_inputs(xs, weights, biases)
    if torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (*xs, *weights, *biases)
    ):
        values = _GroupedLinear.apply(kinds, *xs, *weights, *biases)
    else:
        values, _ = _run_groups(xs, weights, kinds, biases)
    return tuple(
        value.view(*shape, weight.shape[0])
        for value, weight in zip(values, weights, strict=True)
    )


@torch.no_grad()
@tuning_operation
def _tune_grouped_linear(xs, weights, kinds, biases, *, benchmark=None):
    """Verify independent FP32 accumulators and BF16 epilogues before timing."""
    from ..._internal.tuning import compile_candidates, projection_finalists, revalidation_config, save_config, select_config

    xs, weights, biases = _group_inputs(xs, weights, biases)
    cache_key = _group_key(xs, weights, kinds, biases)
    raw_reference = tuple(
        _forward(x, weight, None, torch.float32, (16, 64, 2, 32))
        for x, weight in zip(xs, weights, strict=True)
    )
    value_reference = tuple(
        activation(raw.bfloat16(), {0: "identity", 1: "tanh", 2: "sigmoid"}[kind], bias)
        if kind or bias is not None
        else raw.bfloat16()
        for raw, kind, bias in zip(raw_reference, kinds, biases, strict=True)
    )
    reference = (
        *(value.view(torch.int16) for value in value_reference),
        *(value.view(torch.int32) for value in raw_reference),
    )

    def run(config):
        if benchmark is None:
            return _run_groups(xs, weights, kinds, biases, config=config)[0]
        _GROUP_CONFIGS[cache_key] = config
        return benchmark()

    def check(config):
        values, _ = _run_groups(xs, weights, kinds, biases, config=config)
        _, raw = _run_groups(xs, weights, kinds, biases, raw_mode=2, config=config)
        result = [value.view(torch.int16) for value in values]
        position = 0
        for weight in weights:
            size = xs[0].shape[0] * weight.shape[0]
            value = raw[position : position + size].view(
                xs[0].shape[0], weight.shape[0]
            )
            result.append(value.view(torch.int32))
            position += size
        return tuple(result)

    imported = revalidation_config()
    if imported is not None:
        best, checked = select_config((imported,), run, check=check, reference=reference)
        save_config("tmix/wkv_prepare", cache_key, best)
        return {"config": best, "candidates": checked}

    rows = xs[0].shape[0]
    narrow = max(weight.shape[0] for weight in weights) <= 128
    block_rows = (16,) if rows < 16 else (16, 32, 64, 128)
    if rows > 128:
        block_rows += (256,)
    columns = (8, 16, 32) if narrow else (32, 64, 128)
    candidates = tuple(
        (block_m, block_n, stages, 32)
        for block_m in block_rows
        for block_n in columns
        for stages in (2, 3)
    )
    shared_limit = torch.cuda.get_device_properties(
        xs[0].device
    ).shared_memory_per_block_optin
    measurements = {}
    best = candidates[0]
    for phase in ("tiles", "copies"):
        if phase == "copies":
            spans = (64, 128, 256, 512, 1024) if narrow or rows < 16 else (64, 128)
            copies = tuple(
                span for span in spans if all(x.shape[1] % span == 0 for x in xs)
            )
            if not copies:
                break
            row_tile = min(256, max(16, 1 << (rows - 1).bit_length()))
            matched = [
                item
                for item in measurements.values()
                if item.get("bitwise_equal") and item["config"][0] == row_tile
            ]
            tiles = [best]
            if rows > 128 or rows < 16 or narrow:
                tiles.extend(item["config"] for item in matched)
            elif best[0] != row_tile and matched:
                tiles.append(
                    min(matched, key=lambda item: item["milliseconds"])["config"]
                )
            candidates = tuple(
                dict.fromkeys(
                    (*tile[:2], stages, copy_k)
                    for tile in tiles
                    for stages in (2, 3)
                    for copy_k in (32, *copies)
                )
            )
        eligible, rejected = [], []
        for config in candidates:
            block_m, block_n, stages, copy_k = config
            required = (block_m + block_n) * stages * copy_k * 2
            if required > shared_limit:
                rejected.append(
                    {
                        "config": config,
                        "resource_error": f"shared memory requires {required} bytes; device limit is {shared_limit} bytes",
                    }
                )
            else:
                eligible.append(config)
        compile_candidates(
            (
                _group_call(xs, weights, kinds, biases, mode, config)[0]
                for config in eligible
                for mode in (0, 2)
            ),
            _GROUP_KERNELS,
            _group_spec,
        )
        compiled = []
        for config in eligible:
            try:
                for mode in (0, 2):
                    _compiled_group(
                        *_group_call(xs, weights, kinds, biases, mode, config)[0]
                    )
            except RuntimeError as error:
                rejected.append({"config": config, "compile_error": str(error)})
            else:
                compiled.append(config)
        best, checked = select_config(compiled, run, check=check, reference=reference)
        measurements.update((item["config"], item) for item in (*checked, *rejected))
    best, checked = select_config(
        projection_finalists(measurements.values()),
        run,
        check=check,
        reference=reference,
        repeat=1000,
    )
    measurements.update((item["config"], item) for item in checked)
    _GROUP_CONFIGS[cache_key] = best
    save_config("tmix/wkv_prepare", cache_key, best)
    return {
        "config": best,
        "milliseconds": measurements[best]["milliseconds"],
        "timing_scope": "operator" if benchmark is not None else "grouped_projection",
        "candidates": list(measurements.values()),
    }
