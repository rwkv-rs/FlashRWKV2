# SPDX-License-Identifier: MIT
"""Shared BF16 projections for training, prefill, and decode.

Weights use PyTorch's [out_features, in_features] layout. Each output follows
the same increasing-K sequence of BF16 MMA instructions with FP32 accumulators.
Tuning changes output tiling and copy pipelining, never MMA K tiling or split-K.
"""

import math
from functools import cache
from numbers import Real

import torch
from torch.autograd.function import once_differentiable

_PASS_CONFIG = {"tl.disable_wgmma": True, "tl.enable_fast_math": False}
_DEFAULT_CONFIG = (16, 64, 2, 32)
_TUNED_CONFIGS = {}
_COMPILED_KERNELS = {}


def _matmul_program(
    outputs,
    channels,
    output_dtype,
    bias_dtype,
    has_bias,
    config,
    weight_strides=None,
    input_major=False,
    no_set_max_nreg=False,
):
    import tilelang.language as T

    rows = T.dynamic("rows")
    block_m, block_n, stages, copy_k = config
    block_k = 32
    weight_shape = (channels, outputs) if input_major else (outputs, channels)
    weight_strides = (weight_shape[1], 1) if weight_strides is None else weight_strides

    @T.prim_func
    def matmul(
        X: T.Tensor((rows, channels), T.bfloat16),
        W: T.StridedTensor(weight_shape, weight_strides, T.bfloat16),
        Bias: T.StridedTensor(
            (outputs,) if has_bias else weight_shape,
            (1,) if has_bias else weight_strides,
            bias_dtype,
        ),
        Y: T.Tensor((rows, outputs), output_dtype),
    ):
        with T.Kernel(
            T.ceildiv(outputs, block_n),
            T.ceildiv(rows, block_m),
            threads=min(256, max(128, block_m), block_n * 4),
        ) as (bn, bm):
            if no_set_max_nreg:
                T.no_set_max_nreg()
            xs = T.alloc_shared((block_m, copy_k), T.bfloat16)
            ws = T.alloc_shared(
                (copy_k, block_n) if input_major else (block_n, copy_k), T.bfloat16
            )
            acc = T.alloc_fragment((block_m, block_n), T.float32)
            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(channels, copy_k), num_stages=stages):
                T.copy(X[bm * block_m, k * copy_k], xs)
                if input_major:
                    T.copy(W[k * copy_k, bn * block_n], ws)
                else:
                    T.copy(W[bn * block_n, k * copy_k], ws)
                # Each copy contains consecutive canonical 32-K MMA groups.
                for inner in T.unroll(copy_k // block_k):
                    if input_major:
                        T.gemm(
                            xs[:, inner * block_k : (inner + 1) * block_k],
                            ws[inner * block_k : (inner + 1) * block_k, :],
                            acc,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                    else:
                        T.gemm(
                            xs[:, inner * block_k : (inner + 1) * block_k],
                            ws[:, inner * block_k : (inner + 1) * block_k],
                            acc,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
            for i, j in T.Parallel(block_m, block_n):
                if bm * block_m + i < rows and bn * block_n + j < outputs:
                    if has_bias:
                        Y[bm * block_m + i, bn * block_n + j] = acc[i, j] + T.cast(
                            Bias[bn * block_n + j], T.float32
                        )
                    else:
                        Y[bm * block_m + i, bn * block_n + j] = acc[i, j]

    return matmul


def _compile_spec(
    outputs,
    channels,
    output_dtype,
    bias_dtype,
    has_bias,
    config,
    weight_strides,
    input_major,
    arch,
):
    pass_configs = _PASS_CONFIG
    if arch // 10 == 12:
        # Keep SM12x copies and MMA consumers in one CTA synchronization domain.
        pass_configs = {**pass_configs, "tl.disable_warp_specialized": True}
    weight_width = outputs if input_major else channels
    if channels % 2 or weight_width % 2 or weight_strides[0] % 2 or weight_strides[1] != 1:
        # cp.async requires at least two adjacent BF16 elements. Scalar tails
        # and odd/noncontiguous strides use synchronous copies in the same
        # pipeline; disable both automatic injection and pipeline planning.
        pass_configs = {
            **pass_configs,
            "tl.enable_async_copy": False,
            "tirx.use_async_copy": False,
        }
    return (
        _matmul_program(
            outputs,
            channels,
            output_dtype,
            bias_dtype,
            has_bias,
            config,
            weight_strides,
            input_major,
            # SM12x supports TMA but not CTA register reconfiguration.
            no_set_max_nreg=arch // 10 == 12,
        ),
        {
            "out_idx": [3],
            "target": {"kind": "cuda", "arch": f"sm_{arch}"},
            "pass_configs": pass_configs,
        },
    )


def _compiled(*key):
    import tilelang

    if key not in _COMPILED_KERNELS:
        program, options = _compile_spec(*key)
        _COMPILED_KERNELS[key] = tilelang.compile(program, **options)
    return _COMPILED_KERNELS[key]


def _precompile(keys):
    """Elaborate on the caller thread; compile a bounded batch before timing."""
    from .tuning import compile_candidates

    compile_candidates(keys, _COMPILED_KERNELS, _compile_spec)


def _dtype_name(dtype):
    return "bfloat16" if dtype == torch.bfloat16 else "float32"


def _key(x, weight, output_dtype, bias=None):
    return (
        torch.cuda.get_device_name(x.device),
        x.shape[0],
        weight.shape[0],
        x.shape[1],
        weight.stride(),
        output_dtype,
        None if bias is None else bias.dtype,
    )


def _aligned(tensor):
    # Contiguous views can still start at an unaligned storage offset. Both
    # TMA and the vectorized cp.async copies require 16-byte global alignment.
    return tensor if tensor.data_ptr() % 16 == 0 else tensor.clone()


def _forward(x, weight, bias, output_dtype, config=None):
    x, weight = _aligned(x), _aligned(weight)
    bias = None if bias is None else _aligned(bias)
    if config is None:
        config = _TUNED_CONFIGS.get(_key(x, weight, output_dtype, bias))
        if config is None:
            block_m = 16
            if x.shape[0] > 16:
                max_m = min(128, 1 << (x.shape[0] - 1).bit_length())
                block_n = _DEFAULT_CONFIG[1]
                blocks_n = (weight.shape[0] + block_n - 1) // block_n
                multiprocessors = torch.cuda.get_device_properties(
                    x.device
                ).multi_processor_count
                for candidate in (32, 64, 128):
                    # Keep a full CTA wave while capping row padding.
                    blocks = ((x.shape[0] + candidate - 1) // candidate) * blocks_n
                    if candidate > max_m or blocks < multiprocessors:
                        break
                    block_m = candidate
            config = (block_m, *_DEFAULT_CONFIG[1:])
    major, minor = torch.cuda.get_device_capability(x.device)
    input_major = weight.stride(0) == 1
    physical_weight = weight.t() if input_major else weight
    kernel = _compiled(
        weight.shape[0],
        x.shape[1],
        _dtype_name(output_dtype),
        _dtype_name(bias.dtype) if bias is not None else "bfloat16",
        bias is not None,
        config,
        physical_weight.stride(),
        input_major,
        major * 10 + minor,
    )
    # The bias-free specialization has no bias loads; reusing W avoids allocating
    # a dummy device tensor on every projection.
    return kernel(x, physical_weight, physical_weight if bias is None else bias)


def linear_weight_vjp_fp32(dy, x, *, out=None):
    """Return unrounded [out,in] gradients for FP32 accumulation across chunks.

    BF16 upstream values keep the same MMA arithmetic as ``linear.backward``.
    FP32 logits gradients retain their full input precision. Passing ``out``
    accumulates into an FP32 buffer; FP32 GEMMs then avoid a full-size temporary.
    """
    if dy.dtype == torch.bfloat16:
        partial = _forward(dy.t().contiguous(), x.t(), None, torch.float32)
        return partial if out is None else out.add_(partial)
    if out is None:
        return torch.mm(dy.t(), x.float())
    return out.addmm_(dy.t(), x.float())


def linear_input_vjp(dy, weight):
    """Return the BF16 input VJP, preserving the individual linear cast edge."""
    dy = dy.contiguous()
    if dy.dtype == torch.bfloat16:
        return _forward(dy, weight.t(), None, torch.bfloat16)
    return torch.mm(dy, weight.float()).to(torch.bfloat16)


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, output_dtype):
        ctx.save_for_backward(x, weight)
        ctx.bias_dtype = None if bias is None else bias.dtype
        return _forward(x, weight, bias, output_dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        dy = grad_output.contiguous()
        dx = dw = db = None
        if ctx.needs_input_grad[0]:
            dx = linear_input_vjp(dy, weight)
        if ctx.needs_input_grad[1]:
            if dy.dtype == torch.bfloat16:
                # Preserve the fused BF16 store for an ordinary unchunked VJP.
                dw = _forward(dy.t().contiguous(), x.t(), None, weight.dtype)
            else:
                dw = linear_weight_vjp_fp32(dy, x).to(weight.dtype)
        if ctx.needs_input_grad[2]:
            db = dy.float().sum(dim=0).to(ctx.bias_dtype)
        return dx, dw, db, None


@cache
def _adapter_kernel(outputs, dtype, scale, arch):
    import tilelang
    import tilelang.language as T

    rows = T.dynamic("rows")

    @T.prim_func
    def epilogue(
        Base: T.Tensor((rows, outputs), dtype),
        Update: T.Tensor((rows, outputs), T.float32),
        Y: T.Tensor((rows, outputs), dtype),
    ):
        with T.Kernel(T.ceildiv(rows * outputs, 256), threads=128) as block:
            for i in T.Parallel(256):
                index = block * 256 + i
                if index < rows * outputs:
                    row, col = index // outputs, index % outputs
                    delta = T.ieee_mul(Update[row, col], T.float32(scale))
                    Y[row, col] = T.ieee_add(T.cast(Base[row, col], T.float32), delta)

    return tilelang.compile(
        epilogue,
        out_idx=[2],
        target={"kind": "cuda", "arch": f"sm_{arch}"},
        pass_configs=_PASS_CONFIG,
    )


class _AdapterAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, update, scale):
        ctx.scale = scale
        rows = base.reshape(-1, base.shape[-1])
        major, minor = torch.cuda.get_device_capability(base.device)
        result = _adapter_kernel(
            rows.shape[1], _dtype_name(base.dtype), scale, major * 10 + minor
        )(rows, update.reshape_as(rows))
        return result.reshape_as(base)

    @staticmethod
    def backward(ctx, dy):
        return dy, dy.float() * ctx.scale, None


@cache
def _nora_kernel(rank, channels, dtype, strides, backward, arch):
    import tilelang
    import tilelang.language as T

    @T.prim_func
    def normalize(
        A: T.StridedTensor((rank, channels), strides, dtype),
        DY: T.Tensor((rank, channels), T.bfloat16),
        Inverse: T.Tensor((channels,), T.float32),
        Norm: T.Tensor((channels,), T.float32),
        Out: T.Tensor((rank, channels), dtype if backward else T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(channels, 128), threads=128) as bc:
            total = T.alloc_fragment((128,), T.float32)
            T.clear(total)
            for r in T.serial(rank):
                for j in T.Parallel(128):
                    col = bc * 128 + j
                    if col < channels:
                        value = T.cast(A[r, col], T.float32)
                        if backward:
                            total[j] = T.ieee_fmaf(
                                T.cast(DY[r, col], T.float32), value, total[j]
                            )
                        else:
                            total[j] = T.ieee_fmaf(value, value, total[j])
            for j in T.Parallel(128):
                col = bc * 128 + j
                if col < channels:
                    if backward:
                        inv = Inverse[col]
                        total[j] = T.if_then_else(
                            Norm[col] > 0.0,
                            total[j] * inv * inv / Norm[col],
                            0.0,
                        )
                    else:
                        norm = T.sqrt(total[j])
                        inv = T.ieee_fdiv(1.0, T.ieee_add(norm, 1.0e-6))
                        Norm[col] = norm
                        Inverse[col] = inv
                        total[j] = inv
            for r in T.serial(rank):
                for j in T.Parallel(128):
                    col = bc * 128 + j
                    if col < channels:
                        value = T.cast(A[r, col], T.float32)
                        if backward:
                            Out[r, col] = (
                                T.cast(DY[r, col], T.float32) * Inverse[col]
                                - value * total[j]
                            )
                        else:
                            Out[r, col] = value * total[j]

    return tilelang.compile(
        normalize,
        target={"kind": "cuda", "arch": f"sm_{arch}"},
        pass_configs=_PASS_CONFIG,
    )


def nora_forward(a):
    """Return the common normalized BF16 A and its FP32 inverse/norm for a VJP."""
    normalized = torch.empty(a.shape, device=a.device, dtype=torch.bfloat16)
    inverse = torch.empty(a.shape[1], device=a.device, dtype=torch.float32)
    norm = torch.empty_like(inverse)
    major, minor = torch.cuda.get_device_capability(a.device)
    _nora_kernel(*a.shape, _dtype_name(a.dtype), a.stride(), False, major * 10 + minor)(
        a, normalized, inverse, norm, normalized
    )
    return normalized, inverse, norm


def nora_backward(dy, a, inverse, norm):
    """Apply NoRA's VJP after one BF16 rounding of the complete normalized-A VJP.

    A chunked owner passes its FP32 accumulated ``dy`` here only after all token
    chunks have contributed. The result has the original master parameter dtype.
    """
    da = torch.empty(a.shape, device=a.device, dtype=a.dtype)
    major, minor = torch.cuda.get_device_capability(a.device)
    _nora_kernel(*a.shape, _dtype_name(a.dtype), a.stride(), True, major * 10 + minor)(
        a, dy.to(torch.bfloat16).contiguous(), inverse, norm, da
    )
    return da


class _NoRA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        normalized, inverse, norm = nora_forward(a)
        ctx.save_for_backward(a, inverse, norm)
        return normalized

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        a, inverse, norm = ctx.saved_tensors
        return nora_backward(dy, a, inverse, norm)


def linear(x, weight, bias=None, output_dtype=None):
    """Apply a common projection with BF16 operands and FP32 accumulation.

    ``output_dtype`` is BF16 by default; FP32 is available for logits and fused
    adapter epilogues. The first-order VJP uses the same rounded forward weights.
    Call :func:`tune_linear` before CUDA Graph capture to select a measured tile.
    """
    if not x.is_cuda or not weight.is_cuda or weight.device != x.device:
        raise ValueError("linear operands must share a CUDA device")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("linear inputs and weights must be bfloat16")
    if (
        x.ndim < 2
        or weight.ndim != 2
        or x.shape[-1] != weight.shape[1]
        or x.numel() == 0
        or weight.numel() == 0
    ):
        raise ValueError(
            "linear expects x [...,in_features], weight [out_features,in_features]"
        )
    output_dtype = torch.bfloat16 if output_dtype is None else output_dtype
    if output_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("linear output_dtype must be bfloat16 or float32")
    if bias is not None and (
        bias.shape != (weight.shape[0],)
        or bias.device != x.device
        or bias.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError("linear bias must be [out_features], BF16/FP32, on x's device")
    result = _Linear.apply(
        x.reshape(-1, x.shape[-1]).contiguous(),
        weight,
        None if bias is None else bias.contiguous(),
        output_dtype,
    )
    return result.reshape(*x.shape[:-1], weight.shape[0])


def check_adapter_scale(scale):
    """Validate a public adapter multiplier before any stateful launch."""
    if not isinstance(scale, Real):
        raise TypeError("adapter scale must be a real number")
    value = float(scale)
    if not math.isfinite(value) or abs(value) > torch.finfo(torch.float32).max:
        raise ValueError("adapter scale must be finite and within the float32 range")
    return value


def adapter_linear(
    x,
    weight,
    a=None,
    b=None,
    scale=1.0,
    use_nora=False,
    output_dtype=None,
    *,
    base_result=None,
):
    """Project through a base weight and differentiable LoRA/NoRA parameters.

    NoRA normalizes A columns in FP32, with ``norm + 1e-6`` as in the existing
    training contract. Both modes round A/B and the intermediate features to
    BF16. The adapter's second projection, scaling and addition use FP32.
    """
    if (a is None) != (b is None):
        raise ValueError("adapter A and B must be provided together")
    if a is None:
        return (
            linear(x, weight, output_dtype=output_dtype)
            if base_result is None
            else base_result
        )
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[1] != weight.shape[1]
        or b.shape != (weight.shape[0], a.shape[0])
        or a.device != x.device
        or b.device != x.device
        or a.dtype not in (torch.bfloat16, torch.float32)
        or b.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError("adapter expects A [rank,in], B [out,rank], CUDA BF16/FP32")
    scale = check_adapter_scale(scale)
    result = (
        linear(x, weight, output_dtype=output_dtype)
        if base_result is None
        else base_result
    )
    normalized = _NoRA.apply(a) if use_nora else a.to(torch.bfloat16)
    features = linear(x, normalized)
    update = linear(features, b.to(torch.bfloat16), output_dtype=torch.float32)
    return _AdapterAdd.apply(result, update, scale)


@torch.no_grad()
def tune_linear(x, weight, output_dtype=None, *, bias=None, benchmark=None):
    """Measure output tiles without changing the canonical MMA reduction.

    This explicit warmup uses the actual device and shape. Each candidate must
    equal the canonical kernel in FP32 and the production output dtype, including
    independent one-row FP32 replays, before its latency can select it. Results
    apply to this process, device model, and bias dtype. An optional zero-argument
    ``benchmark`` measures the owning forward operator with each candidate in
    its dispatch cache, preserving the operator's working set during tuning.
    """
    from .tuning import select_config

    output_dtype = torch.bfloat16 if output_dtype is None else output_dtype
    linear(x, weight, bias, output_dtype)
    x = _aligned(x.reshape(-1, x.shape[-1]).contiguous())
    weight = _aligned(weight)
    bias = None if bias is None else _aligned(bias.contiguous())
    check_dtypes = tuple(dict.fromkeys((torch.float32, output_dtype)))
    reference = tuple(
        _forward(x, weight, bias, dtype, _DEFAULT_CONFIG) for dtype in check_dtypes
    )
    probes = sorted({0, x.shape[0] // 2, x.shape[0] - 1})
    for row in probes:
        single = _forward(
            x[row : row + 1], weight, bias, torch.float32, _DEFAULT_CONFIG
        )
        if not torch.equal(reference[0][row : row + 1], single):
            raise RuntimeError("canonical linear failed FP32 batch invariance")
    block_rows = (16,) if x.shape[0] < 16 else (16, 32, 64, 128)
    if x.shape[0] > 128:
        block_rows += (256,)
    narrow_tiles = x.shape[0] < 16 and weight.shape[0] <= 128
    block_cols = (8, 16, 32, 64, 128) if narrow_tiles else (32, 64, 128)
    candidates = (
        (block_m, block_n, stages, 32)
        for block_m in block_rows
        for block_n in block_cols
        for stages in (2, 3)
    )
    major, minor = torch.cuda.get_device_capability(x.device)
    input_major = weight.stride(0) == 1
    weight_strides = (weight.t() if input_major else weight).stride()
    shared_memory_limit = torch.cuda.get_device_properties(
        x.device
    ).shared_memory_per_block_optin
    measurements = {}
    best_config = _DEFAULT_CONFIG

    def run(config):
        if benchmark is None:
            return _forward(x, weight, bias, output_dtype, config)
        _TUNED_CONFIGS[_key(x, weight, output_dtype, bias)] = config
        return benchmark()

    for phase in ("tiles", "copies"):
        if phase == "copies":
            # Copy spans can change the best column tile and pipeline depth.
            spans = (64, 128, 256, 512, 1024) if narrow_tiles else (64, 128)
            copies = [span for span in spans if x.shape[1] % span == 0]
            if not copies:
                break
            tiles = [best_config]
            row_tile = min(256, max(16, 1 << (x.shape[0] - 1).bit_length()))
            matched = [
                item for item in measurements.values()
                if item.get("bitwise_equal") and item["config"][0] == row_tile
            ]
            if x.shape[0] > 128 or narrow_tiles:
                tiles.extend(item["config"] for item in matched)
            elif best_config[0] != row_tile and matched:
                tiles.append(
                    min(matched, key=lambda item: item["milliseconds"])["config"]
                )
            candidates = tuple(
                dict.fromkeys(
                    tile[:2] + (stages, span)
                    for tile in tiles
                    for stages in (2, 3)
                    for span in (32, *copies)
                )
            )
        eligible, rejected = [], []
        for config in candidates:
            block_m, block_n, stages, copy_k = config
            # Both BF16 staging tiles retain one buffer per pipeline stage.
            shared_memory = (block_m + block_n) * copy_k * stages * 2
            if shared_memory > shared_memory_limit:
                rejected.append({
                    "config": config,
                    "resource_error": (
                        f"shared memory requires {shared_memory} bytes; "
                        f"device limit is {shared_memory_limit} bytes"
                    ),
                })
                continue
            eligible.append(config)
        _precompile(
            (
                weight.shape[0], x.shape[1], _dtype_name(dtype),
                _dtype_name(bias.dtype) if bias is not None else "bfloat16",
                bias is not None, config, weight_strides, input_major,
                major * 10 + minor,
            )
            for config in eligible for dtype in check_dtypes
        )
        compiled = []
        for config in eligible:
            try:
                for dtype in check_dtypes:
                    _compiled(
                        weight.shape[0],
                        x.shape[1],
                        _dtype_name(dtype),
                        _dtype_name(bias.dtype) if bias is not None else "bfloat16",
                        bias is not None,
                        config,
                        weight_strides,
                        input_major,
                        major * 10 + minor,
                    )
            except RuntimeError as error:
                # Only compilation is recoverable here. Launch and correctness
                # failures below must still stop tuning.
                rejected.append({"config": config, "compile_error": str(error)})
            else:
                compiled.append(config)
        best_config, checked = select_config(
            compiled,
            run,
            check=lambda config: tuple(
                _forward(x, weight, bias, dtype, config) for dtype in check_dtypes
            ),
            reference=reference,
        )
        measurements.update((item["config"], item) for item in checked + rejected)
    # Finalists use the formal benchmark's graph length. Short graphs can rank
    # copy spans differently from a sustained benchmark.
    finalists = {}
    for item in measurements.values():
        if item.get("bitwise_equal"):
            span = item["config"][3]
            if (
                span not in finalists
                or item["milliseconds"] < finalists[span]["milliseconds"]
            ):
                finalists[span] = item
    best_config, checked = select_config(
        tuple(item["config"] for item in finalists.values()),
        run,
        check=lambda config: tuple(
            _forward(x, weight, bias, dtype, config) for dtype in check_dtypes
        ),
        reference=reference,
        repeat=1000,
    )
    measurements.update((item["config"], item) for item in checked)
    best_ms = measurements[best_config]["milliseconds"]
    _TUNED_CONFIGS[_key(x, weight, output_dtype, bias)] = best_config
    return {
        "config": best_config,
        "milliseconds": best_ms,
        "timing_scope": "operator" if benchmark is not None else "projection",
        "candidates": list(measurements.values()),
    }
