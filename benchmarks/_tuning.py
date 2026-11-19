# SPDX-License-Identifier: MIT
"""Offline candidate verification and graph timing for the operator benchmarks."""

import json
import math
from statistics import median

import torch


def compile_candidates(keys, cache, spec):
    """Elaborate serially, then batch only kernels with identical options."""
    import tilelang

    batches = {}
    for key in dict.fromkeys(keys):
        if key in cache:
            continue
        try:
            program, options = spec(*key)
        except RuntimeError:
            continue
        batch = batches.setdefault(json.dumps(options, sort_keys=True, default=str), (options, [], []))
        batch[1].append(key)
        batch[2].append(program)
    for options, pending, programs in batches.values():
        kernels = tilelang.par_compile(programs, **options, ignore_error=True)
        cache.update(
            (key, kernel)
            for key, kernel in zip(pending, kernels, strict=True)
            if kernel is not None
        )


def projection_configs(rows, channels, outputs, device, measurements=()):
    """At most 12 tiles followed by 12 copy variants of two verified tiles.

    Keep adjacent row tiles on either side of a full SM wave. Copy spans divide
    every input width and fit shared memory before any TIR is constructed.
    """
    props = torch.cuda.get_device_properties(device)
    measured = {item["config"]: item for item in measurements}
    if measured:
        tiles = dict.fromkeys(
            item["config"][:2]
            for item in sorted(
                (item for item in measured.values() if item.get("bitwise_equal")),
                key=lambda item: item["milliseconds"],
            )
        )
        spans = [64, 128]
        if rows < 16 or max(outputs) <= 128:
            spans.extend(
                span for span in (1024, 512, 256)
                if all(width % span == 0 for width in channels)
            )
            spans = spans[:3]
        candidates = (
            (m, n, stages, span)
            for m, n in tuple(tiles)[:2]
            for stages in (2, 3)
            for span in spans
            if all(width % span == 0 for width in channels)
        )
    else:
        columns = (8, 16, 32, 64, 128) if rows < 16 and max(outputs) <= 128 else (32, 64, 128)
        candidates = []
        for n in columns:
            if n > max(32, 1 << (max(outputs) - 1).bit_length()):
                continue
            blocks_n = sum(math.ceil(width / n) for width in outputs)
            limit = min(256, max(16, 1 << (max(1, rows) - 1).bit_length()))
            row_tiles = [m for m in (16, 32, 64, 128, 256) if m <= limit]
            full_wave = [m for m in row_tiles
                         if math.ceil(rows / m) * blocks_n >= props.multi_processor_count]
            center = row_tiles.index(full_wave[-1]) if full_wave else 0
            start = min(center, max(0, len(row_tiles) - 2))
            candidates.extend((m, n, stages, 32) for m in row_tiles[start:start + 2] for stages in (2, 3))
    return tuple(
        config for config in candidates
        if config not in measured
        and (config[0] + config[1]) * config[2] * config[3] * 2 <= props.shared_memory_per_block_optin
    )


def projection_finalists(measurements):
    """Retain at most three verified configurations for sustained graph timing."""
    eligible = (item for item in measurements if item.get("bitwise_equal"))
    ranked = sorted(eligible, key=lambda item: item["milliseconds"])
    return tuple(item["config"] for item in ranked[:3])


@torch.no_grad()
def select_config(candidates, run, *, check=None, reference=None, warmup=3, repeat=20):
    """Return (configuration, measurements), timing CUDA Graph replay.

    ``check`` may return higher-precision outputs than ``run`` to verify
    accumulators before timing the production output dtype. The owning module
    retains its shape/device cache and validates invariance across shapes.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("kernel tuning must finish before CUDA Graph capture")
    candidates = tuple(candidates)
    check = run if check is None else check
    reference = check(candidates[0]) if reference is None else reference
    expected = (reference,) if isinstance(reference, torch.Tensor) else tuple(reference)
    measurements, eligible = [], []
    for config in candidates:
        actual = check(config)
        values = (actual,) if isinstance(actual, torch.Tensor) else tuple(actual)
        equal = len(values) == len(expected) and all(
            torch.equal(value, target) for value, target in zip(values, expected)
        )
        record = {"config": config, "bitwise_equal": equal}
        measurements.append(record)
        if equal:
            eligible.append((config, record))
    if not eligible:
        raise RuntimeError("no kernel configuration satisfies the bitwise contract")
    for config, record in eligible:
        for _ in range(warmup):
            run(config)
        graph = torch.cuda.CUDAGraph()
        # Metadata tickets keep the stream on which the owner prepared them.
        capture_stream = torch.cuda.current_stream()
        if capture_stream == torch.cuda.default_stream():
            capture_stream = None
        with torch.cuda.graph(graph, stream=capture_stream):
            for _ in range(repeat):
                run(config)
        # Instantiate the graph and events before measuring steady device time.
        graph.replay()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        end.record()
        end.synchronize()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        replays = math.ceil(10.0 / start.elapsed_time(end))
        for _ in range(replays):
            graph.replay()
        samples = []
        for _ in range(3):
            start.record()
            for _ in range(replays):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / (repeat * replays))
        record["raw_milliseconds"] = samples
        record["milliseconds"] = median(samples)
        record["launches_per_graph"] = repeat
    best = min(eligible, key=lambda item: item[1]["milliseconds"])[0]
    return best, measurements


@torch.no_grad()
def tune_norm(x, weight, bias, eps=1e-5, res=None):
    """Select row grouping while preserving the canonical per-row FP32 tree."""
    from flashrwkv2.post_norm.tilelang import (
        _ROW_CONFIGS,
        _config_key,
        _forward,
        norm,
    )

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
    from flashrwkv2._internal.linear import (
        _COMPILED_KERNELS,
        _DEFAULT_CONFIG,
        _TUNED_CONFIGS,
        _aligned,
        _compile_spec,
        _compiled,
        _dtype_name,
        _forward,
        _key,
        linear,
    )

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
    major, minor = torch.cuda.get_device_capability(x.device)
    input_major = weight.stride(0) == 1
    weight_strides = (weight.t() if input_major else weight).stride()
    measurements = {}

    def run(config):
        if benchmark is None:
            return _forward(x, weight, bias, output_dtype, config)
        key = _key(x, weight, output_dtype, bias)
        _TUNED_CONFIGS[key] = config
        return benchmark()

    for _ in range(2):
        eligible = projection_configs(
            x.shape[0], (x.shape[1],), (weight.shape[0],), x.device,
            measurements.values(),
        )
        if not eligible:
            break
        rejected = []
        keys = (
            (
                weight.shape[0], x.shape[1], _dtype_name(dtype),
                _dtype_name(bias.dtype) if bias is not None else "bfloat16",
                bias is not None, config, weight_strides, input_major,
                major * 10 + minor,
            )
            for config in eligible for dtype in check_dtypes
        )
        compile_candidates(keys, _COMPILED_KERNELS, _compile_spec)
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
        _, checked = select_config(
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
    best_config, checked = select_config(
        projection_finalists(measurements.values()),
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


@torch.no_grad()
def _tune_grouped_linear(xs, weights, kinds, biases, *, benchmark=None):
    """Verify independent FP32 accumulators and BF16 epilogues before timing."""
    from flashrwkv2._internal.elementwise import activation
    from flashrwkv2.tmix.wkv_prepare.tilelang import (
        _GROUP_CONFIGS,
        _GROUP_KERNELS,
        _compiled_group,
        _forward,
        _group_call,
        _group_inputs,
        _group_key,
        _group_spec,
        _run_groups,
    )

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

    measurements = {}
    for _ in range(2):
        eligible = projection_configs(
            xs[0].shape[0], tuple(x.shape[1] for x in xs),
            tuple(weight.shape[0] for weight in weights), xs[0].device,
            measurements.values(),
        )
        if not eligible:
            break
        rejected = []
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
        _, checked = select_config(compiled, run, check=check, reference=reference)
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
    return {
        "config": best,
        "milliseconds": measurements[best]["milliseconds"],
        "timing_scope": "operator" if benchmark is not None else "grouped_projection",
        "candidates": list(measurements.values()),
    }


@torch.no_grad()
def tune_tokenshift(x, mixes, cu_seqlens, initial_shift=None, *, _status=None):
    """Check functional/status variants and time the requested production path.

    Tuning requires valid, nonempty active metadata before Graph capture.
    A live status is read once; inactive metadata tails are excluded from the
    functional reference, and only defined active status outputs are compared.
    """
    from flashrwkv2.tmix.tokenshift.tilelang import (
        _BLOCK_CONFIGS,
        _config_key,
        _forward,
        _pack_mixes,
        tokenshift,
    )

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
    options = (channels, mixes.shape[0], str(cu_seqlens.dtype).split(".")[-1])
    functional_cu = cu_seqlens
    if active_batch < batch:
        functional_cu = cu_seqlens.clone()
        functional_cu[active_batch + 1:] = active_rows

    def functional(block):
        return _forward(*options, block)(
            x[:active_rows], mixes, functional_cu, initial_shift, cu_seqlens[:0], x[:0]
        )

    def production(block):
        return _forward(*options, block, True)(
            x, mixes, cu_seqlens, initial_shift, status, x[:0]
        )

    def check(block):
        output, final = production(block)
        return (*functional(block), output[:, :active_rows], final[:active_batch])

    canonical = functional(256)
    reference = (*canonical, canonical[0], canonical[1][:active_batch])
    run = functional if _status is None else production
    best, measurements = select_config((256, 128, 512), run, check=check, reference=reference)
    _BLOCK_CONFIGS[_config_key(x, mixes, batch)] = best
    return measurements


@torch.no_grad()
def tune_fused_tokenshift(x, res, weight, bias, mixes, pool, indices, status, eps=1e-5):
    """Tune T=1 row grouping using private state, checking outputs and final slots."""
    from flashrwkv2.tmix.tokenshift.tilelang import (
        _FUSED_CONFIGS,
        _config_key,
        _fused_forward,
        _pack_mixes,
        fused_post_norm_tokenshift,
    )

    mixes = _pack_mixes(x, mixes, pack=False)
    scratch = pool.clone()
    fused_post_norm_tokenshift(x, res, weight, bias, mixes, scratch, indices, status, eps)
    mix_args = mixes + (mixes[-1],) * (6 - len(mixes))

    def run(group):
        return _fused_forward(x.shape[1], len(mixes), eps, group)(
            x, res, weight, bias, *mix_args, scratch, indices, status, x[:0]
        )

    def check(group):
        scratch.copy_(pool)
        total, output = run(group)
        return total, output, scratch.clone()

    best, measurements = select_config((1, 2, 4), run, check=check)
    _FUSED_CONFIGS[_config_key(x, mixes, x.shape[0])] = best
    return measurements


@torch.no_grad()
def tune_tmix_wkv7_recurrent(
    r,
    decay_logits,
    k,
    v,
    a,
    b,
    initial_state,
    cu_seqlens,
    *,
    scale=1.0,
    decay_bias=None,
    indexed=False,
):
    """Check both forward variants and time the requested production mode.

    Every candidate must preserve functional/indexed BF16 outputs and the
    final states. ``indexed=True`` measures in-place inference on a private
    pool with identity slots and valid status. Otherwise it measures the
    functional forward. Both modes preserve the caller's initial state.
    """
    from flashrwkv2.tmix.wkv7.tilelang import (
        _bias_tensor,
        _check_inputs,
        _forward,
        _infer_recurrent,
        tmix_wkv7_recurrent,
    )

    if initial_state.dtype == torch.float16 and not indexed:
        raise ValueError("FP16 state is inference-only; tuning requires indexed=True")

    def functional(config):
        if initial_state.dtype == torch.float16:
            _check_inputs(
                r, decay_logits, k, v, a, b, initial_state, cu_seqlens,
                scale, decay_bias, *config, initial_state.shape[0], inference=True,
            )
            return _forward(
                r, decay_logits, k, v, a, b, initial_state, cu_seqlens,
                _bias_tensor(r, decay_bias),
                torch.empty(0, dtype=torch.int32, device=r.device),
                scale, False, *config,
            )[:2]
        return tmix_wkv7_recurrent(
            r,
            decay_logits,
            k,
            v,
            a,
            b,
            initial_state,
            cu_seqlens,
            scale=scale,
            decay_bias=decay_bias,
            block_values=config[0],
            prefetch_tokens=config[1],
        )

    canonical = functional((64, 1))
    batch = initial_state.shape[0]
    scratch = initial_state.clone()
    indices = torch.arange(batch, dtype=torch.int32, device=r.device)
    status = torch.tensor([0, r.shape[0], batch], dtype=torch.int32, device=r.device)

    def inference(config):
        return _infer_recurrent(
            r, decay_logits, k, v, a, b, scratch, cu_seqlens, indices, status,
            scale=scale, decay_bias=decay_bias,
            block_values=config[0], prefetch_tokens=config[1],
        )

    def check(config):
        scratch.copy_(initial_state)
        output, final = functional(config)
        inferred = inference(config)
        return output, final, inferred, scratch.clone()

    reference = (*canonical, *canonical)
    run = inference if indexed else functional
    prefetches = (1, 4) if r.shape[0] > batch else (1,)
    candidates = tuple((columns, count) for columns in (64, 32) for count in prefetches)
    best, measurements = select_config(candidates, run, check=check, reference=reference)
    return {
        "block_values": best[0], "prefetch_tokens": best[1],
        "indexed": indexed, "candidates": measurements,
        "channels": r.shape[1] * r.shape[2], "state_dtype": str(initial_state.dtype),
        "row_bucket": 1 << (r.shape[0] - 1).bit_length(),
    }


@torch.no_grad()
def tune_readout(x, r, k, v, r_k, weight, bias, gate, head_size=64, eps=64e-5):
    """Measure grouped heads; each head retains one warp and its reduction tree."""
    from flashrwkv2.tmix.readout.tilelang import (
        _ROW_CONFIGS,
        _config_key,
        _forward,
        readout,
    )

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
