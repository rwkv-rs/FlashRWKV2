# SPDX-License-Identifier: MIT

"""Shared RWKV-7 recurrence: BF16 token IO and FP32 state/accumulation.

An explicit inference-only mode stores FP16 state after every token; it does
not change training or claim equality with the default FP32-state trajectory.

State is stored as [sequence, head, key, value].  Every value column follows
the same key reduction and update order, independently of sequence packing,
the number of columns assigned to a block, or whether backward is requested.
The backward differentiates the real-valued recurrence, treating the BF16
output cast as an identity, as in ordinary mixed-precision training.
It saves FP32 boundaries every 16 tokens. For longer head-size-64 inputs,
checkpoint replay saves a vector per token and the reverse pass replays short
FP32 state windows with one writer per gradient. Other inputs use bounded shared-memory replay.
"""

import math

import tilelang
import tilelang.language as T
import torch

from ..._internal.tuning import (
    _AutogradContext,
    _fixed_strategy_device,
    device_class,
    prepared_kernel,
    tuning_operation,
    workload_bucket,
)

_CHECKPOINT = 16
_BACKWARD_SEGMENT = 4096
_LOG2E = 1.4426950408889634
_EXP_NEG_HALF = 0.6065306597126334
_DECAY_EXP2 = -0.8750387749145276
_PASSES = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False}
_COMPILE_FLAGS = ["--fmad=false"]
_TUNED_CONFIGS = {}


@T.macro
def _dot_keys(state, vector, partial, result, offset=0):
    """Fixed key-group FMA scan shared by forward and checkpoint replay."""
    T.clear(partial)
    for step in T.unroll(64):
        for group, value in T.Parallel(partial.shape[0], partial.shape[1]):
            key = step * partial.shape[0] + group
            partial[group, value] = T.fma(
                state[key, value], vector[key + offset], partial[group, value]
            )
    T.reduce_sum(partial, result, dim=0)


@T.macro
def _update_state(state, decay, k, v, b, state_dot_a, offset=0, state_dtype="float32"):
    for key, value in T.Parallel(state.shape[0], state.shape[1]):
        state[key, value] = T.fma(
            k[key + offset],
            v[value],
            T.fma(
                b[key + offset], state_dot_a[value],
                decay[key + offset] * state[key, value],
            ),
        )
        if state_dtype == "float16":
            state[key, value] = T.cast(
                T.cast(state[key, value], "float16"), "float32"
            )


@tilelang.jit(pass_configs=_PASSES, compile_flags=_COMPILE_FLAGS)
def _forward_kernel(
    heads: int,
    dim: int,
    scale: float,
    has_bias: bool,
    save: bool,
    block_values: int = 64,
    has_status: bool = False,
    indexed: bool = False,
    bias_dtype: str = "float32",
    prefetch_tokens: int = 1,
    state_dtype: str = "float32",
):
    n, batch = T.dynamic("n, batch")
    state_slots = T.dynamic("state_slots") if indexed else batch
    columns = min(block_values, dim)
    key_groups = dim // 64
    checkpoints = n // _CHECKPOINT + batch
    threads = columns * key_groups
    within_loop = T.unroll if prefetch_tokens > 1 and dim <= 128 else T.serial

    @T.prim_func
    def kernel(
        R: T.Tensor((n, heads, dim), "bfloat16"),
        W: T.Tensor((n, heads, dim), "bfloat16"),
        K: T.Tensor((n, heads, dim), "bfloat16"),
        V: T.Tensor((n, heads, dim), "bfloat16"),
        A: T.Tensor((n, heads, dim), "bfloat16"),
        B: T.Tensor((n, heads, dim), "bfloat16"),
        S0: T.Tensor((state_slots if indexed else batch, heads, dim, dim), state_dtype),
        CU: T.Tensor((batch + 1,), "int32"),
        Bias: T.Tensor((heads if has_bias else 0, dim), bias_dtype),
        Status: T.Tensor((3 if has_status else 0,), "int32"),
        Indices: T.Tensor((batch if indexed else 0,), "int32"),
        Y: T.Tensor((n, heads, dim), "bfloat16"),
        SN: T.Tensor((0 if indexed else batch, heads, dim, dim), state_dtype),
        Boundaries: T.Tensor((checkpoints if save else 0, heads, dim, dim), state_dtype),
    ):
        with T.Kernel(dim // columns, heads, batch, threads=threads) as (tile, h, seq):
            state = T.alloc_fragment((dim, columns), "float32")
            partial = T.alloc_fragment((key_groups, columns), "float32")
            dot = T.alloc_fragment((columns,), "float32")
            out = T.alloc_fragment((columns,), "float32")
            rv = T.alloc_shared((prefetch_tokens * dim,), "float32")
            wv = T.alloc_shared((prefetch_tokens * dim,), "float32")
            kv = T.alloc_shared((prefetch_tokens * dim,), "float32")
            av = T.alloc_shared((prefetch_tokens * dim,), "float32")
            bv = T.alloc_shared((prefetch_tokens * dim,), "float32")
            vs = T.alloc_shared(
                (prefetch_tokens * columns if prefetch_tokens > 1 else 0,), "float32"
            )
            vv = T.alloc_fragment((columns,), "float32")
            layout = T.Fragment(
                (dim, columns),
                forward_thread_fn=lambda key, value: (
                    (key % key_groups) * columns + value
                ),
                forward_index_fn=lambda key, value: key // key_groups,
            )
            partial_layout = T.Fragment(
                (key_groups, columns),
                forward_thread_fn=lambda group, value: group * columns + value,
                forward_index_fn=lambda group, value: 0,
            )
            T.annotate_layout({state: layout, partial: partial_layout})
            if has_status:
                active = Status[0] == 0 and seq < Status[2]
                start = T.if_then_else(active, CU[seq], 0)
                end = T.if_then_else(active, CU[seq + 1], 0)
                for row in T.serial(T.ceildiv(n, batch)):
                    token = row * batch + seq
                    if token < n and (Status[0] != 0 or token >= Status[1]):
                        for value in T.Parallel(columns):
                            Y[token, h, tile * columns + value] = T.cast(
                                float("nan"), "bfloat16"
                            )
            else:
                active = True
                start = CU[seq]
                end = CU[seq + 1]
            if indexed:
                T.clear(state)
                if active:
                    slot = Indices[seq]
                    for key, value in T.Parallel(dim, columns):
                        state[key, value] = S0[slot, h, key, tile * columns + value]
            else:
                for key, value in T.Parallel(dim, columns):
                    state[key, value] = S0[seq, h, key, tile * columns + value]
            for chunk in T.serial(T.ceildiv(end - start, prefetch_tokens)):
                for prefetched, key in T.Parallel(prefetch_tokens, dim):
                    token = start + chunk * prefetch_tokens + prefetched
                    if prefetch_tokens == 1 or token < end:
                        offset = prefetched * dim + key
                        rv[offset] = T.cast(R[token, h, key], "float32")
                        logits = T.cast(W[token, h, key], "float32")
                        if has_bias:
                            logits = T.cast(
                                T.cast(
                                    logits + T.cast(Bias[h, key], "float32"),
                                    "bfloat16",
                                ),
                                "float32",
                            )
                        wv[offset] = T.exp2(
                            _DECAY_EXP2 / (1.0 + T.exp2(-_LOG2E * logits))
                        )
                        kv[offset] = T.cast(K[token, h, key], "float32")
                        av[offset] = T.cast(A[token, h, key], "float32")
                        bv[offset] = T.cast(B[token, h, key], "float32")
                if prefetch_tokens > 1:
                    for prefetched, value in T.Parallel(prefetch_tokens, columns):
                        token = start + chunk * prefetch_tokens + prefetched
                        if token < end:
                            vs[prefetched * columns + value] = T.cast(
                                V[token, h, tile * columns + value], "float32"
                            )
                for within in within_loop(prefetch_tokens):
                    token = start + chunk * prefetch_tokens + within
                    if prefetch_tokens == 1 or token < end:
                        if save and (token - start) % _CHECKPOINT == 0:
                            checkpoint = (
                                start // _CHECKPOINT + seq
                                + (token - start) // _CHECKPOINT
                            )
                            for key, value in T.Parallel(dim, columns):
                                Boundaries[
                                    checkpoint, h, key, tile * columns + value
                                ] = state[key, value]
                        for value in T.Parallel(columns):
                            if prefetch_tokens == 1:
                                vv[value] = T.cast(
                                    V[token, h, tile * columns + value], "float32"
                                )
                            else:
                                vv[value] = vs[within * columns + value]
                        _dot_keys(state, av, partial, dot, within * dim)
                        _update_state(
                            state, wv, kv, vv, bv, dot, within * dim, state_dtype
                        )
                        _dot_keys(state, rv, partial, out, within * dim)
                        for value in T.Parallel(columns):
                            Y[token, h, tile * columns + value] = T.cast(
                                out[value] * scale, "bfloat16"
                            )
            if indexed:
                if active:
                    slot = Indices[seq]
                    for key, value in T.Parallel(dim, columns):
                        S0[slot, h, key, tile * columns + value] = state[key, value]
            else:
                for key, value in T.Parallel(dim, columns):
                    SN[seq, h, key, tile * columns + value] = state[key, value]

    return kernel


@T.macro
def _backward_sequence(cu, batch, slot, span, seq):
    # Each sequence owns the slots starting at floor(CU[seq] / span) + seq.
    # Padding between sequences keeps their checkpoint and segment indices disjoint.
    hi = T.alloc_var("int32", init=batch)
    seq[0] = 0
    while seq[0] + 1 < hi:
        mid = (seq[0] + hi) // 2
        if cu[mid] // span + mid <= slot:
            seq[0] = mid
        else:
            hi = mid


@T.macro
def _backward_logit(w, bias, token, head, lane, has_bias, logit):
    logit[0] = T.cast(w[token, head, lane], "float32")
    if has_bias:
        logit[0] = T.cast(T.cast(logit[0] + T.cast(bias[head, lane], "float32"), "bfloat16"), "float32")


@tilelang.jit(pass_configs=_PASSES, compile_flags=_COMPILE_FLAGS)
def _prepare_sa_kernel(heads, has_bias, bias_dtype):
    n, batch = T.dynamic("n, batch")
    slots = n // _CHECKPOINT + batch

    @T.prim_func
    def kernel(
        W: T.Tensor((n, heads, 64), "bfloat16"),
        K: T.Tensor((n, heads, 64), "bfloat16"),
        V: T.Tensor((n, heads, 64), "bfloat16"),
        A: T.Tensor((n, heads, 64), "bfloat16"),
        B: T.Tensor((n, heads, 64), "bfloat16"),
        CU: T.Tensor((batch + 1,), "int32"),
        Bias: T.Tensor((heads if has_bias else 0, 64), bias_dtype),
        Boundaries: T.Tensor((slots, heads, 64, 64), "float32"),
        SA: T.Tensor((n, heads, 64), "float32"),
    ):
        with T.Kernel(slots, heads, threads=64) as (slot, h):
            lane = T.get_thread_binding()
            seq = T.alloc_local((1,), "int32")
            _backward_sequence(CU, batch, slot, _CHECKPOINT, seq)
            start = CU[seq[0]]
            end = CU[seq[0] + 1]
            chunk_start = start + (slot - start // _CHECKPOINT - seq[0]) * _CHECKPOINT
            if chunk_start < end:
                state = T.alloc_local((64,), "float32")
                staged = T.alloc_shared((4, 64), "float32")
                logit = T.alloc_local((1,), "float32")
                dot = T.alloc_var("float32")
                for key in T.unroll(64):
                    state[key] = Boundaries[slot, h, key, lane]
                chunk_end = T.min(end, chunk_start + _CHECKPOINT)
                for t in T.serial(chunk_start, chunk_end):
                    _backward_logit(W, Bias, t, h, lane, has_bias, logit)
                    staged[0, lane] = T.exp2(_DECAY_EXP2 / (1.0 + T.exp2(-_LOG2E * logit[0])))
                    staged[1, lane] = T.cast(K[t, h, lane], "float32")
                    staged[2, lane] = T.cast(A[t, h, lane], "float32")
                    staged[3, lane] = T.cast(B[t, h, lane], "float32")
                    T.sync_threads()
                    dot = 0.0
                    for key in T.unroll(64):
                        dot = T.fma(state[key], staged[2, key], dot)
                    SA[t, h, lane] = dot
                    value = T.cast(V[t, h, lane], "float32")
                    for key in T.unroll(64):
                        state[key] = T.fma(
                            staged[1, key], value,
                            T.fma(staged[3, key], dot, staged[0, key] * state[key]),
                        )
                    T.sync_threads()

    return kernel


@tilelang.jit(pass_configs=_PASSES, compile_flags=_COMPILE_FLAGS)
def _adjoint_boundaries_kernel(heads, scale, has_bias, bias_dtype):
    n, batch = T.dynamic("n, batch")
    segments = n // _BACKWARD_SEGMENT + batch

    @T.prim_func
    def kernel(
        R: T.Tensor((n, heads, 64), "bfloat16"),
        W: T.Tensor((n, heads, 64), "bfloat16"),
        A: T.Tensor((n, heads, 64), "bfloat16"),
        B: T.Tensor((n, heads, 64), "bfloat16"),
        DY: T.Tensor((n, heads, 64), "bfloat16"),
        CU: T.Tensor((batch + 1,), "int32"),
        Bias: T.Tensor((heads if has_bias else 0, 64), bias_dtype),
        DSN: T.Tensor((batch, heads, 64, 64), "float32"),
        Incoming: T.Tensor((segments, heads, 64, 64), "float32"),
    ):
        with T.Kernel(2, heads, batch, threads=32) as (tile, h, seq):
            lane = T.get_thread_binding()
            value = tile * 32 + lane
            start = CU[seq]
            end = CU[seq + 1]
            if end - start > _BACKWARD_SEGMENT:
                adjoint = T.alloc_local((64,), "float32")
                x = T.alloc_shared((4, 64), "float32")
                logit = T.alloc_local((1,), "float32")
                dsa = T.alloc_var("float32")
                for key in T.unroll(64):
                    adjoint[key] = DSN[seq, h, key, value]
                for reverse in T.serial(end - start - _BACKWARD_SEGMENT):
                    t = end - 1 - reverse
                    for pair in T.unroll(2):
                        key = pair * 32 + lane
                        _backward_logit(W, Bias, t, h, key, has_bias, logit)
                        x[0, key] = T.exp2(_DECAY_EXP2 / (1.0 + T.exp2(-_LOG2E * logit[0])))
                        x[1, key] = T.cast(A[t, h, key], "float32")
                        x[2, key] = T.cast(B[t, h, key], "float32")
                        x[3, key] = T.cast(R[t, h, key], "float32")
                    dy = T.cast(DY[t, h, value], "float32") * scale
                    T.sync_threads()
                    dsa = 0.0
                    for key in T.unroll(64):
                        adjoint[key] = T.fma(x[3, key], dy, adjoint[key])
                        dsa = T.fma(adjoint[key], x[2, key], dsa)
                    for key in T.unroll(64):
                        adjoint[key] = T.fma(x[1, key], dsa, x[0, key] * adjoint[key])
                    if (t - start) % _BACKWARD_SEGMENT == 0:
                        slot = start // _BACKWARD_SEGMENT + seq + (t - start) // _BACKWARD_SEGMENT - 1
                        for key in T.unroll(64):
                            Incoming[slot, h, key, value] = adjoint[key]
                    T.sync_threads()

    return kernel


@T.macro
def _replay_key_row(state, inputs, local, key):
    for value in T.unroll(64):
        state[value] = T.fma(
            inputs[1, local, key], inputs[2, local, value],
            T.fma(inputs[3, local, key], inputs[4, local, value], inputs[0, local, key] * state[value]),
        )


@tilelang.jit(pass_configs=_PASSES, compile_flags=_COMPILE_FLAGS)
def _direct_backward_kernel(heads, scale, has_bias, bias_dtype, segmented):
    n, batch = T.dynamic("n, batch")
    slots = n // _CHECKPOINT + batch
    segments = n // _BACKWARD_SEGMENT + batch if segmented else batch

    @T.prim_func
    def kernel(
        R: T.Tensor((n, heads, 64), "bfloat16"),
        W: T.Tensor((n, heads, 64), "bfloat16"),
        K: T.Tensor((n, heads, 64), "bfloat16"),
        V: T.Tensor((n, heads, 64), "bfloat16"),
        A: T.Tensor((n, heads, 64), "bfloat16"),
        B: T.Tensor((n, heads, 64), "bfloat16"),
        CU: T.Tensor((batch + 1,), "int32"),
        Bias: T.Tensor((heads if has_bias else 0, 64), bias_dtype),
        Boundaries: T.Tensor((slots, heads, 64, 64), "float32"),
        SA: T.Tensor((n, heads, 64), "float32"),
        DY: T.Tensor((n, heads, 64), "bfloat16"),
        DSN: T.Tensor((batch, heads, 64, 64), "float32"),
        Incoming: T.Tensor((segments if segmented else 0, heads, 64, 64), "float32"),
        Grad: T.Tensor((6, n, heads, 64), "bfloat16"),
        DS0: T.Tensor((batch, heads, 64, 64), "float32"),
    ):
        with T.Kernel(segments, heads, threads=64) as (slot, h):
            lane = T.get_thread_binding()
            seq = T.alloc_local((1,), "int32")
            if segmented:
                _backward_sequence(CU, batch, slot, _BACKWARD_SEGMENT, seq)
            else:
                seq[0] = slot
            start = CU[seq[0]]
            end = CU[seq[0] + 1]
            if segmented:
                begin = start + (slot - start // _BACKWARD_SEGMENT - seq[0]) * _BACKWARD_SEGMENT
                stop = T.min(end, begin + _BACKWARD_SEGMENT)
            else:
                begin = start
                stop = end
            if begin < stop or begin == end == start:
                # Keep both adjoint orientations so each lane owns its key
                # and value gradients. Stage two replay states at a time.
                state = T.alloc_local((64,), "float32")
                first = T.alloc_shared((64, 64), "float32")
                dkey = T.alloc_local((64,), "float32")
                dvalue = T.alloc_local((64,), "float32")
                inputs = T.alloc_shared((6, _CHECKPOINT, 64), "float32")
                x = T.alloc_shared((3, 64), "float32")
                du = T.alloc_shared((64,), "float32")
                logit = T.alloc_local((1,), "float32")
                dr = T.alloc_var("float32")
                dw = T.alloc_var("float32")
                dk = T.alloc_var("float32")
                dv = T.alloc_var("float32")
                da = T.alloc_var("float32")
                db = T.alloc_var("float32")
                dsa = T.alloc_var("float32")
                if segmented and stop != end:
                    for j in T.unroll(64):
                        dkey[j] = Incoming[slot, h, lane, j]
                        dvalue[j] = Incoming[slot, h, j, lane]
                else:
                    for j in T.unroll(64):
                        dkey[j] = DSN[seq[0], h, lane, j]
                        dvalue[j] = DSN[seq[0], h, j, lane]
                chunks = T.ceildiv(stop - begin, _CHECKPOINT)
                for reverse_chunk in T.serial(chunks):
                    chunk_start = begin + (chunks - 1 - reverse_chunk) * _CHECKPOINT
                    checkpoint = start // _CHECKPOINT + seq[0] + (chunk_start - start) // _CHECKPOINT
                    count = T.min(_CHECKPOINT, stop - chunk_start)
                    for local in T.serial(count):
                        t = chunk_start + local
                        _backward_logit(W, Bias, t, h, lane, has_bias, logit)
                        denom = 1.0 + T.exp2(-_LOG2E * logit[0])
                        inputs[0, local, lane] = T.exp2(_DECAY_EXP2 / denom)
                        inputs[1, local, lane] = T.cast(K[t, h, lane], "float32")
                        inputs[2, local, lane] = T.cast(V[t, h, lane], "float32")
                        inputs[3, local, lane] = T.cast(B[t, h, lane], "float32")
                        inputs[4, local, lane] = SA[t, h, lane]
                        inputs[5, local, lane] = 1.0 / denom
                    T.sync_threads()
                    windows = T.ceildiv(count, 2)
                    for reverse_window in T.serial(windows):
                        local_start = (windows - 1 - reverse_window) * 2
                        size = T.min(2, count - local_start)
                        for j in T.unroll(64):
                            state[j] = Boundaries[checkpoint, h, lane, j]
                        for local in T.serial(local_start):
                            _replay_key_row(state, inputs, local, lane)
                        if size == 2:
                            for j in T.unroll(64):
                                first[j, lane] = state[j]
                            _replay_key_row(state, inputs, local_start, lane)
                        for reverse in T.serial(size):
                            local = local_start + size - 1 - reverse
                            t = chunk_start + local
                            if reverse != 0:
                                for j in T.unroll(64):
                                    state[j] = first[j, lane]
                            decay = inputs[0, local, lane]
                            key = inputs[1, local, lane]
                            beta = inputs[3, local, lane]
                            x[0, lane] = T.cast(R[t, h, lane], "float32")
                            x[1, lane] = T.cast(A[t, h, lane], "float32")
                            x[2, lane] = T.cast(DY[t, h, lane], "float32") * scale
                            T.sync_threads()
                            dr = 0.0
                            dw = 0.0
                            dk = 0.0
                            dv = 0.0
                            db = 0.0
                            dsa = 0.0
                            for j in T.unroll(64):
                                v = inputs[2, local, j]
                                sa = inputs[4, local, j]
                                next_state = T.fma(key, v, T.fma(beta, sa, decay * state[j]))
                                dr = T.fma(next_state, x[2, j], dr)
                                dkey[j] = T.fma(x[0, lane], x[2, j], dkey[j])
                                dvalue[j] = T.fma(x[0, j], x[2, lane], dvalue[j])
                                dw = T.fma(dkey[j], state[j], dw)
                                dk = T.fma(dkey[j], v, dk)
                                db = T.fma(dkey[j], sa, db)
                                dv = T.fma(dvalue[j], inputs[1, local, j], dv)
                                dsa = T.fma(dvalue[j], inputs[3, local, j], dsa)
                            du[lane] = dsa
                            T.sync_threads()
                            da = 0.0
                            for j in T.unroll(64):
                                da = T.fma(state[j], du[j], da)
                                dkey[j] = T.fma(x[1, lane], du[j], decay * dkey[j])
                                dvalue[j] = T.fma(x[1, j], dsa, inputs[0, local, j] * dvalue[j])
                            sigmoid = inputs[5, local, lane]
                            Grad[0, t, h, lane] = T.cast(dr, "bfloat16")
                            Grad[1, t, h, lane] = T.cast(
                                dw * (-_EXP_NEG_HALF * sigmoid * (1.0 - sigmoid) * decay), "bfloat16",
                            )
                            Grad[2, t, h, lane] = T.cast(dk, "bfloat16")
                            Grad[3, t, h, lane] = T.cast(dv, "bfloat16")
                            Grad[4, t, h, lane] = T.cast(da, "bfloat16")
                            Grad[5, t, h, lane] = T.cast(db, "bfloat16")
                            T.sync_threads()
                if begin == start:
                    for j in T.unroll(64):
                        DS0[seq[0], h, j, lane] = dvalue[j]

    return kernel


def _direct_vjp(r, w, k, v, a, b, cu, bias, boundary, dy, ds, scale):
    n, heads, dim = r.shape
    batch = ds.shape[0]
    has_bias = bias.numel() != 0
    bias_dtype = "bfloat16" if bias.dtype == torch.bfloat16 else "float32"
    segmented = n > _BACKWARD_SEGMENT and batch * heads < 128
    sa = torch.empty_like(r, dtype=torch.float32)
    ds0 = torch.empty_like(ds)
    gradients = r.new_empty((6, n, heads, dim))
    device = r.device
    prepared_kernel("tmix/wkv7", _prepare_sa_kernel, (heads, has_bias, bias_dtype))(
        w, k, v, a, b, cu, bias, boundary, sa,
    )
    incoming = ds.new_empty((n // _BACKWARD_SEGMENT + batch if segmented else 0, heads, dim, dim))
    if segmented:
        prepared_kernel("tmix/wkv7", _adjoint_boundaries_kernel, (heads, scale, has_bias, bias_dtype))(
            r, w, a, b, dy, cu, bias, ds, incoming,
        )
    prepared_kernel("tmix/wkv7", _direct_backward_kernel, (heads, scale, has_bias, bias_dtype, segmented))(
        r, w, k, v, a, b, cu, bias, boundary, sa, dy, ds, incoming, gradients, ds0,
    )
    return gradients, ds0



@tilelang.jit(pass_configs=_PASSES, compile_flags=_COMPILE_FLAGS)
def _backward_kernel(
    heads: int,
    dim: int,
    scale: float,
    has_bias: bool,
    has_status: bool = False,
    bias_dtype: str = "float32",
):
    n, batch = T.dynamic("n, batch")
    checkpoints = n // _CHECKPOINT + batch
    columns = 1024 // dim
    key_groups = dim // 64
    replicas = 128 // (key_groups * columns)

    @T.prim_func
    def kernel(
        R: T.Tensor((n, heads, dim), "bfloat16"),
        W: T.Tensor((n, heads, dim), "bfloat16"),
        K: T.Tensor((n, heads, dim), "bfloat16"),
        V: T.Tensor((n, heads, dim), "bfloat16"),
        A: T.Tensor((n, heads, dim), "bfloat16"),
        B: T.Tensor((n, heads, dim), "bfloat16"),
        CU: T.Tensor((batch + 1,), "int32"),
        Bias: T.Tensor((heads if has_bias else 0, dim), bias_dtype),
        Status: T.Tensor((3 if has_status else 0,), "int32"),
        Boundaries: T.Tensor((checkpoints, heads, dim, dim), "float32"),
        DY: T.Tensor((n, heads, dim), "bfloat16"),
        DSN: T.Tensor((batch, heads, dim, dim), "float32"),
        Grad: T.Tensor((6, n, heads, dim), "float32"),
        DS0: T.Tensor((batch, heads, dim, dim), "float32"),
    ):
        with T.Kernel(dim // columns, heads, batch, threads=128) as (tile, h, seq):
            # A 16-token history of 1024 state elements occupies 68 KiB.
            # This bounds workspace independently of the sequence length.
            history = T.alloc_shared((_CHECKPOINT + 1, dim, columns), "float32")
            history_sa = T.alloc_shared((_CHECKPOINT, columns), "float32")
            replay_state = T.alloc_fragment((dim, columns), "float32")
            replay_partial = T.alloc_fragment((key_groups, columns), "float32")
            replay_sa = T.alloc_fragment((columns,), "float32")
            state = T.alloc_fragment((dim, columns), "float32")
            adjoint = T.alloc_fragment((dim, columns), "float32")
            product = T.alloc_fragment((dim, columns), "float32")
            rv = T.alloc_shared((dim,), "float32")
            wv = T.alloc_shared((dim,), "float32")
            dwv = T.alloc_shared((dim,), "float32")
            kv = T.alloc_shared((dim,), "float32")
            vv = T.alloc_shared((columns,), "float32")
            av = T.alloc_shared((dim,), "float32")
            bv = T.alloc_shared((dim,), "float32")
            dy = T.alloc_shared((columns,), "float32")
            sa = T.alloc_fragment((columns,), "float32")
            ds_a = T.alloc_fragment((columns,), "float32")
            grad_key = T.alloc_fragment((dim,), "float32")
            grad_value = T.alloc_fragment((columns,), "float32")
            replay_layout = T.Fragment(
                (dim, columns),
                replicate=replicas,
                forward_thread_fn=lambda key, value, rep: (
                    (key % key_groups) * columns + value + rep * key_groups * columns
                ),
                forward_index_fn=lambda key, value: key // key_groups,
            )
            partial_layout = T.Fragment(
                (key_groups, columns),
                replicate=replicas,
                forward_thread_fn=lambda group, value, rep: (
                    group * columns + value + rep * key_groups * columns
                ),
                forward_index_fn=lambda group, value: 0,
            )
            T.annotate_layout(
                {replay_state: replay_layout, replay_partial: partial_layout}
            )
            for key, value in T.Parallel(dim, columns):
                adjoint[key, value] = DSN[seq, h, key, tile * columns + value]
            if has_status:
                active = Status[0] == 0 and seq < Status[2]
            else:
                active = True
            if active:
                start = CU[seq]
                end = CU[seq + 1]
                chunks = T.ceildiv(end - start, _CHECKPOINT)
                for reverse_chunk in T.serial(chunks):
                    chunk = chunks - reverse_chunk - 1
                    chunk_start = start + chunk * _CHECKPOINT
                    count = T.min(_CHECKPOINT, end - chunk_start)
                    checkpoint = start // _CHECKPOINT + seq + chunk
                    for key, value in T.Parallel(dim, columns):
                        replay_state[key, value] = Boundaries[
                            checkpoint, h, key, tile * columns + value
                        ]
                    # Give shared history one writer per element; replay_state
                    # is replicated to preserve the forward reduction order.
                    for key, value in T.Parallel(dim, columns):
                        history[0, key, value] = replay_state[key, value]
                    # Replay from the saved boundary with the shared FMA order.
                    for local in T.serial(count):
                        token = chunk_start + local
                        for key in T.Parallel(dim):
                            logits = T.cast(W[token, h, key], "float32")
                            if has_bias:
                                logits = T.cast(
                                    T.cast(
                                        logits + T.cast(Bias[h, key], "float32"),
                                        "bfloat16",
                                    ),
                                    "float32",
                                )
                            wv[key] = T.exp2(
                                _DECAY_EXP2 / (1.0 + T.exp2(-_LOG2E * logits))
                            )
                            kv[key] = T.cast(K[token, h, key], "float32")
                            av[key] = T.cast(A[token, h, key], "float32")
                            bv[key] = T.cast(B[token, h, key], "float32")
                        for value in T.Parallel(columns):
                            vv[value] = T.cast(
                                V[token, h, tile * columns + value], "float32"
                            )
                        _dot_keys(replay_state, av, replay_partial, replay_sa)
                        for value in T.Parallel(columns):
                            history_sa[local, value] = replay_sa[value]
                        _update_state(replay_state, wv, kv, vv, bv, replay_sa)
                        for key, value in T.Parallel(dim, columns):
                            history[local + 1, key, value] = replay_state[key, value]
                    for reverse in T.serial(count):
                        local = count - 1 - reverse
                        token = chunk_start + local
                        for key in T.Parallel(dim):
                            rv[key] = T.cast(R[token, h, key], "float32")
                            logits = T.cast(W[token, h, key], "float32")
                            if has_bias:
                                logits = T.cast(
                                    T.cast(
                                        logits + T.cast(Bias[h, key], "float32"),
                                        "bfloat16",
                                    ),
                                    "float32",
                                )
                            denominator = 1.0 + T.exp2(-_LOG2E * logits)
                            sigmoid = 1.0 / denominator
                            wv[key] = T.exp2(_DECAY_EXP2 / denominator)
                            dwv[key] = (
                                -_EXP_NEG_HALF * sigmoid * (1.0 - sigmoid) * wv[key]
                            )
                            kv[key] = T.cast(K[token, h, key], "float32")
                            av[key] = T.cast(A[token, h, key], "float32")
                            bv[key] = T.cast(B[token, h, key], "float32")
                        for value in T.Parallel(columns):
                            vv[value] = T.cast(
                                V[token, h, tile * columns + value], "float32"
                            )
                            dy[value] = (
                                T.cast(DY[token, h, tile * columns + value], "float32")
                                * scale
                            )
                        for key, value in T.Parallel(dim, columns):
                            state[key, value] = history[local + 1, key, value]
                            product[key, value] = state[key, value] * dy[value]
                            adjoint[key, value] = T.fma(
                                rv[key], dy[value], adjoint[key, value]
                            )
                        T.reduce_sum(product, grad_key, dim=1)
                        for key in T.Parallel(dim):
                            T.atomic_add(Grad[0, token, h, key], grad_key[key])
                        for key, value in T.Parallel(dim, columns):
                            product[key, value] = adjoint[key, value] * vv[value]
                        T.reduce_sum(product, grad_key, dim=1)
                        for key in T.Parallel(dim):
                            T.atomic_add(Grad[2, token, h, key], grad_key[key])
                        for key, value in T.Parallel(dim, columns):
                            product[key, value] = adjoint[key, value] * kv[key]
                        T.reduce_sum(product, grad_value, dim=0)
                        for value in T.Parallel(columns):
                            Grad[3, token, h, tile * columns + value] = grad_value[
                                value
                            ]
                        for key, value in T.Parallel(dim, columns):
                            product[key, value] = adjoint[key, value] * bv[key]
                        T.reduce_sum(product, ds_a, dim=0)
                        for key, value in T.Parallel(dim, columns):
                            state[key, value] = history[local, key, value]
                        for value in T.Parallel(columns):
                            sa[value] = history_sa[local, value]
                        for key, value in T.Parallel(dim, columns):
                            product[key, value] = adjoint[key, value] * sa[value]
                        T.reduce_sum(product, grad_key, dim=1)
                        for key in T.Parallel(dim):
                            T.atomic_add(Grad[5, token, h, key], grad_key[key])
                        for key, value in T.Parallel(dim, columns):
                            product[key, value] = (
                                adjoint[key, value] * state[key, value]
                            )
                        T.reduce_sum(product, grad_key, dim=1)
                        for key in T.Parallel(dim):
                            T.atomic_add(
                                Grad[1, token, h, key], grad_key[key] * dwv[key]
                            )
                        for key, value in T.Parallel(dim, columns):
                            product[key, value] = state[key, value] * ds_a[value]
                        T.reduce_sum(product, grad_key, dim=1)
                        for key in T.Parallel(dim):
                            T.atomic_add(Grad[4, token, h, key], grad_key[key])
                        for key, value in T.Parallel(dim, columns):
                            adjoint[key, value] = T.fma(
                                av[key], ds_a[value], wv[key] * adjoint[key, value]
                            )
            for key, value in T.Parallel(dim, columns):
                DS0[seq, h, key, tile * columns + value] = adjoint[key, value]

    return kernel


def _prepared_forward_kernel(device, *options):
    variants = (options[:9] + (prefetch,) + options[10:] for prefetch in (1, 4)) if (
        options[1] == 64 and options[-1] == "float32"
        and _fixed_strategy_device(device_class(device))
    ) else (options,)
    return prepared_kernel("tmix/wkv7", _forward_kernel, options, variants)


def _forward(
    r,
    decay_logits,
    k,
    v,
    a,
    b,
    initial_state,
    cu_seqlens,
    bias,
    status,
    scale,
    save,
    block_values,
    prefetch_tokens,
):
    n, heads, dim = r.shape
    batch = initial_state.shape[0]
    output = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)
    boundary = torch.empty(
        (n // _CHECKPOINT + batch if save else 0, heads, dim, dim),
        dtype=initial_state.dtype,
        device=r.device,
    )
    _prepared_forward_kernel(
        r.device, heads,
        dim,
        scale,
        bias.numel() != 0,
        save,
        block_values,
        status.numel() != 0,
        False,
        "bfloat16" if bias.dtype == torch.bfloat16 else "float32",
        prefetch_tokens,
        str(initial_state.dtype).removeprefix("torch."),
    )(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        initial_state,
        cu_seqlens,
        bias,
        status,
        torch.empty(0, dtype=torch.int32, device=r.device),
        output,
        final_state,
        boundary,
    )
    return output, final_state, boundary


class _Recurrent(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, r, w, k, v, a, b, state, cu, bias, status, scale, block_values,
        prefetch_tokens,
    ):
        output, final_state, boundary = _forward(
            r, w, k, v, a, b, state, cu, bias, status, scale, True, block_values,
            prefetch_tokens,
        )
        ctx.save_for_backward(r, w, k, v, a, b, cu, bias, status, boundary)
        ctx.state_shape = state.shape
        ctx.scale = scale
        ctx.set_materialize_grads(False)
        return output, final_state

    @staticmethod
    def backward(ctx, grad_output, grad_final_state):
        gradients, ds0, dbias = _recurrent_vjp(ctx, grad_output, grad_final_state)
        return *gradients.unbind(0), ds0, None, dbias, None, None, None, None


def _recurrent_vjp(ctx, grad_output, grad_final_state):
    r, w, k, v, a, b, cu, bias, status, boundary = ctx.saved_tensors
    dy = torch.zeros_like(v) if grad_output is None else grad_output.contiguous()
    ds = (
        torch.zeros(ctx.state_shape, dtype=torch.float32, device=r.device)
        if grad_final_state is None
        else grad_final_state.contiguous()
    )
    # The extra replay launch pays off on longer sequences. Keep the shared
    # history path for short batches, wider heads and status-masked capacity.
    if r.shape[2] == 64 and r.shape[0] >= max(256, _CHECKPOINT * ds.shape[0]) and status.numel() == 0:
        rounded, ds0 = _direct_vjp(r, w, k, v, a, b, cu, bias, boundary, dy, ds, ctx.scale)
        dbias = rounded[1].float().sum(dim=0).to(bias.dtype) if bias.numel() else None
        return rounded, ds0, dbias
    gradients = torch.zeros((6, *r.shape), dtype=torch.float32, device=r.device)
    ds0 = torch.empty_like(ds)
    _, heads, dim = r.shape
    prepared_kernel("tmix/wkv7", _backward_kernel, (
        heads,
        dim,
        ctx.scale,
        bias.numel() != 0,
        status.numel() != 0,
        "bfloat16" if bias.dtype == torch.bfloat16 else "float32",
    ))(
        r,
        w,
        k,
        v,
        a,
        b,
        cu,
        bias,
        status,
        boundary,
        dy,
        ds,
        gradients,
        ds0,
    )
    rounded = gradients.to(torch.bfloat16)
    dbias = rounded[1].float().sum(dim=0).to(bias.dtype) if bias.numel() else None
    return rounded, ds0, dbias


def _check_inputs(
    r,
    decay_logits,
    k,
    v,
    a,
    b,
    initial_state,
    cu_seqlens,
    scale,
    decay_bias,
    block_values,
    prefetch_tokens,
    batch,
    *,
    inference=False,
):
    if r.ndim != 3 or r.shape[1] < 1 or r.shape[2] not in (64, 128, 256):
        raise ValueError("r must have shape [N,H,D], with D in {64,128,256}")
    for name, value in zip(
        ("r", "decay_logits", "k", "v", "a", "b"), (r, decay_logits, k, v, a, b)
    ):
        if value.shape != r.shape or value.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be BF16 with the same [N,H,D] shape as r")
        if not value.is_cuda or value.device != r.device or not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA on the r device")
    if (
        initial_state.ndim != 4
        or initial_state.shape[1:] != (r.shape[1], r.shape[2], r.shape[2])
        or initial_state.shape[0] < 1
    ):
        raise ValueError("initial_state must have shape [B,H,D,D] with B >= 1")
    if (
        initial_state.dtype not in ((torch.float32, torch.float16) if inference else (torch.float32,))
        or initial_state.device != r.device
        or not initial_state.is_contiguous()
    ):
        dtype = "FP16/FP32" if inference else "FP32"
        raise ValueError(f"initial_state must be contiguous {dtype} on the r device")
    if (
        cu_seqlens.shape != (batch + 1,)
        or cu_seqlens.dtype != torch.int32
        or cu_seqlens.device != r.device
        or not cu_seqlens.is_contiguous()
    ):
        raise ValueError("cu_seqlens must be contiguous CUDA int32 with shape [B+1]")
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    if block_values not in (32, 64):
        raise ValueError("block_values must be 32 or 64")
    if prefetch_tokens not in (1, 4):
        raise ValueError("prefetch_tokens must be 1 or 4")
    if decay_bias is not None and (
        decay_bias.shape not in (r.shape[1:], (r.shape[1] * r.shape[2],))
        or decay_bias.dtype not in (torch.float32, torch.bfloat16)
        or decay_bias.device != r.device
        or not decay_bias.is_contiguous()
    ):
        raise ValueError(
            "decay_bias must be contiguous BF16/FP32 [H,D] or [H*D] on the r device"
        )


def _bias_tensor(r, decay_bias):
    if decay_bias is None:
        return torch.empty((0, r.shape[2]), dtype=torch.float32, device=r.device)
    return decay_bias.reshape(r.shape[1:])


def _config_key(r, batch, bias, *, indexed=False, state_dtype=torch.float32):
    return (
        device_class(r.device),
        (workload_bucket(r.shape[0]), *r.shape[1:]),
        workload_bucket(batch),
        r.shape[0] > batch,
        None if bias is None else bias.dtype,
        indexed,
        state_dtype,
    )


def _fixed_recurrent_config(r, batch, state_dtype):
    # Each value column retains the same key reduction and FP32 update order.
    # Multi-token prefetch overlaps input loads; decode needs no prefetch.
    device = device_class(r.device)
    if not _fixed_strategy_device(device) or r.shape[-1] != 64 or state_dtype != torch.float32:
        return None
    return 64, 4 if r.shape[0] > batch else 1


def _infer_recurrent(
    r,
    decay_logits,
    k,
    v,
    a,
    b,
    state_pool,
    cu_seqlens,
    state_indices,
    status,
    *,
    scale=1.0,
    decay_bias=None,
    block_values=None,
    prefetch_tokens=None,
):
    """Launch the common forward directly against an indexed inference pool."""
    if (
        state_indices.ndim != 1
        or state_indices.numel() == 0
        or state_indices.dtype != torch.int32
        or state_indices.device != r.device
        or not state_indices.is_contiguous()
    ):
        raise ValueError("state_indices must be contiguous CUDA int32 [B]")
    batch = state_indices.numel()
    _check_inputs(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        state_pool,
        cu_seqlens,
        scale,
        decay_bias,
        64 if block_values is None else block_values,
        1 if prefetch_tokens is None else prefetch_tokens,
        batch,
        inference=True,
    )
    from ..._internal.tuning import get_config

    config = (64, 1)
    if block_values is None or prefetch_tokens is None:
        fixed = _fixed_recurrent_config(r, batch, state_pool.dtype)
        config = get_config(
            _TUNED_CONFIGS, "tmix/wkv7",
            _config_key(r, batch, decay_bias, indexed=True, state_dtype=state_pool.dtype),
            fixed or config,
            tune=lambda: tune_tmix_wkv7_recurrent(
                r, decay_logits, k, v, a, b,
                state_pool, cu_seqlens,
                scale=scale, decay_bias=decay_bias, indexed=True,
                state_indices=state_indices, status=status,
            ),
            fixed=fixed is not None,
        )
    block_values = config[0] if block_values is None else block_values
    prefetch_tokens = config[1] if prefetch_tokens is None else prefetch_tokens
    _, heads, dim = r.shape
    bias = _bias_tensor(r, decay_bias)
    output = torch.empty_like(v)
    empty = torch.empty((0, heads, dim, dim), dtype=state_pool.dtype, device=r.device)
    _prepared_forward_kernel(
        r.device, heads,
        dim,
        float(scale),
        bias.numel() != 0,
        False,
        block_values,
        True,
        True,
        "bfloat16" if bias.dtype == torch.bfloat16 else "float32",
        prefetch_tokens,
        str(state_pool.dtype).removeprefix("torch."),
    )(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        state_pool,
        cu_seqlens,
        bias,
        status,
        state_indices,
        output,
        empty,
        empty,
    )
    return output


@torch.library.custom_op("flashrwkv2::tmix_wkv7_forward", mutates_args=())
def _compiled_recurrent(
    r: torch.Tensor, w: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    a: torch.Tensor, b: torch.Tensor, state: torch.Tensor, cu: torch.Tensor,
    bias: torch.Tensor, status: torch.Tensor, scale: float,
    block_values: int | None, prefetch_tokens: int | None, save: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    block_values, prefetch_tokens = _functional_config(
        r, w, k, v, a, b, state, cu, scale, bias if bias.numel() else None, block_values, prefetch_tokens,
    )
    return _forward(r, w, k, v, a, b, state, cu, bias, status,
                    scale, save, block_values, prefetch_tokens)


@_compiled_recurrent.register_fake
def _compiled_recurrent_fake(r, w, k, v, a, b, state, cu, bias, status, scale, block_values, prefetch_tokens, save):
    boundary = state.new_empty((r.shape[0] // _CHECKPOINT + state.shape[0] if save else 0, *state.shape[1:]))
    return torch.empty_like(v), torch.empty_like(state), boundary


def _compiled_recurrent_context(ctx, inputs, output):
    r, w, k, v, a, b, state, cu, bias, status, scale, _, _, _ = inputs
    ctx.save_for_backward(r, w, k, v, a, b, cu, bias, status, output[2])
    ctx.state_shape, ctx.scale = state.shape, scale
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(output[2])


@torch.library.custom_op(
    "flashrwkv2::tmix_wkv7_backward", mutates_args=(),
    schema="(Tensor? dy, Tensor? ds, Tensor[] saved, SymInt[] state_shape, float scale) -> (Tensor, Tensor, Tensor?)",
)
def _compiled_recurrent_vjp(dy, ds, saved, state_shape, scale):
    return _recurrent_vjp(_AutogradContext(saved_tensors=saved, state_shape=state_shape, scale=scale), dy, ds)


@_compiled_recurrent_vjp.register_fake
def _compiled_recurrent_vjp_fake(dy, ds, saved, state_shape, scale):
    r, _, _, _, _, _, _, bias, _, _ = saved
    return r.new_empty((6, *r.shape)), r.new_empty(state_shape, dtype=torch.float32), torch.empty_like(bias) if bias.numel() else None


def _compiled_recurrent_backward(ctx, dy, ds, unused_boundary):
    gradients, ds0, dbias = _compiled_recurrent_vjp(dy, ds, list(ctx.saved_tensors), list(ctx.state_shape), ctx.scale)
    return *gradients.unbind(0), ds0, None, dbias, None, None, None, None, None


_compiled_recurrent.register_autograd(_compiled_recurrent_backward, setup_context=_compiled_recurrent_context)


def _functional_config(r, decay_logits, k, v, a, b, initial_state, cu_seqlens, scale, decay_bias, block_values, prefetch_tokens):
    from ..._internal.tuning import get_config
    config = (64, 1)
    if block_values is None or prefetch_tokens is None:
        fixed = _fixed_recurrent_config(r, initial_state.shape[0], initial_state.dtype)
        config = get_config(
            _TUNED_CONFIGS, "tmix/wkv7", _config_key(r, initial_state.shape[0], decay_bias),
            fixed or config,
            tune=lambda: tune_tmix_wkv7_recurrent(r, decay_logits, k, v, a, b, initial_state, cu_seqlens,
                                                 scale=scale, decay_bias=decay_bias),
            fixed=fixed is not None,
        )
    return config[0] if block_values is None else block_values, config[1] if prefetch_tokens is None else prefetch_tokens


def tmix_wkv7_recurrent(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale: float = 1.0,
    decay_bias: torch.Tensor | None = None,
    block_values: int | None = None,
    prefetch_tokens: int | None = None,
    _status: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run packed recurrent forward, with the same arithmetic in training.

    Token inputs are contiguous BF16 ``[N,H,D]``; state is FP32 ``[B,H,D,D]``.
    ``cu_seqlens`` is contiguous CUDA int32 ``[B+1]`` spanning ``[0,N]``.
    Empty sequences preserve their initial state.  Optional BF16/FP32 decay bias
    ``[H,D]`` or ``[H*D]`` is added in FP32 and rounded to BF16 before the
    retention transform, matching a prepare stage that materializes logits.
    State is functional: the caller-owned initial state is never mutated.

    ``block_values`` chooses independent value columns per CTA, leaving the
    key reduction unchanged. ``prefetch_tokens`` selects one or four tokens
    per shared-memory load. The validated RTX 4090 head-size-64 FP32-state
    path uses 64 columns and selects prefetch by workload; other paths use a
    verified profile or the default of 64 columns and one token.
    """
    batch = initial_state.shape[0] if initial_state.ndim else 0
    _check_inputs(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        initial_state,
        cu_seqlens,
        scale,
        decay_bias,
        64 if block_values is None else block_values,
        1 if prefetch_tokens is None else prefetch_tokens,
        batch,
    )
    if torch.compiler.is_compiling():
        status = _status if _status is not None else r.new_empty(0, dtype=torch.int32)
        save = torch.is_grad_enabled() and any(value.requires_grad for value in (r, decay_logits, k, v, a, b, initial_state))
        if decay_bias is not None:
            save = save or (torch.is_grad_enabled() and decay_bias.requires_grad)
        return _compiled_recurrent(r, decay_logits, k, v, a, b, initial_state, cu_seqlens,
                                   _bias_tensor(r, decay_bias), status, float(scale), block_values, prefetch_tokens, save)[:2]
    block_values, prefetch_tokens = _functional_config(
        r, decay_logits, k, v, a, b, initial_state, cu_seqlens, scale, decay_bias, block_values, prefetch_tokens,
    )
    bias = _bias_tensor(r, decay_bias)
    status = (
        _status
        if _status is not None
        else torch.empty(0, dtype=torch.int32, device=r.device)
    )
    inputs = (r, decay_logits, k, v, a, b, initial_state, cu_seqlens, bias, status)
    if torch.is_grad_enabled() and any(value.requires_grad for value in inputs):
        return _Recurrent.apply(*inputs, float(scale), block_values, prefetch_tokens)
    return _forward(*inputs, float(scale), False, block_values, prefetch_tokens)[:2]


@torch.no_grad()
@tuning_operation
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
    state_indices=None,
    status=None,
):
    """Check both forward variants and time the requested production mode.

    Every candidate must preserve functional/indexed BF16 outputs and the
    final states. ``indexed=True`` measures in-place inference on a private
    pool with identity slots and valid status. Otherwise it measures the
    functional forward. The two modes retain separate configuration caches;
    neither mode mutates the caller's initial state.
    """
    from ..._internal.tuning import save_config, select_config

    if state_indices is not None:
        if status.tolist() != [0, r.shape[0], state_indices.numel()]:
            raise ValueError("WKV tuning requires valid fully active representative metadata for the capture shape")
        initial_state = initial_state.index_select(0, state_indices.long())
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
    key = _config_key(
        r, batch, decay_bias, indexed=indexed, state_dtype=initial_state.dtype
    )
    _TUNED_CONFIGS[key] = best
    save_config("tmix/wkv7", key, best)
    return {
        "block_values": best[0], "prefetch_tokens": best[1],
        "indexed": indexed, "candidates": measurements,
    }


__all__ = ["tmix_wkv7_recurrent", "tune_tmix_wkv7_recurrent"]
