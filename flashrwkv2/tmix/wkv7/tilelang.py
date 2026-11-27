# SPDX-License-Identifier: MIT

"""Shared RWKV-7 recurrence: BF16 token IO and FP32 state/accumulation.

An explicit inference-only mode stores FP16 state after every token; it does
not change training or claim equality with the default FP32-state trajectory.

State is stored as [sequence, head, key, value].  Each state mode has a fixed key reduction
and update order independent of sequence packing and value tiling. D64 FP16
serving with 40 or 64 heads reduces four contiguous groups of 16 keys.
The backward differentiates the real-valued recurrence, treating the BF16
output cast as an identity, as in ordinary mixed-precision training.
It saves FP32 boundaries every 16 tokens. For longer head-size-64 inputs,
checkpoint replay saves a vector per token and the reverse pass replays short
FP32 state windows with one writer per gradient. Other inputs use bounded shared-memory replay.
"""

import math
from functools import cache

import tilelang
import tilelang.language as T
import torch

from ..._internal.execution import (
    _AutogradContext,
    _fixed_strategy_device,
    device_class,
)

_CHECKPOINT = 16
_BACKWARD_SEGMENT = 4096
_LOG2E = 1.4426950408889634
_EXP_NEG_HALF = 0.6065306597126334
_DECAY_EXP2 = -0.8750387749145276
_PASSES = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False}
_COMPILE_FLAGS = ["--fmad=false"]

# FP16 serving token staging slots and DeltaLog layout constants.
_TOK_R, _TOK_DECAY, _TOK_K, _TOK_V, _TOK_A, _TOK_B = 0, 1, 2, 3, 4, 5
_LOG_DECAY, _LOG_K, _LOG_V, _LOG_B, _LOG_SA = 0, 1, 2, 3, 4
_DELTALOG_STANDARD_FIELDS, _DELTALOG_COMPACT_FIELDS = 5, 6


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


@T.macro
def _load_deltalog(log, slot, head, previous, fields, record):
    if fields == _DELTALOG_COMPACT_FIELDS:
        for key in T.Parallel(64):
            wk = T.reinterpret(log[slot, head, 3 * record, key], "uint32")
            vb = T.reinterpret(log[slot, head, 3 * record + 1, key], "uint32")
            w = T.reinterpret(wk & T.uint32(0xffff0000), "float32")
            previous[_LOG_DECAY, key] = T.exp2(_DECAY_EXP2 / (1.0 + T.exp2(-_LOG2E * w)))
            previous[_LOG_K, key] = T.reinterpret(wk << 16, "float32")
            previous[_LOG_V, key] = T.reinterpret(vb & T.uint32(0xffff0000), "float32")
            previous[_LOG_B, key] = T.reinterpret(vb << 16, "float32")
            previous[_LOG_SA, key] = log[slot, head, 3 * record + 2, key]
    else:
        for field, key in T.Parallel(_DELTALOG_STANDARD_FIELDS, 64):
            previous[field, key] = log[slot, head, field, key]


@T.macro
def _replay_deltalog(state, log, slot, head, fields, record):
    previous = T.alloc_shared((_DELTALOG_STANDARD_FIELDS, 64), "float32")
    _load_deltalog(log, slot, head, previous, fields, record)
    T.sync_threads()
    for key, col in T.Parallel(64, 64):
        replay = T.fma(previous[_LOG_K, key], previous[_LOG_V, col],
                       T.fma(previous[_LOG_B, key], previous[_LOG_SA, col],
                             previous[_LOG_DECAY, key] * state[key, col]))
        state[key, col] = T.cast(T.cast(replay, "float16"), "float32")


@tilelang.jit(pass_configs=_PASSES, compile_flags=_COMPILE_FLAGS)
def _copy_recurrent_slots(channels, heads, dtype, packed_source, with_log):
    src_capacity, dst_capacity, batch = T.dynamic("src_capacity, dst_capacity, batch")
    tile = 2048

    @T.prim_func
    def kernel(
        Source: T.Tensor((src_capacity, channels), dtype),
        Destination: T.Tensor((dst_capacity, channels), dtype),
        SrcSlots: T.Tensor((batch,), "int32"),
        DstSlots: T.Tensor((batch,), "int32"),
        Status: T.Tensor((3,), "int32"),
        Phase: T.Tensor((dst_capacity if with_log else 0, heads), "int32"),
    ):
        with T.Kernel(channels // tile, batch, threads=128) as (block, seq):
            if Status[0] == 0 and seq < Status[2]:
                src = seq if packed_source else SrcSlots[seq]
                dst = DstSlots[seq]
                for c in T.Parallel(tile):
                    channel = block * tile + c
                    Destination[dst, channel] = Source[src, channel]
                if with_log and block == 0:
                    for h in T.Parallel(heads):
                        Phase[dst, h] = 0

    return kernel


@tilelang.jit(pass_configs=_PASSES, compile_flags=_COMPILE_FLAGS)
def _materialize_deltalog(heads, fields, has_status):
    batch, capacity = T.dynamic("batch, capacity")
    records = 2 if fields == 6 else 1
    count_mask = 0xf if fields == 6 else 1
    claim = """
    __device__ __forceinline__ int rwkv_claim_log(int* phase, long long offset) {
        return atomicExch(phase + offset, 0);
    }
    """

    @T.prim_func
    def kernel(
        Pool: T.Tensor((capacity, heads, 64, 64), "float16"),
        Log: T.Tensor((capacity, heads, fields, 64), "float32"),
        Phase: T.Tensor((capacity, heads), "int32"),
        Slots: T.Tensor((batch,), "int32"),
        Status: T.Tensor((3 if has_status else 0,), "int32"),
    ):
        with T.Kernel(heads, batch, threads=64, prelude=claim) as (h, seq):
            if not has_status or (Status[0] == 0 and seq < Status[2]):
                slot = Slots[seq]
                pending = T.alloc_shared((1,), "int32")
                if T.get_thread_binding() == 0:
                    # Repeated source slots may be requested by prefix or beam copies.
                    pending[0] = T.call_extern("int32", "rwkv_claim_log",
                                               Phase.access_ptr("rw"), T.int64(slot) * heads + h)
                T.sync_threads()
                if (pending[0] & count_mask) != 0:
                    state = T.alloc_fragment((64, 64), "float32")
                    T.annotate_layout({state: T.Fragment((64, 64), forward_thread_fn=lambda k, c: c,
                                                         forward_index_fn=lambda k, c: k)})
                    for key, col in T.Parallel(64, 64):
                        state[key, col] = Pool[slot, h, key, col]
                    for record in T.serial(records):
                        if record < (pending[0] & count_mask):
                            _replay_deltalog(state, Log, slot, h, fields, record)
                            T.sync_threads()
                    for key, col in T.Parallel(64, 64):
                        Pool[slot, h, key, col] = T.cast(state[key, col], "float16")

    return kernel


@tilelang.jit(pass_configs=_PASSES, compile_flags=_COMPILE_FLAGS)
def _decode_traversal(heads, ordered, compact):
    batch, capacity = T.dynamic("batch, capacity")

    @T.prim_func
    def kernel(Phase: T.Tensor((capacity, heads), "int32"),
               Slots: T.Tensor((batch,), "int32"), Status: T.Tensor((3,), "int32"),
               Order: T.Tensor((batch + 1 if ordered else 0,), "int32"),
               Reverse: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=32):
            if T.get_thread_binding() == 0:
                first = Order[batch] if ordered else 0
                Reverse[0] = 0
                if first < Status[2]:
                    seq = Order[first] if ordered else 0
                    Reverse[0] = (Phase[Slots[seq], 0] >> (16 if compact else 0)) & 1
    return kernel


@tilelang.jit(pass_configs={**_PASSES, tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True},
              compile_flags=_COMPILE_FLAGS)
def _fp16_serving_kernel(
    heads, scale, bias_dtype, has_bias, ordered, with_log, allow_append,
    alternate=False, bulk_threshold=0, fields=5, rows="all", checkpoint_interval=0,
    write_output=True,
):
    """D64 recurrence: each warp sums fixed contiguous key groups for its value columns."""
    n, batch, slots = T.dynamic("n, batch, slots")
    stride = checkpoint_interval or 1
    checkpoints = n // stride + batch if checkpoint_interval else 0
    records = 2 if fields == 6 else 1
    count_mask = 0xf if fields == 6 else 1
    groups, columns = 4, 16
    lanes = 32 // groups
    keys = 64 // groups
    threads = columns * groups
    tiles = 64 // columns
    reduction = r"""
    __device__ __forceinline__ int rwkv_row(const int* order, int rank, int tiled,
                                           int prefills, int active, int reverse, int ordered) {
      int index;
      if (rank < 4 * tiled) index = rank / 4;
      else {
        index = rank - 3 * tiled;
        if (reverse && index >= prefills) index = active - 1 - (index - prefills);
      }
      return ordered ? order[index] : index;
    }
    __device__ __forceinline__ int rwkv_claim_prefill_tile(int* phases, long long index, int tile, int bits) {
      const int shift = bits * tile, mask = ((1 << bits) - 1) << shift;
      return (atomicAnd(phases + index, ~mask) & mask) >> shift;
    }
    __device__ __forceinline__ float rwkv_sum_groups(float x) {
      #pragma unroll
      for (int delta = 8; delta < 32; delta *= 2)
        x = __fadd_rn(x, __shfl_xor_sync(0xffffffff, x, delta));
      return x;
    }
    """

    if rows == "decode":
        # Streaming state accesses leave space for the repeatedly read token and journal vectors.
        reduction += r"""
        __device__ __forceinline__ float rwkv_stream_load(const void* pool, long long index) {
          unsigned short bits;
          asm("ld.global.cs.b16 %0, [%1];" : "=h"(bits)
              : "l"(static_cast<const unsigned short*>(pool) + index));
          float value;
          asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(bits));
          return value;
        }
        __device__ __forceinline__ void rwkv_stream_store(void* pool, long long index, float value) {
          unsigned short bits;
          asm("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(value));
          asm volatile("st.global.cs.b16 [%0], %1;" ::
                       "l"(static_cast<unsigned short*>(pool) + index), "h"(bits) : "memory");
        }
        """

    @T.macro
    def load_token(R, W, K, V, A, B, Bias, vectors, token, h, t, key):
        if write_output:
            vectors[t, _TOK_R, key] = T.cast(R[token, h, key], "float32")
        logits = T.cast(W[token, h, key], "float32")
        if has_bias:
            logits = T.cast(T.cast(logits + T.cast(Bias[h, key], "float32"), "bfloat16"), "float32")
        vectors[t, _TOK_DECAY, key] = T.exp2(_DECAY_EXP2 / (1.0 + T.exp2(-_LOG2E * logits)))
        vectors[t, _TOK_K, key] = T.cast(K[token, h, key], "float32")
        vectors[t, _TOK_V, key] = T.cast(V[token, h, key], "float32")
        vectors[t, _TOK_A, key] = T.cast(A[token, h, key], "float32")
        vectors[t, _TOK_B, key] = T.cast(B[token, h, key], "float32")

    @T.macro
    def step(state, vectors, Y, Boundaries, token, seq, start, h, t, group, col, dot):
        if checkpoint_interval and (token - start) % stride == 0:
            checkpoint = start // stride + seq + (token - start) // stride
            for j in T.unroll(keys):
                Boundaries[checkpoint, h, group * keys + j, col] = T.cast(state[j], "float16")
        accum = T.alloc_var("float32", init=0.0)
        for j in T.unroll(keys):
            accum = T.fma(state[j], vectors[t, _TOK_A, group * keys + j], accum)
        dot[0] = T.call_pure_extern("float32", "rwkv_sum_groups", accum)
        accum = 0.0
        for j in T.unroll(keys):
            k = group * keys + j
            updated = T.fma(vectors[t, _TOK_K, k], vectors[t, _TOK_V, col],
                            T.fma(vectors[t, _TOK_B, k], dot[0], vectors[t, _TOK_DECAY, k] * state[j]))
            state[j] = T.cast(T.cast(updated, "float16"), "float32")
            if write_output:
                accum = T.fma(state[j], vectors[t, _TOK_R, k], accum)
        if write_output:
            output = T.call_pure_extern("float32", "rwkv_sum_groups", accum)
            if group == 0:
                Y[token, h, col] = T.cast(output * scale, "bfloat16")

    @T.macro
    def column_step(state, values, Y, Boundaries, token, seq, start, h, t, tid, sa):
        if checkpoint_interval and (token - start) % stride == 0:
            checkpoint = start // stride + seq + (token - start) // stride
            for key in T.unroll(64):
                Boundaries[checkpoint, h, key, tid] = T.cast(state[key], "float16")
        sy = T.alloc_local((groups if write_output else 0,), "float32")
        for g in T.unroll(groups):
            sa[g] = 0.0
            if write_output:
                sy[g] = 0.0
            for j in T.unroll(keys):
                sa[g] = T.fma(state[g * keys + j], values[t, _TOK_A, g * keys + j], sa[g])
        sa[0] = (sa[0] + sa[1]) + (sa[2] + sa[3])
        for g in T.unroll(groups):
            for j in T.unroll(keys):
                k = g * keys + j
                val = T.fma(values[t, _TOK_K, k], values[t, _TOK_V, tid],
                            T.fma(values[t, _TOK_B, k], sa[0], values[t, _TOK_DECAY, k] * state[k]))
                state[k] = T.cast(T.cast(val, "float16"), "float32")
                if write_output:
                    sy[g] = T.fma(state[k], values[t, _TOK_R, k], sy[g])
        if write_output:
            sy[0] = (sy[0] + sy[1]) + (sy[2] + sy[3])
            Y[token, h, tid] = T.cast(sy[0] * scale, "bfloat16")

    @T.macro
    def full_head(R, W, K, V, A, B, Bias, Pool, Log, Phase, Y, Boundaries, seq, start, end, slot, h, tid, single):
        vectors = T.alloc_shared((1 if single else 4, 6, 64), "float32")
        past = T.alloc_shared((5 if with_log else 0, 64), "float32")
        pending = T.alloc_shared((1 if with_log else 0,), "int32")
        state = T.alloc_local((64,), "float32")
        sa = T.alloc_local((groups,), "float32")
        offset = (T.int64(slot) * heads + h) * 4096 + tid
        for key in T.unroll(64):
            if rows == "decode":
                state[key] = T.call_extern("float32", "rwkv_stream_load", Pool.access_ptr("r"), offset + key * 64)
            else:
                state[key] = T.cast(Pool[slot, h, key, tid], "float32")
        if single:
            load_token(R, W, K, V, A, B, Bias, vectors, start, h, 0, tid)
        if with_log:
            if tid == 0:
                pending[0] = Phase[slot, h]
        T.sync_threads()
        if with_log:
            for record in T.serial(records):
                if record < (pending[0] & count_mask):
                    _load_deltalog(Log, slot, h, past, fields, record)
                    T.sync_threads()
                    for key in T.unroll(64):
                        updated = T.fma(past[1, key], past[2, tid],
                                        T.fma(past[3, key], past[4, tid], past[0, key] * state[key]))
                        state[key] = T.cast(T.cast(updated, "float16"), "float32")
                    T.sync_threads()
        if single:
            column_step(state, vectors, Y, Boundaries, start, seq, start, h, 0, tid, sa)
        else:
            for chunk in T.serial((end - start) // 4):
                for t, key in T.Parallel(4, 64):
                    load_token(R, W, K, V, A, B, Bias, vectors, start + chunk * 4 + t, h, t, key)
                T.sync_threads()
                for t in T.unroll(4):
                    column_step(state, vectors, Y, Boundaries, start + chunk * 4 + t, seq, start, h, t, tid, sa)
                T.sync_threads()
            for tail in T.serial((end - start) % 4):
                token = end - (end - start) % 4 + tail
                load_token(R, W, K, V, A, B, Bias, vectors, token, h, 0, tid)
                T.sync_threads()
                column_step(state, vectors, Y, Boundaries, token, seq, start, h, 0, tid, sa)
                T.sync_threads()
        # Offset the three-step cycle by head so each call writes back part of the pool.
        append = False
        if with_log and allow_append and single:
            if fields == 6:
                tick = (pending[0] >> 17) + h * 3 // heads
                append = (pending[0] & count_mask) < records and (rows != "decode" or tick % 3 != 2)
            else:
                append = (pending[0] & 1) == 0
            if append:
                if fields == 6:
                    # The four BF16 operands take two words. The state dot keeps all 32 bits.
                    w = T.cast(W[start, h, tid], "float32")
                    if has_bias:
                        w = T.cast(T.cast(w + T.cast(Bias[h, tid], "float32"), "bfloat16"), "float32")
                    wk = ((T.reinterpret(w, "uint32") & T.uint32(0xffff0000))
                          | (T.reinterpret(vectors[0, 2, tid], "uint32") >> 16))
                    vb = ((T.reinterpret(vectors[0, 3, tid], "uint32") & T.uint32(0xffff0000))
                          | (T.reinterpret(vectors[0, 5, tid], "uint32") >> 16))
                    log_offset = 3 * (pending[0] & count_mask)
                    Log[slot, h, log_offset, tid] = T.reinterpret(wk, "float32")
                    Log[slot, h, log_offset + 1, tid] = T.reinterpret(vb, "float32")
                    Log[slot, h, log_offset + 2, tid] = sa[0]
                else:
                    Log[slot, h, 0, tid] = vectors[0, 1, tid]
                    Log[slot, h, 1, tid] = vectors[0, 2, tid]
                    Log[slot, h, 2, tid] = vectors[0, 3, tid]
                    Log[slot, h, 3, tid] = vectors[0, 5, tid]
                    Log[slot, h, 4, tid] = sa[0]
            else:
                for key in T.unroll(64):
                    if rows == "decode":
                        T.call_extern("void", "rwkv_stream_store", Pool.access_ptr("w"), offset + key * 64, state[key])
                    else:
                        Pool[slot, h, key, tid] = T.cast(state[key], "float16")
        else:
            for key in T.unroll(64):
                Pool[slot, h, key, tid] = T.cast(state[key], "float16")
        if with_log:
            if tid == 0:
                if allow_append and single:
                    if fields == 6:
                        count = T.if_then_else(append, (pending[0] & count_mask) + 1, 0)
                        epoch = ((pending[0] >> 17) + 1) % 3
                        # Four independently claimable counts, the traversal bit, then the tick.
                        direction = (pending[0] ^ 0x10000) & 0x10000
                        Phase[slot, h] = (epoch << 17) | direction | count * 0x1111
                    else:
                        Phase[slot, h] = T.if_then_else(append, (1 << tiles) - 1, 0)
                else:
                    Phase[slot, h] = 0

    @T.prim_func
    def kernel(
        R: T.Tensor((n if write_output else 0, heads, 64), "bfloat16"), W: T.Tensor((n, heads, 64), "bfloat16"),
        K: T.Tensor((n, heads, 64), "bfloat16"), V: T.Tensor((n, heads, 64), "bfloat16"),
        A: T.Tensor((n, heads, 64), "bfloat16"), B: T.Tensor((n, heads, 64), "bfloat16"),
        Pool: T.Tensor((slots, heads, 64, 64), "float16"), CU: T.Tensor((batch + 1,), "int32"),
        Bias: T.Tensor((heads if has_bias else 0, 64), bias_dtype),
        Status: T.Tensor((3,), "int32"), Indices: T.Tensor((batch,), "int32"),
        Y: T.Tensor((n if write_output else 0, heads, 64), "bfloat16"),
        Order: T.Tensor((batch + 1 if ordered else 0,), "int32"),
        Log: T.Tensor((slots if with_log else 0, heads, fields, 64), "float32"),
        Phase: T.Tensor((slots if with_log else 0, heads), "int32"),
        Reverse: T.Tensor((1 if alternate else 0,), "int32"),
        Boundaries: T.Tensor((checkpoints, heads, 64, 64), "float16"),
    ):
        # Full heads need one row per grid slot. A sparse prefill uses four value tiles.
        extra = T.min(batch, 4) * (tiles - 1) if ordered else T.if_then_else(batch == 1 and n > 1, tiles - 1, 0)
        width = batch if bulk_threshold else batch + extra
        if rows == "prefill":
            # Four value tiles per sparse prompt. Dense prompts stride the same grid.
            width = T.min(batch, 4) * tiles
        with T.Kernel(heads, width, threads=threads, prelude=reduction) as (h, priority):
            tid = T.get_thread_binding()
            group = tid % 32 // lanes
            local_col = tid // 32 * lanes + tid % lanes
            state = T.alloc_local((keys,), "float32")
            dot = T.alloc_local((1,), "float32")
            vectors = T.alloc_shared((4, 6, 64), "float32")
            previous = T.alloc_shared((5 if with_log else 0, 64), "float32")
            phase = T.alloc_shared((1 if with_log else 0,), "int32")
            if write_output and rows != "prefill" and Status[1] < n and priority < batch:
                if tid < 64:
                    for i in T.serial(T.ceildiv(n - Status[1], batch)):
                        row = Status[1] + i * batch + priority
                        if row < n:
                            Y[row, h, tid] = T.cast(float("nan"), "bfloat16")
            num_prefills = Order[batch] if ordered else T.if_then_else(batch == 1 and Status[1] > 1, 1, 0)
            tiled = T.if_then_else(num_prefills >= bulk_threshold, 0, num_prefills) if bulk_threshold else num_prefills
            prefill_tasks = num_prefills + (tiles - 1) * tiled
            if rows == "decode":
                tasks = Status[2] - num_prefills
            elif rows == "prefill":
                tasks = prefill_tasks
            else:
                tasks = Status[2] + (tiles - 1) * tiled
            visits = T.cast(priority < tasks, "int32") if rows == "decode" else T.ceildiv(tasks - priority, width)
            for offset in T.serial(visits):
                rank = priority + offset * width
                narrow = rank < tiles * tiled
                is_prefill = True if rows == "prefill" else False if rows == "decode" else rank < prefill_tasks
                direction = Reverse[0] if alternate else 0
                if rows == "decode":
                    index = T.if_then_else(direction != 0, Status[2] - 1 - priority, num_prefills + priority)
                    seq = Order[index] if ordered else index
                else:
                    seq = T.call_pure_extern("int32", "rwkv_row", Order.access_ptr("r"), rank, tiled,
                                             num_prefills, Status[2], direction, int(ordered))
                start, end = CU[seq], CU[seq + 1]
                slot = Indices[seq]
                if not is_prefill:
                    full_head(R, W, K, V, A, B, Bias, Pool, Log, Phase, Y, Boundaries, seq, start, end, slot, h, tid, True)
                elif bulk_threshold and not narrow:
                    full_head(R, W, K, V, A, B, Bias, Pool, Log, Phase, Y, Boundaries, seq, start, end, slot, h, tid, False)
                else:
                    tile = rank % tiles
                    col = tile * columns + local_col
                    for j in T.unroll(keys):
                        state[j] = T.cast(Pool[slot, h, group * keys + j, col], "float32")
                    if with_log:
                        if tid == 0:
                            phase[0] = T.call_extern("int32", "rwkv_claim_prefill_tile", Phase.access_ptr("rw"), T.int64(slot) * heads + h, tile, 4 if fields == 6 else 1)
                        T.sync_threads()
                        for record in T.serial(records):
                            if record < phase[0]:
                                _load_deltalog(Log, slot, h, previous, fields, record)
                                T.sync_threads()
                                for j in T.unroll(keys):
                                    k = group * keys + j
                                    replay = T.fma(previous[1, k], previous[2, col],
                                                   T.fma(previous[3, k], previous[4, col],
                                                         previous[0, k] * state[j]))
                                    state[j] = T.cast(T.cast(replay, "float16"), "float32")
                                T.sync_threads()
                    for chunk in T.serial((end - start) // 4):
                        for t, k in T.Parallel(4, 64):
                            load_token(R, W, K, V, A, B, Bias, vectors, start + chunk * 4 + t, h, t, k)
                        T.sync_threads()
                        for t in T.unroll(4):
                            step(state, vectors, Y, Boundaries, start + chunk * 4 + t, seq, start, h, t, group, col, dot)
                        T.sync_threads()
                    for tail in T.serial((end - start) % 4):
                        token = end - (end - start) % 4 + tail
                        if tid < 64:
                            load_token(R, W, K, V, A, B, Bias, vectors, token, h, 0, tid)
                        T.sync_threads()
                        step(state, vectors, Y, Boundaries, token, seq, start, h, 0, group, col, dot)
                        T.sync_threads()
                    for j in T.unroll(keys):
                        Pool[slot, h, group * keys + j, col] = T.cast(state[j], "float16")
    return kernel


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
    checkpoint_interval: int = _CHECKPOINT,
    write_output: bool = True,
):
    n, batch = T.dynamic("n, batch")
    state_slots = T.dynamic("state_slots") if indexed else batch
    columns = min(block_values, dim)
    key_groups = dim // 64
    checkpoints = n // checkpoint_interval + batch
    threads = columns * key_groups
    within_loop = T.unroll if prefetch_tokens > 1 and dim <= 128 else T.serial

    @T.prim_func
    def kernel(
        R: T.Tensor((n if write_output else 0, heads, dim), "bfloat16"),
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
        Y: T.Tensor((n if write_output else 0, heads, dim), "bfloat16"),
        SN: T.Tensor((0 if indexed else batch, heads, dim, dim), state_dtype),
        Boundaries: T.Tensor((checkpoints if save else 0, heads, dim, dim), state_dtype),
    ):
        with T.Kernel(dim // columns, heads, batch, threads=threads) as (tile, h, seq):
            state = T.alloc_fragment((dim, columns), "float32")
            partial = T.alloc_fragment((key_groups, columns), "float32")
            dot = T.alloc_fragment((columns,), "float32")
            out = T.alloc_fragment((columns,), "float32")
            rv = T.alloc_shared((prefetch_tokens * dim if write_output else 0,), "float32")
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
                if write_output:
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
                        if write_output:
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
                        if save and (token - start) % checkpoint_interval == 0:
                            checkpoint = (
                                start // checkpoint_interval + seq
                                + (token - start) // checkpoint_interval
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
                        if write_output:
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


def _direct_vjp(r, w, k, v, a, b, cu, bias, boundary, dy, ds, options, segmented):
    n, heads, dim = r.shape
    batch = ds.shape[0]
    _, _, has_bias, bias_dtype = options
    sa = torch.empty_like(r, dtype=torch.float32)
    ds0 = torch.empty_like(ds)
    gradients = r.new_empty((6, n, heads, dim))
    _prepare_sa_kernel(heads, has_bias, bias_dtype)(
        w, k, v, a, b, cu, bias, boundary, sa,
    )
    incoming = ds.new_empty((n // _BACKWARD_SEGMENT + batch if segmented else 0, heads, dim, dim))
    if segmented:
        _adjoint_boundaries_kernel(*options)(
            r, w, a, b, dy, cu, bias, ds, incoming,
        )
    _direct_backward_kernel(*options, segmented)(
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
            active = Status[0] == 0 and seq < Status[2] if has_status else True
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
    _forward_kernel(
        heads,
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
    n, heads, dim = r.shape
    batch = ds.shape[0]
    has_bias = bias.numel() != 0
    has_status = status.numel() != 0
    bias_dtype = "bfloat16" if bias.dtype == torch.bfloat16 else "float32"
    options = heads, dim, ctx.scale, has_bias, has_status, bias_dtype
    direct_options = heads, ctx.scale, has_bias, bias_dtype
    can_direct = dim == 64 and not has_status
    # The extra replay launch pays off on longer sequences.
    direct = can_direct and n >= max(4096, _CHECKPOINT * batch)
    segmented = n > _BACKWARD_SEGMENT and batch * heads < 128
    if direct:
        rounded, ds0 = _direct_vjp(r, w, k, v, a, b, cu, bias, boundary, dy, ds, direct_options, segmented)
        dbias = rounded[1].float().sum(dim=0).to(bias.dtype) if bias.numel() else None
        return rounded, ds0, dbias
    gradients = torch.zeros((6, *r.shape), dtype=torch.float32, device=r.device)
    ds0 = torch.empty_like(ds)
    _backward_kernel(*options)(
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
    geometry = decay_logits if r is None else r
    reference = "decay_logits" if r is None else "r"
    if geometry.ndim != 3 or geometry.shape[1] < 1 or geometry.shape[2] not in (64, 128, 256):
        raise ValueError(f"{reference} must have shape [N,H,D], with D in {{64,128,256}}")
    for name, value in zip(
        ("r", "decay_logits", "k", "v", "a", "b"), (r, decay_logits, k, v, a, b), strict=True
    ):
        if name == "r" and r is None:
            continue
        if value.shape != geometry.shape or value.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be BF16 with the same [N,H,D] shape as {reference}")
        if not value.is_cuda or value.device != geometry.device or not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA on the {reference} device")
    if (
        initial_state.ndim != 4
        or initial_state.shape[1:] != (geometry.shape[1], geometry.shape[2], geometry.shape[2])
        or initial_state.shape[0] < 1
    ):
        raise ValueError("initial_state must have shape [B,H,D,D] with B >= 1")
    if (
        initial_state.dtype not in ((torch.float32, torch.float16) if inference else (torch.float32,))
        or initial_state.device != geometry.device
        or not initial_state.is_contiguous()
    ):
        dtype = "FP16/FP32" if inference else "FP32"
        raise ValueError(f"initial_state must be contiguous {dtype} on the {reference} device")
    if (
        cu_seqlens.shape != (batch + 1,)
        or cu_seqlens.dtype != torch.int32
        or cu_seqlens.device != geometry.device
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
        decay_bias.shape not in (geometry.shape[1:], (geometry.shape[1] * geometry.shape[2],))
        or decay_bias.dtype not in (torch.float32, torch.bfloat16)
        or decay_bias.device != geometry.device
        or not decay_bias.is_contiguous()
    ):
        raise ValueError(
            f"decay_bias must be contiguous BF16/FP32 [H,D] or [H*D] on the {reference} device"
        )


def _bias_tensor(r, decay_bias):
    if decay_bias is None:
        return torch.empty((0, r.shape[2]), dtype=torch.float32, device=r.device)
    return decay_bias.reshape(r.shape[1:])


def _recurrent_config(r, batch, columns, prefetch):
    if columns is not None and prefetch is not None:
        return columns, prefetch
    tile = (64, 1)
    if r.shape[-1] == 64 and r.shape[0] > batch:
        device = device_class(r.device)
        if _fixed_strategy_device(device):
            tile = (64, 4)
    return tile[0] if columns is None else columns, tile[1] if prefetch is None else prefetch


@cache
def _prefill_stream(caller):
    return torch.cuda.Stream(device=caller.device, priority=caller.priority)


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
    row_order=None,
    delta_buffers=None,
    append_log=False,
    save=False,
    boundaries=None,
    checkpoint_interval=_CHECKPOINT,
):
    """Launch the common forward directly against an indexed inference pool."""
    write_output = r is not None
    geometry = r if write_output else decay_logits
    if (
        state_indices.ndim != 1
        or state_indices.numel() == 0
        or state_indices.dtype != torch.int32
        or state_indices.device != geometry.device
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

    tokens, heads, dim = geometry.shape
    device = geometry.device
    if save:
        if (not isinstance(checkpoint_interval, int) or isinstance(checkpoint_interval, bool)
                or checkpoint_interval <= 0):
            raise ValueError("checkpoint_interval must be a positive integer")
        shape = (tokens // checkpoint_interval + batch, heads, dim, dim)
        if boundaries is None:
            boundaries = state_pool.new_empty(shape)
        elif (
            not isinstance(boundaries, torch.Tensor) or boundaries.shape != shape
            or boundaries.dtype != state_pool.dtype or boundaries.device != device
            or not boundaries.is_contiguous()
        ):
            raise ValueError(
                f"boundaries must be contiguous {state_pool.dtype} with shape {shape} on {device}"
            )
        start = boundaries.data_ptr()
        end = start + boundaries.nbytes
        inputs = (r, decay_logits, k, v, a, b, decay_bias, state_pool, cu_seqlens, state_indices, status)
        for tensor in inputs:
            if tensor is not None and tensor.nbytes:
                address = tensor.data_ptr()
                if start < address + tensor.nbytes and address < end:
                    raise ValueError("boundaries must not overlap inputs or state_pool")
    else:
        checkpoint_interval = _CHECKPOINT
    kernel_r = r if write_output else geometry[:0]
    kernel_y = torch.empty_like(v) if write_output else kernel_r
    output = kernel_y if write_output else None
    bias = _bias_tensor(geometry, decay_bias)
    if state_pool.dtype == torch.float16 and dim == 64 and heads in (40, 64):
        order = row_order
        if order is None:
            lengths = torch.where(torch.arange(batch, device=device) < status[2],
                                  cu_seqlens[1:] - cu_seqlens[:-1], 0)
            order = torch.cat((lengths.argsort(descending=True, stable=True).int(),
                               (lengths > 1).sum(dtype=torch.int32).view(1)))
        if delta_buffers is None:
            log = torch.empty((0, heads, 5, 64), device=device, dtype=torch.float32)
            phase = status[:0].view(0, heads)
        else:
            log, phase = delta_buffers
        alternate = append_log and batch >= 128
        reverse = status.new_empty(1) if alternate else status[:0]
        if alternate:
            _decode_traversal(heads, order.numel() != 0, log.shape[2] == 6)(phase, state_indices, status, order, reverse)
        config = (
            heads, float(scale), "bfloat16" if bias.dtype == torch.bfloat16 else "float32",
            bias.numel() != 0, order.numel() != 0, delta_buffers is not None, append_log, alternate,
            4 if batch >= 4 and order.numel() else 0, log.shape[2],
        )
        tensors = (kernel_r, decay_logits, k, v, a, b, state_pool, cu_seqlens, bias, status, state_indices,
                   kernel_y, order, log, phase, reverse,
                   boundaries if save else state_pool[:0])
        if alternate and log.shape[2] == 6:
            if order.numel():
                caller = torch.cuda.current_stream(device)
                prefill = _prefill_stream(caller)
                prefill.wait_stream(caller)
                with torch.cuda.stream(prefill):
                    _fp16_serving_kernel(*config, rows="prefill", write_output=write_output)(*tensors)
                _fp16_serving_kernel(*config, rows="decode", write_output=write_output)(*tensors)
                caller.wait_stream(prefill)
            else:
                _fp16_serving_kernel(*config, rows="decode", write_output=write_output)(*tensors)
        else:
            _fp16_serving_kernel(
                *config, checkpoint_interval=checkpoint_interval if save else 0, write_output=write_output,
            )(*tensors)
        return (output, boundaries) if save else output

    block_values, prefetch_tokens = _recurrent_config(
        geometry, batch, block_values, prefetch_tokens,
    )
    empty = state_pool.new_empty((0, heads, dim, dim))
    _forward_kernel(
        heads,
        dim,
        float(scale),
        bias.numel() != 0,
        save,
        block_values,
        True,
        True,
        "bfloat16" if bias.dtype == torch.bfloat16 else "float32",
        prefetch_tokens,
        str(state_pool.dtype).removeprefix("torch."),
        checkpoint_interval=checkpoint_interval,
        write_output=write_output,
    )(
        kernel_r,
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
        kernel_y,
        empty,
        boundaries if save else empty,
    )
    return (output, boundaries) if save else output


@torch.library.custom_op("flashrwkv2::tmix_wkv7_forward", mutates_args=())
def _compiled_recurrent(
    r: torch.Tensor, w: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    a: torch.Tensor, b: torch.Tensor, state: torch.Tensor, cu: torch.Tensor,
    bias: torch.Tensor, status: torch.Tensor, scale: float,
    block_values: int | None, prefetch_tokens: int | None, save: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    block_values, prefetch_tokens = _recurrent_config(
        r, state.shape[0], block_values, prefetch_tokens,
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
    per shared-memory load. The committed SM89 and SM120 configurations use
    four-token prefetch for D64 multi-token workloads and one for decode.
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
    block_values, prefetch_tokens = _recurrent_config(
        r, batch, block_values, prefetch_tokens,
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


__all__ = ["tmix_wkv7_recurrent"]
