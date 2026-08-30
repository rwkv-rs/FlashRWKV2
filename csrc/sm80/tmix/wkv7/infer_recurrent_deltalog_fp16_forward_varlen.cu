// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Albatross project
//
// Canonical algorithm source:
//   BlinkDL/Albatross/faster3a_2607/cuda/rwkv7_wkv_deltalog_v3a.cu
//   revision 3e41bc43ed5e8332927ddd7e0ce4816cf200a6ea
//   Apache-2.0
//
// Local adaptation:
//   - packed token storage and live metadata status;
//   - state_indices-owned [K,V] base state, elapsed, phase, and log bundle;
//   - per-slot append/merge phase instead of one rectangular batch phase;
//   - optional decay bias, output scaling, CUDA Graph capacity tails, and
//     fail-closed prevalidation before any state mutation.

#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

#include "recurrent_decay.cuh"

namespace {

namespace cg = cooperative_groups;

constexpr int kHeadSize = 64;
constexpr int kHalf2HeadSize = kHeadSize / 2;
constexpr int kKinds = 5;
constexpr int kDeltaKind = 0;
constexpr int kUKind = 1;
constexpr int kBKind = 2;
constexpr int kKKind = 3;
constexpr int kVKind = 4;

using flashrwkv2::wkv7::recurrent_fp16_delta;

template <typename io_t>
__device__ __forceinline__ io_t invalid_value() {
  return static_cast<io_t>(__int_as_float(0x7fffffff));
}

template <typename io_t>
__device__ __forceinline__ void fill_invalid_output(
    int64_t block_index,
    int64_t block_count,
    int64_t output_elements,
    io_t* output_ptr) {
  for (int64_t index =
           block_index * static_cast<int64_t>(blockDim.x) + threadIdx.x;
       index < output_elements;
       index += block_count * static_cast<int64_t>(blockDim.x)) {
    output_ptr[index] = invalid_value<io_t>();
  }
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ half warp_dot(half2 left, half2 right) {
  const half2 product = __hmul2(left, right);
  const float local =
      __half2float(product.x) + __half2float(product.y);
  return __float2half_rn(warp_sum(local));
}

__device__ __forceinline__ half2 load_state_kv(
    const half* state_base,
    int value_index,
    int key_pair) {
  return __halves2half2(
      __ldg(state_base + (2 * key_pair) * kHeadSize + value_index),
      __ldg(state_base + (2 * key_pair + 1) * kHeadSize + value_index));
}

__device__ __forceinline__ void store_state_kv(
    half* state_base,
    int value_index,
    int key_pair,
    half2 value) {
  state_base[(2 * key_pair) * kHeadSize + value_index] = value.x;
  state_base[(2 * key_pair + 1) * kHeadSize + value_index] = value.y;
}

__device__ __forceinline__ int64_t log_index(
    int log_slot,
    int kind,
    int state_slot,
    int head_index,
    int channel,
    int state_pool_slots,
    int num_heads) {
  return (((static_cast<int64_t>(log_slot) * kKinds + kind) *
                state_pool_slots +
            state_slot) *
               num_heads +
           head_index) *
          kHeadSize +
      channel;
}

__global__ void validate_materialize_slots_kernel(
    const int* __restrict__ state_indices,
    const int* __restrict__ phase_pool,
    const int* __restrict__ metadata_status,
    int* __restrict__ deltalog_status,
    int num_entries,
    int state_pool_slots,
    int merge_interval) {
  if (threadIdx.x == 0) {
    deltalog_status[0] = metadata_status == nullptr ? 0 : metadata_status[0];
    if (metadata_status != nullptr &&
        (metadata_status[2] < 0 || metadata_status[2] > num_entries)) {
      deltalog_status[0] = 1;
    }
  }
  __syncthreads();
  if (deltalog_status[0] != 0) {
    return;
  }
  const int active_entries =
      metadata_status == nullptr ? num_entries : metadata_status[2];
  for (int index = static_cast<int>(threadIdx.x);
       index < active_entries;
       index += static_cast<int>(blockDim.x)) {
    const int state_slot = state_indices[index];
    if (state_slot < 0 || state_slot >= state_pool_slots) {
      atomicCAS(deltalog_status, 0, 1);
      continue;
    }
    const int phase = phase_pool[state_slot];
    if (phase < 0 || phase >= merge_interval) {
      atomicCAS(deltalog_status, 0, 1);
    }
    for (int other = index + 1; other < active_entries; ++other) {
      if (state_indices[other] == state_slot) {
        atomicCAS(deltalog_status, 0, 1);
      }
    }
  }
}

template <int M>
__global__ __launch_bounds__(kHeadSize, 1) void materialize_slots_kernel(
    int num_heads,
    int state_pool_slots,
    const int* __restrict__ state_indices,
    int* __restrict__ phase_pool,
    const int* __restrict__ metadata_status,
    const int* __restrict__ deltalog_status,
    half* __restrict__ state_pool,
    half* __restrict__ deltalog_pool) {
  const int head_index = static_cast<int>(blockIdx.x);
  const int entry = static_cast<int>(blockIdx.y);
  const int thread = static_cast<int>(threadIdx.x);
  if (deltalog_status[0] != 0 ||
      (metadata_status != nullptr && entry >= metadata_status[2])) {
    return;
  }
  const int state_slot = state_indices[entry];
  const int phase = phase_pool[state_slot];
  half* const state_base =
      state_pool +
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
          kHeadSize * kHeadSize;
  half2 state[kHalf2HeadSize];
#pragma unroll
  for (int key_pair = 0; key_pair < kHalf2HeadSize; ++key_pair) {
    state[key_pair] = load_state_kv(state_base, thread, key_pair);
  }
  for (int log_slot = 0; log_slot < phase; ++log_slot) {
    const half old_u = deltalog_pool[log_index(
        log_slot, kUKind, state_slot, head_index, thread,
        state_pool_slots, num_heads)];
    const half old_v = deltalog_pool[log_index(
        log_slot, kVKind, state_slot, head_index, thread,
        state_pool_slots, num_heads)];
    // Match the public FP16-state contract: accumulate one recurrence in
    // FP32, then quantize once when the pending log becomes physical state.
#pragma unroll
    for (int key_pair = 0; key_pair < kHalf2HeadSize; ++key_pair) {
      const int key = 2 * key_pair;
      const half2 old_state = state[key_pair];
      const half2 delta = __ldg(reinterpret_cast<const half2*>(
          deltalog_pool +
          log_index(log_slot, kDeltaKind, state_slot, head_index, key,
                    state_pool_slots, num_heads)));
      const half2 old_b = __ldg(reinterpret_cast<const half2*>(
          deltalog_pool +
          log_index(log_slot, kBKind, state_slot, head_index, key,
                    state_pool_slots, num_heads)));
      const half2 old_k = __ldg(reinterpret_cast<const half2*>(
          deltalog_pool +
          log_index(log_slot, kKKind, state_slot, head_index, key,
                    state_pool_slots, num_heads)));
      const float old_state0 = __half2float(old_state.x);
      const float old_state1 = __half2float(old_state.y);
      float updated0 = fmaf(
          old_state0, __half2float(delta.x), old_state0);
      float updated1 = fmaf(
          old_state1, __half2float(delta.y), old_state1);
      updated0 = fmaf(
          __half2float(old_u), __half2float(old_b.x), updated0);
      updated1 = fmaf(
          __half2float(old_u), __half2float(old_b.y), updated1);
      updated0 = fmaf(
          __half2float(old_k.x), __half2float(old_v), updated0);
      updated1 = fmaf(
          __half2float(old_k.y), __half2float(old_v), updated1);
      state[key_pair] = __floats2half2_rn(updated0, updated1);
    }
  }
  if (phase != 0) {
#pragma unroll
    for (int key_pair = 0; key_pair < kHalf2HeadSize; ++key_pair) {
      store_state_kv(state_base, thread, key_pair, state[key_pair]);
    }
  }
  __syncthreads();
  const half zero = __float2half_rn(0.0f);
#pragma unroll
  for (int log_slot = 0; log_slot < M - 1; ++log_slot) {
#pragma unroll
    for (int kind = 0; kind < kKinds; ++kind) {
      deltalog_pool[log_index(
          log_slot, kind, state_slot, head_index, thread,
          state_pool_slots, num_heads)] = zero;
    }
  }
}

// phase_pool is slot-wide.  Reset it only after every per-head materialize
// block in the preceding launch has consumed the same phase snapshot.
__global__ void reset_materialized_phases_kernel(
    const int* __restrict__ state_indices,
    int* __restrict__ phase_pool,
    const int* __restrict__ metadata_status,
    const int* __restrict__ deltalog_status,
    int num_entries) {
  if (deltalog_status[0] != 0) {
    return;
  }
  const int active_entries =
      metadata_status == nullptr ? num_entries : metadata_status[2];
  for (int entry = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
       entry < active_entries;
       entry += static_cast<int>(gridDim.x) * blockDim.x) {
    phase_pool[state_indices[entry]] = 0;
  }
}

// The regular path snapshots each active slot before the recurrent launch.
// Small grids perform the same check inside a cooperative launch.  Both paths
// establish the fail-closed result before any base/log/elapsed/phase write and
// let the recurrent kernel advance a slot from the immutable snapshot.
__global__ void validate_deltalog_slots_kernel(
    const int* __restrict__ query_start_loc,
    const int* __restrict__ state_indices,
    const int* __restrict__ elapsed_pool,
    const int* __restrict__ phase_pool,
    const int* __restrict__ metadata_status,
    int* __restrict__ deltalog_status,
    int num_sequences,
    int merge_interval) {
  if (threadIdx.x == 0) {
    deltalog_status[0] = metadata_status[0];
  }
  __syncthreads();
  if (metadata_status[0] != 0) {
    return;
  }
  for (int sequence_index = static_cast<int>(threadIdx.x);
       sequence_index < num_sequences;
       sequence_index += static_cast<int>(blockDim.x)) {
    if (sequence_index >= metadata_status[2]) {
      continue;
    }
    const int token_count =
        query_start_loc[sequence_index + 1] -
        query_start_loc[sequence_index];
    const int state_slot = state_indices[sequence_index];
    const int phase = phase_pool[state_slot];
    deltalog_status[1 + 2 * sequence_index] = phase;
    deltalog_status[2 + 2 * sequence_index] = elapsed_pool[state_slot];
    if (token_count != 1 || phase < 0 || phase >= merge_interval) {
      atomicCAS(deltalog_status, 0, 1);
    }
  }
}

template <int M, bool AddBias, bool Cooperative>
__global__ __launch_bounds__(kHeadSize, 1) void deltalog_slot_kernel(
    int num_heads,
    int state_pool_slots,
    int64_t output_elements,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ state_indices,
    int* __restrict__ elapsed_pool,
    int* __restrict__ phase_pool,
    const int* __restrict__ metadata_status,
    int* __restrict__ deltalog_status,
    half* __restrict__ state_pool,
    half* __restrict__ deltalog_pool,
    const half* __restrict__ r_ptr,
    const half* __restrict__ decay_ptr,
    const half* __restrict__ decay_bias_ptr,
    const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr,
    const half* __restrict__ a_ptr,
    const half* __restrict__ b_ptr,
    half* __restrict__ output_ptr,
    float scale) {
  static_assert(M == 2 || M == 3 || M == 4 || M == 6 || M == 8);
  const int head_index = static_cast<int>(blockIdx.x);
  const int sequence_index = static_cast<int>(blockIdx.y);
  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int lane = thread & 31;
  const int64_t block_index =
      static_cast<int64_t>(sequence_index) * num_heads + head_index;
  const int64_t block_count =
      static_cast<int64_t>(gridDim.x) * gridDim.y;
  __shared__ int cooperative_precheck_failed;
  if constexpr (Cooperative) {
    const cg::grid_group grid = cg::this_grid();
    if (metadata_status[0] == 0 && head_index == 0 && thread == 0 &&
        sequence_index < metadata_status[2]) {
      const int token_count =
          query_start_loc[sequence_index + 1] -
          query_start_loc[sequence_index];
      const int cooperative_state_slot = state_indices[sequence_index];
      const int cooperative_phase = phase_pool[cooperative_state_slot];
      deltalog_status[1 + 2 * sequence_index] =
          token_count == 1 ? cooperative_phase : M;
      deltalog_status[2 + 2 * sequence_index] =
          elapsed_pool[cooperative_state_slot];
    }
    grid.sync();
    if (thread == 0) {
      cooperative_precheck_failed = metadata_status[0];
      if (cooperative_precheck_failed == 0) {
        for (int active_sequence = 0;
             active_sequence < metadata_status[2];
             ++active_sequence) {
          const int active_phase =
              deltalog_status[1 + 2 * active_sequence];
          if (active_phase < 0 || active_phase >= M) {
            cooperative_precheck_failed = 1;
            break;
          }
        }
      }
    }
    __syncthreads();
  }
  bool precheck_failed = false;
  if constexpr (Cooperative) {
    precheck_failed = cooperative_precheck_failed != 0;
  } else {
    precheck_failed = deltalog_status[0] != 0;
  }
  if (precheck_failed) {
    fill_invalid_output(
        block_index, block_count, output_elements, output_ptr);
    return;
  }
  if (sequence_index >= metadata_status[2]) {
    return;
  }

  __shared__ int token_index;
  __shared__ int state_slot;
  __shared__ int slot_phase;
  __shared__ int elapsed_base;
  if (thread == 0) {
    token_index = query_start_loc[sequence_index];
    state_slot = state_indices[sequence_index];
    slot_phase = deltalog_status[1 + 2 * sequence_index];
    elapsed_base = deltalog_status[2 + 2 * sequence_index];
  }
  __syncthreads();

  const int64_t token_base =
      (static_cast<int64_t>(token_index) * num_heads + head_index) *
      kHeadSize;
  half* const state_base =
      state_pool +
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
          kHeadSize * kHeadSize;

  __shared__ __align__(128) half2 current_r[kHalf2HeadSize];
  __shared__ __align__(128) half2 current_a[kHalf2HeadSize];
  __shared__ __align__(128) half2 current_b[kHalf2HeadSize];
  __shared__ __align__(128) half2 current_k[kHalf2HeadSize];
  __shared__ __align__(128) half2 current_delta[kHalf2HeadSize];
  __shared__ __align__(128) half2 history_delta[M - 1][kHalf2HeadSize];
  __shared__ __align__(128) half2 history_b[M - 1][kHalf2HeadSize];
  __shared__ __align__(128) half2 history_k[M - 1][kHalf2HeadSize];
  __shared__ __align__(128) half2 final_query[2][kHalf2HeadSize];
  __shared__ half coeff_b[2][M - 1];
  __shared__ half coeff_k[2][M - 1];
  __shared__ half current_br;
  __shared__ half current_kr;

  if (thread < kHalf2HeadSize) {
    const int64_t pair_base = token_base / 2 + thread;
    current_r[thread] =
        __ldg(reinterpret_cast<const half2*>(r_ptr) + pair_base);
    current_a[thread] =
        __ldg(reinterpret_cast<const half2*>(a_ptr) + pair_base);
    current_b[thread] =
        __ldg(reinterpret_cast<const half2*>(b_ptr) + pair_base);
    current_k[thread] =
        __ldg(reinterpret_cast<const half2*>(k_ptr) + pair_base);
    const half2 raw_decay =
        __ldg(reinterpret_cast<const half2*>(decay_ptr) + pair_base);
    float decay0 = __half2float(raw_decay.x);
    float decay1 = __half2float(raw_decay.y);
    if constexpr (AddBias) {
      const half2 bias = __ldg(
          reinterpret_cast<const half2*>(
              decay_bias_ptr + head_index * kHeadSize) + thread);
      decay0 += __half2float(bias.x);
      decay1 += __half2float(bias.y);
    }
    const int phase0 =
        elapsed_base + head_index * kHeadSize + 2 * thread;
    current_delta[thread] = __halves2half2(
        __float2half_rn(recurrent_fp16_delta(decay0, phase0)),
        __float2half_rn(recurrent_fp16_delta(decay1, phase0 + 1)));
  }
  const int history_count = slot_phase == M - 1 ? M - 1 : slot_phase;
  for (int flat = thread;
       flat < history_count * kHalf2HeadSize;
       flat += kHeadSize) {
    const int log_slot = flat / kHalf2HeadSize;
    const int key_pair = flat % kHalf2HeadSize;
    const int key = 2 * key_pair;
    history_delta[log_slot][key_pair] = __ldg(
        reinterpret_cast<const half2*>(
            deltalog_pool +
            log_index(log_slot, kDeltaKind, state_slot, head_index, key,
                      state_pool_slots, num_heads)));
    history_b[log_slot][key_pair] = __ldg(
        reinterpret_cast<const half2*>(
            deltalog_pool +
            log_index(log_slot, kBKind, state_slot, head_index, key,
                      state_pool_slots, num_heads)));
    history_k[log_slot][key_pair] = __ldg(
        reinterpret_cast<const half2*>(
            deltalog_pool +
            log_index(log_slot, kKKind, state_slot, head_index, key,
                      state_pool_slots, num_heads)));
  }
  __syncthreads();

  half2 state[kHalf2HeadSize];
#pragma unroll
  for (int key_pair = 0; key_pair < kHalf2HeadSize; ++key_pair) {
    state[key_pair] = load_state_kv(state_base, thread, key_pair);
  }

  if (slot_phase != M - 1) {
    half2 query = warp == 0
        ? current_a[lane]
        : __hfma2(current_r[lane], current_delta[lane], current_r[lane]);
    for (int log_slot = history_count - 1; log_slot >= 0; --log_slot) {
      const half b_coeff = warp_dot(history_b[log_slot][lane], query);
      const half k_coeff = warp_dot(history_k[log_slot][lane], query);
      if (lane == 0) {
        coeff_b[warp][log_slot] = b_coeff;
        coeff_k[warp][log_slot] = k_coeff;
      }
      query = __hfma2(
          query, history_delta[log_slot][lane], query);
    }
    final_query[warp][lane] = query;
    const half current_dot = warp == 0
        ? warp_dot(current_b[lane], current_r[lane])
        : warp_dot(current_k[lane], current_r[lane]);
    if (lane == 0) {
      if (warp == 0) {
        current_br = current_dot;
      } else {
        current_kr = current_dot;
      }
    }
    __syncthreads();

    half2 base_u = __float2half2_rn(0.0f);
    half2 base_output = __float2half2_rn(0.0f);
#pragma unroll
    for (int key_pair = 0; key_pair < kHalf2HeadSize; ++key_pair) {
      base_u = __hfma2(state[key_pair], final_query[0][key_pair], base_u);
      base_output = __hfma2(
          state[key_pair], final_query[1][key_pair], base_output);
    }
    half logical_u = __hadd(base_u.x, base_u.y);
    half logical_output = __hadd(base_output.x, base_output.y);
    for (int log_slot = history_count - 1; log_slot >= 0; --log_slot) {
      const half old_u = deltalog_pool[log_index(
          log_slot, kUKind, state_slot, head_index, thread,
          state_pool_slots, num_heads)];
      const half old_v = deltalog_pool[log_index(
          log_slot, kVKind, state_slot, head_index, thread,
          state_pool_slots, num_heads)];
      logical_u = __hfma(old_u, coeff_b[0][log_slot], logical_u);
      logical_u = __hfma(old_v, coeff_k[0][log_slot], logical_u);
      logical_output =
          __hfma(old_u, coeff_b[1][log_slot], logical_output);
      logical_output =
          __hfma(old_v, coeff_k[1][log_slot], logical_output);
    }
    const half current_v = v_ptr[token_base + thread];
    logical_output = __hfma(logical_u, current_br, logical_output);
    logical_output = __hfma(current_v, current_kr, logical_output);
    output_ptr[token_base + thread] = __float2half_rn(
        __half2float(logical_output) * scale);

    deltalog_pool[log_index(
        slot_phase, kDeltaKind, state_slot, head_index, thread,
        state_pool_slots, num_heads)] =
        reinterpret_cast<half*>(current_delta)[thread];
    deltalog_pool[log_index(
        slot_phase, kUKind, state_slot, head_index, thread,
        state_pool_slots, num_heads)] = logical_u;
    deltalog_pool[log_index(
        slot_phase, kBKind, state_slot, head_index, thread,
        state_pool_slots, num_heads)] =
        reinterpret_cast<half*>(current_b)[thread];
    deltalog_pool[log_index(
        slot_phase, kKKind, state_slot, head_index, thread,
        state_pool_slots, num_heads)] =
        reinterpret_cast<half*>(current_k)[thread];
    deltalog_pool[log_index(
        slot_phase, kVKind, state_slot, head_index, thread,
        state_pool_slots, num_heads)] = current_v;
    if (head_index == 0 && thread == 0) {
      elapsed_pool[state_slot] = elapsed_base + 1;
      phase_pool[state_slot] = slot_phase + 1;
    }
    return;
  }

  // Merge all existing log entries into the physical state, then execute the
  // current token as the final recurrence of the cycle.
#pragma unroll
  for (int log_slot = 0; log_slot < M - 1; ++log_slot) {
    const half old_u = deltalog_pool[log_index(
        log_slot, kUKind, state_slot, head_index, thread,
        state_pool_slots, num_heads)];
    const half old_v = deltalog_pool[log_index(
        log_slot, kVKind, state_slot, head_index, thread,
        state_pool_slots, num_heads)];
    const half2 old_u_pair = __halves2half2(old_u, old_u);
    const half2 old_v_pair = __halves2half2(old_v, old_v);
#pragma unroll
    for (int key_pair = 0; key_pair < kHalf2HeadSize; ++key_pair) {
      const half2 old_state = state[key_pair];
      state[key_pair] = __hfma2(
          old_state, history_delta[log_slot][key_pair],
          __hfma2(
              history_k[log_slot][key_pair], old_v_pair,
              __hfma2(old_u_pair, history_b[log_slot][key_pair], old_state)));
    }
  }

  half2 current_u_pair = __float2half2_rn(0.0f);
#pragma unroll
  for (int key_pair = 0; key_pair < kHalf2HeadSize; ++key_pair) {
    current_u_pair = __hfma2(
        current_a[key_pair], state[key_pair], current_u_pair);
  }
  const half current_u = __hadd(current_u_pair.x, current_u_pair.y);
  const half2 current_u_broadcast =
      __halves2half2(current_u, current_u);
  const half current_v = v_ptr[token_base + thread];
  const half2 current_v_broadcast =
      __halves2half2(current_v, current_v);
  half2 result = __float2half2_rn(0.0f);
#pragma unroll
  for (int key_pair = 0; key_pair < kHalf2HeadSize; ++key_pair) {
    const half2 old_state = state[key_pair];
    state[key_pair] = __hfma2(
        old_state, current_delta[key_pair],
        __hfma2(
            current_k[key_pair], current_v_broadcast,
            __hfma2(current_u_broadcast, current_b[key_pair], old_state)));
    result = __hfma2(state[key_pair], current_r[key_pair], result);
  }
  output_ptr[token_base + thread] = __float2half_rn(
      (__half2float(result.x) + __half2float(result.y)) * scale);
#pragma unroll
  for (int key_pair = 0; key_pair < kHalf2HeadSize; ++key_pair) {
    store_state_kv(state_base, thread, key_pair, state[key_pair]);
  }
  if (head_index == 0 && thread == 0) {
    elapsed_pool[state_slot] = elapsed_base + 1;
    phase_pool[state_slot] = 0;
  }
}

template <int M, bool AddBias>
bool cooperative_grid_fits(int block_count, int device) {
  thread_local int cached_device = -1;
  thread_local int cached_resident_blocks = 0;
  if (cached_device == device) {
    return block_count <= cached_resident_blocks;
  }
  int cooperative_launch = 0;
  int multiprocessor_count = 0;
  int blocks_per_multiprocessor = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &cooperative_launch, cudaDevAttrCooperativeLaunch, device));
  if (cooperative_launch == 0) {
    cached_device = device;
    cached_resident_blocks = 0;
    return false;
  }
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_multiprocessor,
      deltalog_slot_kernel<M, AddBias, true>, kHeadSize, 0));
  cached_device = device;
  cached_resident_blocks =
      multiprocessor_count * blocks_per_multiprocessor;
  return block_count <= cached_resident_blocks;
}

template <int M, bool AddBias>
void launch_deltalog_specialization(
    int num_sequences,
    int num_heads,
    int state_pool_slots,
    int device,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& state_indices,
    torch::Tensor& elapsed_pool,
    torch::Tensor& phase_pool,
    torch::Tensor& state_pool,
    torch::Tensor& deltalog_pool,
    const torch::Tensor& r,
    const torch::Tensor& decay,
    const torch::Tensor& decay_bias,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& output,
    const torch::Tensor& metadata_status,
    torch::Tensor& deltalog_status,
    float scale,
    cudaStream_t stream) {
  const dim3 grid(num_heads, num_sequences);
  const dim3 block(kHeadSize);
  const int block_count = num_sequences * num_heads;
  if (cooperative_grid_fits<M, AddBias>(block_count, device)) {
    int64_t output_elements = output.numel();
    const int* query_start_loc_ptr = query_start_loc.data_ptr<int>();
    const int* state_indices_ptr = state_indices.data_ptr<int>();
    int* elapsed_pool_ptr = elapsed_pool.data_ptr<int>();
    int* phase_pool_ptr = phase_pool.data_ptr<int>();
    const int* metadata_status_ptr = metadata_status.data_ptr<int>();
    int* deltalog_status_ptr = deltalog_status.data_ptr<int>();
    half* state_pool_ptr = reinterpret_cast<half*>(state_pool.data_ptr());
    half* deltalog_pool_ptr =
        reinterpret_cast<half*>(deltalog_pool.data_ptr());
    const half* r_ptr = reinterpret_cast<const half*>(r.data_ptr());
    const half* decay_ptr = reinterpret_cast<const half*>(decay.data_ptr());
    const half* decay_bias_ptr = AddBias
        ? reinterpret_cast<const half*>(decay_bias.data_ptr())
        : nullptr;
    const half* k_ptr = reinterpret_cast<const half*>(k.data_ptr());
    const half* v_ptr = reinterpret_cast<const half*>(v.data_ptr());
    const half* a_ptr = reinterpret_cast<const half*>(a.data_ptr());
    const half* b_ptr = reinterpret_cast<const half*>(b.data_ptr());
    half* output_ptr = reinterpret_cast<half*>(output.data_ptr());
    void* kernel_arguments[] = {
        &num_heads,
        &state_pool_slots,
        &output_elements,
        &query_start_loc_ptr,
        &state_indices_ptr,
        &elapsed_pool_ptr,
        &phase_pool_ptr,
        &metadata_status_ptr,
        &deltalog_status_ptr,
        &state_pool_ptr,
        &deltalog_pool_ptr,
        &r_ptr,
        &decay_ptr,
        &decay_bias_ptr,
        &k_ptr,
        &v_ptr,
        &a_ptr,
        &b_ptr,
        &output_ptr,
        &scale,
    };
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        deltalog_slot_kernel<M, AddBias, true>,
        grid, block, kernel_arguments, 0, stream));
    return;
  }

  constexpr int validation_threads = 256;
  validate_deltalog_slots_kernel<<<1, validation_threads, 0, stream>>>(
      query_start_loc.data_ptr<int>(), state_indices.data_ptr<int>(),
      elapsed_pool.data_ptr<int>(), phase_pool.data_ptr<int>(),
      metadata_status.data_ptr<int>(), deltalog_status.data_ptr<int>(),
      num_sequences, M);
  deltalog_slot_kernel<M, AddBias, false>
      <<<grid, block, 0, stream>>>(
          num_heads, state_pool_slots, output.numel(),
          query_start_loc.data_ptr<int>(), state_indices.data_ptr<int>(),
          elapsed_pool.data_ptr<int>(), phase_pool.data_ptr<int>(),
          metadata_status.data_ptr<int>(), deltalog_status.data_ptr<int>(),
          reinterpret_cast<half*>(state_pool.data_ptr()),
          reinterpret_cast<half*>(deltalog_pool.data_ptr()),
          reinterpret_cast<const half*>(r.data_ptr()),
          reinterpret_cast<const half*>(decay.data_ptr()),
          AddBias ? reinterpret_cast<const half*>(decay_bias.data_ptr())
                  : nullptr,
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a.data_ptr()),
          reinterpret_cast<const half*>(b.data_ptr()),
          reinterpret_cast<half*>(output.data_ptr()), scale);
}

template <int M>
void launch_deltalog_step(
    bool add_bias,
    int num_sequences,
    int num_heads,
    int state_pool_slots,
    int device,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& state_indices,
    torch::Tensor& elapsed_pool,
    torch::Tensor& phase_pool,
    torch::Tensor& state_pool,
    torch::Tensor& deltalog_pool,
    const torch::Tensor& r,
    const torch::Tensor& decay,
    const torch::Tensor& decay_bias,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& output,
    const torch::Tensor& metadata_status,
    torch::Tensor& deltalog_status,
    float scale,
    cudaStream_t stream) {
  if (add_bias) {
    launch_deltalog_specialization<M, true>(
        num_sequences, num_heads, state_pool_slots, device, query_start_loc,
        state_indices, elapsed_pool, phase_pool, state_pool, deltalog_pool, r,
        decay, decay_bias, k, v, a, b, output, metadata_status,
        deltalog_status, scale, stream);
  } else {
    launch_deltalog_specialization<M, false>(
        num_sequences, num_heads, state_pool_slots, device, query_start_loc,
        state_indices, elapsed_pool, phase_pool, state_pool, deltalog_pool, r,
        decay, decay_bias, k, v, a, b, output, metadata_status,
        deltalog_status, scale, stream);
  }
}

}  // namespace

void tmix_wkv7_recurrent_deltalog_fp16_from_decay_logits_cuda(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor elapsed_pool,
    torch::Tensor phase_pool,
    torch::Tensor state_pool,
    torch::Tensor deltalog_pool,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor decay_bias,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor metadata_status,
    torch::Tensor deltalog_status,
    double scale) {
  const c10::cuda::CUDAGuard device_guard(state_pool.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int num_sequences = static_cast<int>(state_indices.numel());
  const int num_heads = static_cast<int>(state_pool.size(1));
  const int state_pool_slots = static_cast<int>(state_pool.size(0));
  const int device = state_pool.get_device();
  const int merge_interval = static_cast<int>(deltalog_pool.size(0) + 1);
#define DISPATCH_M(Value) \
  case Value: \
    launch_deltalog_step<Value>( \
        decay_bias.defined(), num_sequences, num_heads, state_pool_slots, \
        device, \
        query_start_loc, state_indices, elapsed_pool, phase_pool, state_pool, \
        deltalog_pool, r, decay_logits, decay_bias, k, v, a, b, output, \
        metadata_status, deltalog_status, static_cast<float>(scale), stream); \
    break
  switch (merge_interval) {
    DISPATCH_M(2);
    DISPATCH_M(3);
    DISPATCH_M(4);
    DISPATCH_M(6);
    DISPATCH_M(8);
    default:
      TORCH_CHECK(false, "unsupported DeltaLog merge interval");
  }
#undef DISPATCH_M
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tmix_wkv7_recurrent_deltalog_fp16_materialize_slots_cuda(
    torch::Tensor state_indices,
    torch::Tensor phase_pool,
    torch::Tensor state_pool,
    torch::Tensor deltalog_pool,
    torch::Tensor deltalog_status,
    torch::Tensor metadata_status) {
  const c10::cuda::CUDAGuard device_guard(state_pool.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int num_entries = static_cast<int>(state_indices.numel());
  const int num_heads = static_cast<int>(state_pool.size(1));
  const int state_pool_slots = static_cast<int>(state_pool.size(0));
  const int merge_interval = static_cast<int>(deltalog_pool.size(0) + 1);
  constexpr int validation_threads = 256;
  validate_materialize_slots_kernel<<<1, validation_threads, 0, stream>>>(
      state_indices.data_ptr<int>(), phase_pool.data_ptr<int>(),
      metadata_status.defined() ? metadata_status.data_ptr<int>() : nullptr,
      deltalog_status.data_ptr<int>(), num_entries, state_pool_slots,
      merge_interval);
  const dim3 grid(num_heads, num_entries);
  const dim3 block(kHeadSize);
#define DISPATCH_MATERIALIZE_M(Value) \
  case Value: \
    materialize_slots_kernel<Value><<<grid, block, 0, stream>>>( \
        num_heads, state_pool_slots, state_indices.data_ptr<int>(), \
        phase_pool.data_ptr<int>(), \
        metadata_status.defined() ? metadata_status.data_ptr<int>() : nullptr, \
        deltalog_status.data_ptr<int>(), \
        reinterpret_cast<half*>(state_pool.data_ptr()), \
        reinterpret_cast<half*>(deltalog_pool.data_ptr())); \
    break
  switch (merge_interval) {
    DISPATCH_MATERIALIZE_M(2);
    DISPATCH_MATERIALIZE_M(3);
    DISPATCH_MATERIALIZE_M(4);
    DISPATCH_MATERIALIZE_M(6);
    DISPATCH_MATERIALIZE_M(8);
    default:
      TORCH_CHECK(false, "unsupported DeltaLog merge interval");
  }
#undef DISPATCH_MATERIALIZE_M
  reset_materialized_phases_kernel<<<1, validation_threads, 0, stream>>>(
      state_indices.data_ptr<int>(), phase_pool.data_ptr<int>(),
      metadata_status.defined() ? metadata_status.data_ptr<int>() : nullptr,
      deltalog_status.data_ptr<int>(), num_entries);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
