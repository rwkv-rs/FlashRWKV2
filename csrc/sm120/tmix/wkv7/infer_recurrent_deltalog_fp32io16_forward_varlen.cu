// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Albatross project
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
//
// DeltaLog algebra adapted from:
//   BlinkDL/Albatross/faster3a_2607/cuda/rwkv7_wkv_deltalog_v3a.cu
//   revision 3e41bc43ed5e8332927ddd7e0ce4816cf200a6ea
//   Apache-2.0
//
// This numerical-mode owner keeps the public FP32 state and its private logs
// in FP32 while retaining FP16 token IO.  Packed metadata, slot-local phase,
// CUDA Graph capacity tails, and fail-closed validation are local contracts.

#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include "validation.h"

#include <cstdint>

#include "recurrent_decay.cuh"

namespace {

namespace cg = cooperative_groups;

constexpr int kHeadSize = 64;
constexpr int kWarpSize = 32;
constexpr int kKinds = 5;
constexpr int kDeltaKind = 0;
constexpr int kUKind = 1;
constexpr int kBKind = 2;
constexpr int kKKind = 3;
constexpr int kVKind = 4;

// Match the ordinary FP32IO16 translation unit under --use_fast_math.
__device__ __forceinline__ float recurrent_retention(float decay_logits) {
  return exp2f(__fdividef(
      flashrwkv2::wkv7::kNegativeExpHalfLog2E,
      1.0f + exp2f(
          flashrwkv2::wkv7::kNegativeLog2E * decay_logits)));
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
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
__global__ __launch_bounds__(kHeadSize, 2) void materialize_slots_kernel(
    int num_heads,
    int state_pool_slots,
    const int* __restrict__ state_indices,
    int* __restrict__ phase_pool,
    const int* __restrict__ metadata_status,
    const int* __restrict__ deltalog_status,
    float* __restrict__ state_pool,
    float* __restrict__ deltalog_pool) {
  const int head_index = static_cast<int>(blockIdx.x);
  const int entry = static_cast<int>(blockIdx.y);
  const int thread = static_cast<int>(threadIdx.x);
  if (deltalog_status[0] != 0 ||
      (metadata_status != nullptr && entry >= metadata_status[2])) {
    return;
  }
  const int state_slot = state_indices[entry];
  const int phase = phase_pool[state_slot];
  float* const state_base =
      state_pool +
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
          kHeadSize * kHeadSize;
  float state[kHeadSize];
#pragma unroll
  for (int key = 0; key < kHeadSize; ++key) {
    state[key] = state_base[key * kHeadSize + thread];
  }
  for (int log_slot = 0; log_slot < phase; ++log_slot) {
    const float old_u = deltalog_pool[log_index(
        log_slot, kUKind, state_slot, head_index, thread,
        state_pool_slots, num_heads)];
    const float old_v = deltalog_pool[log_index(
        log_slot, kVKind, state_slot, head_index, thread,
        state_pool_slots, num_heads)];
#pragma unroll
    for (int key = 0; key < kHeadSize; ++key) {
      const float old_state = state[key];
      float updated = fmaf(
          old_state,
          deltalog_pool[log_index(
              log_slot, kDeltaKind, state_slot, head_index, key,
              state_pool_slots, num_heads)],
          old_state);
      updated = fmaf(
          old_u,
          deltalog_pool[log_index(
              log_slot, kBKind, state_slot, head_index, key,
              state_pool_slots, num_heads)],
          updated);
      state[key] = fmaf(
          deltalog_pool[log_index(
              log_slot, kKKind, state_slot, head_index, key,
              state_pool_slots, num_heads)],
          old_v, updated);
    }
  }
  if (phase != 0) {
#pragma unroll
    for (int key = 0; key < kHeadSize; ++key) {
      state_base[key * kHeadSize + thread] = state[key];
    }
  }
  __syncthreads();
#pragma unroll
  for (int log_slot = 0; log_slot < M - 1; ++log_slot) {
#pragma unroll
    for (int kind = 0; kind < kKinds; ++kind) {
      deltalog_pool[log_index(
          log_slot, kind, state_slot, head_index, thread,
          state_pool_slots, num_heads)] = 0.0f;
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

__device__ __forceinline__ void fill_invalid_output(
    int64_t block_index,
    int64_t block_count,
    int64_t output_elements,
    half* output_ptr) {
  const half invalid = __float2half(__int_as_float(0x7fffffff));
  for (int64_t index =
           block_index * static_cast<int64_t>(blockDim.x) + threadIdx.x;
       index < output_elements;
       index += block_count * static_cast<int64_t>(blockDim.x)) {
    output_ptr[index] = invalid;
  }
}

__global__ void validate_deltalog_slots_kernel(
    const int* __restrict__ query_start_loc,
    const int* __restrict__ state_indices,
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
    if (token_count != 1 || phase < 0 || phase >= merge_interval) {
      atomicCAS(deltalog_status, 0, 1);
    }
  }
}

template <int M, bool AddBias, bool Cooperative>
__global__ __launch_bounds__(kHeadSize, 2) void deltalog_slot_kernel(
    int num_heads,
    int state_pool_slots,
    int64_t output_elements,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ state_indices,
    int* __restrict__ phase_pool,
    const int* __restrict__ metadata_status,
    int* __restrict__ deltalog_status,
    float* __restrict__ state_pool,
    float* __restrict__ deltalog_pool,
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
  const int warp = thread / kWarpSize;
  const int lane = thread & (kWarpSize - 1);
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
      const int state_slot = state_indices[sequence_index];
      const int phase = phase_pool[state_slot];
      deltalog_status[1 + 2 * sequence_index] =
          token_count == 1 ? phase : M;
    }
    grid.sync();
    if (thread == 0) {
      cooperative_precheck_failed = metadata_status[0];
      if (cooperative_precheck_failed == 0) {
        for (int active_sequence = 0;
             active_sequence < metadata_status[2];
             ++active_sequence) {
          const int phase = deltalog_status[1 + 2 * active_sequence];
          if (phase < 0 || phase >= M) {
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
  if (thread == 0) {
    token_index = query_start_loc[sequence_index];
    state_slot = state_indices[sequence_index];
    slot_phase = deltalog_status[1 + 2 * sequence_index];
  }
  __syncthreads();

  const int64_t token_base =
      (static_cast<int64_t>(token_index) * num_heads + head_index) *
      kHeadSize;
  float* const state_base = state_pool +
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
          kHeadSize * kHeadSize;

  __shared__ float current_r[kHeadSize];
  __shared__ float current_a[kHeadSize];
  __shared__ float current_b[kHeadSize];
  __shared__ float current_k[kHeadSize];
  __shared__ float current_v[kHeadSize];
  __shared__ float current_delta[kHeadSize];
  __shared__ float history_delta[M - 1][kHeadSize];
  __shared__ float history_b[M - 1][kHeadSize];
  __shared__ float history_k[M - 1][kHeadSize];
  __shared__ float final_query[2][kHeadSize];
  __shared__ float coeff_b[2][M - 1];
  __shared__ float coeff_k[2][M - 1];
  __shared__ float current_br;
  __shared__ float current_kr;

  current_r[thread] = __half2float(r_ptr[token_base + thread]);
  current_a[thread] = __half2float(a_ptr[token_base + thread]);
  current_b[thread] = __half2float(b_ptr[token_base + thread]);
  current_k[thread] = __half2float(k_ptr[token_base + thread]);
  current_v[thread] = __half2float(v_ptr[token_base + thread]);
  float decay = __half2float(decay_ptr[token_base + thread]);
  if constexpr (AddBias) {
    decay += __half2float(
        decay_bias_ptr[head_index * kHeadSize + thread]);
  }
  current_delta[thread] = recurrent_retention(decay) - 1.0f;

  const int history_count = slot_phase == M - 1 ? M - 1 : slot_phase;
  for (int flat = thread;
       flat < history_count * kHeadSize;
       flat += kHeadSize) {
    const int log_slot = flat / kHeadSize;
    const int key = flat % kHeadSize;
    history_delta[log_slot][key] = deltalog_pool[log_index(
        log_slot, kDeltaKind, state_slot, head_index, key,
        state_pool_slots, num_heads)];
    history_b[log_slot][key] = deltalog_pool[log_index(
        log_slot, kBKind, state_slot, head_index, key,
        state_pool_slots, num_heads)];
    history_k[log_slot][key] = deltalog_pool[log_index(
        log_slot, kKKind, state_slot, head_index, key,
        state_pool_slots, num_heads)];
  }
  __syncthreads();

  float state[kHeadSize];
#pragma unroll
  for (int key = 0; key < kHeadSize; ++key) {
    state[key] = state_base[key * kHeadSize + thread];
  }

  if (slot_phase != M - 1) {
    float query0 = warp == 0
        ? current_a[lane]
        : current_r[lane] * (1.0f + current_delta[lane]);
    float query1 = warp == 0
        ? current_a[lane + kWarpSize]
        : current_r[lane + kWarpSize] *
            (1.0f + current_delta[lane + kWarpSize]);
    for (int log_slot = history_count - 1; log_slot >= 0; --log_slot) {
      const float b_coeff = warp_sum(
          history_b[log_slot][lane] * query0 +
          history_b[log_slot][lane + kWarpSize] * query1);
      const float k_coeff = warp_sum(
          history_k[log_slot][lane] * query0 +
          history_k[log_slot][lane + kWarpSize] * query1);
      if (lane == 0) {
        coeff_b[warp][log_slot] = b_coeff;
        coeff_k[warp][log_slot] = k_coeff;
      }
      query0 *= 1.0f + history_delta[log_slot][lane];
      query1 *= 1.0f + history_delta[log_slot][lane + kWarpSize];
    }
    final_query[warp][lane] = query0;
    final_query[warp][lane + kWarpSize] = query1;
    const float current_dot = warp == 0
        ? warp_sum(
              current_b[lane] * current_r[lane] +
              current_b[lane + kWarpSize] *
                  current_r[lane + kWarpSize])
        : warp_sum(
              current_k[lane] * current_r[lane] +
              current_k[lane + kWarpSize] *
                  current_r[lane + kWarpSize]);
    if (lane == 0) {
      if (warp == 0) {
        current_br = current_dot;
      } else {
        current_kr = current_dot;
      }
    }
    __syncthreads();

    float logical_u = 0.0f;
    float logical_output = 0.0f;
#pragma unroll
    for (int key = 0; key < kHeadSize; ++key) {
      logical_u = fmaf(state[key], final_query[0][key], logical_u);
      logical_output =
          fmaf(state[key], final_query[1][key], logical_output);
    }
    for (int log_slot = history_count - 1; log_slot >= 0; --log_slot) {
      const float old_u = deltalog_pool[log_index(
          log_slot, kUKind, state_slot, head_index, thread,
          state_pool_slots, num_heads)];
      const float old_v = deltalog_pool[log_index(
          log_slot, kVKind, state_slot, head_index, thread,
          state_pool_slots, num_heads)];
      logical_u = fmaf(old_u, coeff_b[0][log_slot], logical_u);
      logical_u = fmaf(old_v, coeff_k[0][log_slot], logical_u);
      logical_output =
          fmaf(old_u, coeff_b[1][log_slot], logical_output);
      logical_output =
          fmaf(old_v, coeff_k[1][log_slot], logical_output);
    }
    logical_output = fmaf(logical_u, current_br, logical_output);
    logical_output = fmaf(current_v[thread], current_kr, logical_output);
    output_ptr[token_base + thread] =
        __float2half_rn(logical_output * scale);

    deltalog_pool[log_index(
        slot_phase, kDeltaKind, state_slot, head_index, thread,
        state_pool_slots, num_heads)] = current_delta[thread];
    deltalog_pool[log_index(
        slot_phase, kUKind, state_slot, head_index, thread,
        state_pool_slots, num_heads)] = logical_u;
    deltalog_pool[log_index(
        slot_phase, kBKind, state_slot, head_index, thread,
        state_pool_slots, num_heads)] = current_b[thread];
    deltalog_pool[log_index(
        slot_phase, kKKind, state_slot, head_index, thread,
        state_pool_slots, num_heads)] = current_k[thread];
    deltalog_pool[log_index(
        slot_phase, kVKind, state_slot, head_index, thread,
        state_pool_slots, num_heads)] = current_v[thread];
    if (head_index == 0 && thread == 0) {
      phase_pool[state_slot] = slot_phase + 1;
    }
    return;
  }

#pragma unroll
  for (int log_slot = 0; log_slot < M - 1; ++log_slot) {
    const float old_u = deltalog_pool[log_index(
        log_slot, kUKind, state_slot, head_index, thread,
        state_pool_slots, num_heads)];
    const float old_v = deltalog_pool[log_index(
        log_slot, kVKind, state_slot, head_index, thread,
        state_pool_slots, num_heads)];
#pragma unroll
    for (int key = 0; key < kHeadSize; ++key) {
      const float old_state = state[key];
      float updated = fmaf(
          old_state, history_delta[log_slot][key], old_state);
      updated = fmaf(old_u, history_b[log_slot][key], updated);
      state[key] = fmaf(history_k[log_slot][key], old_v, updated);
    }
  }

  float current_u = 0.0f;
#pragma unroll
  for (int key = 0; key < kHeadSize; ++key) {
    current_u = fmaf(current_a[key], state[key], current_u);
  }
  float result = 0.0f;
#pragma unroll
  for (int key = 0; key < kHeadSize; ++key) {
    const float old_state = state[key];
    float updated = fmaf(old_state, current_delta[key], old_state);
    updated = fmaf(current_u, current_b[key], updated);
    updated = fmaf(current_k[key], current_v[thread], updated);
    state[key] = updated;
    result = fmaf(updated, current_r[key], result);
  }
  output_ptr[token_base + thread] = __float2half_rn(result * scale);
#pragma unroll
  for (int key = 0; key < kHeadSize; ++key) {
    state_base[key * kHeadSize + thread] = state[key];
  }
  if (head_index == 0 && thread == 0) {
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
  FLASHRWKV_CUDA_CHECK(cudaDeviceGetAttribute(
      &cooperative_launch, cudaDevAttrCooperativeLaunch, device));
  if (cooperative_launch == 0) {
    cached_device = device;
    cached_resident_blocks = 0;
    return false;
  }
  FLASHRWKV_CUDA_CHECK(cudaDeviceGetAttribute(
      &multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
  FLASHRWKV_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
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
    const torch::stable::Tensor& query_start_loc,
    const torch::stable::Tensor& state_indices,
    torch::stable::Tensor& phase_pool,
    torch::stable::Tensor& state_pool,
    torch::stable::Tensor& deltalog_pool,
    const torch::stable::Tensor& r,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& decay_bias,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    torch::stable::Tensor& output,
    const torch::stable::Tensor& metadata_status,
    torch::stable::Tensor& deltalog_status,
    float scale,
    cudaStream_t stream) {
  const dim3 grid(num_heads, num_sequences);
  const dim3 block(kHeadSize);
  const int block_count = num_sequences * num_heads;
  auto* state_ptr = state_pool.mutable_data_ptr<float>();
  auto* log_ptr = deltalog_pool.mutable_data_ptr<float>();
  const auto* r_ptr = reinterpret_cast<const half*>(r.mutable_data_ptr());
  const auto* decay_ptr =
      reinterpret_cast<const half*>(decay.mutable_data_ptr());
  const auto* bias_ptr = AddBias
      ? reinterpret_cast<const half*>(decay_bias.mutable_data_ptr())
      : nullptr;
  const auto* k_ptr = reinterpret_cast<const half*>(k.mutable_data_ptr());
  const auto* v_ptr = reinterpret_cast<const half*>(v.mutable_data_ptr());
  const auto* a_ptr = reinterpret_cast<const half*>(a.mutable_data_ptr());
  const auto* b_ptr = reinterpret_cast<const half*>(b.mutable_data_ptr());
  auto* output_ptr = reinterpret_cast<half*>(output.mutable_data_ptr());

  if (cooperative_grid_fits<M, AddBias>(block_count, device)) {
    int64_t output_elements = output.numel();
    const int* query_start_loc_ptr = query_start_loc.mutable_data_ptr<int>();
    const int* state_indices_ptr = state_indices.mutable_data_ptr<int>();
    int* phase_pool_ptr = phase_pool.mutable_data_ptr<int>();
    const int* metadata_status_ptr = metadata_status.mutable_data_ptr<int>();
    int* deltalog_status_ptr = deltalog_status.mutable_data_ptr<int>();
    void* kernel_arguments[] = {
        &num_heads,
        &state_pool_slots,
        &output_elements,
        &query_start_loc_ptr,
        &state_indices_ptr,
        &phase_pool_ptr,
        &metadata_status_ptr,
        &deltalog_status_ptr,
        &state_ptr,
        &log_ptr,
        &r_ptr,
        &decay_ptr,
        &bias_ptr,
        &k_ptr,
        &v_ptr,
        &a_ptr,
        &b_ptr,
        &output_ptr,
        &scale,
    };
    FLASHRWKV_CUDA_CHECK(cudaLaunchCooperativeKernel(
        deltalog_slot_kernel<M, AddBias, true>,
        grid, block, kernel_arguments, 0, stream));
    return;
  }

  constexpr int validation_threads = 256;
  validate_deltalog_slots_kernel<<<1, validation_threads, 0, stream>>>(
      query_start_loc.mutable_data_ptr<int>(), state_indices.mutable_data_ptr<int>(),
      phase_pool.mutable_data_ptr<int>(), metadata_status.mutable_data_ptr<int>(),
      deltalog_status.mutable_data_ptr<int>(), num_sequences, M);
  deltalog_slot_kernel<M, AddBias, false><<<grid, block, 0, stream>>>(
      num_heads, state_pool_slots, output.numel(),
      query_start_loc.mutable_data_ptr<int>(), state_indices.mutable_data_ptr<int>(),
      phase_pool.mutable_data_ptr<int>(), metadata_status.mutable_data_ptr<int>(),
      deltalog_status.mutable_data_ptr<int>(), state_ptr, log_ptr, r_ptr, decay_ptr,
      bias_ptr, k_ptr, v_ptr, a_ptr, b_ptr, output_ptr, scale);
}

template <int M>
void launch_deltalog_step(
    bool add_bias,
    int num_sequences,
    int num_heads,
    int state_pool_slots,
    int device,
    const torch::stable::Tensor& query_start_loc,
    const torch::stable::Tensor& state_indices,
    torch::stable::Tensor& phase_pool,
    torch::stable::Tensor& state_pool,
    torch::stable::Tensor& deltalog_pool,
    const torch::stable::Tensor& r,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& decay_bias,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    torch::stable::Tensor& output,
    const torch::stable::Tensor& metadata_status,
    torch::stable::Tensor& deltalog_status,
    float scale,
    cudaStream_t stream) {
  if (add_bias) {
    launch_deltalog_specialization<M, true>(
        num_sequences, num_heads, state_pool_slots, device, query_start_loc,
        state_indices, phase_pool, state_pool, deltalog_pool, r, decay,
        decay_bias, k, v, a, b, output, metadata_status, deltalog_status,
        scale, stream);
  } else {
    launch_deltalog_specialization<M, false>(
        num_sequences, num_heads, state_pool_slots, device, query_start_loc,
        state_indices, phase_pool, state_pool, deltalog_pool, r, decay,
        decay_bias, k, v, a, b, output, metadata_status, deltalog_status,
        scale, stream);
  }
}

}  // namespace

void tmix_wkv7_recurrent_deltalog_fp32io16_from_decay_logits_cuda(
    torch::stable::Tensor query_start_loc,
    torch::stable::Tensor state_indices,
    torch::stable::Tensor phase_pool,
    torch::stable::Tensor state_pool,
    torch::stable::Tensor deltalog_pool,
    torch::stable::Tensor r,
    torch::stable::Tensor decay_logits,
    torch::stable::Tensor decay_bias,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor a,
    torch::stable::Tensor b,
    torch::stable::Tensor output,
    torch::stable::Tensor metadata_status,
    torch::stable::Tensor deltalog_status,
    double scale) {
  const torch::stable::accelerator::DeviceGuard device_guard(state_pool.device().index());
  const auto stream = flashrwkv2::validation::current_cuda_stream();
  const int num_sequences = static_cast<int>(state_indices.numel());
  const int num_heads = static_cast<int>(state_pool.size(1));
  const int state_pool_slots = static_cast<int>(state_pool.size(0));
  const int device = state_pool.device().index();
  const int merge_interval = static_cast<int>(deltalog_pool.size(0) + 1);
#define DISPATCH_M(Value) \
  case Value: \
    launch_deltalog_step<Value>( \
        decay_bias.defined(), num_sequences, num_heads, state_pool_slots, \
        device, query_start_loc, state_indices, phase_pool, state_pool, \
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
      STD_TORCH_CHECK(false, "unsupported DeltaLog merge interval");
  }
#undef DISPATCH_M
  FLASHRWKV_CUDA_CHECK(cudaGetLastError());
}

void tmix_wkv7_recurrent_deltalog_fp32io16_materialize_slots_cuda(
    torch::stable::Tensor state_indices,
    torch::stable::Tensor phase_pool,
    torch::stable::Tensor state_pool,
    torch::stable::Tensor deltalog_pool,
    torch::stable::Tensor deltalog_status,
    torch::stable::Tensor metadata_status) {
  const torch::stable::accelerator::DeviceGuard device_guard(state_pool.device().index());
  const auto stream = flashrwkv2::validation::current_cuda_stream();
  const int num_entries = static_cast<int>(state_indices.numel());
  const int num_heads = static_cast<int>(state_pool.size(1));
  const int state_pool_slots = static_cast<int>(state_pool.size(0));
  const int merge_interval = static_cast<int>(deltalog_pool.size(0) + 1);
  constexpr int validation_threads = 256;
  validate_materialize_slots_kernel<<<1, validation_threads, 0, stream>>>(
      state_indices.mutable_data_ptr<int>(), phase_pool.mutable_data_ptr<int>(),
      metadata_status.defined() ? metadata_status.mutable_data_ptr<int>() : nullptr,
      deltalog_status.mutable_data_ptr<int>(), num_entries, state_pool_slots,
      merge_interval);
  const dim3 grid(num_heads, num_entries);
  const dim3 block(kHeadSize);
#define DISPATCH_MATERIALIZE_M(Value) \
  case Value: \
    materialize_slots_kernel<Value><<<grid, block, 0, stream>>>( \
        num_heads, state_pool_slots, state_indices.mutable_data_ptr<int>(), \
        phase_pool.mutable_data_ptr<int>(), \
        metadata_status.defined() ? metadata_status.mutable_data_ptr<int>() : nullptr, \
        deltalog_status.mutable_data_ptr<int>(), state_pool.mutable_data_ptr<float>(), \
        deltalog_pool.mutable_data_ptr<float>()); \
    break
  switch (merge_interval) {
    DISPATCH_MATERIALIZE_M(2);
    DISPATCH_MATERIALIZE_M(3);
    DISPATCH_MATERIALIZE_M(4);
    DISPATCH_MATERIALIZE_M(6);
    DISPATCH_MATERIALIZE_M(8);
    default:
      STD_TORCH_CHECK(false, "unsupported DeltaLog merge interval");
  }
#undef DISPATCH_MATERIALIZE_M
  reset_materialized_phases_kernel<<<1, validation_threads, 0, stream>>>(
      state_indices.mutable_data_ptr<int>(), phase_pool.mutable_data_ptr<int>(),
      metadata_status.defined() ? metadata_status.mutable_data_ptr<int>() : nullptr,
      deltalog_status.mutable_data_ptr<int>(), num_entries);
  FLASHRWKV_CUDA_CHECK(cudaGetLastError());
}
