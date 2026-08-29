// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Albatross project
//
// Canonical body source:
//   BlinkDL/Albatross/faster3a_2607/cuda/rwkv7_wkv_fp32_v2.cu
//   revision 3e41bc43ed5e8332927ddd7e0ce4816cf200a6ea
//   Apache-2.0
//
// Local adaptation:
//   - packed [total_tokens,H,D] inputs selected by cu_seqlens;
//   - state_indices-backed state slots in canonical [K,V] layout;
//   - raw decay logits plus optional decay bias are transformed in-kernel;
//   - rectangular FP16 workloads preserve the Albatross large/small-warp
//     selector; a canonical-[K,V] B1 tile and group4 t-loop cover remaining
//     optimized shapes, while the forced-only short-block body is retained as
//     an unselected source reference;
//   - metadata status remains fail-closed for the low-level binding.
//
// vllm-rwkv at 6d683f9e49a2997e405c47edc147872c8609513b is a packed-varlen
// contract reference only; it is not the kernel-body source of this file.

#include <cuda_fp16.h>
#include "validation.h"

#include <limits>

#include "recurrent_decay.cuh"

namespace {

constexpr int kWarpThreads = 32;
constexpr int kBlockThreads = 32;
constexpr int kDecayAddThreads = 256;
constexpr int kKvTileThreads = kWarpThreads;
constexpr int kKvValueTile = 4;
constexpr int kKvKeysPerWarp = kWarpThreads / kKvValueTile;
constexpr int kSmallValuesPerBlock = 8;
constexpr int kSmallWarpsPerBlock = kSmallValuesPerBlock;
constexpr int kSmallThreads = kWarpThreads * kSmallWarpsPerBlock;

template <typename io_t>
__device__ __forceinline__ float to_float(io_t value) {
  return static_cast<float>(value);
}

template <typename io_t>
__device__ __forceinline__ io_t from_float(float value) {
  return static_cast<io_t>(value);
}

template <typename io_t>
__global__ void add_decay_bias_flat_kernel(
    const io_t* __restrict__ decay_logits,
    const io_t* __restrict__ decay_bias,
    io_t* __restrict__ decay_raw,
    int64_t elements,
    int channels) {
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    decay_raw[index] = from_float<io_t>(
        to_float(decay_logits[index]) +
        to_float(decay_bias[index % channels]));
  }
}

template <typename io_t>
__global__ void add_decay_bias_2d_kernel(
    const io_t* __restrict__ decay_logits,
    const io_t* __restrict__ decay_bias,
    io_t* __restrict__ decay_raw,
    int rows,
    int channels) {
  const int row = static_cast<int>(blockIdx.y);
  for (int channel =
           static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
       channel < channels;
       channel += static_cast<int>(gridDim.x) * blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * channels + channel;
    decay_raw[index] = from_float<io_t>(
        to_float(decay_logits[index]) + to_float(decay_bias[channel]));
  }
}

__global__ void add_decay_bias_half2_flat_kernel(
    const __half* __restrict__ decay_logits,
    const __half* __restrict__ decay_bias,
    __half* __restrict__ decay_raw,
    int64_t pairs,
    int channel_pairs) {
  for (int64_t pair =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       pair < pairs;
       pair += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int channel_pair = static_cast<int>(pair % channel_pairs);
    const auto logits = *reinterpret_cast<const __half2*>(decay_logits + pair * 2);
    const auto bias = *reinterpret_cast<const __half2*>(
        decay_bias + static_cast<int64_t>(channel_pair) * 2);
    *reinterpret_cast<__half2*>(decay_raw + pair * 2) = __hadd2(logits, bias);
  }
}

__global__ void add_decay_bias_half2_2d_kernel(
    const __half* __restrict__ decay_logits,
    const __half* __restrict__ decay_bias,
    __half* __restrict__ decay_raw,
    int channel_pairs) {
  const int row = static_cast<int>(blockIdx.y);
  for (int channel_pair =
           static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
       channel_pair < channel_pairs;
       channel_pair += static_cast<int>(gridDim.x) * blockDim.x) {
    const int64_t pair =
        static_cast<int64_t>(row) * channel_pairs + channel_pair;
    const auto logits = *reinterpret_cast<const __half2*>(decay_logits + pair * 2);
    const auto bias = *reinterpret_cast<const __half2*>(
        decay_bias + static_cast<int64_t>(channel_pair) * 2);
    *reinterpret_cast<__half2*>(decay_raw + pair * 2) = __hadd2(logits, bias);
  }
}

template <typename io_t>
void launch_decay_bias_add(
    const torch::stable::Tensor& decay_logits,
    const torch::stable::Tensor& decay_bias,
    torch::stable::Tensor& decay_raw,
    cudaStream_t stream) {
  const int rows = static_cast<int>(decay_logits.size(0));
  const int channels = static_cast<int>(decay_logits.numel() / rows);
  if (channels == 4096 && rows >= 17 && rows <= 65535) {
    const dim3 grid(
        static_cast<unsigned int>((channels + kDecayAddThreads - 1) /
                                  kDecayAddThreads),
        static_cast<unsigned int>(rows));
    add_decay_bias_2d_kernel<io_t><<<grid, kDecayAddThreads, 0, stream>>>(
        decay_logits.mutable_data_ptr<io_t>(), decay_bias.mutable_data_ptr<io_t>(),
        decay_raw.mutable_data_ptr<io_t>(), rows, channels);
  } else {
    const int64_t elements = decay_logits.numel();
    const int blocks = static_cast<int>(
        (elements + kDecayAddThreads - 1) / kDecayAddThreads);
    add_decay_bias_flat_kernel<io_t><<<
        blocks, kDecayAddThreads, 0, stream>>>(
        decay_logits.mutable_data_ptr<io_t>(), decay_bias.mutable_data_ptr<io_t>(),
        decay_raw.mutable_data_ptr<io_t>(), elements, channels);
  }
}

template <>
void launch_decay_bias_add<torch::headeronly::Half>(
    const torch::stable::Tensor& decay_logits,
    const torch::stable::Tensor& decay_bias,
    torch::stable::Tensor& decay_raw,
    cudaStream_t stream) {
  const int rows = static_cast<int>(decay_logits.size(0));
  const int channels = static_cast<int>(decay_logits.numel() / rows);
  const int channel_pairs = channels / 2;
  const auto* logits = reinterpret_cast<const __half*>(
      decay_logits.mutable_data_ptr<torch::headeronly::Half>());
  const auto* bias = reinterpret_cast<const __half*>(
      decay_bias.mutable_data_ptr<torch::headeronly::Half>());
  auto* raw = reinterpret_cast<__half*>(decay_raw.mutable_data_ptr<torch::headeronly::Half>());
  if (channels == 4096 && rows >= 17 && rows <= 65535) {
    const dim3 grid(
        static_cast<unsigned int>(
            (channel_pairs + kDecayAddThreads - 1) / kDecayAddThreads),
        static_cast<unsigned int>(rows));
    add_decay_bias_half2_2d_kernel<<<grid, kDecayAddThreads, 0, stream>>>(
        logits, bias, raw, channel_pairs);
  } else {
    const int64_t pairs = decay_logits.numel() / 2;
    const int blocks = static_cast<int>(
        (pairs + kDecayAddThreads - 1) / kDecayAddThreads);
    add_decay_bias_half2_flat_kernel<<<
        blocks, kDecayAddThreads, 0, stream>>>(
        logits, bias, raw, pairs, channel_pairs);
  }
}

// ee3308 builds this inference translation unit with --use_fast_math.  Keep
// its retention division explicit here instead of changing the shared decay
// helper used by StateTune, RL and the other inference numerical modes.
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

__device__ __forceinline__ float warp_sum_broadcast(float value) {
  return __shfl_sync(0xffffffffu, warp_sum(value), 0);
}

__device__ __forceinline__ float block_sum_broadcast(float value) {
  __shared__ float partial;
  value = warp_sum(value);
  if (threadIdx.x == 0) {
    partial = value;
  }
  __syncthreads();
  return partial;
}

template <typename io_t>
__device__ __forceinline__ void fill_invalid_output(
    int64_t block_index,
    int64_t block_count,
    int64_t output_elements,
    io_t* __restrict__ output_ptr) {
  const io_t nan_value = from_float<io_t>(__int_as_float(0x7fffffff));
  for (int64_t output_index =
           block_index * static_cast<int64_t>(blockDim.x) + threadIdx.x;
       output_index < output_elements;
       output_index += block_count * static_cast<int64_t>(blockDim.x)) {
    output_ptr[output_index] = nan_value;
  }
}

// Albatross large family, adapted from wkv_fp32_v2_kernel.  FlashRWKV2 stores
// state as [slot,head,key,value], whereas the upstream row-local state is
// addressed as [slot,head,value,key]; the cooperative load/store below is the
// only layout adaptation.
template <int HeadSize, typename io_t>
__global__ __launch_bounds__(HeadSize, HeadSize == 64 ? 2 : 1)
void wkv_fp32_v2_kernel(
    int num_heads,
    int64_t output_elements,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ state_indices,
    const int* __restrict__ metadata_status,
    float* __restrict__ state_ptr,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ decay_logits_ptr,
    const io_t* __restrict__ decay_bias_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr,
    const io_t* __restrict__ b_ptr,
    io_t* __restrict__ output_ptr,
    float scale) {
  const int head_index = static_cast<int>(blockIdx.x);
  const int sequence_index = static_cast<int>(blockIdx.y);
  const int value_index = static_cast<int>(threadIdx.x);
  const int64_t block_index =
      static_cast<int64_t>(sequence_index) * num_heads + head_index;
  const int64_t block_count =
      static_cast<int64_t>(gridDim.x) * gridDim.y;

  if (metadata_status[0] != 0) {
    fill_invalid_output(
        block_index, block_count, output_elements, output_ptr);
    return;
  }
  if (sequence_index >= metadata_status[2]) {
    return;
  }

  __shared__ int token_start;
  __shared__ int token_end;
  __shared__ int state_slot;
  if (value_index == 0) {
    token_start = query_start_loc[sequence_index];
    token_end = query_start_loc[sequence_index + 1];
    state_slot = state_indices[sequence_index];
  }
  __syncthreads();

  float* state_base =
      state_ptr +
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
          HeadSize * HeadSize;
  float state[HeadSize];
#pragma unroll
  for (int key_index = 0; key_index < HeadSize; ++key_index) {
    state[key_index] = state_base[key_index * HeadSize + value_index];
  }

  __shared__ float r[HeadSize];
  __shared__ float decay[HeadSize];
  __shared__ float k[HeadSize];
  __shared__ float a[HeadSize];
  __shared__ float b[HeadSize];

  for (int token_index = token_start; token_index < token_end; ++token_index) {
    const int64_t input_index =
        (static_cast<int64_t>(token_index) * num_heads + head_index) *
            HeadSize +
        value_index;
    r[value_index] = to_float(r_ptr[input_index]);
    float decay_logits = to_float(decay_logits_ptr[input_index]);
    if (decay_bias_ptr != nullptr) {
      decay_logits +=
          to_float(decay_bias_ptr[head_index * HeadSize + value_index]);
    }
    decay[value_index] = recurrent_retention(decay_logits);
    k[value_index] = to_float(k_ptr[input_index]);
    a[value_index] = to_float(a_ptr[input_index]);
    b[value_index] = to_float(b_ptr[input_index]);
    __syncthreads();

    float a_state = 0.0f;
#pragma unroll
    for (int key_index = 0; key_index < HeadSize; ++key_index) {
      a_state += a[key_index] * state[key_index];
    }

    const float value = to_float(v_ptr[input_index]);
    float result = 0.0f;
#pragma unroll
    for (int key_index = 0; key_index < HeadSize; ++key_index) {
      const float updated =
          decay[key_index] * state[key_index] +
          b[key_index] * a_state + k[key_index] * value;
      state[key_index] = updated;
      result += r[key_index] * updated;
    }
    output_ptr[input_index] = from_float<io_t>(scale * result);
    __syncthreads();
  }

#pragma unroll
  for (int key_index = 0; key_index < HeadSize; ++key_index) {
    state_base[key_index * HeadSize + value_index] = state[key_index];
  }
}

// Canonical [K,V] state needs a layout-native small workload path.  One warp
// owns a value tile: lane groups traverse K while consecutive lanes
// remain contiguous in V.  Segmented warp reductions reproduce a_state[V] and
// y[V] without block barriers or transposing the public state pool.
template <int HeadSize, typename io_t>
__global__ __launch_bounds__(kKvTileThreads, 4)
void wkv_fp32_v2_kv_tile_kernel(
    int num_heads,
    int64_t output_elements,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ state_indices,
    const int* __restrict__ metadata_status,
    float* __restrict__ state_ptr,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ decay_logits_ptr,
    const io_t* __restrict__ decay_bias_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr,
    const io_t* __restrict__ b_ptr,
    io_t* __restrict__ output_ptr,
    float scale) {
  const int head_index = static_cast<int>(blockIdx.x);
  const int sequence_index = static_cast<int>(blockIdx.y);
  const int lane = static_cast<int>(threadIdx.x) & (kWarpThreads - 1);
  const int value_lane = lane & (kKvValueTile - 1);
  const int key_lane = lane / kKvValueTile;
  const int value_index =
      static_cast<int>(blockIdx.z) * kKvValueTile + value_lane;
  const int64_t block_index =
      (static_cast<int64_t>(sequence_index) * num_heads + head_index) *
          gridDim.z +
      blockIdx.z;
  const int64_t block_count =
      static_cast<int64_t>(gridDim.x) * gridDim.y * gridDim.z;

  if (metadata_status[0] != 0) {
    fill_invalid_output(
        block_index, block_count, output_elements, output_ptr);
    return;
  }
  if (sequence_index >= metadata_status[2]) {
    return;
  }

  int token_start = lane == 0 ? query_start_loc[sequence_index] : 0;
  int token_end = lane == 0 ? query_start_loc[sequence_index + 1] : 0;
  int state_slot = lane == 0 ? state_indices[sequence_index] : 0;
  token_start = __shfl_sync(0xffffffffu, token_start, 0);
  token_end = __shfl_sync(0xffffffffu, token_end, 0);
  state_slot = __shfl_sync(0xffffffffu, state_slot, 0);

  float* state_base =
      state_ptr +
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
          HeadSize * HeadSize;
  for (int token_index = token_start; token_index < token_end; ++token_index) {
    const int64_t token_base =
        (static_cast<int64_t>(token_index) * num_heads + head_index) *
        HeadSize;
    float a_state = 0.0f;
#pragma unroll
    for (int key_base = 0; key_base < HeadSize;
         key_base += kKvKeysPerWarp) {
      const int key_index = key_base + key_lane;
      float contribution = 0.0f;
      if (value_index < HeadSize) {
        contribution =
            state_base[static_cast<int64_t>(key_index) * HeadSize +
                       value_index] *
            to_float(a_ptr[token_base + key_index]);
      }
#pragma unroll
      for (int offset = kWarpThreads / 2; offset >= kKvValueTile;
           offset >>= 1) {
        contribution +=
            __shfl_down_sync(0xffffffffu, contribution, offset);
      }
      if (lane < kKvValueTile) {
        a_state += contribution;
      }
    }
    a_state = __shfl_sync(0xffffffffu, a_state, value_lane);
    float value = lane < kKvValueTile && value_index < HeadSize
        ? to_float(v_ptr[token_base + value_index])
        : 0.0f;
    value = __shfl_sync(0xffffffffu, value, value_lane);
    float result = 0.0f;
#pragma unroll
    for (int key_base = 0; key_base < HeadSize;
         key_base += kKvKeysPerWarp) {
      const int key_index = key_base + key_lane;
      float contribution = 0.0f;
      if (value_index < HeadSize) {
        const int64_t input_index = token_base + key_index;
        float decay_logits = to_float(decay_logits_ptr[input_index]);
        if (decay_bias_ptr != nullptr) {
          decay_logits += to_float(decay_bias_ptr[
              head_index * HeadSize + key_index]);
        }
        const int64_t state_index =
            static_cast<int64_t>(key_index) * HeadSize + value_index;
        const float updated =
            recurrent_retention(decay_logits) * state_base[state_index] +
            to_float(b_ptr[input_index]) * a_state +
            to_float(k_ptr[input_index]) * value;
        state_base[state_index] = updated;
        contribution = to_float(r_ptr[input_index]) * updated;
      }
#pragma unroll
      for (int offset = kWarpThreads / 2; offset >= kKvValueTile;
           offset >>= 1) {
        contribution +=
            __shfl_down_sync(0xffffffffu, contribution, offset);
      }
      if (lane < kKvValueTile) {
        result += contribution;
      }
    }
    if (lane < kKvValueTile && value_index < HeadSize) {
      output_ptr[token_base + value_index] =
          from_float<io_t>(scale * result);
    }
  }
}

// Albatross's four-warp t-loop, adapted to packed request boundaries and
// slot-indexed [K,V] state.  Four independent value tiles share one copy of
// the token vectors, avoiding the one-warp CTA slot limit for short-sequence
// shapes outside ee3308's rectangular FP16 selector.
template <int ValueTile>
__global__ __launch_bounds__(kWarpThreads * 4, 2)
void wkv_fp32_v2_kv_tloop_group4_kernel(
    int num_heads,
    int64_t output_elements,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ state_indices,
    const int* __restrict__ metadata_status,
    float* __restrict__ state_ptr,
    const torch::headeronly::Half* __restrict__ r_ptr,
    const torch::headeronly::Half* __restrict__ decay_logits_ptr,
    const torch::headeronly::Half* __restrict__ k_ptr,
    const torch::headeronly::Half* __restrict__ v_ptr,
    const torch::headeronly::Half* __restrict__ a_ptr,
    const torch::headeronly::Half* __restrict__ b_ptr,
    torch::headeronly::Half* __restrict__ output_ptr,
    float scale) {
  static_assert(ValueTile == 4 || ValueTile == 8);
  constexpr int kHeadSize = 64;
  constexpr int kWarpsPerBlock = 4;
  constexpr int kKeyLanes = kWarpThreads / ValueTile;
  constexpr int kStatesPerLane = kHeadSize / kKeyLanes;

  const int head_index = static_cast<int>(blockIdx.x);
  const int sequence_index = static_cast<int>(blockIdx.y);
  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread / kWarpThreads;
  const int lane = thread & (kWarpThreads - 1);
  const int value_lane = lane & (ValueTile - 1);
  const int key_lane = lane / ValueTile;
  const int value_tile = static_cast<int>(blockIdx.z) * kWarpsPerBlock + warp;
  const int value_index = value_tile * ValueTile + value_lane;
  const int64_t block_index =
      (static_cast<int64_t>(sequence_index) * num_heads + head_index) *
          gridDim.z +
      blockIdx.z;
  const int64_t block_count =
      static_cast<int64_t>(gridDim.x) * gridDim.y * gridDim.z;

  if (metadata_status[0] != 0) {
    fill_invalid_output(
        block_index, block_count, output_elements, output_ptr);
    return;
  }
  if (sequence_index >= metadata_status[2]) {
    return;
  }

  // These three values are warp-uniform and remain hot in L1.  Keeping them
  // in registers avoids an otherwise unconditional CTA barrier before the
  // short t-loop starts.
  const int token_start = query_start_loc[sequence_index];
  const int token_end = query_start_loc[sequence_index + 1];
  const int state_slot = state_indices[sequence_index];

  float* state_base =
      state_ptr +
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
          kHeadSize * kHeadSize;
  float state[kStatesPerLane];
#pragma unroll
  for (int slot = 0; slot < kStatesPerLane; ++slot) {
    const int key_index = slot * kKeyLanes + key_lane;
    state[slot] = state_base[key_index * kHeadSize + value_index];
  }

  __shared__ float shared_r[kHeadSize];
  __shared__ float shared_decay[kHeadSize];
  __shared__ float shared_k[kHeadSize];
  __shared__ float shared_v[kHeadSize];
  __shared__ float shared_a[kHeadSize];
  __shared__ float shared_b[kHeadSize];

  for (int token_index = token_start; token_index < token_end; ++token_index) {
    const int64_t token_base =
        (static_cast<int64_t>(token_index) * num_heads + head_index) *
        kHeadSize;
    // This is the consumer barrier for the preceding token.  The first
    // iteration has no shared-vector producer to wait for, while the barrier
    // after the loads below already covers the initial state loads.
    if (token_index != token_start) {
      __syncthreads();
    }
    if (thread < kHeadSize) {
      const int64_t input_index = token_base + thread;
      shared_r[thread] = to_float(r_ptr[input_index]);
      // The owner pre-adds an optional decay bias before dispatch, so the
      // selected group4 body never needs an inner-loop nullable-pointer branch.
      shared_decay[thread] =
          recurrent_retention(to_float(decay_logits_ptr[input_index]));
      shared_k[thread] = to_float(k_ptr[input_index]);
      shared_v[thread] = to_float(v_ptr[input_index]);
      shared_a[thread] = to_float(a_ptr[input_index]);
      shared_b[thread] = to_float(b_ptr[input_index]);
    }
    __syncthreads();

    float a_state = 0.0f;
#pragma unroll
    for (int slot = 0; slot < kStatesPerLane; ++slot) {
      const int key_index = slot * kKeyLanes + key_lane;
      a_state += state[slot] * shared_a[key_index];
    }
#pragma unroll
    for (int offset = kWarpThreads / 2; offset >= ValueTile; offset >>= 1) {
      a_state += __shfl_down_sync(0xffffffffu, a_state, offset);
    }
    a_state = __shfl_sync(0xffffffffu, a_state, value_lane);

    const float value = shared_v[value_index];
    float result = 0.0f;
#pragma unroll
    for (int slot = 0; slot < kStatesPerLane; ++slot) {
      const int key_index = slot * kKeyLanes + key_lane;
      const float updated = state[slot] * shared_decay[key_index] +
          a_state * shared_b[key_index] + shared_k[key_index] * value;
      state[slot] = updated;
      result += updated * shared_r[key_index];
    }
#pragma unroll
    for (int offset = kWarpThreads / 2; offset >= ValueTile; offset >>= 1) {
      result += __shfl_down_sync(0xffffffffu, result, offset);
    }
    if (lane < ValueTile) {
      output_ptr[token_base + value_index] =
          from_float<torch::headeronly::Half>(scale * result);
    }
  }

#pragma unroll
  for (int slot = 0; slot < kStatesPerLane; ++slot) {
    const int key_index = slot * kKeyLanes + key_lane;
    state_base[key_index * kHeadSize + value_index] = state[slot];
  }
}

// Albatross small-warp arithmetic adapted to canonical [K,V] state.  Each
// warp still owns one V coordinate and traverses K in the upstream lane order,
// preserving the reduction tree and FP32 update order.  Eight warps share a
// coalesced [K,V-tile] load/store so the public layout does not turn every
// state access into a stride-HeadSize transaction.
template <int HeadSize, typename io_t>
__global__ __launch_bounds__(kSmallThreads, 2)
void wkv_fp32_v2_small_warp_kernel(
    int num_heads,
    int64_t output_elements,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ state_indices,
    const int* __restrict__ metadata_status,
    float* __restrict__ state_ptr,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ decay_logits_ptr,
    const io_t* __restrict__ decay_bias_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr,
    const io_t* __restrict__ b_ptr,
    io_t* __restrict__ output_ptr,
    float scale) {
  const int value_tile = static_cast<int>(blockIdx.x);
  const int head_index = static_cast<int>(blockIdx.y);
  const int sequence_index = static_cast<int>(blockIdx.z);
  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread / kWarpThreads;
  const int lane = thread & (kWarpThreads - 1);
  const int value_index = value_tile * kSmallValuesPerBlock + warp;
  const int64_t block_index =
      (static_cast<int64_t>(sequence_index) * num_heads + head_index) *
          gridDim.x +
      value_tile;
  const int64_t block_count = static_cast<int64_t>(gridDim.x) * gridDim.y *
      gridDim.z;

  if (metadata_status[0] != 0) {
    fill_invalid_output(
        block_index, block_count, output_elements, output_ptr);
    return;
  }
  if (sequence_index >= metadata_status[2]) {
    return;
  }

  const int token_start = query_start_loc[sequence_index];
  const int token_end = query_start_loc[sequence_index + 1];
  const int state_slot = state_indices[sequence_index];
  float* state_base = state_ptr +
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
          HeadSize * HeadSize;
  __shared__ float state_tile[kSmallValuesPerBlock * HeadSize];
  for (int linear = thread;
       linear < kSmallValuesPerBlock * HeadSize;
       linear += kSmallThreads) {
    const int key_index = linear / kSmallValuesPerBlock;
    const int value_offset = linear % kSmallValuesPerBlock;
    state_tile[linear] = state_base[
        static_cast<int64_t>(key_index) * HeadSize +
        value_tile * kSmallValuesPerBlock + value_offset];
  }
  __syncthreads();

  constexpr int kStatesPerLane = HeadSize / kWarpThreads;
  float state[kStatesPerLane];
#pragma unroll
  for (int slot = 0; slot < kStatesPerLane; ++slot) {
    const int key_index = slot * kWarpThreads + lane;
    state[slot] = state_tile[
        key_index * kSmallValuesPerBlock + warp];
  }

  __shared__ float shared_r[HeadSize];
  __shared__ float shared_decay[HeadSize];
  __shared__ float shared_k[HeadSize];
  __shared__ float shared_v[HeadSize];
  __shared__ float shared_a[HeadSize];
  __shared__ float shared_b[HeadSize];

  for (int token_index = token_start; token_index < token_end; ++token_index) {
    const int64_t token_base =
        static_cast<int64_t>(token_index) * num_heads * HeadSize +
        static_cast<int64_t>(head_index) * HeadSize;
    if (thread < HeadSize) {
      const int64_t index = token_base + thread;
      float decay_logits = to_float(decay_logits_ptr[index]);
      if (decay_bias_ptr != nullptr) {
        decay_logits +=
            to_float(decay_bias_ptr[head_index * HeadSize + thread]);
      }
      shared_r[thread] = to_float(r_ptr[index]);
      shared_decay[thread] = recurrent_retention(decay_logits);
      shared_k[thread] = to_float(k_ptr[index]);
      shared_v[thread] = to_float(v_ptr[index]);
      shared_a[thread] = to_float(a_ptr[index]);
      shared_b[thread] = to_float(b_ptr[index]);
    }
    __syncthreads();

    float a_state = 0.0f;
#pragma unroll
    for (int slot = 0; slot < kStatesPerLane; ++slot) {
      const int key_index = slot * kWarpThreads + lane;
      a_state += state[slot] * shared_a[key_index];
    }
    a_state = warp_sum_broadcast(a_state);

    const float value = shared_v[value_index];
    float result = 0.0f;
#pragma unroll
    for (int slot = 0; slot < kStatesPerLane; ++slot) {
      const int key_index = slot * kWarpThreads + lane;
      const float updated =
          state[slot] * shared_decay[key_index] +
          value * shared_k[key_index] +
          a_state * shared_b[key_index];
      state[slot] = updated;
      result += updated * shared_r[key_index];
    }
    result = warp_sum(result);
    if (lane == 0) {
      output_ptr[token_base + value_index] =
          from_float<io_t>(scale * result);
    }
    __syncthreads();
  }

#pragma unroll
  for (int slot = 0; slot < kStatesPerLane; ++slot) {
    const int key_index = slot * kWarpThreads + lane;
    state_tile[key_index * kSmallValuesPerBlock + warp] = state[slot];
  }
  __syncthreads();
  for (int linear = thread;
       linear < kSmallValuesPerBlock * HeadSize;
       linear += kSmallThreads) {
    const int key_index = linear / kSmallValuesPerBlock;
    const int value_offset = linear % kSmallValuesPerBlock;
    state_base[
        static_cast<int64_t>(key_index) * HeadSize +
        value_tile * kSmallValuesPerBlock + value_offset] = state_tile[linear];
  }
}

// Upstream status: Albatross revision
// 3e41bc43ed5e8332927ddd7e0ce4816cf200a6ea exposes this body through the
// registered forward_block operator (mode==3).  Local status: FlashRWKV's
// public packed-varlen API intentionally has no forced mode selector.  Keep
// the upstream adaptation as disabled reference code; do not re-enable it
// without a real selector, packed-varlen correctness and launch-boundary
// coverage, and benchmark evidence.
#if 0
template <int HeadSize, typename io_t>
__global__ __launch_bounds__(kBlockThreads, 4)
void wkv_fp32_v2_short_block_kernel(
    int num_heads,
    int64_t output_elements,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ state_indices,
    const int* __restrict__ metadata_status,
    float* __restrict__ state_ptr,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ decay_logits_ptr,
    const io_t* __restrict__ decay_bias_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr,
    const io_t* __restrict__ b_ptr,
    io_t* __restrict__ output_ptr,
    float scale) {
  const int row = static_cast<int>(blockIdx.x);
  const int head_index = static_cast<int>(blockIdx.y);
  const int sequence_index = static_cast<int>(blockIdx.z);
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t block_index =
      (static_cast<int64_t>(sequence_index) * num_heads + head_index) *
          HeadSize +
      row;
  const int64_t block_count = static_cast<int64_t>(gridDim.x) * gridDim.y *
      gridDim.z;

  if (metadata_status[0] != 0) {
    fill_invalid_output(
        block_index, block_count, output_elements, output_ptr);
    return;
  }
  if (sequence_index >= metadata_status[2]) {
    return;
  }

  const int token_start = query_start_loc[sequence_index];
  const int token_end = query_start_loc[sequence_index + 1];
  const int state_slot = state_indices[sequence_index];
  // Keep the same [K,V] state addressing as the large and small-warp
  // families; row is the fixed V coordinate for this block.
  float* state_base = state_ptr +
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
          HeadSize * HeadSize;

  for (int token_index = token_start; token_index < token_end; ++token_index) {
    const int64_t token_base =
        static_cast<int64_t>(token_index) * num_heads * HeadSize +
        static_cast<int64_t>(head_index) * HeadSize;
    float a_state = 0.0f;
    for (int key_index = tid; key_index < HeadSize;
         key_index += kBlockThreads) {
      a_state +=
          state_base[static_cast<int64_t>(key_index) * HeadSize + row] *
          to_float(a_ptr[token_base + key_index]);
    }
    a_state = block_sum_broadcast(a_state);

    const float value = to_float(v_ptr[token_base + row]);
    float result = 0.0f;
    for (int key_index = tid; key_index < HeadSize;
         key_index += kBlockThreads) {
      const int64_t index = token_base + key_index;
      float decay_logits = to_float(decay_logits_ptr[index]);
      if (decay_bias_ptr != nullptr) {
        decay_logits +=
            to_float(decay_bias_ptr[head_index * HeadSize + key_index]);
      }
      const float updated =
          state_base[static_cast<int64_t>(key_index) * HeadSize + row] *
              recurrent_retention(decay_logits) +
          a_state * to_float(b_ptr[index]) +
          value * to_float(k_ptr[index]);
      state_base[static_cast<int64_t>(key_index) * HeadSize + row] = updated;
      result += updated * to_float(r_ptr[index]);
    }
    result = block_sum_broadcast(result);
    if (tid == 0) {
      output_ptr[token_base + row] = from_float<io_t>(scale * result);
    }
    __syncthreads();
  }
}
#endif

bool use_small_auto(
    int batch_size,
    int total_tokens,
    int max_seqlen,
    bool io_fp16) {
  if (!io_fp16 || max_seqlen <= 0 ||
      static_cast<int64_t>(batch_size) * max_seqlen != total_tokens) {
    return false;
  }
  // Preserve ee3308's FP32IO16 arithmetic selector for rectangular requests.
  // The [K,V] tile keeps its reduction tree while restoring coalesced state
  // traffic; selecting another tree changes every later layer's FP16 inputs
  // and amplifies into model-level state and logits drift. Ragged requests keep
  // the other layout-native families because ee3308 has no matching contract.
  return (max_seqlen == 1 && batch_size <= 96) ||
      (max_seqlen == 2 && batch_size <= 21) ||
      (max_seqlen == 3 && batch_size <= 3) ||
      (max_seqlen == 4 && (batch_size == 1 || batch_size == 3)) ||
      (batch_size == 1 && max_seqlen >= 5 && max_seqlen <= 11);
}

int group4_value_tile_auto(
    int sequence_capacity,
    int num_heads,
    int max_seqlen,
    bool io_fp16) {
  if (!io_fp16 || max_seqlen <= 0) {
    return 0;
  }
  const int64_t sequence_heads =
      static_cast<int64_t>(sequence_capacity) * num_heads;
  if (max_seqlen == 5 && sequence_heads <= 80) {
    return 8;
  }
  if (max_seqlen >= 6 && max_seqlen <= 9 && sequence_heads <= 80) {
    return sequence_heads <= 40 ? 4 : 8;
  }
  if (max_seqlen >= 10 && sequence_heads <= 48) {
    return 4;
  }
  return 0;
}

template <int HeadSize, typename io_t>
void launch_recurrent_fp32(
    int num_sequences,
    int num_heads,
    int total_tokens,
    int max_seqlen,
    const torch::stable::Tensor& query_start_loc,
    const torch::stable::Tensor& state_indices,
    torch::stable::Tensor& state,
    const torch::stable::Tensor& r,
    const torch::stable::Tensor& decay_logits,
    const torch::stable::Tensor& decay_bias,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    torch::stable::Tensor& output,
    const torch::stable::Tensor& metadata_status,
    float scale,
    cudaStream_t stream) {
  const int64_t output_elements = output.numel();
  const bool io_fp16 = r.scalar_type() == torch::headeronly::ScalarType::Half;
  const bool use_small = HeadSize == 64 && use_small_auto(
      num_sequences, total_tokens, max_seqlen, io_fp16);
  const int group4_value_tile = group4_value_tile_auto(
      num_sequences, num_heads, max_seqlen, io_fp16);
  const auto* query_ptr = query_start_loc.mutable_data_ptr<int>();
  const auto* state_indices_ptr = state_indices.mutable_data_ptr<int>();
  const auto* status_ptr = metadata_status.mutable_data_ptr<int>();
  auto* state_ptr = state.mutable_data_ptr<float>();
  const auto* decay_bias_ptr =
      decay_bias.defined() ? decay_bias.mutable_data_ptr<io_t>() : nullptr;
  // Disabled upstream forced-mode launch retained beside the automatic
  // selector.  See the #if 0 kernel body above for provenance.
#if 0
  if (use_short_block_mode) {
    wkv_fp32_v2_short_block_kernel<HeadSize, io_t>
        <<<dim3(HeadSize, num_heads, num_sequences), dim3(kBlockThreads), 0,
               stream>>>(
            num_heads, output_elements, query_ptr, state_indices_ptr,
            status_ptr, state_ptr, r.mutable_data_ptr<io_t>(),
            decay_logits.mutable_data_ptr<io_t>(), decay_bias_ptr, k.mutable_data_ptr<io_t>(),
            v.mutable_data_ptr<io_t>(), a.mutable_data_ptr<io_t>(), b.mutable_data_ptr<io_t>(),
            output.mutable_data_ptr<io_t>(), scale);
  }
#endif
  if (use_small) {
    wkv_fp32_v2_small_warp_kernel<HeadSize, io_t>
        <<<dim3(HeadSize / kSmallValuesPerBlock, num_heads, num_sequences),
               dim3(kSmallThreads), 0, stream>>>(
            num_heads, output_elements, query_ptr, state_indices_ptr,
            status_ptr, state_ptr, r.mutable_data_ptr<io_t>(),
            decay_logits.mutable_data_ptr<io_t>(), decay_bias_ptr, k.mutable_data_ptr<io_t>(),
            v.mutable_data_ptr<io_t>(), a.mutable_data_ptr<io_t>(), b.mutable_data_ptr<io_t>(),
            output.mutable_data_ptr<io_t>(), scale);
    return;
  }
  if constexpr (HeadSize == 64) {
    if (num_sequences == 1 && total_tokens == 1 && max_seqlen == 1) {
      wkv_fp32_v2_kv_tile_kernel<HeadSize, io_t>
          <<<dim3(num_heads, num_sequences, HeadSize / kKvValueTile),
             dim3(kKvTileThreads), 0, stream>>>(
              num_heads, output_elements, query_ptr, state_indices_ptr,
              status_ptr, state_ptr, r.mutable_data_ptr<io_t>(),
              decay_logits.mutable_data_ptr<io_t>(), decay_bias_ptr,
              k.mutable_data_ptr<io_t>(), v.mutable_data_ptr<io_t>(), a.mutable_data_ptr<io_t>(),
              b.mutable_data_ptr<io_t>(), output.mutable_data_ptr<io_t>(), scale);
      return;
    }
    if (group4_value_tile == 4) {
      wkv_fp32_v2_kv_tloop_group4_kernel<4>
          <<<dim3(num_heads, num_sequences, HeadSize / (4 * 4)),
             dim3(kWarpThreads * 4), 0, stream>>>(
              num_heads, output_elements, query_ptr, state_indices_ptr,
              status_ptr, state_ptr,
              reinterpret_cast<const torch::headeronly::Half*>(r.mutable_data_ptr<io_t>()),
              reinterpret_cast<const torch::headeronly::Half*>(
                  decay_logits.mutable_data_ptr<io_t>()),
              reinterpret_cast<const torch::headeronly::Half*>(k.mutable_data_ptr<io_t>()),
              reinterpret_cast<const torch::headeronly::Half*>(v.mutable_data_ptr<io_t>()),
              reinterpret_cast<const torch::headeronly::Half*>(a.mutable_data_ptr<io_t>()),
              reinterpret_cast<const torch::headeronly::Half*>(b.mutable_data_ptr<io_t>()),
              reinterpret_cast<torch::headeronly::Half*>(output.mutable_data_ptr<io_t>()), scale);
      return;
    }
    if (group4_value_tile == 8) {
      wkv_fp32_v2_kv_tloop_group4_kernel<8>
          <<<dim3(num_heads, num_sequences, HeadSize / (8 * 4)),
             dim3(kWarpThreads * 4), 0, stream>>>(
              num_heads, output_elements, query_ptr, state_indices_ptr,
              status_ptr, state_ptr,
              reinterpret_cast<const torch::headeronly::Half*>(r.mutable_data_ptr<io_t>()),
              reinterpret_cast<const torch::headeronly::Half*>(
                  decay_logits.mutable_data_ptr<io_t>()),
              reinterpret_cast<const torch::headeronly::Half*>(k.mutable_data_ptr<io_t>()),
              reinterpret_cast<const torch::headeronly::Half*>(v.mutable_data_ptr<io_t>()),
              reinterpret_cast<const torch::headeronly::Half*>(a.mutable_data_ptr<io_t>()),
              reinterpret_cast<const torch::headeronly::Half*>(b.mutable_data_ptr<io_t>()),
              reinterpret_cast<torch::headeronly::Half*>(output.mutable_data_ptr<io_t>()), scale);
      return;
    }
  }
  wkv_fp32_v2_kernel<HeadSize, io_t>
      <<<dim3(num_heads, num_sequences), dim3(HeadSize), 0, stream>>>(
          num_heads, output_elements, query_ptr, state_indices_ptr,
          status_ptr, state_ptr, r.mutable_data_ptr<io_t>(),
          decay_logits.mutable_data_ptr<io_t>(), decay_bias_ptr, k.mutable_data_ptr<io_t>(),
          v.mutable_data_ptr<io_t>(), a.mutable_data_ptr<io_t>(), b.mutable_data_ptr<io_t>(),
          output.mutable_data_ptr<io_t>(), scale);
}

}  // namespace

void tmix_wkv7_recurrent_fp32_from_decay_logits_cuda(
    torch::stable::Tensor query_start_loc,
    torch::stable::Tensor state_indices,
    torch::stable::Tensor state,
    torch::stable::Tensor r,
    torch::stable::Tensor decay_logits,
    torch::stable::Tensor decay_bias,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor a,
    torch::stable::Tensor b,
    torch::stable::Tensor output,
    torch::stable::Tensor metadata_status,
    double scale,
    int64_t max_seqlen) {
  const torch::stable::accelerator::DeviceGuard device_guard(state.device().index());
  const auto stream = flashrwkv2::validation::current_cuda_stream();
  const int num_sequences = static_cast<int>(state_indices.numel());
  const int num_heads = static_cast<int>(state.size(1));
  const int total_tokens = static_cast<int>(r.size(0));
  // The public Python path always supplies a prepared ticket.  A direct
  // low-level launch without one has no synchronous metadata snapshot, so use
  // the canonical large-family dispatch rather than turning fail-closed
  // metadata handling into an argument error.
  if (max_seqlen <= 0) {
    max_seqlen = std::numeric_limits<int>::max();
  }

  auto dispatch_launch = [&](auto scalar_value) {
    using scalar_t = decltype(scalar_value);
        torch::stable::Tensor decay_raw = decay_logits;
        torch::stable::Tensor recurrent_decay_bias = decay_bias;
        if (decay_bias.defined()) {
          decay_raw = torch::stable::empty_like(decay_logits);
          launch_decay_bias_add<scalar_t>(
              decay_logits, decay_bias, decay_raw, stream);
          FLASHRWKV_CUDA_CHECK(cudaGetLastError());
          recurrent_decay_bias = torch::stable::Tensor();
        }
        switch (state.size(2)) {
          case 64:
            launch_recurrent_fp32<64, scalar_t>(
                num_sequences, num_heads, total_tokens,
                static_cast<int>(max_seqlen), query_start_loc, state_indices,
                state, r, decay_raw, recurrent_decay_bias, k, v, a, b,
                output, metadata_status,
                static_cast<float>(scale), stream);
            break;
          case 128:
            launch_recurrent_fp32<128, scalar_t>(
                num_sequences, num_heads, total_tokens,
                static_cast<int>(max_seqlen), query_start_loc, state_indices,
                state, r, decay_raw, recurrent_decay_bias, k, v, a, b,
                output, metadata_status,
                static_cast<float>(scale), stream);
            break;
          case 256:
            launch_recurrent_fp32<256, scalar_t>(
                num_sequences, num_heads, total_tokens,
                static_cast<int>(max_seqlen), query_start_loc, state_indices,
                state, r, decay_raw, recurrent_decay_bias, k, v, a, b,
                output, metadata_status,
                static_cast<float>(scale), stream);
            break;
          default:
            STD_TORCH_CHECK(false, "unsupported recurrent head size");
        }
      };
  switch (r.scalar_type()) {
    case torch::headeronly::ScalarType::Half:
      dispatch_launch(torch::headeronly::Half{});
      break;
    case torch::headeronly::ScalarType::BFloat16:
      dispatch_launch(torch::headeronly::BFloat16{});
      break;
    default:
      STD_TORCH_CHECK(false, "token dtype must be float16 or bfloat16");
  }
  FLASHRWKV_CUDA_CHECK(cudaGetLastError());
}
