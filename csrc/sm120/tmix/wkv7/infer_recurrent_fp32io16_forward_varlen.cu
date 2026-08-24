// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Albatross project
//
// Canonical body source:
//   BlinkDL/Albatross/faster3a_2607/cuda/rwkv7_wkv_fp32_v2.cu
//   revision ee3308f6922e59f2166c7fac3c5a192340a2b48e
//   Apache-2.0
//
// Local adaptation:
//   - packed [total_tokens,H,D] inputs selected by cu_seqlens;
//   - state_indices-backed state slots in canonical [K,V] layout;
//   - raw decay logits plus optional decay bias are transformed in-kernel;
//   - the Albatross large implementation and a canonical-[K,V] B1 tile are
//     automatic dispatch families; the small-warp and forced-only short-block
//     bodies are retained as source references, but are not selected because
//     their fixed-V lane traversal is strided under the public state layout;
//   - metadata status remains fail-closed for the low-level binding.
//
// vllm-rwkv at 6d683f9e49a2997e405c47edc147872c8609513b is a packed-varlen
// contract reference only; it is not the kernel-body source of this file.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <limits>

#include "recurrent_decay.cuh"

namespace {

constexpr int kWarpThreads = 32;
constexpr int kBlockThreads = 32;
constexpr int kDecayAddThreads = 256;
constexpr int kKvTileThreads = kWarpThreads;
constexpr int kKvValueTile = 4;
constexpr int kKvKeysPerWarp = kWarpThreads / kKvValueTile;

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
    const torch::Tensor& decay_logits,
    const torch::Tensor& decay_bias,
    torch::Tensor& decay_raw,
    cudaStream_t stream) {
  const int rows = static_cast<int>(decay_logits.size(0));
  const int channels = static_cast<int>(decay_logits.numel() / rows);
  if (channels == 4096 && rows >= 17 && rows <= 65535) {
    const dim3 grid(
        static_cast<unsigned int>((channels + kDecayAddThreads - 1) /
                                  kDecayAddThreads),
        static_cast<unsigned int>(rows));
    add_decay_bias_2d_kernel<io_t><<<grid, kDecayAddThreads, 0, stream>>>(
        decay_logits.data_ptr<io_t>(), decay_bias.data_ptr<io_t>(),
        decay_raw.data_ptr<io_t>(), rows, channels);
  } else {
    const int64_t elements = decay_logits.numel();
    const int blocks = static_cast<int>(
        (elements + kDecayAddThreads - 1) / kDecayAddThreads);
    add_decay_bias_flat_kernel<io_t><<<
        blocks, kDecayAddThreads, 0, stream>>>(
        decay_logits.data_ptr<io_t>(), decay_bias.data_ptr<io_t>(),
        decay_raw.data_ptr<io_t>(), elements, channels);
  }
}

template <>
void launch_decay_bias_add<c10::Half>(
    const torch::Tensor& decay_logits,
    const torch::Tensor& decay_bias,
    torch::Tensor& decay_raw,
    cudaStream_t stream) {
  const int rows = static_cast<int>(decay_logits.size(0));
  const int channels = static_cast<int>(decay_logits.numel() / rows);
  const int channel_pairs = channels / 2;
  const auto* logits = reinterpret_cast<const __half*>(
      decay_logits.data_ptr<c10::Half>());
  const auto* bias = reinterpret_cast<const __half*>(
      decay_bias.data_ptr<c10::Half>());
  auto* raw = reinterpret_cast<__half*>(decay_raw.data_ptr<c10::Half>());
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

using flashrwkv2::wkv7::recurrent_retention;

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

// Albatross small-warp family.  Each block owns one output row and uses the
// 32-lane warp to traverse arbitrary D in 32-lane strips.  Unlike the fixed
// length upstream launch, token_start/token_end are read per request.
template <int HeadSize, typename io_t>
__global__ __launch_bounds__(kWarpThreads, 4)
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
  const int row = static_cast<int>(blockIdx.x);
  const int head_index = static_cast<int>(blockIdx.y);
  const int sequence_index = static_cast<int>(blockIdx.z);
  const int lane = static_cast<int>(threadIdx.x);
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
  // The canonical FlashRWKV2 state is [K,V].  This row-parallel family owns
  // one fixed V coordinate, so the K loop must stride by HeadSize from the
  // state column rather than treating the row as K and transposing state.
  float* state_base = state_ptr +
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
          HeadSize * HeadSize;

  for (int token_index = token_start; token_index < token_end; ++token_index) {
    const int64_t token_base =
        static_cast<int64_t>(token_index) * num_heads * HeadSize +
        static_cast<int64_t>(head_index) * HeadSize;
    float a_state = 0.0f;
    for (int key_index = lane; key_index < HeadSize;
         key_index += kWarpThreads) {
      a_state +=
          state_base[static_cast<int64_t>(key_index) * HeadSize + row] *
          to_float(a_ptr[token_base + key_index]);
    }
    a_state = warp_sum_broadcast(a_state);

    const float value = to_float(v_ptr[token_base + row]);
    float result = 0.0f;
    for (int key_index = lane; key_index < HeadSize;
         key_index += kWarpThreads) {
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
    result = warp_sum(result);
    if (lane == 0) {
      output_ptr[token_base + row] = from_float<io_t>(scale * result);
    }
  }
}

// Upstream status: Albatross revision
// ee3308f6922e59f2166c7fac3c5a192340a2b48e exposes this body through the
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
  // Albatross tunes this family for its row-major [V,K] state.  FlashRWKV2's
  // canonical [K,V] state turns the fixed-V warp's lane traversal into
  // HeadSize-strided loads and stores.  On SM120 this loses to the cooperative
  // large family at every rectangular selector point, including B=1/16/64,
  // while the large family already exceeds Albatross at B=320/960.  Keep the
  // imported body above for provenance, but fail closed to the measured family
  // until a layout-native small kernel has its own correctness and benchmark
  // evidence.
  (void)batch_size;
  (void)total_tokens;
  (void)max_seqlen;
  (void)io_fp16;
  return false;
}

template <int HeadSize, typename io_t>
void launch_recurrent_fp32(
    int num_sequences,
    int num_heads,
    int total_tokens,
    int max_seqlen,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& state_indices,
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& decay_logits,
    const torch::Tensor& decay_bias,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& output,
    const torch::Tensor& metadata_status,
    float scale,
    cudaStream_t stream) {
  const int64_t output_elements = output.numel();
  const bool io_fp16 = r.scalar_type() == at::ScalarType::Half;
  const bool use_small = use_small_auto(
      num_sequences, total_tokens, max_seqlen, io_fp16);
  const auto* query_ptr = query_start_loc.data_ptr<int>();
  const auto* state_indices_ptr = state_indices.data_ptr<int>();
  const auto* status_ptr = metadata_status.data_ptr<int>();
  auto* state_ptr = state.data_ptr<float>();
  const auto* decay_bias_ptr =
      decay_bias.defined() ? decay_bias.data_ptr<io_t>() : nullptr;
  // Disabled upstream forced-mode launch retained beside the automatic
  // selector.  See the #if 0 kernel body above for provenance.
#if 0
  if (use_short_block_mode) {
    wkv_fp32_v2_short_block_kernel<HeadSize, io_t>
        <<<dim3(HeadSize, num_heads, num_sequences), dim3(kBlockThreads), 0,
               stream>>>(
            num_heads, output_elements, query_ptr, state_indices_ptr,
            status_ptr, state_ptr, r.data_ptr<io_t>(),
            decay_logits.data_ptr<io_t>(), decay_bias_ptr, k.data_ptr<io_t>(),
            v.data_ptr<io_t>(), a.data_ptr<io_t>(), b.data_ptr<io_t>(),
            output.data_ptr<io_t>(), scale);
  }
#endif
  if constexpr (HeadSize == 64) {
    if (num_sequences == 1 && total_tokens == 1 && max_seqlen == 1) {
      wkv_fp32_v2_kv_tile_kernel<HeadSize, io_t>
          <<<dim3(num_heads, num_sequences, HeadSize / kKvValueTile),
             dim3(kKvTileThreads), 0, stream>>>(
              num_heads, output_elements, query_ptr, state_indices_ptr,
              status_ptr, state_ptr, r.data_ptr<io_t>(),
              decay_logits.data_ptr<io_t>(), decay_bias_ptr,
              k.data_ptr<io_t>(), v.data_ptr<io_t>(), a.data_ptr<io_t>(),
              b.data_ptr<io_t>(), output.data_ptr<io_t>(), scale);
      return;
    }
  }
  if (use_small) {
    wkv_fp32_v2_small_warp_kernel<HeadSize, io_t>
        <<<dim3(HeadSize, num_heads, num_sequences), dim3(kWarpThreads), 0,
               stream>>>(
            num_heads, output_elements, query_ptr, state_indices_ptr,
            status_ptr, state_ptr, r.data_ptr<io_t>(),
            decay_logits.data_ptr<io_t>(), decay_bias_ptr, k.data_ptr<io_t>(),
            v.data_ptr<io_t>(), a.data_ptr<io_t>(), b.data_ptr<io_t>(),
            output.data_ptr<io_t>(), scale);
  } else {
    wkv_fp32_v2_kernel<HeadSize, io_t>
        <<<dim3(num_heads, num_sequences), dim3(HeadSize), 0, stream>>>(
            num_heads, output_elements, query_ptr, state_indices_ptr,
            status_ptr, state_ptr, r.data_ptr<io_t>(),
            decay_logits.data_ptr<io_t>(), decay_bias_ptr, k.data_ptr<io_t>(),
            v.data_ptr<io_t>(), a.data_ptr<io_t>(), b.data_ptr<io_t>(),
            output.data_ptr<io_t>(), scale);
  }
}

}  // namespace

void tmix_wkv7_recurrent_fp32_from_decay_logits_cuda(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor decay_bias,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor metadata_status,
    double scale,
    int64_t max_seqlen) {
  const c10::cuda::CUDAGuard device_guard(state.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
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

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      r.scalar_type(),
      "flashrwkv2_tmix_wkv7_recurrent_fp32_from_decay_logits",
      [&] {
        torch::Tensor decay_raw = decay_logits;
        torch::Tensor recurrent_decay_bias = decay_bias;
        if (decay_bias.defined()) {
          decay_raw = torch::empty_like(decay_logits);
          launch_decay_bias_add<scalar_t>(
              decay_logits, decay_bias, decay_raw, stream);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
          recurrent_decay_bias = torch::Tensor();
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
            TORCH_CHECK(false, "unsupported recurrent head size");
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
