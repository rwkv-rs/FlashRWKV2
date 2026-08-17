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
//   - Albatross large and small-warp implementations remain active automatic
//     dispatch families; the forced-only short-block body is retained under
//     #if 0 because this API intentionally has no forced mode selector;
//   - metadata status remains fail-closed for the low-level binding.
//
// vllm-rwkv at 6d683f9e49a2997e405c47edc147872c8609513b is a packed-varlen
// contract reference only; it is not the kernel-body source of this file.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <limits>

#include "recurrent_decay.cuh"

namespace {

constexpr int kWarpThreads = 32;
constexpr int kBlockThreads = 32;

template <typename io_t>
__device__ __forceinline__ float to_float(io_t value) {
  return static_cast<float>(value);
}

template <typename io_t>
__device__ __forceinline__ io_t from_float(float value) {
  return static_cast<io_t>(value);
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

bool use_small_auto(int batch_size, int max_seqlen, bool io_fp16) {
  if (io_fp16) {
    return (max_seqlen == 1 && batch_size <= 96) ||
        (max_seqlen == 2 && batch_size <= 21) ||
        (max_seqlen == 3 && batch_size <= 3) ||
        (max_seqlen == 4 && (batch_size == 1 || batch_size == 3)) ||
        (batch_size == 1 && max_seqlen >= 5 && max_seqlen <= 11);
  }
  return (max_seqlen == 1) ||
      (max_seqlen == 2 && batch_size <= 96) ||
      (max_seqlen == 3 && (batch_size <= 4 || batch_size == 6)) ||
      (max_seqlen == 4 && (batch_size == 1 || batch_size == 3)) ||
      (batch_size == 1 && max_seqlen >= 5 && max_seqlen <= 9);
}

template <int HeadSize, typename io_t>
void launch_recurrent_fp32(
    int num_sequences,
    int num_heads,
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
  const bool use_small = use_small_auto(num_sequences, max_seqlen, io_fp16);
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
        switch (state.size(2)) {
          case 64:
            launch_recurrent_fp32<64, scalar_t>(
                num_sequences, num_heads, static_cast<int>(max_seqlen),
                query_start_loc, state_indices, state, r, decay_logits,
                decay_bias, k, v, a, b, output, metadata_status,
                static_cast<float>(scale), stream);
            break;
          case 128:
            launch_recurrent_fp32<128, scalar_t>(
                num_sequences, num_heads, static_cast<int>(max_seqlen),
                query_start_loc, state_indices, state, r, decay_logits,
                decay_bias, k, v, a, b, output, metadata_status,
                static_cast<float>(scale), stream);
            break;
          case 256:
            launch_recurrent_fp32<256, scalar_t>(
                num_sequences, num_heads, static_cast<int>(max_seqlen),
                query_start_loc, state_indices, state, r, decay_logits,
                decay_bias, k, v, a, b, output, metadata_status,
                static_cast<float>(scale), stream);
            break;
          default:
            TORCH_CHECK(false, "unsupported recurrent head size");
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
