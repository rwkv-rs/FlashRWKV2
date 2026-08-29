// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
// FlashKDA source revision: 1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b.
// K1 prepares RWKV-7 chunk transforms and K2 applies them to the request
// state.  The packed boundary is supplied by cu_seqlens; no padding is used.

#include "validation.h"

#include "recurrent_decay.cuh"

namespace {

constexpr int kHeadSize = 64;

using flashrwkv2::wkv7::recurrent_retention;

template <typename io_t>
__device__ __forceinline__ float to_float(io_t value) {
  return static_cast<float>(value);
}

template <typename io_t>
__device__ __forceinline__ io_t from_float(float value) {
  return static_cast<io_t>(value);
}

struct K1Shared {
  float transform[kHeadSize][kHeadSize];
  float bias[kHeadSize][kHeadSize];
  float r[kHeadSize];
  float decay[kHeadSize];
  float k[kHeadSize];
  float v[kHeadSize];
  float a[kHeadSize];
  float b[kHeadSize];
};

template <typename io_t>
__global__ __launch_bounds__(kHeadSize, 1)
void infer_chunk_bf16_k1_kernel(
    int num_heads,
    int chunk_size,
    int max_chunks,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ metadata_status,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ decay_ptr,
    const io_t* __restrict__ decay_bias_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr,
    const io_t* __restrict__ b_ptr,
    float* __restrict__ chunk_transform_ptr,
    float* __restrict__ chunk_bias_ptr,
    float* __restrict__ token_transform_ptr,
    float* __restrict__ token_bias_ptr,
    float scale) {
  if (metadata_status[0] != 0) {
    return;
  }
  const int linear_block = static_cast<int>(blockIdx.x);
  const int sequence_index = linear_block / num_heads;
  if (sequence_index >= metadata_status[2]) {
    return;
  }
  const int head_index = linear_block % num_heads;
  const int column = static_cast<int>(threadIdx.x);
  __shared__ K1Shared shared;

#pragma unroll
  for (int row = 0; row < kHeadSize; ++row) {
    shared.transform[row][column] = row == column ? 1.0f : 0.0f;
    shared.bias[row][column] = 0.0f;
  }
  __syncthreads();

  const int token_start = query_start_loc[sequence_index];
  const int token_end = query_start_loc[sequence_index + 1];
  const int sequence_length = token_end - token_start;
  const int chunk_count = (sequence_length + chunk_size - 1) / chunk_size;
  for (int chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
    const int chunk_token_start = token_start + chunk_index * chunk_size;
    const int chunk_token_end = min(chunk_token_start + chunk_size, token_end);
    for (int token_index = chunk_token_start;
         token_index < chunk_token_end;
         ++token_index) {
      const int64_t input_index =
          (static_cast<int64_t>(token_index) * num_heads + head_index) *
          kHeadSize + column;
      shared.r[column] = to_float(r_ptr[input_index]);
      float decay_input = to_float(decay_ptr[input_index]);
      if (decay_bias_ptr != nullptr) {
        decay_input += to_float(
            decay_bias_ptr[head_index * kHeadSize + column]);
      }
      shared.decay[column] = recurrent_retention(decay_input);
      shared.k[column] = to_float(k_ptr[input_index]);
      shared.v[column] = to_float(v_ptr[input_index]);
      shared.a[column] = to_float(a_ptr[input_index]);
      shared.b[column] = to_float(b_ptr[input_index]);
      __syncthreads();

      float transform_dot_a = 0.0f;
      float bias_dot_a = 0.0f;
#pragma unroll
      for (int row = 0; row < kHeadSize; ++row) {
        transform_dot_a = fmaf(
            shared.a[row], shared.transform[row][column], transform_dot_a);
        bias_dot_a = fmaf(
            shared.a[row], shared.bias[row][column], bias_dot_a);
      }
      const float value = shared.v[column];
#pragma unroll
      for (int row = 0; row < kHeadSize; ++row) {
        shared.transform[row][column] = fmaf(
            shared.b[row],
            transform_dot_a,
            shared.decay[row] * shared.transform[row][column]);
        shared.bias[row][column] = fmaf(
            shared.k[row],
            value,
            fmaf(
                shared.b[row],
                bias_dot_a,
                shared.decay[row] * shared.bias[row][column]));
      }
      __syncthreads();

      float output_transform = 0.0f;
      float output_bias = 0.0f;
#pragma unroll
      for (int row = 0; row < kHeadSize; ++row) {
        output_transform = fmaf(
            shared.r[row], shared.transform[row][column], output_transform);
        output_bias = fmaf(
            shared.r[row], shared.bias[row][column], output_bias);
      }
      token_transform_ptr[input_index] = scale * output_transform;
      token_bias_ptr[input_index] = scale * output_bias;
      __syncthreads();
    }

    const int64_t workspace_base =
        ((static_cast<int64_t>(sequence_index) * max_chunks + chunk_index) *
             num_heads +
         head_index) *
        kHeadSize * kHeadSize;
#pragma unroll
    for (int row = 0; row < kHeadSize; ++row) {
      const int64_t workspace_index =
          workspace_base + row * kHeadSize + column;
      chunk_transform_ptr[workspace_index] = shared.transform[row][column];
      chunk_bias_ptr[workspace_index] = shared.bias[row][column];
    }
  }
}

struct K2Shared {
  float token_transform[kHeadSize];
  float chunk_transform[kHeadSize][kHeadSize];
  float next_state[kHeadSize][kHeadSize];
};

template <typename io_t>
__global__ __launch_bounds__(kHeadSize, 1)
void infer_chunk_bf16_k2_kernel(
    int num_heads,
    int chunk_size,
    int max_chunks,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ state_indices,
    const int* __restrict__ metadata_status,
    io_t* __restrict__ state_ptr,
    io_t* __restrict__ output_ptr,
    const float* __restrict__ chunk_transform_ptr,
    const float* __restrict__ chunk_bias_ptr,
    const float* __restrict__ token_transform_ptr,
    const float* __restrict__ token_bias_ptr) {
  if (metadata_status[0] != 0) {
    return;
  }
  const int linear_block = static_cast<int>(blockIdx.x);
  const int sequence_index = linear_block / num_heads;
  if (sequence_index >= metadata_status[2]) {
    return;
  }
  const int head_index = linear_block % num_heads;
  const int value_index = static_cast<int>(threadIdx.x);
  __shared__ K2Shared shared;

  const int state_slot = state_indices[sequence_index];
  const int64_t state_base =
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
      kHeadSize * kHeadSize;
  float state[kHeadSize];
#pragma unroll
  for (int key_index = 0; key_index < kHeadSize; ++key_index) {
    state[key_index] = to_float(
        state_ptr[state_base + key_index * kHeadSize + value_index]);
  }

  const int token_start = query_start_loc[sequence_index];
  const int token_end = query_start_loc[sequence_index + 1];
  const int sequence_length = token_end - token_start;
  const int chunk_count = (sequence_length + chunk_size - 1) / chunk_size;
  for (int chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
    const int chunk_token_start = token_start + chunk_index * chunk_size;
    const int chunk_token_end = min(chunk_token_start + chunk_size, token_end);
    for (int token_index = chunk_token_start;
         token_index < chunk_token_end;
         ++token_index) {
      const int64_t token_base =
          (static_cast<int64_t>(token_index) * num_heads + head_index) *
          kHeadSize;
      shared.token_transform[value_index] =
          token_transform_ptr[token_base + value_index];
      __syncthreads();

      float output = token_bias_ptr[token_base + value_index];
#pragma unroll
      for (int key_index = 0; key_index < kHeadSize; ++key_index) {
        output = fmaf(
            shared.token_transform[key_index], state[key_index], output);
      }
      output_ptr[token_base + value_index] = from_float<io_t>(output);
      __syncthreads();
    }

    const int64_t workspace_base =
        ((static_cast<int64_t>(sequence_index) * max_chunks + chunk_index) *
             num_heads +
         head_index) *
        kHeadSize * kHeadSize;
#pragma unroll
    for (int row = 0; row < kHeadSize; ++row) {
      shared.chunk_transform[row][value_index] =
          chunk_transform_ptr[workspace_base + row * kHeadSize + value_index];
    }
    __syncthreads();

#pragma unroll
    for (int row = 0; row < kHeadSize; ++row) {
      float updated =
          chunk_bias_ptr[workspace_base + row * kHeadSize + value_index];
#pragma unroll
      for (int input_key = 0; input_key < kHeadSize; ++input_key) {
        updated = fmaf(
            shared.chunk_transform[row][input_key],
            state[input_key],
            updated);
      }
      shared.next_state[row][value_index] = updated;
    }
    __syncthreads();

#pragma unroll
    for (int key_index = 0; key_index < kHeadSize; ++key_index) {
      state[key_index] = shared.next_state[key_index][value_index];
    }
    __syncthreads();
  }

#pragma unroll
  for (int key_index = 0; key_index < kHeadSize; ++key_index) {
    state_ptr[state_base + key_index * kHeadSize + value_index] =
        from_float<io_t>(state[key_index]);
  }
}

}  // namespace

void infer_tmix_wkv7_chunk_bf16_forward_varlen_cuda(
    torch::stable::Tensor query_start_loc,
    torch::stable::Tensor state_indices,
    torch::stable::Tensor state_pool,
    torch::stable::Tensor r,
    torch::stable::Tensor decay_logits,
    torch::stable::Tensor decay_bias,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor a,
    torch::stable::Tensor b,
    torch::stable::Tensor output,
    torch::stable::Tensor chunk_transform,
    torch::stable::Tensor chunk_bias,
    torch::stable::Tensor token_transform,
    torch::stable::Tensor token_bias,
    int64_t chunk_size,
    int64_t max_seqlen,
    double scale,
    torch::stable::Tensor metadata_status) {
  const torch::stable::accelerator::DeviceGuard device_guard(r.device().index());
  const auto stream = flashrwkv2::validation::current_cuda_stream();
  const int num_sequences = static_cast<int>(state_indices.numel());
  const int num_heads = static_cast<int>(r.size(1));
  const int max_chunks = static_cast<int>(
      (max_seqlen + chunk_size - 1) / chunk_size);
  using io_t = torch::headeronly::BFloat16;

  infer_chunk_bf16_k1_kernel<io_t><<<
      num_sequences * num_heads, kHeadSize, 0, stream>>>(
      num_heads,
      static_cast<int>(chunk_size),
      max_chunks,
      query_start_loc.mutable_data_ptr<int>(),
      metadata_status.mutable_data_ptr<int>(),
      r.mutable_data_ptr<io_t>(),
      decay_logits.mutable_data_ptr<io_t>(),
      decay_bias.defined() ? decay_bias.mutable_data_ptr<io_t>() : nullptr,
      k.mutable_data_ptr<io_t>(),
      v.mutable_data_ptr<io_t>(),
      a.mutable_data_ptr<io_t>(),
      b.mutable_data_ptr<io_t>(),
      chunk_transform.mutable_data_ptr<float>(),
      chunk_bias.mutable_data_ptr<float>(),
      token_transform.mutable_data_ptr<float>(),
      token_bias.mutable_data_ptr<float>(),
      static_cast<float>(scale));
  FLASHRWKV_CUDA_CHECK(cudaGetLastError());

  infer_chunk_bf16_k2_kernel<io_t><<<
      num_sequences * num_heads, kHeadSize, 0, stream>>>(
      num_heads,
      static_cast<int>(chunk_size),
      max_chunks,
      query_start_loc.mutable_data_ptr<int>(),
      state_indices.mutable_data_ptr<int>(),
      metadata_status.mutable_data_ptr<int>(),
      state_pool.mutable_data_ptr<io_t>(),
      output.mutable_data_ptr<io_t>(),
      chunk_transform.mutable_data_ptr<float>(),
      chunk_bias.mutable_data_ptr<float>(),
      token_transform.mutable_data_ptr<float>(),
      token_bias.mutable_data_ptr<float>());
  FLASHRWKV_CUDA_CHECK(cudaGetLastError());
}
