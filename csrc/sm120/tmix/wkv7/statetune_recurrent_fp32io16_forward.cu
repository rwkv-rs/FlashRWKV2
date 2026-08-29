// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the RWKV-LM project
// Mechanically copied from RWKV-LM train_temp at revision
// 952102498e9ed367ea0a59ee64106916d474d30f.
// StateTune owns this body so its public binding is independent of pretrain.
// Raw-decay training recurrence: the public boundary is decay_logits;
// checkpoint boundaries, final state and state_dot_a stay training-local.

#include "validation.h"

#include "../../../sm120/tmix/wkv7/recurrent_decay.cuh"

namespace {

using flashrwkv2::wkv7::recurrent_retention;

template <typename io_t>
__device__ __forceinline__ float to_float(io_t value) {
  return static_cast<float>(value);
}

template <typename io_t>
__device__ __forceinline__ io_t from_float(float value) {
  return static_cast<io_t>(value);
}

template <int HeadSize, typename io_t>
__global__ __launch_bounds__(HeadSize, 1)
void statetune_tmix_wkv7_recurrent_fp32io16_forward_kernel(
    int num_heads,
    const int* __restrict__ sequence_chunk_offsets,
    const int* __restrict__ chunk_token_starts,
    const int* __restrict__ chunk_token_ends,
    float* __restrict__ state_ptr,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ decay_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr,
    const io_t* __restrict__ b_ptr,
    io_t* __restrict__ output_ptr,
    float* __restrict__ boundary_ptr,
    float* __restrict__ state_dot_a_ptr,
    float scale) {
  const int head_index = static_cast<int>(blockIdx.x);
  const int sequence_index = static_cast<int>(blockIdx.y);
  const int value_index = static_cast<int>(threadIdx.x);

  const int64_t state_base =
      (static_cast<int64_t>(sequence_index) * num_heads + head_index) *
      HeadSize * HeadSize;
  float state[HeadSize];
#pragma unroll
  for (int key_index = 0; key_index < HeadSize; ++key_index) {
    state[key_index] =
        state_ptr[state_base + key_index * HeadSize + value_index];
  }

  __shared__ float r[HeadSize];
  __shared__ float decay[HeadSize];
  __shared__ float k[HeadSize];
  __shared__ float a[HeadSize];
  __shared__ float b[HeadSize];

  const int chunk_start = sequence_chunk_offsets[sequence_index];
  const int chunk_end = sequence_chunk_offsets[sequence_index + 1];
  for (int chunk_index = chunk_start;
       chunk_index < chunk_end;
       ++chunk_index) {
    const int64_t boundary_base =
        (static_cast<int64_t>(chunk_index) * num_heads + head_index) *
        HeadSize * HeadSize;
#pragma unroll
    for (int key_index = 0; key_index < HeadSize; ++key_index) {
      boundary_ptr[
          boundary_base + key_index * HeadSize + value_index] =
          state[key_index];
    }

    const int token_start = chunk_token_starts[chunk_index];
    const int token_end = chunk_token_ends[chunk_index];
    for (int token_index = token_start;
         token_index < token_end;
         ++token_index) {
      const int64_t input_index =
          (static_cast<int64_t>(token_index) * num_heads + head_index) *
              HeadSize +
          value_index;
      r[value_index] = to_float(r_ptr[input_index]);
      decay[value_index] = recurrent_retention(
          to_float(decay_ptr[input_index]));
      k[value_index] = to_float(k_ptr[input_index]);
      a[value_index] = to_float(a_ptr[input_index]);
      b[value_index] = to_float(b_ptr[input_index]);
      __syncthreads();

      float state_dot_a = 0.0f;
#pragma unroll
      for (int key_index = 0; key_index < HeadSize; ++key_index) {
        state_dot_a = fmaf(a[key_index], state[key_index], state_dot_a);
      }
      state_dot_a_ptr[input_index] = state_dot_a;

      const float value = to_float(v_ptr[input_index]);
      float output = 0.0f;
#pragma unroll
      for (int key_index = 0; key_index < HeadSize; ++key_index) {
        const float updated =
            decay[key_index] * state[key_index] +
            b[key_index] * state_dot_a +
            k[key_index] * value;
        state[key_index] = updated;
        output = fmaf(r[key_index], updated, output);
      }
      output_ptr[input_index] = from_float<io_t>(scale * output);
      __syncthreads();
    }
  }

#pragma unroll
  for (int key_index = 0; key_index < HeadSize; ++key_index) {
    state_ptr[state_base + key_index * HeadSize + value_index] =
        state[key_index];
  }
}

template <int HeadSize, typename io_t>
void launch_statetune_tmix_wkv7_recurrent_fp32io16_forward(
    int num_sequences,
    int num_heads,
    const torch::stable::Tensor& sequence_chunk_offsets,
    const torch::stable::Tensor& chunk_token_starts,
    const torch::stable::Tensor& chunk_token_ends,
    torch::stable::Tensor& state,
    const torch::stable::Tensor& r,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    torch::stable::Tensor& output,
    torch::stable::Tensor& boundary,
    torch::stable::Tensor& state_dot_a,
    float scale,
    cudaStream_t stream) {
  statetune_tmix_wkv7_recurrent_fp32io16_forward_kernel<HeadSize, io_t>
      <<<dim3(num_heads, num_sequences), HeadSize, 0, stream>>>(
          num_heads,
          sequence_chunk_offsets.mutable_data_ptr<int>(),
          chunk_token_starts.mutable_data_ptr<int>(),
          chunk_token_ends.mutable_data_ptr<int>(),
          state.mutable_data_ptr<float>(),
          r.mutable_data_ptr<io_t>(),
          decay.mutable_data_ptr<io_t>(),
          k.mutable_data_ptr<io_t>(),
          v.mutable_data_ptr<io_t>(),
          a.mutable_data_ptr<io_t>(),
          b.mutable_data_ptr<io_t>(),
          output.mutable_data_ptr<io_t>(),
          boundary.mutable_data_ptr<float>(),
          state_dot_a.mutable_data_ptr<float>(),
          scale);
}

}  // namespace

void statetune_tmix_wkv7_recurrent_fp32io16_forward_cuda_impl(
    torch::stable::Tensor sequence_chunk_offsets,
    torch::stable::Tensor chunk_token_starts,
    torch::stable::Tensor chunk_token_ends,
    torch::stable::Tensor state,
    torch::stable::Tensor r,
    torch::stable::Tensor decay,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor a,
    torch::stable::Tensor b,
    torch::stable::Tensor output,
    torch::stable::Tensor boundary,
    torch::stable::Tensor state_dot_a,
    double scale) {
  const torch::stable::accelerator::DeviceGuard device_guard(state.device().index());
  const auto stream = flashrwkv2::validation::current_cuda_stream();
  const int num_sequences =
      static_cast<int>(sequence_chunk_offsets.numel() - 1);
  const int num_heads = static_cast<int>(state.size(1));

  auto dispatch_launch = [&](auto scalar_value) {
    using scalar_t = decltype(scalar_value);
        switch (state.size(2)) {
          case 64:
            launch_statetune_tmix_wkv7_recurrent_fp32io16_forward<64, scalar_t>(
                num_sequences, num_heads, sequence_chunk_offsets,
                chunk_token_starts, chunk_token_ends, state, r, decay,
                k, v, a, b, output, boundary, state_dot_a,
                static_cast<float>(scale), stream);
            break;
          case 128:
            launch_statetune_tmix_wkv7_recurrent_fp32io16_forward<128, scalar_t>(
                num_sequences, num_heads, sequence_chunk_offsets,
                chunk_token_starts, chunk_token_ends, state, r, decay,
                k, v, a, b, output, boundary, state_dot_a,
                static_cast<float>(scale), stream);
            break;
          case 256:
            launch_statetune_tmix_wkv7_recurrent_fp32io16_forward<256, scalar_t>(
                num_sequences, num_heads, sequence_chunk_offsets,
                chunk_token_starts, chunk_token_ends, state, r, decay,
                k, v, a, b, output, boundary, state_dot_a,
                static_cast<float>(scale), stream);
            break;
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

void statetune_tmix_wkv7_recurrent_fp32io16_forward_cuda(
    torch::stable::Tensor sequence_chunk_offsets,
    torch::stable::Tensor chunk_token_starts,
    torch::stable::Tensor chunk_token_ends,
    torch::stable::Tensor state,
    torch::stable::Tensor r,
    torch::stable::Tensor decay_logits,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor a,
    torch::stable::Tensor b,
    torch::stable::Tensor output,
    torch::stable::Tensor boundary,
    torch::stable::Tensor state_dot_a,
    double scale) {
  statetune_tmix_wkv7_recurrent_fp32io16_forward_cuda_impl(
      sequence_chunk_offsets, chunk_token_starts, chunk_token_ends, state, r,
      decay_logits, k, v, a, b, output, boundary, state_dot_a, scale);
}
