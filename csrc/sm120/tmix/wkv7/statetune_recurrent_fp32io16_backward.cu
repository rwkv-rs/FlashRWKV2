// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the RWKV-LM project
// Mechanically copied from RWKV-LM train_temp at revision
// 952102498e9ed367ea0a59ee64106916d474d30f.
// StateTune owns this body so its public binding is independent of pretrain.
// Raw-decay training backward: initial-state gradients, checkpoint replay
// and decay_logits gradients remain part of the training-only contract.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "../../../sm120/tmix/wkv7/recurrent_decay.cuh"

namespace {

constexpr int kHeadSize = 64;

using flashrwkv2::wkv7::recurrent_retention;
using flashrwkv2::wkv7::recurrent_retention_log_derivative;

template <typename io_t>
__device__ __forceinline__ float to_float(io_t value) {
  return static_cast<float>(value);
}

template <typename io_t>
__device__ __forceinline__ io_t from_float(float value) {
  return static_cast<io_t>(value);
}

struct BackwardShared {
  float r[kHeadSize];
  float decay[kHeadSize];
  float k[kHeadSize];
  float v[kHeadSize];
  float a[kHeadSize];
  float b[kHeadSize];
  float grad_output[kHeadSize];
  float state_dot_a[kHeadSize];
  float adjoint_dot_b[kHeadSize];
  float state[kHeadSize][kHeadSize];
  float adjoint[kHeadSize][kHeadSize];
};

template <typename io_t>
__global__ __launch_bounds__(kHeadSize, 1)
void statetune_tmix_wkv7_recurrent_fp32io16_backward_kernel(
    int num_heads,
    const int* __restrict__ sequence_chunk_offsets,
    const int* __restrict__ chunk_token_starts,
    const int* __restrict__ chunk_token_ends,
    const float* __restrict__ final_state_ptr,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ decay_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr,
    const io_t* __restrict__ b_ptr,
    const float* __restrict__ state_dot_a_ptr,
    const io_t* __restrict__ grad_output_ptr,
    const float* __restrict__ grad_final_state_ptr,
    const float* __restrict__ boundary_ptr,
    io_t* __restrict__ grad_r_ptr,
    io_t* __restrict__ grad_decay_ptr,
    io_t* __restrict__ grad_k_ptr,
    io_t* __restrict__ grad_v_ptr,
    io_t* __restrict__ grad_a_ptr,
    io_t* __restrict__ grad_b_ptr,
    float* __restrict__ grad_initial_state_ptr,
    float scale) {
  const int head_index = static_cast<int>(blockIdx.x);
  const int sequence_index = static_cast<int>(blockIdx.y);
  const int value_index = static_cast<int>(threadIdx.x);
  __shared__ BackwardShared shared;

  const int64_t state_base =
      (static_cast<int64_t>(sequence_index) * num_heads + head_index) *
      kHeadSize * kHeadSize;
  float state[kHeadSize];
  float adjoint[kHeadSize];
#pragma unroll
  for (int key_index = 0; key_index < kHeadSize; ++key_index) {
    state[key_index] =
        final_state_ptr[
            state_base + key_index * kHeadSize + value_index];
    adjoint[key_index] =
        grad_final_state_ptr == nullptr
        ? 0.0f
        : grad_final_state_ptr[
              state_base + key_index * kHeadSize + value_index];
  }

  const int sequence_chunk_start =
      sequence_chunk_offsets[sequence_index];
  const int sequence_chunk_end =
      sequence_chunk_offsets[sequence_index + 1];
  for (int chunk_index = sequence_chunk_end - 1;
       chunk_index >= sequence_chunk_start;
       --chunk_index) {
    if (chunk_index + 1 < sequence_chunk_end) {
      const int64_t boundary_base =
          (static_cast<int64_t>(chunk_index + 1) * num_heads + head_index) *
          kHeadSize * kHeadSize;
#pragma unroll
      for (int key_index = 0; key_index < kHeadSize; ++key_index) {
        state[key_index] =
            boundary_ptr[
                boundary_base + key_index * kHeadSize + value_index];
      }
    }

    const int token_start = chunk_token_starts[chunk_index];
    const int token_end = chunk_token_ends[chunk_index];
    for (int token_index = token_end - 1;
         token_index >= token_start;
         --token_index) {
      const int64_t input_index =
          (static_cast<int64_t>(token_index) * num_heads + head_index) *
              kHeadSize +
          value_index;
      shared.r[value_index] = to_float(r_ptr[input_index]);
      shared.decay[value_index] = recurrent_retention(
          to_float(decay_ptr[input_index]));
      shared.k[value_index] = to_float(k_ptr[input_index]);
      shared.v[value_index] = to_float(v_ptr[input_index]);
      shared.a[value_index] = to_float(a_ptr[input_index]);
      shared.b[value_index] = to_float(b_ptr[input_index]);
      shared.grad_output[value_index] =
          grad_output_ptr == nullptr
          ? 0.0f
          : to_float(grad_output_ptr[input_index]);
      shared.state_dot_a[value_index] = state_dot_a_ptr[input_index];
      __syncthreads();

      const float output_adjoint = shared.grad_output[value_index];
#pragma unroll
      for (int key_index = 0; key_index < kHeadSize; ++key_index) {
        adjoint[key_index] = fmaf(
            scale * shared.r[key_index],
            output_adjoint,
            adjoint[key_index]);
        shared.state[key_index][value_index] = state[key_index];
        shared.adjoint[key_index][value_index] = adjoint[key_index];
      }
      __syncthreads();

      float grad_r = 0.0f;
      float grad_k = 0.0f;
      float grad_b = 0.0f;
#pragma unroll
      for (int other_value = 0;
           other_value < kHeadSize;
           ++other_value) {
        grad_r = fmaf(
            shared.state[value_index][other_value],
            shared.grad_output[other_value],
            grad_r);
        grad_k = fmaf(
            shared.adjoint[value_index][other_value],
            shared.v[other_value],
            grad_k);
        grad_b = fmaf(
            shared.adjoint[value_index][other_value],
            shared.state_dot_a[other_value],
            grad_b);
      }
      __syncthreads();

      float grad_v = 0.0f;
      float adjoint_dot_b = 0.0f;
#pragma unroll
      for (int key_index = 0; key_index < kHeadSize; ++key_index) {
        grad_v = fmaf(
            adjoint[key_index],
            shared.k[key_index],
            grad_v);
        adjoint_dot_b = fmaf(
            adjoint[key_index],
            shared.b[key_index],
            adjoint_dot_b);
        state[key_index] =
            (
                state[key_index] -
                shared.k[key_index] * shared.v[value_index] -
                shared.b[key_index] *
                    shared.state_dot_a[value_index]
            ) /
            shared.decay[key_index];
        shared.state[key_index][value_index] = state[key_index];
      }
      shared.adjoint_dot_b[value_index] = adjoint_dot_b;
      __syncthreads();

      float grad_decay_logits = 0.0f;
      float grad_a = 0.0f;
#pragma unroll
      for (int other_value = 0;
           other_value < kHeadSize;
           ++other_value) {
        grad_decay_logits = fmaf(
            shared.adjoint[value_index][other_value],
            shared.state[value_index][other_value],
            grad_decay_logits);
        grad_a = fmaf(
            shared.state[value_index][other_value],
            shared.adjoint_dot_b[other_value],
            grad_a);
      }
      grad_decay_logits *= shared.decay[value_index];
      grad_decay_logits *= recurrent_retention_log_derivative(
          to_float(decay_ptr[input_index]));

      if (grad_r_ptr != nullptr) {
        grad_r_ptr[input_index] = from_float<io_t>(scale * grad_r);
      }
      if (grad_decay_ptr != nullptr) {
        grad_decay_ptr[input_index] = from_float<io_t>(grad_decay_logits);
      }
      if (grad_k_ptr != nullptr) {
        grad_k_ptr[input_index] = from_float<io_t>(grad_k);
      }
      if (grad_v_ptr != nullptr) {
        grad_v_ptr[input_index] = from_float<io_t>(grad_v);
      }
      if (grad_a_ptr != nullptr) {
        grad_a_ptr[input_index] = from_float<io_t>(grad_a);
      }
      if (grad_b_ptr != nullptr) {
        grad_b_ptr[input_index] = from_float<io_t>(grad_b);
      }

#pragma unroll
      for (int key_index = 0; key_index < kHeadSize; ++key_index) {
        adjoint[key_index] = fmaf(
            shared.a[key_index],
            adjoint_dot_b,
            shared.decay[key_index] * adjoint[key_index]);
      }
      __syncthreads();
    }
  }

  if (grad_initial_state_ptr != nullptr) {
#pragma unroll
    for (int key_index = 0; key_index < kHeadSize; ++key_index) {
      grad_initial_state_ptr[
          state_base + key_index * kHeadSize + value_index] =
          adjoint[key_index];
    }
  }
}

template <int HeadSize>
struct LargeBackwardShared {
  static constexpr int kWarps = HeadSize / 32;
  float r[HeadSize];
  float decay[HeadSize];
  float k[HeadSize];
  float v[HeadSize];
  float a[HeadSize];
  float b[HeadSize];
  float grad_output[HeadSize];
  float state_dot_a[HeadSize];
  float reduction[3][kWarps];
};

struct Reduction3 {
  float x;
  float y;
  float z;
};

template <int HeadSize>
__device__ __forceinline__ Reduction3 block_sum3(
    Reduction3 value,
    float (&scratch)[3][HeadSize / 32]) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value.x += __shfl_down_sync(0xffffffffu, value.x, offset);
    value.y += __shfl_down_sync(0xffffffffu, value.y, offset);
    value.z += __shfl_down_sync(0xffffffffu, value.z, offset);
  }
  if (lane == 0) {
    scratch[0][warp] = value.x;
    scratch[1][warp] = value.y;
    scratch[2][warp] = value.z;
  }
  __syncthreads();

  if (warp == 0) {
    constexpr int warps = HeadSize / 32;
    value.x = lane < warps ? scratch[0][lane] : 0.0f;
    value.y = lane < warps ? scratch[1][lane] : 0.0f;
    value.z = lane < warps ? scratch[2][lane] : 0.0f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value.x += __shfl_down_sync(0xffffffffu, value.x, offset);
      value.y += __shfl_down_sync(0xffffffffu, value.y, offset);
      value.z += __shfl_down_sync(0xffffffffu, value.z, offset);
    }
    if (lane == 0) {
      scratch[0][0] = value.x;
      scratch[1][0] = value.y;
      scratch[2][0] = value.z;
    }
  }
  __syncthreads();
  return Reduction3{scratch[0][0], scratch[1][0], scratch[2][0]};
}

template <int HeadSize, typename io_t>
__global__ __launch_bounds__(HeadSize, 1)
void statetune_tmix_wkv7_recurrent_fp32io16_backward_large_kernel(
    int num_heads,
    const int* __restrict__ sequence_chunk_offsets,
    const int* __restrict__ chunk_token_starts,
    const int* __restrict__ chunk_token_ends,
    const float* __restrict__ final_state_ptr,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ decay_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr,
    const io_t* __restrict__ b_ptr,
    const float* __restrict__ state_dot_a_ptr,
    const io_t* __restrict__ grad_output_ptr,
    const float* __restrict__ grad_final_state_ptr,
    const float* __restrict__ boundary_ptr,
    io_t* __restrict__ grad_r_ptr,
    io_t* __restrict__ grad_decay_ptr,
    io_t* __restrict__ grad_k_ptr,
    io_t* __restrict__ grad_v_ptr,
    io_t* __restrict__ grad_a_ptr,
    io_t* __restrict__ grad_b_ptr,
    float* __restrict__ grad_initial_state_ptr,
    float scale) {
  const int head_index = static_cast<int>(blockIdx.x);
  const int sequence_index = static_cast<int>(blockIdx.y);
  const int value_index = static_cast<int>(threadIdx.x);
  __shared__ LargeBackwardShared<HeadSize> shared;

  const int64_t state_base =
      (static_cast<int64_t>(sequence_index) * num_heads + head_index) *
      HeadSize * HeadSize;
  float state[HeadSize];
  float adjoint[HeadSize];
#pragma unroll
  for (int key_index = 0; key_index < HeadSize; ++key_index) {
    state[key_index] =
        final_state_ptr[state_base + key_index * HeadSize + value_index];
    adjoint[key_index] =
        grad_final_state_ptr == nullptr
        ? 0.0f
        : grad_final_state_ptr[
              state_base + key_index * HeadSize + value_index];
  }

  const int sequence_chunk_start =
      sequence_chunk_offsets[sequence_index];
  const int sequence_chunk_end =
      sequence_chunk_offsets[sequence_index + 1];
  for (int chunk_index = sequence_chunk_end - 1;
       chunk_index >= sequence_chunk_start;
       --chunk_index) {
    if (chunk_index + 1 < sequence_chunk_end) {
      const int64_t boundary_base =
          (static_cast<int64_t>(chunk_index + 1) * num_heads + head_index) *
          HeadSize * HeadSize;
#pragma unroll
      for (int key_index = 0; key_index < HeadSize; ++key_index) {
        state[key_index] = boundary_ptr[
            boundary_base + key_index * HeadSize + value_index];
      }
    }

    const int token_start = chunk_token_starts[chunk_index];
    const int token_end = chunk_token_ends[chunk_index];
    for (int token_index = token_end - 1;
         token_index >= token_start;
         --token_index) {
      const int64_t input_base =
          (static_cast<int64_t>(token_index) * num_heads + head_index) *
          HeadSize;
      const int64_t input_index = input_base + value_index;
      shared.r[value_index] = to_float(r_ptr[input_index]);
      shared.decay[value_index] = recurrent_retention(
          to_float(decay_ptr[input_index]));
      shared.k[value_index] = to_float(k_ptr[input_index]);
      shared.v[value_index] = to_float(v_ptr[input_index]);
      shared.a[value_index] = to_float(a_ptr[input_index]);
      shared.b[value_index] = to_float(b_ptr[input_index]);
      shared.grad_output[value_index] =
          grad_output_ptr == nullptr
          ? 0.0f
          : to_float(grad_output_ptr[input_index]);
      shared.state_dot_a[value_index] = state_dot_a_ptr[input_index];
      __syncthreads();

      const float output_adjoint = shared.grad_output[value_index];
#pragma unroll
      for (int key_index = 0; key_index < HeadSize; ++key_index) {
        adjoint[key_index] = fmaf(
            scale * shared.r[key_index],
            output_adjoint,
            adjoint[key_index]);
      }

#pragma unroll
      for (int key_index = 0; key_index < HeadSize; ++key_index) {
        const Reduction3 gradients = block_sum3<HeadSize>(
            Reduction3{
                state[key_index] * shared.grad_output[value_index],
                adjoint[key_index] * shared.v[value_index],
                adjoint[key_index] * shared.state_dot_a[value_index],
            },
            shared.reduction);
        if (value_index == 0) {
          if (grad_r_ptr != nullptr) {
            grad_r_ptr[input_base + key_index] =
                from_float<io_t>(scale * gradients.x);
          }
          if (grad_k_ptr != nullptr) {
            grad_k_ptr[input_base + key_index] =
                from_float<io_t>(gradients.y);
          }
          if (grad_b_ptr != nullptr) {
            grad_b_ptr[input_base + key_index] =
                from_float<io_t>(gradients.z);
          }
        }
      }

      float grad_v = 0.0f;
      float adjoint_dot_b = 0.0f;
#pragma unroll
      for (int key_index = 0; key_index < HeadSize; ++key_index) {
        grad_v = fmaf(
            adjoint[key_index], shared.k[key_index], grad_v);
        adjoint_dot_b = fmaf(
            adjoint[key_index], shared.b[key_index], adjoint_dot_b);
        state[key_index] =
            (
                state[key_index] -
                shared.k[key_index] * shared.v[value_index] -
                shared.b[key_index] * shared.state_dot_a[value_index]
            ) /
            shared.decay[key_index];
      }
      if (grad_v_ptr != nullptr) {
        grad_v_ptr[input_index] = from_float<io_t>(grad_v);
      }

#pragma unroll
      for (int key_index = 0; key_index < HeadSize; ++key_index) {
        const Reduction3 gradients = block_sum3<HeadSize>(
            Reduction3{
                adjoint[key_index] * state[key_index],
                state[key_index] * adjoint_dot_b,
                0.0f,
            },
            shared.reduction);
        if (value_index == 0) {
          if (grad_decay_ptr != nullptr) {
            float grad_decay = gradients.x * shared.decay[key_index];
            grad_decay *= recurrent_retention_log_derivative(
                to_float(decay_ptr[input_base + key_index]));
            grad_decay_ptr[input_base + key_index] =
                from_float<io_t>(grad_decay);
          }
          if (grad_a_ptr != nullptr) {
            grad_a_ptr[input_base + key_index] =
                from_float<io_t>(gradients.y);
          }
        }
      }

#pragma unroll
      for (int key_index = 0; key_index < HeadSize; ++key_index) {
        adjoint[key_index] = fmaf(
            shared.a[key_index],
            adjoint_dot_b,
            shared.decay[key_index] * adjoint[key_index]);
      }
      __syncthreads();
    }
  }

  if (grad_initial_state_ptr != nullptr) {
#pragma unroll
    for (int key_index = 0; key_index < HeadSize; ++key_index) {
      grad_initial_state_ptr[
          state_base + key_index * HeadSize + value_index] =
          adjoint[key_index];
    }
  }
}

template <int HeadSize, typename io_t>
void launch_statetune_tmix_wkv7_recurrent_fp32io16_backward(
    int num_sequences,
    int num_heads,
    const torch::Tensor& sequence_chunk_offsets,
    const torch::Tensor& chunk_token_starts,
    const torch::Tensor& chunk_token_ends,
    const torch::Tensor& final_state,
    const torch::Tensor& r,
    const torch::Tensor& decay,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& state_dot_a,
    const torch::Tensor& grad_output,
    const torch::Tensor& grad_final_state,
    const torch::Tensor& boundary,
    torch::Tensor& grad_r,
    torch::Tensor& grad_decay,
    torch::Tensor& grad_k,
    torch::Tensor& grad_v,
    torch::Tensor& grad_a,
    torch::Tensor& grad_b,
    torch::Tensor& grad_initial_state,
    float scale,
    cudaStream_t stream) {
  if constexpr (HeadSize == kHeadSize) {
    statetune_tmix_wkv7_recurrent_fp32io16_backward_kernel<io_t>
        <<<dim3(num_heads, num_sequences), HeadSize, 0, stream>>>(
            num_heads,
            sequence_chunk_offsets.data_ptr<int>(),
            chunk_token_starts.data_ptr<int>(),
            chunk_token_ends.data_ptr<int>(),
            final_state.data_ptr<float>(),
            r.data_ptr<io_t>(),
            decay.data_ptr<io_t>(),
            k.data_ptr<io_t>(),
            v.data_ptr<io_t>(),
            a.data_ptr<io_t>(),
            b.data_ptr<io_t>(),
            state_dot_a.data_ptr<float>(),
            grad_output.defined() ? grad_output.data_ptr<io_t>() : nullptr,
            grad_final_state.defined()
                ? grad_final_state.data_ptr<float>()
                : nullptr,
            boundary.data_ptr<float>(),
            grad_r.defined() ? grad_r.data_ptr<io_t>() : nullptr,
            grad_decay.defined()
                ? grad_decay.data_ptr<io_t>()
                : nullptr,
            grad_k.defined() ? grad_k.data_ptr<io_t>() : nullptr,
            grad_v.defined() ? grad_v.data_ptr<io_t>() : nullptr,
            grad_a.defined() ? grad_a.data_ptr<io_t>() : nullptr,
            grad_b.defined() ? grad_b.data_ptr<io_t>() : nullptr,
            grad_initial_state.defined()
                ? grad_initial_state.data_ptr<float>()
                : nullptr,
            scale);
  } else {
    statetune_tmix_wkv7_recurrent_fp32io16_backward_large_kernel<HeadSize, io_t>
        <<<dim3(num_heads, num_sequences), HeadSize, 0, stream>>>(
            num_heads,
            sequence_chunk_offsets.data_ptr<int>(),
            chunk_token_starts.data_ptr<int>(),
            chunk_token_ends.data_ptr<int>(),
            final_state.data_ptr<float>(),
            r.data_ptr<io_t>(),
            decay.data_ptr<io_t>(),
            k.data_ptr<io_t>(),
            v.data_ptr<io_t>(),
            a.data_ptr<io_t>(),
            b.data_ptr<io_t>(),
            state_dot_a.data_ptr<float>(),
            grad_output.defined() ? grad_output.data_ptr<io_t>() : nullptr,
            grad_final_state.defined()
                ? grad_final_state.data_ptr<float>()
                : nullptr,
            boundary.data_ptr<float>(),
            grad_r.defined() ? grad_r.data_ptr<io_t>() : nullptr,
            grad_decay.defined()
                ? grad_decay.data_ptr<io_t>()
                : nullptr,
            grad_k.defined() ? grad_k.data_ptr<io_t>() : nullptr,
            grad_v.defined() ? grad_v.data_ptr<io_t>() : nullptr,
            grad_a.defined() ? grad_a.data_ptr<io_t>() : nullptr,
            grad_b.defined() ? grad_b.data_ptr<io_t>() : nullptr,
            grad_initial_state.defined()
                ? grad_initial_state.data_ptr<float>()
                : nullptr,
            scale);
  }
}

}  // namespace

void statetune_tmix_wkv7_recurrent_fp32io16_backward_cuda_impl(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor final_state,
    torch::Tensor r,
    torch::Tensor decay,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor state_dot_a,
    torch::Tensor grad_output,
    torch::Tensor grad_final_state,
    torch::Tensor boundary,
    torch::Tensor grad_r,
    torch::Tensor grad_decay,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor grad_a,
    torch::Tensor grad_b,
    torch::Tensor grad_initial_state,
    double scale) {
  const c10::cuda::CUDAGuard device_guard(final_state.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int num_sequences =
      static_cast<int>(sequence_chunk_offsets.numel() - 1);
  const int num_heads = static_cast<int>(final_state.size(1));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      r.scalar_type(),
      "flashrwkv2_statetune_tmix_wkv7_recurrent_fp32io16_backward",
      [&] {
        const auto launch = [&]<int HeadSize>() {
          launch_statetune_tmix_wkv7_recurrent_fp32io16_backward<HeadSize, scalar_t>(
              num_sequences,
              num_heads,
              sequence_chunk_offsets,
              chunk_token_starts,
              chunk_token_ends,
              final_state,
              r,
              decay,
              k,
              v,
              a,
              b,
              state_dot_a,
              grad_output,
              grad_final_state,
              boundary,
              grad_r,
              grad_decay,
              grad_k,
              grad_v,
              grad_a,
              grad_b,
              grad_initial_state,
              static_cast<float>(scale),
              stream);
        };
        switch (final_state.size(2)) {
          case 64:
            launch.template operator()<64>();
            break;
          case 128:
            launch.template operator()<128>();
            break;
          case 256:
            launch.template operator()<256>();
            break;
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void statetune_tmix_wkv7_recurrent_fp32io16_backward_cuda(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor final_state,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor state_dot_a,
    torch::Tensor grad_output,
    torch::Tensor grad_final_state,
    torch::Tensor boundary,
    torch::Tensor grad_r,
    torch::Tensor grad_decay_logits,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor grad_a,
    torch::Tensor grad_b,
    torch::Tensor grad_initial_state,
    double scale) {
  statetune_tmix_wkv7_recurrent_fp32io16_backward_cuda_impl(
      sequence_chunk_offsets, chunk_token_starts, chunk_token_ends,
      final_state, r, decay_logits, k, v, a, b, state_dot_a, grad_output,
      grad_final_state, boundary, grad_r, grad_decay_logits, grad_k, grad_v,
      grad_a, grad_b, grad_initial_state, scale);
}
