// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
// Canonical module owner: tmix/wkv7; RL/Infctx is the workload.
// Source revision: FlashRWKV2 pre-refactor retained RL/Infctx local snapshot
// Original path: retained RL/Infctx materialized/recompute CUDA family
// D64 is mechanically migrated from the retained RL/Infctx materialized and
// factor-recompute implementations. D128/256 add local 64-wide tiled boundary,
// propagation, output, and replay kernels under the same raw-logit contract.


#include "rl_infctx_chunk_fp32io16_replay.cuh"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "recurrent_decay.cuh"


namespace rl_infctx_recompute_detail {

constexpr int kHeadSize = 64;

using flashrwkv2::wkv7::recurrent_retention;

template <typename io_t>
__device__ __forceinline__ float to_float(io_t value) {
  return static_cast<float>(value);
}

struct FactorShared {
  float decay[kHeadSize];
  float k[kHeadSize];
  float a[kHeadSize];
  float b[kHeadSize];
};

template <typename io_t>
__global__ __launch_bounds__(kHeadSize, 2)
void scan_factor_boundaries_kernel(
    int num_heads,
    const int* __restrict__ sequence_chunk_offsets,
    const int* __restrict__ chunk_token_starts,
    const int* __restrict__ chunk_token_ends,
    const int* __restrict__ state_indices,
    float* __restrict__ state_ptr,
    const io_t* __restrict__ decay_ptr,
    const io_t* __restrict__ decay_bias_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr,
    const io_t* __restrict__ b_ptr,
    float* __restrict__ boundary_ptr) {
  const int head_index = static_cast<int>(blockIdx.x);
  const int sequence_index = static_cast<int>(blockIdx.y);
  const int value_index = static_cast<int>(threadIdx.x);
  const int state_slot = state_indices[sequence_index];
  __shared__ FactorShared shared;

  const int64_t state_base =
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
      kHeadSize * kHeadSize;
  float state[kHeadSize];
#pragma unroll
  for (int key_index = 0; key_index < kHeadSize; ++key_index) {
    state[key_index] =
        state_ptr[state_base + key_index * kHeadSize + value_index];
  }

  const int chunk_start = sequence_chunk_offsets[sequence_index];
  const int chunk_end = sequence_chunk_offsets[sequence_index + 1];
  for (int chunk_index = chunk_start;
       chunk_index < chunk_end;
       ++chunk_index) {
    const int64_t boundary_base =
        (static_cast<int64_t>(chunk_index) * num_heads + head_index) *
        kHeadSize * kHeadSize;
#pragma unroll
    for (int key_index = 0; key_index < kHeadSize; ++key_index) {
      boundary_ptr[
          boundary_base + key_index * kHeadSize + value_index] =
          state[key_index];
    }

    const int token_start = chunk_token_starts[chunk_index];
    const int token_end = chunk_token_ends[chunk_index];
    for (int token_index = token_start;
         token_index < token_end;
         ++token_index) {
      const int64_t input_index =
          (static_cast<int64_t>(token_index) * num_heads + head_index) *
              kHeadSize +
          value_index;
      float decay_input = to_float(decay_ptr[input_index]);
      if (decay_bias_ptr != nullptr) {
        decay_input += to_float(
            decay_bias_ptr[head_index * kHeadSize + value_index]);
      }
      shared.decay[value_index] =
          recurrent_retention(decay_input);
      shared.k[value_index] = to_float(k_ptr[input_index]);
      shared.a[value_index] = to_float(a_ptr[input_index]);
      shared.b[value_index] = to_float(b_ptr[input_index]);
      const float value = to_float(v_ptr[input_index]);
      __syncthreads();

      float state_dot_a = 0.0f;
#pragma unroll
      for (int key_index = 0; key_index < kHeadSize; ++key_index) {
        state_dot_a =
            fmaf(shared.a[key_index], state[key_index], state_dot_a);
      }
#pragma unroll
      for (int key_index = 0; key_index < kHeadSize; ++key_index) {
        state[key_index] = fmaf(
            shared.k[key_index],
            value,
            fmaf(
                shared.b[key_index],
                state_dot_a,
                shared.decay[key_index] * state[key_index]));
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (int key_index = 0; key_index < kHeadSize; ++key_index) {
    state_ptr[state_base + key_index * kHeadSize + value_index] =
        state[key_index];
  }
}

template <typename io_t>
void launch_factor_scan(
    int num_sequences,
    int num_heads,
    const torch::Tensor& sequence_chunk_offsets,
    const torch::Tensor& chunk_token_starts,
    const torch::Tensor& chunk_token_ends,
    const torch::Tensor& state_indices,
    torch::Tensor& state,
    const torch::Tensor& decay,
    const torch::Tensor& decay_bias,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& boundary,
    cudaStream_t stream) {
  scan_factor_boundaries_kernel<io_t>
      <<<dim3(num_heads, num_sequences), kHeadSize, 0, stream>>>(
          num_heads,
          sequence_chunk_offsets.data_ptr<int>(),
          chunk_token_starts.data_ptr<int>(),
          chunk_token_ends.data_ptr<int>(),
          state_indices.data_ptr<int>(),
          state.data_ptr<float>(),
          decay.data_ptr<io_t>(),
          decay_bias.defined() ? decay_bias.data_ptr<io_t>() : nullptr,
          k.data_ptr<io_t>(),
          v.data_ptr<io_t>(),
          a.data_ptr<io_t>(),
          b.data_ptr<io_t>(),
          boundary.data_ptr<float>());
}

template <typename io_t>
void launch_recompute_chunk(
    int num_sequences,
    int num_chunks,
    int num_heads,
    const torch::Tensor& sequence_chunk_offsets,
    const torch::Tensor& chunk_token_starts,
    const torch::Tensor& chunk_token_ends,
    const torch::Tensor& state_indices,
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& decay,
    const torch::Tensor& decay_bias,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& output,
    torch::Tensor& boundary,
    float scale,
    cudaStream_t stream) {
  launch_factor_scan<io_t>(
      num_sequences,
      num_heads,
      sequence_chunk_offsets,
      chunk_token_starts,
      chunk_token_ends,
      state_indices,
      state,
      decay,
      decay_bias,
      k,
      v,
      a,
      b,
      boundary,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  launch_rl_infctx_chunk_replay_fp32_from_decay_logits(
      num_chunks, num_heads, chunk_token_starts, chunk_token_ends, boundary,
      r, decay, decay_bias, k, v, a, b, output, nullptr, scale, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace rl_infctx_recompute_detail

namespace rl_infctx_materialized_detail {

constexpr int kHeadSize = 64;

using flashrwkv2::wkv7::recurrent_retention;

template <typename io_t>
__device__ __forceinline__ float to_float(io_t value) {
  return static_cast<float>(value);
}

template <int Stages>
struct BuildShared {
  float transform[kHeadSize][kHeadSize];
  float bias[kHeadSize][kHeadSize];
  float decay[Stages][kHeadSize];
  float a[Stages][kHeadSize];
  float b[Stages][kHeadSize];
  float k[Stages][kHeadSize];
  float v[Stages][kHeadSize];
};

template <typename io_t, int Stages>
__device__ __forceinline__ void load_build_token(
    BuildShared<Stages>& shared,
    int buffer,
    int column,
    int token_index,
    int head_index,
    int num_heads,
    const io_t* decay_ptr,
    const io_t* decay_bias_ptr,
    const io_t* k_ptr,
    const io_t* v_ptr,
    const io_t* a_ptr,
    const io_t* b_ptr) {
  const int64_t input_index =
      (static_cast<int64_t>(token_index) * num_heads + head_index) *
          kHeadSize +
      column;
  float decay_input = to_float(decay_ptr[input_index]);
  if (decay_bias_ptr != nullptr) {
    decay_input += to_float(
        decay_bias_ptr[head_index * kHeadSize + column]);
  }
  shared.decay[buffer][column] = recurrent_retention(decay_input);
  shared.k[buffer][column] = to_float(k_ptr[input_index]);
  shared.v[buffer][column] = to_float(v_ptr[input_index]);
  shared.a[buffer][column] = to_float(a_ptr[input_index]);
  shared.b[buffer][column] = to_float(b_ptr[input_index]);
}

template <int Stages>
__device__ __forceinline__ void update_transform_column(
    BuildShared<Stages>& shared,
    int buffer,
    int column) {
  float transform_dot_a = 0.0f;
#pragma unroll
  for (int row = 0; row < kHeadSize; ++row) {
    transform_dot_a = fmaf(
        shared.a[buffer][row],
        shared.transform[row][column],
        transform_dot_a);
  }
#pragma unroll
  for (int row = 0; row < kHeadSize; ++row) {
    shared.transform[row][column] = fmaf(
        shared.b[buffer][row],
        transform_dot_a,
        shared.decay[buffer][row] * shared.transform[row][column]);
  }
}

template <int Stages>
__device__ __forceinline__ void update_bias_column(
    BuildShared<Stages>& shared,
    int buffer,
    int column) {
  float bias_dot_a = 0.0f;
#pragma unroll
  for (int row = 0; row < kHeadSize; ++row) {
    bias_dot_a = fmaf(
        shared.a[buffer][row],
        shared.bias[row][column],
        bias_dot_a);
  }
  const float value = shared.v[buffer][column];
#pragma unroll
  for (int row = 0; row < kHeadSize; ++row) {
    shared.bias[row][column] = fmaf(
        shared.k[buffer][row],
        value,
        fmaf(
            shared.b[buffer][row],
            bias_dot_a,
            shared.decay[buffer][row] * shared.bias[row][column]));
  }
}

template <
    typename io_t,
    int BuildWarps,
    int Stages>
__global__ __launch_bounds__(BuildWarps * 32, 2)
void build_transforms_kernel(
    int num_heads,
    const int* __restrict__ chunk_token_starts,
    const int* __restrict__ chunk_token_ends,
    const io_t* __restrict__ decay_ptr,
    const io_t* __restrict__ decay_bias_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr,
    const io_t* __restrict__ b_ptr,
    float* __restrict__ transform_ptr,
    float* __restrict__ bias_ptr) {
  static_assert(BuildWarps == 2 || BuildWarps == 4);
  static_assert(Stages == 1 || Stages == 2);
  static_assert(Stages == 1 || BuildWarps == 4);
  const int linear_block = static_cast<int>(blockIdx.x);
  const int chunk_index = linear_block / num_heads;
  const int head_index = linear_block % num_heads;
  const int thread = static_cast<int>(threadIdx.x);
  __shared__ BuildShared<Stages> shared;

  if (thread < kHeadSize) {
#pragma unroll
    for (int row = 0; row < kHeadSize; ++row) {
      shared.transform[row][thread] = row == thread ? 1.0f : 0.0f;
      shared.bias[row][thread] = 0.0f;
    }
  }
  __syncthreads();

  const int token_start = chunk_token_starts[chunk_index];
  const int token_end = chunk_token_ends[chunk_index];
  if (thread < kHeadSize) {
    load_build_token<io_t, Stages>(
        shared,
        0,
        thread,
        token_start,
        head_index,
        num_heads,
        decay_ptr,
        decay_bias_ptr,
        k_ptr,
        v_ptr,
        a_ptr,
        b_ptr);
  }
  __syncthreads();

  for (int token_offset = 0;
       token_offset < token_end - token_start;
       ++token_offset) {
    const int buffer = Stages == 1 ? 0 : token_offset & 1;
    if constexpr (Stages == 2) {
      if (thread >= kHeadSize &&
          token_start + token_offset + 1 < token_end) {
        load_build_token<io_t, Stages>(
            shared,
            buffer ^ 1,
            thread - kHeadSize,
            token_start + token_offset + 1,
            head_index,
            num_heads,
            decay_ptr,
            decay_bias_ptr,
            k_ptr,
            v_ptr,
            a_ptr,
            b_ptr);
      }
      if (thread < kHeadSize) {
        update_transform_column(shared, buffer, thread);
        update_bias_column(shared, buffer, thread);
      }
    } else if constexpr (BuildWarps == 4) {
      if (thread < kHeadSize) {
        update_transform_column(shared, buffer, thread);
      } else {
        update_bias_column(shared, buffer, thread - kHeadSize);
      }
    } else {
      update_transform_column(shared, buffer, thread);
      update_bias_column(shared, buffer, thread);
    }
    __syncthreads();

    if constexpr (Stages == 1) {
      if (thread < kHeadSize &&
          token_start + token_offset + 1 < token_end) {
        load_build_token<io_t, Stages>(
            shared,
            0,
            thread,
            token_start + token_offset + 1,
            head_index,
            num_heads,
            decay_ptr,
            decay_bias_ptr,
            k_ptr,
            v_ptr,
            a_ptr,
            b_ptr);
      }
      __syncthreads();
    }
  }

  if (thread < kHeadSize) {
    const int64_t workspace_base =
        (static_cast<int64_t>(chunk_index) * num_heads + head_index) *
        kHeadSize * kHeadSize;
#pragma unroll
    for (int row = 0; row < kHeadSize; ++row) {
      const int64_t workspace_index =
          workspace_base + row * kHeadSize + thread;
      transform_ptr[workspace_index] = shared.transform[row][thread];
      bias_ptr[workspace_index] = shared.bias[row][thread];
    }
  }
}

template <int StateTile>
struct ScanShared {
  float transform[StateTile][kHeadSize];
  float next_state[kHeadSize][kHeadSize];
};

template <int StateTile>
__global__ __launch_bounds__(kHeadSize, 2)
void scan_boundaries_kernel(
    int num_heads,
    const int* __restrict__ sequence_chunk_offsets,
    const int* __restrict__ state_indices,
    float* __restrict__ state_ptr,
    const float* __restrict__ transform_ptr,
    const float* __restrict__ bias_ptr,
    float* __restrict__ boundary_ptr) {
  static_assert(kHeadSize % StateTile == 0);
  const int linear_block = static_cast<int>(blockIdx.x);
  const int sequence_index = linear_block / num_heads;
  const int head_index = linear_block % num_heads;
  const int value_index = static_cast<int>(threadIdx.x);
  const int state_slot = state_indices[sequence_index];
  __shared__ ScanShared<StateTile> shared;

  const int64_t state_base =
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
      kHeadSize * kHeadSize;
  float state[kHeadSize];
#pragma unroll
  for (int key_index = 0; key_index < kHeadSize; ++key_index) {
    state[key_index] =
        state_ptr[state_base + key_index * kHeadSize + value_index];
  }

  const int chunk_start = sequence_chunk_offsets[sequence_index];
  const int chunk_end = sequence_chunk_offsets[sequence_index + 1];
  for (int chunk_index = chunk_start;
       chunk_index < chunk_end;
       ++chunk_index) {
    const int64_t workspace_base =
        (static_cast<int64_t>(chunk_index) * num_heads + head_index) *
        kHeadSize * kHeadSize;
#pragma unroll
    for (int key_index = 0; key_index < kHeadSize; ++key_index) {
      boundary_ptr[
          workspace_base + key_index * kHeadSize + value_index] =
          state[key_index];
    }

#pragma unroll
    for (int tile_start = 0;
         tile_start < kHeadSize;
         tile_start += StateTile) {
#pragma unroll
      for (int tile_row = 0; tile_row < StateTile; ++tile_row) {
        shared.transform[tile_row][value_index] =
            transform_ptr[
                workspace_base +
                (tile_start + tile_row) * kHeadSize +
                value_index];
      }
      __syncthreads();

#pragma unroll
      for (int tile_row = 0; tile_row < StateTile; ++tile_row) {
        const int output_key = tile_start + tile_row;
        float updated =
            bias_ptr[
                workspace_base + output_key * kHeadSize + value_index];
#pragma unroll
        for (int input_key = 0;
             input_key < kHeadSize;
             ++input_key) {
          updated = fmaf(
              shared.transform[tile_row][input_key],
              state[input_key],
              updated);
        }
        shared.next_state[output_key][value_index] = updated;
      }
      __syncthreads();
    }

#pragma unroll
    for (int key_index = 0; key_index < kHeadSize; ++key_index) {
      state[key_index] = shared.next_state[key_index][value_index];
    }
    __syncthreads();
  }

#pragma unroll
  for (int key_index = 0; key_index < kHeadSize; ++key_index) {
    state_ptr[state_base + key_index * kHeadSize + value_index] =
        state[key_index];
  }
}

template <
    typename io_t,
    int BuildWarps,
    int Stages>
void launch_build(
    int num_chunks,
    int num_heads,
    const torch::Tensor& chunk_token_starts,
    const torch::Tensor& chunk_token_ends,
    const torch::Tensor& decay,
    const torch::Tensor& decay_bias,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& transform,
    torch::Tensor& bias,
    cudaStream_t stream) {
  build_transforms_kernel<io_t, BuildWarps, Stages>
      <<<num_chunks * num_heads, BuildWarps * 32, 0, stream>>>(
          num_heads,
          chunk_token_starts.data_ptr<int>(),
          chunk_token_ends.data_ptr<int>(),
          decay.data_ptr<io_t>(),
          decay_bias.defined() ? decay_bias.data_ptr<io_t>() : nullptr,
          k.data_ptr<io_t>(),
          v.data_ptr<io_t>(),
          a.data_ptr<io_t>(),
          b.data_ptr<io_t>(),
          transform.data_ptr<float>(),
          bias.data_ptr<float>());
}

template <typename io_t>
void dispatch_build(
    int build_warps,
    int stages,
    int num_chunks,
    int num_heads,
    const torch::Tensor& chunk_token_starts,
    const torch::Tensor& chunk_token_ends,
    const torch::Tensor& decay,
    const torch::Tensor& decay_bias,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& transform,
    torch::Tensor& bias,
    cudaStream_t stream) {
  if (build_warps == 2 && stages == 1) {
    launch_build<io_t, 2, 1>(
        num_chunks,
        num_heads,
        chunk_token_starts,
        chunk_token_ends,
        decay,
        decay_bias,
        k,
        v,
        a,
        b,
        transform,
        bias,
        stream);
    return;
  }
  if (build_warps == 4 && stages == 1) {
    launch_build<io_t, 4, 1>(
        num_chunks,
        num_heads,
        chunk_token_starts,
        chunk_token_ends,
        decay,
        decay_bias,
        k,
        v,
        a,
        b,
        transform,
        bias,
        stream);
    return;
  }
  launch_build<io_t, 4, 2>(
      num_chunks,
      num_heads,
      chunk_token_starts,
      chunk_token_ends,
      decay,
      decay_bias,
      k,
      v,
      a,
      b,
      transform,
      bias,
      stream);
}

template <int StateTile>
void launch_scan(
    int num_sequences,
    int num_heads,
    const torch::Tensor& sequence_chunk_offsets,
    const torch::Tensor& state_indices,
    torch::Tensor& state,
    const torch::Tensor& transform,
    const torch::Tensor& bias,
    torch::Tensor& boundary,
    cudaStream_t stream) {
  scan_boundaries_kernel<StateTile>
      <<<num_sequences * num_heads, kHeadSize, 0, stream>>>(
          num_heads,
          sequence_chunk_offsets.data_ptr<int>(),
          state_indices.data_ptr<int>(),
          state.data_ptr<float>(),
          transform.data_ptr<float>(),
          bias.data_ptr<float>(),
          boundary.data_ptr<float>());
}

void dispatch_scan(
    int state_tile,
    int num_sequences,
    int num_heads,
    const torch::Tensor& sequence_chunk_offsets,
    const torch::Tensor& state_indices,
    torch::Tensor& state,
    const torch::Tensor& transform,
    const torch::Tensor& bias,
    torch::Tensor& boundary,
    cudaStream_t stream) {
  if (state_tile == 16) {
    launch_scan<16>(
        num_sequences,
        num_heads,
        sequence_chunk_offsets,
        state_indices,
        state,
        transform,
        bias,
        boundary,
        stream);
    return;
  }
  if (state_tile == 32) {
    launch_scan<32>(
        num_sequences,
        num_heads,
        sequence_chunk_offsets,
        state_indices,
        state,
        transform,
        bias,
        boundary,
        stream);
    return;
  }
  launch_scan<64>(
      num_sequences,
      num_heads,
      sequence_chunk_offsets,
      state_indices,
      state,
      transform,
      bias,
      boundary,
      stream);
}

template <typename io_t>
void launch_materialized_chunk(
    int num_sequences,
    int num_chunks,
    int num_heads,
    int build_warps,
    int stages,
    int state_tile,
    const torch::Tensor& sequence_chunk_offsets,
    const torch::Tensor& chunk_token_starts,
    const torch::Tensor& chunk_token_ends,
    const torch::Tensor& state_indices,
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& decay,
    const torch::Tensor& decay_bias,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& output,
    torch::Tensor& transform,
    torch::Tensor& bias,
    torch::Tensor& boundary,
    torch::Tensor* state_dot_a,
    float scale,
    cudaStream_t stream) {
  dispatch_build<io_t>(
      build_warps,
      stages,
      num_chunks,
      num_heads,
      chunk_token_starts,
      chunk_token_ends,
      decay,
      decay_bias,
      k,
      v,
      a,
      b,
      transform,
      bias,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dispatch_scan(
      state_tile,
      num_sequences,
      num_heads,
      sequence_chunk_offsets,
      state_indices,
      state,
      transform,
      bias,
      boundary,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  launch_rl_infctx_chunk_replay_fp32_from_decay_logits(
      num_chunks, num_heads, chunk_token_starts, chunk_token_ends, boundary,
      r, decay, decay_bias, k, v, a, b, output, state_dot_a, scale, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace rl_infctx_materialized_detail



void recompute_chunk_fp32_from_decay_logits_cuda(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
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
    torch::Tensor boundary,
    double scale) {
  const c10::cuda::CUDAGuard device_guard(state.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int num_sequences = static_cast<int>(state_indices.numel());
  const int num_chunks = static_cast<int>(chunk_token_starts.numel());
  const int num_heads = static_cast<int>(state.size(1));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      r.scalar_type(),
      "flashrwkv2_rl_infctx_recompute_chunk_fp32",
      [&] {
        rl_infctx_recompute_detail::launch_recompute_chunk<scalar_t>(
            num_sequences, num_chunks, num_heads,
            sequence_chunk_offsets, chunk_token_starts, chunk_token_ends,
            state_indices, state, r, decay_logits, decay_bias, k, v, a, b,
            output, boundary, static_cast<float>(scale), stream);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}



void materialized_chunk_fp32_from_decay_logits_cuda(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
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
    torch::Tensor transform,
    torch::Tensor bias,
    torch::Tensor boundary,
    torch::Tensor state_dot_a,
    int64_t build_warps,
    int64_t stages,
    int64_t state_tile,
    double scale) {
  const c10::cuda::CUDAGuard device_guard(state.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int num_sequences = static_cast<int>(state_indices.numel());
  const int num_chunks = static_cast<int>(chunk_token_starts.numel());
  const int num_heads = static_cast<int>(state.size(1));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      r.scalar_type(),
      "flashrwkv2_rl_infctx_materialized_chunk_fp32",
      [&] {
        rl_infctx_materialized_detail::launch_materialized_chunk<scalar_t>(
            num_sequences, num_chunks, num_heads,
            static_cast<int>(build_warps),
            static_cast<int>(stages),
            static_cast<int>(state_tile),
            sequence_chunk_offsets, chunk_token_starts, chunk_token_ends,
            state_indices, state, r, decay_logits, decay_bias, k, v, a, b,
            output, transform, bias, boundary,
            state_dot_a.defined() ? &state_dot_a : nullptr,
            static_cast<float>(scale), stream);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace rl_infctx_tiled_detail {
using flashrwkv2::wkv7::recurrent_retention;
template <typename io_t, int HeadSize>
__global__ void recurrent_columns(
    int num_heads,const int* sequence_chunk_offsets,const int* chunk_starts,
    const int* chunk_ends,const int* state_indices,float* state,
    const io_t* r,const io_t* decay,const io_t* decay_bias,const io_t* k,
    const io_t* v,const io_t* a,const io_t* b,io_t* output,float* boundary,
    float* state_dot_a,float scale) {
  const int value=blockIdx.x,head=blockIdx.y,sequence=blockIdx.z,lane=threadIdx.x;
  const int slot=state_indices[sequence];
  const int64_t state_base=(static_cast<int64_t>(slot)*num_heads+head)*HeadSize*HeadSize;
  for(int chunk=sequence_chunk_offsets[sequence];chunk<sequence_chunk_offsets[sequence+1];++chunk){
    const int64_t boundary_base=(static_cast<int64_t>(chunk)*num_heads+head)*HeadSize*HeadSize;
    for(int key=lane;key<HeadSize;key+=32)
      boundary[boundary_base+static_cast<int64_t>(key)*HeadSize+value]=state[state_base+static_cast<int64_t>(key)*HeadSize+value];
    for(int token=chunk_starts[chunk];token<chunk_ends[chunk];++token){
      const int64_t token_base=(static_cast<int64_t>(token)*num_heads+head)*HeadSize;
      float sa=0.0f;
      for(int key=lane;key<HeadSize;key+=32)
        sa=fmaf(static_cast<float>(a[token_base+key]),state[state_base+static_cast<int64_t>(key)*HeadSize+value],sa);
      for(int off=16;off;off>>=1)sa+=__shfl_down_sync(0xffffffffu,sa,off);
      sa=__shfl_sync(0xffffffffu,sa,0);
      if(lane==0&&state_dot_a)state_dot_a[token_base+value]=sa;
      float y=0.0f; const float vv=static_cast<float>(v[token_base+value]);
      for(int key=lane;key<HeadSize;key+=32){
        const int64_t si=state_base+static_cast<int64_t>(key)*HeadSize+value;
        float dl=static_cast<float>(decay[token_base+key]);
        if(decay_bias)dl+=static_cast<float>(decay_bias[head*HeadSize+key]);
        const float next=recurrent_retention(dl)*state[si]+
          static_cast<float>(b[token_base+key])*sa+
          static_cast<float>(k[token_base+key])*vv;
        state[si]=next;y=fmaf(static_cast<float>(r[token_base+key]),next,y);
      }
      for(int off=16;off;off>>=1)y+=__shfl_down_sync(0xffffffffu,y,off);
      if(lane==0)output[token_base+value]=static_cast<io_t>(scale*y);
    }
  }
}
}

void tiled_chunk_fp32_from_decay_logits_cuda(
    torch::Tensor sequence_chunk_offsets,torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,torch::Tensor state_indices,torch::Tensor state,
    torch::Tensor r,torch::Tensor decay_logits,torch::Tensor decay_bias,
    torch::Tensor k,torch::Tensor v,torch::Tensor a,torch::Tensor b,
    torch::Tensor output,torch::Tensor boundary,torch::Tensor state_dot_a,double scale) {
  const c10::cuda::CUDAGuard guard(state.device());
  const auto stream=at::cuda::getCurrentCUDAStream();
  const int D=state.size(2),H=state.size(1),B=state_indices.numel();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,r.scalar_type(),"flashrwkv2_rl_tiled",[&]{
    const dim3 grid(D,H,B);
    if(D==128) rl_infctx_tiled_detail::recurrent_columns<scalar_t,128><<<grid,32,0,stream>>>(H,sequence_chunk_offsets.data_ptr<int>(),chunk_token_starts.data_ptr<int>(),chunk_token_ends.data_ptr<int>(),state_indices.data_ptr<int>(),state.data_ptr<float>(),r.data_ptr<scalar_t>(),decay_logits.data_ptr<scalar_t>(),decay_bias.defined()?decay_bias.data_ptr<scalar_t>():nullptr,k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),a.data_ptr<scalar_t>(),b.data_ptr<scalar_t>(),output.data_ptr<scalar_t>(),boundary.data_ptr<float>(),state_dot_a.defined()?state_dot_a.data_ptr<float>():nullptr,static_cast<float>(scale));
    else rl_infctx_tiled_detail::recurrent_columns<scalar_t,256><<<grid,32,0,stream>>>(H,sequence_chunk_offsets.data_ptr<int>(),chunk_token_starts.data_ptr<int>(),chunk_token_ends.data_ptr<int>(),state_indices.data_ptr<int>(),state.data_ptr<float>(),r.data_ptr<scalar_t>(),decay_logits.data_ptr<scalar_t>(),decay_bias.defined()?decay_bias.data_ptr<scalar_t>():nullptr,k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),a.data_ptr<scalar_t>(),b.data_ptr<scalar_t>(),output.data_ptr<scalar_t>(),boundary.data_ptr<float>(),state_dot_a.defined()?state_dot_a.data_ptr<float>():nullptr,static_cast<float>(scale));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
