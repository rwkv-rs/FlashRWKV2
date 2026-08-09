// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Albatross project
//
// Canonical body source:
//   BlinkDL/Albatross/faster3a_2607/cuda/rwkv7_wkv_fp16_v2.cu
//   revision ee3308f6922e59f2166c7fac3c5a192340a2b48e
//   Apache-2.0
//
// Additional exact helper body source:
//   BlinkDL/Albatross/faster3a_2607/cuda/rwkv7_fast_ops_fp16.cu
//   revision ee3308f6922e59f2166c7fac3c5a192340a2b48e
//
// Local adaptation:
//   - packed [total_tokens,H,D] token storage and cu_seqlens boundaries;
//   - state_indices-backed FP16 state pool in FlashRWKV2's [K,V] layout;
//   - raw decay logits and optional bias fused into the retention transform;
//   - clone, exact, seq-v2, one-cp and one-direct family symbols retained for
//     Albatross-shaped dispatch;
//   - Albatross half2 arithmetic, cp.async token staging and swizzled shared
//     state staging are preserved for the canonical D=64 implementation;
//   - D=128/256 use a local 64-key-tiled recurrent family. Each warp owns one
//     value column, keeps only the active 64-key tile live, and updates the
//     FP16 state pool directly; no FP32-state fallback is selected.
//
// vllm-rwkv at 6d683f9e49a2997e405c47edc147872c8609513b is a varlen and
// state-layout reference only, not the primary implementation source.

#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <climits>
#include <cstdint>

#include "recurrent_decay.cuh"

namespace {

constexpr int kHeadSize = 64;
constexpr int kHalf2HeadSize = kHeadSize / 2;
constexpr int kHalfPerInt4 = sizeof(int4) / sizeof(half);

using flashrwkv2::wkv7::recurrent_retention;
using flashrwkv2::wkv7::recurrent_fp16_delta;

inline int64_t ceil_div(int64_t n, int64_t d) {
  return (n + d - 1) / d;
}

__global__ void advance_i32_kernel(int* __restrict__ x, int amount, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] += amount;
  }
}

// Varlen adaptation of the same Albatross advance_i32 body.  Metadata
// validation is produced on the same stream; an invalid ticket leaves every
// elapsed slot untouched, just like the recurrent output/state kernels.
__global__ void advance_i32_varlen_kernel(
    const int* __restrict__ query_start_loc,
    const int* __restrict__ state_indices,
    const int* __restrict__ metadata_status,
    int* __restrict__ elapsed_state,
    int num_sequences) {
  const int sequence_index =
      static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (sequence_index >= num_sequences || metadata_status[0] != 0 ||
      sequence_index >= metadata_status[2]) {
    return;
  }
  const int token_count =
      query_start_loc[sequence_index + 1] - query_start_loc[sequence_index];
  elapsed_state[state_indices[sequence_index]] += token_count;
}

// Exact Albatross add_vec_kernel.  This helper is tokenwise, so packed rows
// are already the complete varlen adaptation and no request metadata is read.
__global__ void recurrent_add_vec_kernel(
    int C,
    const half* __restrict__ x,
    const half* __restrict__ vec,
    half* __restrict__ out,
    int64_t total_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }
  const int c = static_cast<int>((pair_idx % (C >> 1)) << 1);
  const int64_t idx = pair_idx * 2;
  const float2 x_value = __half22float2(*reinterpret_cast<const half2*>(x + idx));
  const float2 vec_value = __half22float2(*reinterpret_cast<const half2*>(vec + c));
  *reinterpret_cast<half2*>(out + idx) =
      __floats2half2_rn(x_value.x + vec_value.x, x_value.y + vec_value.y);
}

// Exact Albatross add_vec_2d_kernel.  The row index removes the flat modulo
// from each half2 owner; the caller-owned tuned guard below admits only the
// canonical C=4096,row window.
__global__ void recurrent_add_vec_2d_kernel(
    int C,
    const half* __restrict__ x,
    const half* __restrict__ vec,
    half* __restrict__ out) {
  const int c_pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int c_pairs = C >> 1;
  if (c_pair >= c_pairs) {
    return;
  }
  const int64_t row = blockIdx.y;
  const int c = c_pair << 1;
  const int64_t idx = row * C + c;
  const float2 x_value = __half22float2(*reinterpret_cast<const half2*>(x + idx));
  const float2 vec_value = __half22float2(*reinterpret_cast<const half2*>(vec + c));
  *reinterpret_cast<half2*>(out + idx) =
      __floats2half2_rn(x_value.x + vec_value.x, x_value.y + vec_value.y);
}

template <int Bytes>
__device__ __forceinline__ void cp_async(
    void* shared,
    const void* global,
    bool predicate) {
  static_assert(Bytes == 16 || Bytes == 8 || Bytes == 4);
  const int copied_bytes = predicate ? Bytes : 0;
  const unsigned int shared_address = __cvta_generic_to_shared(shared);
  if constexpr (Bytes == 16) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2, %3;"
        :
        : "r"(shared_address),
          "l"(global),
          "n"(Bytes),
          "r"(copied_bytes));
  } else {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2, %3;"
        :
        : "r"(shared_address),
          "l"(global),
          "n"(Bytes),
          "r"(copied_bytes));
  }
}

// The clone family keeps the separately named Albatross helper sequence.  It
// is instruction-equivalent to the pinned clone_cp_async body, but remains a
// distinct source path so later upstream clone tuning cannot silently collapse
// into exact/seq-v2 dispatch.
template <int Bytes>
__device__ __forceinline__ void clone_cp_async(
    void* shared,
    const void* global,
    bool predicate) {
  static_assert(Bytes == 16 || Bytes == 8 || Bytes == 4);
  const int copied_bytes = predicate ? Bytes : 0;
  const unsigned int shared_address = __cvta_generic_to_shared(shared);
  if constexpr (Bytes == 16) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2, %3;"
        :
        : "r"(shared_address),
          "l"(global),
          "n"(Bytes),
          "r"(copied_bytes));
  } else {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2, %3;"
        :
        : "r"(shared_address),
          "l"(global),
          "n"(Bytes),
          "r"(copied_bytes));
  }
}

__device__ __forceinline__ void clone_cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int NumPending>
__device__ __forceinline__ void clone_cp_wait() {
  if constexpr (NumPending == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" : : "n"(NumPending));
  }
}

__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int NumPending>
__device__ __forceinline__ void cp_wait() {
  if constexpr (NumPending == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" : : "n"(NumPending));
  }
}

template <bool Grid2D>
__device__ __forceinline__ void decode_sequence_head(
    int num_heads,
    int& sequence_index,
    int& head_index) {
  if constexpr (Grid2D) {
    head_index = static_cast<int>(blockIdx.x);
    sequence_index = static_cast<int>(blockIdx.y);
  } else {
    const int sequence_head = static_cast<int>(blockIdx.x);
    sequence_index = sequence_head / num_heads;
    head_index = sequence_head - sequence_index * num_heads;
  }
}

__device__ __forceinline__ void prefetch_token(
    int thread,
    int lane,
    int64_t token_base,
    half2* r,
    half2* decay,
    half2* k,
    half2* a,
    half2* b,
    half2* b_dummy,
    const half* r_ptr,
    const half* decay_ptr,
    const half* k_ptr,
    const half* a_ptr,
    const half* b_ptr) {
  cp_async<4>(
      (thread < 32 ? decay : a) + lane,
      reinterpret_cast<const half2*>(
          thread < 32 ? decay_ptr + token_base : a_ptr + token_base) +
          lane,
      true);
  cp_commit();
  cp_async<4>(
      (thread < 32 ? r : k) + lane,
      reinterpret_cast<const half2*>(
          thread < 32 ? r_ptr + token_base : k_ptr + token_base) +
          lane,
      true);
  cp_async<4>(
      (thread < 32 ? b : b_dummy) + lane,
      reinterpret_cast<const half2*>(b_ptr + token_base) + lane,
      thread < 32);
  cp_commit();
}

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
  for (int64_t output_index =
           block_index * static_cast<int64_t>(blockDim.x) + threadIdx.x;
       output_index < output_elements;
       output_index += block_count * static_cast<int64_t>(blockDim.x)) {
    output_ptr[output_index] = invalid_value<io_t>();
  }
}

__device__ __forceinline__ float warp_sum_fp32(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

// D128/D256 local extension.  One warp owns one [K] state column and the
// launch tiles the value dimension through grid.x.  Keeping the recurrent
// column in the FP16 state pool avoids the per-thread float[D] register array
// used by the FP32-state family and preserves the Albatross FP16 quantization
// and elapsed-phase contract after every token.
template <int HeadSize>
__global__ __launch_bounds__(32, 4) void wkv_fp16_tiled_column_kernel(
    int num_heads,
    int64_t output_elements,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ state_indices,
    const int* __restrict__ elapsed_state_ptr,
    const int* __restrict__ metadata_status,
    half* __restrict__ state_ptr,
    const half* __restrict__ r_ptr,
    const half* __restrict__ decay_ptr,
    const half* __restrict__ decay_bias_ptr,
    const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr,
    const half* __restrict__ a_ptr,
    const half* __restrict__ b_ptr,
    half* __restrict__ output_ptr,
    float scale) {
  const int value_index = static_cast<int>(blockIdx.x);
  const int head_index = static_cast<int>(blockIdx.y);
  const int sequence_index = static_cast<int>(blockIdx.z);
  const int lane = static_cast<int>(threadIdx.x);
  const int64_t block_index =
      (static_cast<int64_t>(sequence_index) * num_heads + head_index) *
          HeadSize +
      value_index;
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

  const int state_slot = state_indices[sequence_index];
  const int token_start = query_start_loc[sequence_index];
  const int token_end = query_start_loc[sequence_index + 1];
  const int elapsed_base = elapsed_state_ptr[state_slot];
  const int64_t state_head_base =
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
      HeadSize * HeadSize;
  for (int token_index = token_start; token_index < token_end; ++token_index) {
    const int token_offset = token_index - token_start;
    const int64_t token_base =
        (static_cast<int64_t>(token_index) * num_heads + head_index) *
        HeadSize;
    float state_dot_a = 0.0f;
    for (int key_index = lane; key_index < HeadSize; key_index += 32) {
      const float state_value = __half2float(
          state_ptr[state_head_base +
                    static_cast<int64_t>(key_index) * HeadSize + value_index]);
      state_dot_a = fmaf(
          state_value, __half2float(a_ptr[token_base + key_index]),
          state_dot_a);
    }
    state_dot_a = warp_sum_fp32(state_dot_a);
    state_dot_a = __shfl_sync(0xffffffffu, state_dot_a, 0);

    float output_value = 0.0f;
    const float value = __half2float(v_ptr[token_base + value_index]);
    for (int key_index = lane; key_index < HeadSize; key_index += 32) {
      const int64_t state_index =
          state_head_base + static_cast<int64_t>(key_index) * HeadSize +
          value_index;
      const float old_state = __half2float(state_ptr[state_index]);
      float decay_input = __half2float(decay_ptr[token_base + key_index]);
      if (decay_bias_ptr != nullptr) {
        decay_input +=
            __half2float(decay_bias_ptr[head_index * HeadSize + key_index]);
      }
      const int phase =
          elapsed_base + head_index * HeadSize + key_index + token_offset;
      const float delta = __half2float(
          __float2half_rn(recurrent_fp16_delta(decay_input, phase)));
      const float updated =
          old_state + old_state * delta +
          __half2float(b_ptr[token_base + key_index]) * state_dot_a +
          __half2float(k_ptr[token_base + key_index]) * value;
      const half quantized = __float2half_rn(updated);
      state_ptr[state_index] = quantized;
      output_value = fmaf(
          __half2float(quantized), __half2float(r_ptr[token_base + key_index]),
          output_value);
    }
    output_value = warp_sum_fp32(output_value);
    if (lane == 0) {
      output_ptr[token_base + value_index] =
          __float2half_rn(scale * output_value);
    }
  }
}

template <int HeadSize>
void launch_fp16_tiled_column(
    int num_sequences,
    int num_heads,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& state_indices,
    const torch::Tensor& elapsed_state,
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
  wkv_fp16_tiled_column_kernel<HeadSize>
      <<<dim3(HeadSize, num_heads, num_sequences), 32, 0, stream>>>(
          num_heads, output.numel(), query_start_loc.data_ptr<int>(),
          state_indices.data_ptr<int>(), elapsed_state.data_ptr<int>(),
          metadata_status.data_ptr<int>(),
          reinterpret_cast<half*>(state.data_ptr()),
          reinterpret_cast<const half*>(r.data_ptr()),
          reinterpret_cast<const half*>(decay_logits.data_ptr()),
          decay_bias.defined()
              ? reinterpret_cast<const half*>(decay_bias.data_ptr())
              : nullptr,
          reinterpret_cast<const half*>(k.data_ptr()),
          reinterpret_cast<const half*>(v.data_ptr()),
          reinterpret_cast<const half*>(a.data_ptr()),
          reinterpret_cast<const half*>(b.data_ptr()),
          reinterpret_cast<half*>(output.data_ptr()), scale);
}

// The state pool is [K,V].  Shared state is staged as [K,V/2] with the
// Albatross XOR swizzle; each thread then reconstructs the two adjacent-key
// values for one fixed V coordinate.  This is the local state-layout adapter;
// the five public bodies below retain the distinct Albatross token pipelines.
__device__ __forceinline__ void load_state_swizzled(
    half* state_base,
    half2* state_shared,
    half2* state) {
  const int thread = static_cast<int>(threadIdx.x);
#pragma unroll
  for (int vector_iteration = 0;
       vector_iteration < kHeadSize / kHalfPerInt4; ++vector_iteration) {
    const int vector_index = vector_iteration * kHeadSize + thread;
    const int4 state_vector =
        reinterpret_cast<const int4*>(state_base)[vector_index];
#pragma unroll
    for (int pair_index = 0; pair_index < kHalfPerInt4 / 2; ++pair_index) {
      const int key_index =
          vector_iteration * kHalfPerInt4 + thread / kHalfPerInt4;
      const int value_pair =
          (thread % kHalfPerInt4) * (kHalfPerInt4 / 2) + pair_index;
      state_shared[key_index * kHalf2HeadSize +
                   ((key_index & 31) ^ value_pair)] =
          reinterpret_cast<const half2*>(&state_vector)[pair_index];
    }
  }
  __syncthreads();

#pragma unroll
  for (int pair_index = 0; pair_index < kHalf2HeadSize; ++pair_index) {
    const int key0 = pair_index * 2;
    const int key1 = key0 + 1;
    const int value_pair = thread >> 1;
    const int value_lane = thread & 1;
    const half value0 = reinterpret_cast<const half*>(
        &state_shared[key0 * kHalf2HeadSize +
                      ((key0 & 31) ^ value_pair)])[value_lane];
    const half value1 = reinterpret_cast<const half*>(
        &state_shared[key1 * kHalf2HeadSize +
                      ((key1 & 31) ^ value_pair)])[value_lane];
    state[pair_index] = __halves2half2(value0, value1);
  }
}

__device__ __forceinline__ void store_state_swizzled(
    half* state_base,
    half2* state_shared,
    const half2* state) {
  const int thread = static_cast<int>(threadIdx.x);
#pragma unroll
  for (int pair_index = 0; pair_index < kHalf2HeadSize; ++pair_index) {
    const int key0 = pair_index * 2;
    const int key1 = key0 + 1;
    const int value_pair = thread >> 1;
    const int value_lane = thread & 1;
    reinterpret_cast<half*>(
        &state_shared[key0 * kHalf2HeadSize +
                      ((key0 & 31) ^ value_pair)])[value_lane] = state[pair_index].x;
    reinterpret_cast<half*>(
        &state_shared[key1 * kHalf2HeadSize +
                      ((key1 & 31) ^ value_pair)])[value_lane] = state[pair_index].y;
  }
  __syncthreads();

#pragma unroll
  for (int vector_iteration = 0;
       vector_iteration < kHeadSize / kHalfPerInt4; ++vector_iteration) {
    const int vector_index = vector_iteration * kHeadSize + thread;
    const int key_index =
        vector_iteration * kHalfPerInt4 + thread / kHalfPerInt4;
    const int value_pair =
        (thread % kHalfPerInt4) * (kHalfPerInt4 / 2);
    int4 state_vector;
#pragma unroll
    for (int pair_index = 0; pair_index < kHalfPerInt4 / 2; ++pair_index) {
      reinterpret_cast<half2*>(&state_vector)[pair_index] = state_shared[
          key_index * kHalf2HeadSize +
          ((key_index & 31) ^ (value_pair + pair_index))];
    }
    reinterpret_cast<int4*>(state_base)[vector_index] = state_vector;
  }
}

template <bool Tis1 = false, bool AddW0 = false, bool Grid2D = false>
__global__ __launch_bounds__(kHeadSize, 2)
void wkv_fp16_v1_clone_kernel(
    int num_heads,
    int64_t output_elements,
    const int* query_start_loc,
    const int* state_indices,
    const int* elapsed_state_ptr,
    const int* metadata_status,
    half* state_ptr,
    const half* r_ptr,
    const half* decay_ptr,
    const half* decay_bias_ptr,
    const half* k_ptr,
    const half* v_ptr,
    const half* a_ptr,
    const half* b_ptr,
    half* output_ptr,
    float scale) {
  int sequence_index;
  int head_index;
  decode_sequence_head<Grid2D>(num_heads, sequence_index, head_index);
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;
  const int64_t block_index =
      static_cast<int64_t>(sequence_index) * num_heads + head_index;
  const int64_t block_count =
      static_cast<int64_t>(gridDim.x) * gridDim.y;
  if (metadata_status[0] != 0) {
    fill_invalid_output(block_index, block_count, output_elements, output_ptr);
    return;
  }
  if (sequence_index >= metadata_status[2]) {
    return;
  }

  int token_start = 0;
  int token_end = 0;
  int state_slot = 0;
  int elapsed_base = 0;
  if (lane == 0) {
    token_start = query_start_loc[sequence_index];
    token_end = query_start_loc[sequence_index + 1];
    state_slot = state_indices[sequence_index];
    elapsed_base = elapsed_state_ptr[state_slot];
  }
  token_start = __shfl_sync(0xffffffffu, token_start, 0);
  token_end = __shfl_sync(0xffffffffu, token_end, 0);
  state_slot = __shfl_sync(0xffffffffu, state_slot, 0);
  elapsed_base = __shfl_sync(0xffffffffu, elapsed_base, 0);
  const int token_count = token_end - token_start;
  if constexpr (Tis1) {
    __builtin_assume(token_count == 1);
  }
  if (token_count <= 0) {
    return;
  }

  half* state_base = state_ptr +
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
      kHeadSize * kHeadSize;
  __shared__ __align__(256) half2 state_shared[kHeadSize][kHalf2HeadSize];
  half2* state_shared_ptr = &state_shared[0][0];
  half2 state[kHalf2HeadSize];
  load_state_swizzled(state_base, state_shared_ptr, state);

  // Mechanical Albatross clone body: a single shared token buffer and the
  // clone-specific cp.async sequence are intentionally retained here.
  __shared__ __align__(128) half2
      r[kHalf2HeadSize], k[kHalf2HeadSize], decay[kHalf2HeadSize],
      a[kHalf2HeadSize], b[kHalf2HeadSize], b_dummy[kHalf2HeadSize];
  int64_t token_base =
      (static_cast<int64_t>(token_start) * num_heads + head_index) * kHeadSize;
  for (int token_offset = 0; token_offset < token_count; ++token_offset) {
    __syncthreads();
    clone_cp_async<4>(
        (thread < 32 ? decay : a) + lane,
        reinterpret_cast<const half2*>(
            (thread < 32 ? decay_ptr : a_ptr) + token_base) + lane,
        true);
    clone_cp_commit();
    clone_cp_async<4>(
        (thread < 32 ? r : k) + lane,
        reinterpret_cast<const half2*>(
            (thread < 32 ? r_ptr : k_ptr) + token_base) + lane,
        true);
    clone_cp_async<4>(
        (thread < 32 ? b : b_dummy) + lane,
        reinterpret_cast<const half2*>(b_ptr + token_base) + lane,
        thread < 32);
    clone_cp_commit();

    const half value = v_ptr[token_base + thread];
    const half2 value2 = __halves2half2(value, value);
    half2 output2 = __float2half2_rn(0.0f);
    half2 state_dot_a2 = __float2half2_rn(0.0f);
    clone_cp_wait<1>();
    __syncthreads();
#pragma unroll
    for (int pair_index = 0; pair_index < kHalf2HeadSize; ++pair_index) {
      state_dot_a2 = __hfma2(a[pair_index], state[pair_index], state_dot_a2);
    }
    const half state_dot_a = __hadd(state_dot_a2.x, state_dot_a2.y);
    const half2 state_dot_a_pair = __halves2half2(state_dot_a, state_dot_a);
    float decay_input = __half2float(
        reinterpret_cast<half*>(decay)[thread]);
    if constexpr (AddW0) {
      decay_input += __half2float(decay_bias_ptr[head_index * kHeadSize + thread]);
    }
    const int phase = elapsed_base + head_index * kHeadSize + thread + token_offset;
    reinterpret_cast<half*>(decay)[thread] =
        __float2half_rn(recurrent_fp16_delta(decay_input, phase));
    clone_cp_wait<0>();
    __syncthreads();
#pragma unroll
    for (int pair_index = 0; pair_index < kHalf2HeadSize; ++pair_index) {
      half2& state_pair = state[pair_index];
      state_pair = __hfma2(
          state_pair, decay[pair_index],
          __hfma2(k[pair_index], value2,
                  __hfma2(state_dot_a_pair, b[pair_index], state_pair)));
      output2 = __hfma2(state_pair, r[pair_index], output2);
    }
    output_ptr[token_base + thread] = __float2half_rn(
        (__half2float(output2.x) + __half2float(output2.y)) * scale);
    token_base += static_cast<int64_t>(num_heads) * kHeadSize;
  }
  store_state_swizzled(state_base, state_shared_ptr, state);
}

template <bool Tis1 = false, bool AddW0 = false, bool Grid2D = false>
__global__ __launch_bounds__(kHeadSize, 2)
void wkv_fp16_v1_exact_kernel(
    int num_heads,
    int64_t output_elements,
    const int* query_start_loc,
    const int* state_indices,
    const int* elapsed_state_ptr,
    const int* metadata_status,
    half* state_ptr,
    const half* r_ptr,
    const half* decay_ptr,
    const half* decay_bias_ptr,
    const half* k_ptr,
    const half* v_ptr,
    const half* a_ptr,
    const half* b_ptr,
    half* output_ptr,
    float scale) {
  int sequence_index;
  int head_index;
  decode_sequence_head<Grid2D>(num_heads, sequence_index, head_index);
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;
  const int64_t block_index =
      static_cast<int64_t>(sequence_index) * num_heads + head_index;
  const int64_t block_count = static_cast<int64_t>(gridDim.x) * gridDim.y;
  if (metadata_status[0] != 0) {
    fill_invalid_output(block_index, block_count, output_elements, output_ptr);
    return;
  }
  if (sequence_index >= metadata_status[2]) {
    return;
  }
  int token_start = 0;
  int token_end = 0;
  int state_slot = 0;
  int elapsed_base = 0;
  if (lane == 0) {
    token_start = query_start_loc[sequence_index];
    token_end = query_start_loc[sequence_index + 1];
    state_slot = state_indices[sequence_index];
    elapsed_base = elapsed_state_ptr[state_slot];
  }
  token_start = __shfl_sync(0xffffffffu, token_start, 0);
  token_end = __shfl_sync(0xffffffffu, token_end, 0);
  state_slot = __shfl_sync(0xffffffffu, state_slot, 0);
  elapsed_base = __shfl_sync(0xffffffffu, elapsed_base, 0);
  const int token_count = token_end - token_start;
  if constexpr (Tis1) {
    __builtin_assume(token_count == 1);
  }
  if (token_count <= 0) {
    return;
  }

  half* state_base = state_ptr +
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
      kHeadSize * kHeadSize;
  __shared__ __align__(256) half2 state_shared[kHeadSize][kHalf2HeadSize];
  half2* state_shared_ptr = &state_shared[0][0];
  half2 state[kHalf2HeadSize];
  load_state_swizzled(state_base, state_shared_ptr, state);

  // Mechanical Albatross exact body: the single-buffer cp.async pipeline is
  // kept separate from clone and seq-v2 instead of being routed through a
  // one shared-buffer recurrent body.
  __shared__ __align__(128) half2
      r[kHalf2HeadSize], k[kHalf2HeadSize], decay[kHalf2HeadSize],
      a[kHalf2HeadSize], b[kHalf2HeadSize], b_dummy[kHalf2HeadSize];
  int64_t token_base =
      (static_cast<int64_t>(token_start) * num_heads + head_index) * kHeadSize;
  for (int token_offset = 0; token_offset < token_count; ++token_offset) {
    __syncthreads();
    prefetch_token(
        thread, lane, token_base, r, decay, k, a, b, b_dummy,
        r_ptr, decay_ptr, k_ptr, a_ptr, b_ptr);
    const half value = v_ptr[token_base + thread];
    const half2 value2 = __halves2half2(value, value);
    half2 output2 = __float2half2_rn(0.0f);
    half2 state_dot_a2 = __float2half2_rn(0.0f);
    cp_wait<1>();
    __syncthreads();
#pragma unroll
    for (int pair_index = 0; pair_index < kHalf2HeadSize; ++pair_index) {
      state_dot_a2 = __hfma2(a[pair_index], state[pair_index], state_dot_a2);
    }
    const half state_dot_a = __hadd(state_dot_a2.x, state_dot_a2.y);
    const half2 state_dot_a_pair = __halves2half2(state_dot_a, state_dot_a);
    float decay_input = __half2float(
        reinterpret_cast<half*>(decay)[thread]);
    if constexpr (AddW0) {
      decay_input += __half2float(decay_bias_ptr[head_index * kHeadSize + thread]);
    }
    const int phase = elapsed_base + head_index * kHeadSize + thread + token_offset;
    reinterpret_cast<half*>(decay)[thread] =
        __float2half_rn(recurrent_fp16_delta(decay_input, phase));
    cp_wait<0>();
    __syncthreads();
#pragma unroll
    for (int pair_index = 0; pair_index < kHalf2HeadSize; ++pair_index) {
      half2& state_pair = state[pair_index];
      state_pair = __hfma2(
          state_pair, decay[pair_index],
          __hfma2(k[pair_index], value2,
                  __hfma2(state_dot_a_pair, b[pair_index], state_pair)));
      output2 = __hfma2(state_pair, r[pair_index], output2);
    }
    output_ptr[token_base + thread] = __float2half_rn(
        (__half2float(output2.x) + __half2float(output2.y)) * scale);
    token_base += static_cast<int64_t>(num_heads) * kHeadSize;
  }
  store_state_swizzled(state_base, state_shared_ptr, state);
}

template <bool AddW0 = false, bool Grid2D = false>
__global__ __launch_bounds__(kHeadSize, 2)
void wkv_fp16_seq_v2_kernel(
    int num_heads,
    int64_t output_elements,
    const int* query_start_loc,
    const int* state_indices,
    const int* elapsed_state_ptr,
    const int* metadata_status,
    half* state_ptr,
    const half* r_ptr,
    const half* decay_ptr,
    const half* decay_bias_ptr,
    const half* k_ptr,
    const half* v_ptr,
    const half* a_ptr,
    const half* b_ptr,
    half* output_ptr,
    float scale) {
  int sequence_index;
  int head_index;
  decode_sequence_head<Grid2D>(num_heads, sequence_index, head_index);
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;
  const int64_t block_index =
      static_cast<int64_t>(sequence_index) * num_heads + head_index;
  const int64_t block_count = static_cast<int64_t>(gridDim.x) * gridDim.y;
  if (metadata_status[0] != 0) {
    fill_invalid_output(block_index, block_count, output_elements, output_ptr);
    return;
  }
  if (sequence_index >= metadata_status[2]) {
    return;
  }
  int token_start = 0;
  int token_end = 0;
  int state_slot = 0;
  int elapsed_base = 0;
  if (lane == 0) {
    token_start = query_start_loc[sequence_index];
    token_end = query_start_loc[sequence_index + 1];
    state_slot = state_indices[sequence_index];
    elapsed_base = elapsed_state_ptr[state_slot];
  }
  token_start = __shfl_sync(0xffffffffu, token_start, 0);
  token_end = __shfl_sync(0xffffffffu, token_end, 0);
  state_slot = __shfl_sync(0xffffffffu, state_slot, 0);
  elapsed_base = __shfl_sync(0xffffffffu, elapsed_base, 0);
  const int token_count = token_end - token_start;
  if (token_count <= 0) {
    return;
  }

  half* state_base = state_ptr +
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
      kHeadSize * kHeadSize;
  __shared__ __align__(256) half2 state_shared[kHeadSize][kHalf2HeadSize];
  half2* state_shared_ptr = &state_shared[0][0];
  half2 state[kHalf2HeadSize];
  load_state_swizzled(state_base, state_shared_ptr, state);

  // Mechanical Albatross seq-v2 body: double-buffered token staging and the
  // upstream prefetch/wait ordering are preserved for the tuned sequence path.
  __shared__ __align__(128) half2
      r[2][kHalf2HeadSize], k[2][kHalf2HeadSize],
      decay[2][kHalf2HeadSize], a[2][kHalf2HeadSize],
      b[2][kHalf2HeadSize], b_dummy[kHalf2HeadSize];
  const int64_t sequence_stride = static_cast<int64_t>(num_heads) * kHeadSize;
  int64_t token_base =
      (static_cast<int64_t>(token_start) * num_heads + head_index) * kHeadSize;
  prefetch_token(
      thread, lane, token_base, r[0], decay[0], k[0], a[0], b[0], b_dummy,
      r_ptr, decay_ptr, k_ptr, a_ptr, b_ptr);
  for (int token_offset = 0; token_offset < token_count; ++token_offset) {
    const int current = token_offset & 1;
    cp_wait<0>();
    __syncthreads();
    half2 state_dot_a2 = __float2half2_rn(0.0f);
#pragma unroll
    for (int pair_index = 0; pair_index < kHalf2HeadSize; ++pair_index) {
      state_dot_a2 = __hfma2(
          a[current][pair_index], state[pair_index], state_dot_a2);
    }
    const half state_dot_a = __hadd(state_dot_a2.x, state_dot_a2.y);
    const half2 state_dot_a_pair = __halves2half2(state_dot_a, state_dot_a);
    float decay_input = __half2float(
        reinterpret_cast<half*>(decay[current])[thread]);
    if constexpr (AddW0) {
      decay_input += __half2float(decay_bias_ptr[head_index * kHeadSize + thread]);
    }
    const int phase = elapsed_base + head_index * kHeadSize + thread + token_offset;
    reinterpret_cast<half*>(decay[current])[thread] =
        __float2half_rn(recurrent_fp16_delta(decay_input, phase));
    __syncthreads();
    if (token_offset + 1 < token_count) {
      prefetch_token(
          thread, lane, token_base + sequence_stride, r[current ^ 1],
          decay[current ^ 1], k[current ^ 1], a[current ^ 1],
          b[current ^ 1], b_dummy, r_ptr, decay_ptr, k_ptr, a_ptr, b_ptr);
    }
    const half value = v_ptr[token_base + thread];
    const half2 value2 = __halves2half2(value, value);
    half2 output2 = __float2half2_rn(0.0f);
#pragma unroll
    for (int pair_index = 0; pair_index < kHalf2HeadSize; ++pair_index) {
      half2& state_pair = state[pair_index];
      state_pair = __hfma2(
          state_pair, decay[current][pair_index],
          __hfma2(k[current][pair_index], value2,
                  __hfma2(state_dot_a_pair, b[current][pair_index], state_pair)));
      output2 = __hfma2(state_pair, r[current][pair_index], output2);
    }
    output_ptr[token_base + thread] = __float2half_rn(
        (__half2float(output2.x) + __half2float(output2.y)) * scale);
    token_base += sequence_stride;
  }
  store_state_swizzled(state_base, state_shared_ptr, state);
}

template <bool AddW0 = false, bool Grid2D = false>
__global__ __launch_bounds__(kHeadSize, 2)
void wkv_fp16_one_cp_kernel(
    int num_heads,
    int64_t output_elements,
    const int* query_start_loc,
    const int* state_indices,
    const int* elapsed_state_ptr,
    const int* metadata_status,
    half* state_ptr,
    const half* r_ptr,
    const half* decay_ptr,
    const half* decay_bias_ptr,
    const half* k_ptr,
    const half* v_ptr,
    const half* a_ptr,
    const half* b_ptr,
    half* output_ptr,
    float scale) {
  int sequence_index;
  int head_index;
  decode_sequence_head<Grid2D>(num_heads, sequence_index, head_index);
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;
  const int64_t block_index =
      static_cast<int64_t>(sequence_index) * num_heads + head_index;
  const int64_t block_count = static_cast<int64_t>(gridDim.x) * gridDim.y;
  if (metadata_status[0] != 0) {
    fill_invalid_output(block_index, block_count, output_elements, output_ptr);
    return;
  }
  if (sequence_index >= metadata_status[2]) {
    return;
  }
  int token_start = 0;
  int token_end = 0;
  int state_slot = 0;
  int elapsed_base = 0;
  if (lane == 0) {
    token_start = query_start_loc[sequence_index];
    token_end = query_start_loc[sequence_index + 1];
    state_slot = state_indices[sequence_index];
    elapsed_base = elapsed_state_ptr[state_slot];
  }
  token_start = __shfl_sync(0xffffffffu, token_start, 0);
  token_end = __shfl_sync(0xffffffffu, token_end, 0);
  state_slot = __shfl_sync(0xffffffffu, state_slot, 0);
  elapsed_base = __shfl_sync(0xffffffffu, elapsed_base, 0);
  if (token_end - token_start <= 0) {
    return;
  }
  half* state_base = state_ptr +
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
      kHeadSize * kHeadSize;
  __shared__ __align__(256) half2 state_shared[kHeadSize][kHalf2HeadSize];
  half2* state_shared_ptr = &state_shared[0][0];
  half2 state[kHalf2HeadSize];
  load_state_swizzled(state_base, state_shared_ptr, state);

  // Mechanical Albatross one-direct body: the T=1 path intentionally uses
  // direct __ldg loads and no cp.async token pipeline.
  __shared__ __align__(128) half2
      r[kHalf2HeadSize], k[kHalf2HeadSize], decay[kHalf2HeadSize],
      a[kHalf2HeadSize], b[kHalf2HeadSize];
  const int64_t token_base =
      (static_cast<int64_t>(token_start) * num_heads + head_index) * kHeadSize;
  if (thread < kHalf2HeadSize) {
    const int64_t pair_base = token_base / 2 + thread;
    r[thread] = __ldg(reinterpret_cast<const half2*>(r_ptr) + pair_base);
    decay[thread] = __ldg(reinterpret_cast<const half2*>(decay_ptr) + pair_base);
    k[thread] = __ldg(reinterpret_cast<const half2*>(k_ptr) + pair_base);
    a[thread] = __ldg(reinterpret_cast<const half2*>(a_ptr) + pair_base);
    b[thread] = __ldg(reinterpret_cast<const half2*>(b_ptr) + pair_base);
  }
  __syncthreads();
  half2 state_dot_a2 = __float2half2_rn(0.0f);
#pragma unroll
  for (int pair_index = 0; pair_index < kHalf2HeadSize; ++pair_index) {
    state_dot_a2 = __hfma2(a[pair_index], state[pair_index], state_dot_a2);
  }
  const half state_dot_a = __hadd(state_dot_a2.x, state_dot_a2.y);
  const half2 state_dot_a_pair = __halves2half2(state_dot_a, state_dot_a);
  float decay_input = __half2float(
      reinterpret_cast<half*>(decay)[thread]);
  if constexpr (AddW0) {
    decay_input += __half2float(decay_bias_ptr[head_index * kHeadSize + thread]);
  }
  const int phase = elapsed_base + head_index * kHeadSize + thread;
  reinterpret_cast<half*>(decay)[thread] =
      __float2half_rn(recurrent_fp16_delta(decay_input, phase));
  __syncthreads();
  const half value = __ldg(v_ptr + token_base + thread);
  const half2 value2 = __halves2half2(value, value);
  half2 output2 = __float2half2_rn(0.0f);
#pragma unroll
  for (int pair_index = 0; pair_index < kHalf2HeadSize; ++pair_index) {
    half2& state_pair = state[pair_index];
    state_pair = __hfma2(
        state_pair, decay[pair_index],
        __hfma2(k[pair_index], value2,
                __hfma2(state_dot_a_pair, b[pair_index], state_pair)));
    output2 = __hfma2(state_pair, r[pair_index], output2);
  }
  output_ptr[token_base + thread] = __float2half_rn(
      (__half2float(output2.x) + __half2float(output2.y)) * scale);
  store_state_swizzled(state_base, state_shared_ptr, state);
}

template <bool AddW0 = false, bool Grid2D = false>
__global__ __launch_bounds__(kHeadSize, 2)
void wkv_fp16_one_direct_kernel(
    int num_heads,
    int64_t output_elements,
    const int* query_start_loc,
    const int* state_indices,
    const int* elapsed_state_ptr,
    const int* metadata_status,
    half* state_ptr,
    const half* r_ptr,
    const half* decay_ptr,
    const half* decay_bias_ptr,
    const half* k_ptr,
    const half* v_ptr,
    const half* a_ptr,
    const half* b_ptr,
    half* output_ptr,
    float scale) {
  int sequence_index;
  int head_index;
  decode_sequence_head<Grid2D>(num_heads, sequence_index, head_index);
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;
  const int64_t block_index =
      static_cast<int64_t>(sequence_index) * num_heads + head_index;
  const int64_t block_count = static_cast<int64_t>(gridDim.x) * gridDim.y;
  if (metadata_status[0] != 0) {
    fill_invalid_output(block_index, block_count, output_elements, output_ptr);
    return;
  }
  if (sequence_index >= metadata_status[2]) {
    return;
  }
  int token_start = 0;
  int token_end = 0;
  int state_slot = 0;
  int elapsed_base = 0;
  if (lane == 0) {
    token_start = query_start_loc[sequence_index];
    token_end = query_start_loc[sequence_index + 1];
    state_slot = state_indices[sequence_index];
    elapsed_base = elapsed_state_ptr[state_slot];
  }
  token_start = __shfl_sync(0xffffffffu, token_start, 0);
  token_end = __shfl_sync(0xffffffffu, token_end, 0);
  state_slot = __shfl_sync(0xffffffffu, state_slot, 0);
  elapsed_base = __shfl_sync(0xffffffffu, elapsed_base, 0);
  if (token_end - token_start <= 0) {
    return;
  }
  half* state_base = state_ptr +
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
      kHeadSize * kHeadSize;
  __shared__ __align__(256) half2 state_shared[kHeadSize][kHalf2HeadSize];
  half2* state_shared_ptr = &state_shared[0][0];
  half2 state[kHalf2HeadSize];
  load_state_swizzled(state_base, state_shared_ptr, state);

  // Mechanical Albatross one-cp body: one token, one cp.async group, and the
  // same half2/shared-state sequence as the pinned upstream implementation.
  __shared__ __align__(128) half2
      r[kHalf2HeadSize], k[kHalf2HeadSize], decay[kHalf2HeadSize],
      a[kHalf2HeadSize], b[kHalf2HeadSize], b_dummy[kHalf2HeadSize];
  const int64_t token_base =
      (static_cast<int64_t>(token_start) * num_heads + head_index) * kHeadSize;
  cp_async<4>(
      (thread < 32 ? decay : a) + lane,
      reinterpret_cast<const half2*>(
          (thread < 32 ? decay_ptr : a_ptr) + token_base) + lane,
      true);
  cp_commit();
  cp_async<4>(
      (thread < 32 ? r : k) + lane,
      reinterpret_cast<const half2*>(
          (thread < 32 ? r_ptr : k_ptr) + token_base) + lane,
      true);
  cp_async<4>(
      (thread < 32 ? b : b_dummy) + lane,
      reinterpret_cast<const half2*>(b_ptr + token_base) + lane,
      thread < 32);
  cp_commit();
  const half value = v_ptr[token_base + thread];
  const half2 value2 = __halves2half2(value, value);
  half2 state_dot_a2 = __float2half2_rn(0.0f);
  cp_wait<1>();
  __syncthreads();
#pragma unroll
  for (int pair_index = 0; pair_index < kHalf2HeadSize; ++pair_index) {
    state_dot_a2 = __hfma2(a[pair_index], state[pair_index], state_dot_a2);
  }
  const half state_dot_a = __hadd(state_dot_a2.x, state_dot_a2.y);
  const half2 state_dot_a_pair = __halves2half2(state_dot_a, state_dot_a);
  float decay_input = __half2float(
      reinterpret_cast<half*>(decay)[thread]);
  if constexpr (AddW0) {
    decay_input += __half2float(decay_bias_ptr[head_index * kHeadSize + thread]);
  }
  const int phase = elapsed_base + head_index * kHeadSize + thread;
  reinterpret_cast<half*>(decay)[thread] =
      __float2half_rn(recurrent_fp16_delta(decay_input, phase));
  cp_wait<0>();
  __syncthreads();
  half2 output2 = __float2half2_rn(0.0f);
#pragma unroll
  for (int pair_index = 0; pair_index < kHalf2HeadSize; ++pair_index) {
    half2& state_pair = state[pair_index];
    state_pair = __hfma2(
        state_pair, decay[pair_index],
        __hfma2(k[pair_index], value2,
                __hfma2(state_dot_a_pair, b[pair_index], state_pair)));
    output2 = __hfma2(state_pair, r[pair_index], output2);
  }
  output_ptr[token_base + thread] = __float2half_rn(
      (__half2float(output2.x) + __half2float(output2.y)) * scale);
  store_state_swizzled(state_base, state_shared_ptr, state);
}

bool use_v2_seq(int batch_size, int max_seqlen) {
  return (batch_size == 1 && max_seqlen >= 8) ||
      (batch_size == 4 && max_seqlen >= 4) ||
      (batch_size == 8 && max_seqlen >= 8) ||
      (batch_size == 64 && max_seqlen == 1) ||
      (batch_size == 128 && max_seqlen == 1);
}

bool use_grid2d(int batch_size, int max_seqlen, int num_heads) {
  // This is the Albatross use_wkv_bh_grid_2d policy with packed sequence
  // boundaries substituted for its fixed B/T launch.  Keep the upstream
  // C=4096,H=64 admission and sparse exact-path overrides; other operator
  // shapes use the flat B*H grid specialization.
  if (num_heads != 64 || batch_size <= 0 || batch_size > 65535) {
    return false;
  }
  if (max_seqlen == 1) {
    return batch_size <= 16;
  }
  if ((batch_size == 2 && max_seqlen == 32) ||
      (batch_size == 4 && (max_seqlen == 16 || max_seqlen == 64)) ||
      (batch_size == 8 && max_seqlen == 8)) {
    return true;
  }
  return !use_v2_seq(batch_size, max_seqlen);
}

template <bool Grid2D, typename Kernel>
void launch_fp16_d64(
    Kernel kernel,
    int num_sequences,
    int num_heads,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& state_indices,
    const torch::Tensor& elapsed_state,
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& decay,
    const torch::Tensor& decay_bias,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& output,
    const torch::Tensor& metadata_status,
    float scale,
    cudaStream_t stream) {
  const dim3 grid = Grid2D
      ? dim3(static_cast<unsigned int>(num_heads),
             static_cast<unsigned int>(num_sequences), 1)
      : dim3(static_cast<unsigned int>(num_heads * num_sequences), 1, 1);
  kernel<<<grid, dim3(kHeadSize), 0, stream>>>(
      num_heads, output.numel(), query_start_loc.data_ptr<int>(),
      state_indices.data_ptr<int>(), elapsed_state.data_ptr<int>(),
      metadata_status.data_ptr<int>(),
      reinterpret_cast<half*>(state.data_ptr()),
      reinterpret_cast<const half*>(r.data_ptr()),
      reinterpret_cast<const half*>(decay.data_ptr()),
      decay_bias.defined()
          ? reinterpret_cast<const half*>(decay_bias.data_ptr())
          : nullptr,
      reinterpret_cast<const half*>(k.data_ptr()),
      reinterpret_cast<const half*>(v.data_ptr()),
      reinterpret_cast<const half*>(a.data_ptr()),
      reinterpret_cast<const half*>(b.data_ptr()),
      reinterpret_cast<half*>(output.data_ptr()), scale);
}

enum class Fp16Family {
  Clone,
  Exact,
  SeqV2,
  OneCp,
  OneDirect,
};

template <Fp16Family Family, bool AddW0, bool Grid2D>
auto select_fp16_kernel() {
  if constexpr (Family == Fp16Family::Clone) {
    return wkv_fp16_v1_clone_kernel<true, AddW0, Grid2D>;
  } else if constexpr (Family == Fp16Family::Exact) {
    return wkv_fp16_v1_exact_kernel<false, AddW0, Grid2D>;
  } else if constexpr (Family == Fp16Family::SeqV2) {
    return wkv_fp16_seq_v2_kernel<AddW0, Grid2D>;
  } else if constexpr (Family == Fp16Family::OneCp) {
    return wkv_fp16_one_cp_kernel<AddW0, Grid2D>;
  } else {
    return wkv_fp16_one_direct_kernel<AddW0, Grid2D>;
  }
}

template <Fp16Family Family>
void dispatch_fp16_family(
    bool add_w0,
    bool grid2d,
    int num_sequences,
    int num_heads,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& state_indices,
    const torch::Tensor& elapsed_state,
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& decay,
    const torch::Tensor& decay_bias,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& output,
    const torch::Tensor& metadata_status,
    float scale,
    cudaStream_t stream) {
  if (grid2d) {
    if (add_w0) {
      launch_fp16_d64<true>(
          select_fp16_kernel<Family, true, true>(), num_sequences, num_heads,
          query_start_loc, state_indices, elapsed_state, state, r, decay,
          decay_bias, k, v, a, b, output, metadata_status, scale, stream);
    } else {
      launch_fp16_d64<true>(
          select_fp16_kernel<Family, false, true>(), num_sequences, num_heads,
          query_start_loc, state_indices, elapsed_state, state, r, decay,
          decay_bias, k, v, a, b, output, metadata_status, scale, stream);
    }
  } else if (add_w0) {
    launch_fp16_d64<false>(
        select_fp16_kernel<Family, true, false>(), num_sequences, num_heads,
        query_start_loc, state_indices, elapsed_state, state, r, decay,
        decay_bias, k, v, a, b, output, metadata_status, scale, stream);
  } else {
    launch_fp16_d64<false>(
        select_fp16_kernel<Family, false, false>(), num_sequences, num_heads,
        query_start_loc, state_indices, elapsed_state, state, r, decay,
        decay_bias, k, v, a, b, output, metadata_status, scale, stream);
  }
}

}  // namespace

void recurrent_fp16_advance_i32_cuda(torch::Tensor x, int64_t amount) {
  TORCH_CHECK(amount >= INT_MIN && amount <= INT_MAX,
              "advance_i32 amount out of int range");
  constexpr int threads = 256;
  const int64_t n = x.numel();
  TORCH_CHECK(n > 0, "advance_i32 requires a non-empty state tensor");
  const c10::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  advance_i32_kernel<<<static_cast<int>(ceil_div(n, threads)), threads, 0, stream>>>(
      x.data_ptr<int>(), static_cast<int>(amount), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void recurrent_fp16_advance_i32_varlen_cuda(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor elapsed_state,
    torch::Tensor metadata_status) {
  const c10::cuda::CUDAGuard device_guard(elapsed_state.device());
  constexpr int threads = 256;
  const int num_sequences = static_cast<int>(state_indices.numel());
  auto stream = at::cuda::getCurrentCUDAStream();
  advance_i32_varlen_kernel<<<
      static_cast<int>(ceil_div(num_sequences, threads)), threads, 0, stream>>>(
      query_start_loc.data_ptr<int>(), state_indices.data_ptr<int>(),
      metadata_status.data_ptr<int>(), elapsed_state.data_ptr<int>(),
      num_sequences);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor recurrent_add_vec_forward_varlen_cuda(
    torch::Tensor x, torch::Tensor vec) {
  const int C = static_cast<int>(x.size(1));
  const int64_t rows = x.size(0);
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (C == 4096 && rows >= 17 && rows <= 65535) {
    recurrent_add_vec_2d_kernel<<<
        dim3(static_cast<unsigned int>((C / 2 + threads - 1) / threads),
             static_cast<unsigned int>(rows),
        1),
        threads,
        0,
        stream>>>(
        C,
        reinterpret_cast<const half*>(x.data_ptr()),
        reinterpret_cast<const half*>(vec.data_ptr()),
        reinterpret_cast<half*>(out.data_ptr()));
  } else {
    const int64_t total_pairs = x.numel() / 2;
    recurrent_add_vec_kernel<<<
        static_cast<int>((total_pairs + threads - 1) / threads),
        threads,
        0,
        stream>>>(
        C,
        reinterpret_cast<const half*>(x.data_ptr()),
        reinterpret_cast<const half*>(vec.data_ptr()),
        reinterpret_cast<half*>(out.data_ptr()),
        total_pairs);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void recurrent_fp16_from_decay_logits_cuda(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor elapsed_state,
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
  if (max_seqlen <= 0) {
    max_seqlen = INT32_MAX;
  }
  const bool all_t1 = max_seqlen == 1;
  const bool seq_v2 = !all_t1 && use_v2_seq(num_sequences, max_seqlen);
  const bool add_w0 = decay_bias.defined();
  const bool grid2d = use_grid2d(num_sequences, max_seqlen, num_heads);

  if (state.size(2) == kHeadSize) {
    if (all_t1) {
      if (num_sequences <= 2) {
        dispatch_fp16_family<Fp16Family::Clone>(
            add_w0, grid2d, num_sequences, num_heads, query_start_loc,
            state_indices, elapsed_state, state, r, decay_logits, decay_bias,
            k, v, a, b, output, metadata_status, static_cast<float>(scale),
            stream);
      } else if (num_sequences <= 64) {
        dispatch_fp16_family<Fp16Family::OneCp>(
            add_w0, grid2d, num_sequences, num_heads, query_start_loc,
            state_indices, elapsed_state, state, r, decay_logits, decay_bias,
            k, v, a, b, output, metadata_status, static_cast<float>(scale),
            stream);
      } else if (num_sequences <= 128) {
        dispatch_fp16_family<Fp16Family::OneDirect>(
            add_w0, grid2d, num_sequences, num_heads, query_start_loc,
            state_indices, elapsed_state, state, r, decay_logits, decay_bias,
            k, v, a, b, output, metadata_status, static_cast<float>(scale),
            stream);
      } else {
        dispatch_fp16_family<Fp16Family::Clone>(
            add_w0, grid2d, num_sequences, num_heads, query_start_loc,
            state_indices, elapsed_state, state, r, decay_logits, decay_bias,
            k, v, a, b, output, metadata_status, static_cast<float>(scale),
            stream);
      }
    } else if (seq_v2) {
      dispatch_fp16_family<Fp16Family::SeqV2>(
          add_w0, grid2d, num_sequences, num_heads, query_start_loc,
          state_indices, elapsed_state, state, r, decay_logits, decay_bias, k,
          v, a, b, output, metadata_status, static_cast<float>(scale), stream);
    } else {
      dispatch_fp16_family<Fp16Family::Exact>(
          add_w0, grid2d, num_sequences, num_heads, query_start_loc,
          state_indices, elapsed_state, state, r, decay_logits, decay_bias, k,
          v, a, b, output, metadata_status, static_cast<float>(scale), stream);
    }
  } else if (state.size(2) == 128) {
    launch_fp16_tiled_column<128>(
        num_sequences, num_heads, query_start_loc, state_indices,
        elapsed_state, state, r, decay_logits, decay_bias, k, v, a, b,
        output, metadata_status, static_cast<float>(scale), stream);
  } else if (state.size(2) == 256) {
    launch_fp16_tiled_column<256>(
        num_sequences, num_heads, query_start_loc, state_indices,
        elapsed_state, state, r, decay_logits, decay_bias, k, v, a, b,
        output, metadata_status, static_cast<float>(scale), stream);
  } else {
    TORCH_CHECK(
        false,
        "FP16 recurrent family supports head sizes 64, 128, and 256");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
