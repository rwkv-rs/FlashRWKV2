// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Albatross project
//
// Canonical body source:
//   BlinkDL/Albatross/faster3a_2607/cuda/rwkv7_wkv_fp16_v2.cu
//   revision 3e41bc43ed5e8332927ddd7e0ce4816cf200a6ea
//   Apache-2.0
//
// Additional exact helper body source:
//   BlinkDL/Albatross/faster3a_2607/cuda/rwkv7_fast_ops_fp16.cu
//   revision 3e41bc43ed5e8332927ddd7e0ce4816cf200a6ea
//
// Local adaptation:
//   - packed [total_tokens,H,D] token storage and cu_seqlens boundaries;
//   - state_indices-backed FP16 state pool in FlashRWKV2's [K,V] layout;
//   - raw decay logits and optional bias fused into the retention transform;
//   - clone, exact, seq-v2, one-cp and one-direct token pipelines retained for
//     Albatross-shaped dispatch, with their old shared state transpose replaced
//     by the 3e41bc4 direct [K,V] load/store body;
//   - Albatross half2 arithmetic and cp.async token staging are preserved for
//     the canonical D=64 implementation;
//   - D=128/256 use a local 64-key-tiled recurrent family. Each warp owns one
//     value column, keeps only the active 64-key tile live, and updates the
//     FP16 state pool directly; no FP32-state fallback is selected.
//
// vllm-rwkv at 6d683f9e49a2997e405c47edc147872c8609513b is a varlen and
// state-layout reference only, not the primary implementation source.

#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include <cuda_fp16.h>
#include "validation.h"

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
    const torch::stable::Tensor& query_start_loc,
    const torch::stable::Tensor& state_indices,
    const torch::stable::Tensor& elapsed_state,
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
  wkv_fp16_tiled_column_kernel<HeadSize>
      <<<dim3(HeadSize, num_heads, num_sequences), 32, 0, stream>>>(
          num_heads, output.numel(), query_start_loc.mutable_data_ptr<int>(),
          state_indices.mutable_data_ptr<int>(), elapsed_state.mutable_data_ptr<int>(),
          metadata_status.mutable_data_ptr<int>(),
          reinterpret_cast<half*>(state.mutable_data_ptr()),
          reinterpret_cast<const half*>(r.mutable_data_ptr()),
          reinterpret_cast<const half*>(decay_logits.mutable_data_ptr()),
          decay_bias.defined()
              ? reinterpret_cast<const half*>(decay_bias.mutable_data_ptr())
              : nullptr,
          reinterpret_cast<const half*>(k.mutable_data_ptr()),
          reinterpret_cast<const half*>(v.mutable_data_ptr()),
          reinterpret_cast<const half*>(a.mutable_data_ptr()),
          reinterpret_cast<const half*>(b.mutable_data_ptr()),
          reinterpret_cast<half*>(output.mutable_data_ptr()), scale);
}

// Albatross 3e41bc4 replaces the old layout-compatibility transpose outright:
// the public state is already [K,V], and thread V directly loads/stores its
// adjacent K pairs.  The five established token pipelines below all use this
// native implementation; there is no remaining slow transpose fallback.
__device__ __forceinline__ void load_state_kv(
    const half* state_base,
    half2* state) {
  const int thread = static_cast<int>(threadIdx.x);
#pragma unroll
  for (int pair_index = 0; pair_index < kHalf2HeadSize; ++pair_index) {
    state[pair_index] = __halves2half2(
        __ldg(state_base + (2 * pair_index) * kHeadSize + thread),
        __ldg(state_base + (2 * pair_index + 1) * kHeadSize + thread));
  }
}

__device__ __forceinline__ void store_state_kv(
    half* state_base,
    const half2* state) {
  const int thread = static_cast<int>(threadIdx.x);
#pragma unroll
  for (int pair_index = 0; pair_index < kHalf2HeadSize; ++pair_index) {
    state_base[(2 * pair_index) * kHeadSize + thread] = state[pair_index].x;
    state_base[(2 * pair_index + 1) * kHeadSize + thread] = state[pair_index].y;
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
  half2 state[kHalf2HeadSize];
  load_state_kv(state_base, state);

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
  store_state_kv(state_base, state);
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
  half2 state[kHalf2HeadSize];
  load_state_kv(state_base, state);

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
  store_state_kv(state_base, state);
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
  half2 state[kHalf2HeadSize];
  load_state_kv(state_base, state);

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
  store_state_kv(state_base, state);
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
  half2 state[kHalf2HeadSize];
  load_state_kv(state_base, state);

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
  store_state_kv(state_base, state);
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
  half2 state[kHalf2HeadSize];
  load_state_kv(state_base, state);

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
  store_state_kv(state_base, state);
}

// Native [K,V] family from Albatross 3e41bc4.  Each lane owns two adjacent V
// columns and keeps all K rows in registers, so state traffic stays coalesced
// without the compatibility transpose used by the older FP16 families.
template <bool AddW0, bool StreamState>
__global__ __launch_bounds__(32, 8) void wkv_fp16_kv_warp_pair_kernel(
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
  const int head_index = static_cast<int>(blockIdx.x);
  const int sequence_index = static_cast<int>(blockIdx.y);
  const int lane = static_cast<int>(threadIdx.x);
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

  int token_start = lane == 0 ? query_start_loc[sequence_index] : 0;
  int token_end = lane == 0 ? query_start_loc[sequence_index + 1] : 0;
  int state_slot = lane == 0 ? state_indices[sequence_index] : 0;
  int elapsed_base = lane == 0 ? elapsed_state_ptr[state_slot] : 0;
  token_start = __shfl_sync(0xffffffffu, token_start, 0);
  token_end = __shfl_sync(0xffffffffu, token_end, 0);
  state_slot = __shfl_sync(0xffffffffu, state_slot, 0);
  elapsed_base = __shfl_sync(0xffffffffu, elapsed_base, 0);

  half* const state_base =
      state_ptr +
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
          kHeadSize * kHeadSize;
  half2 state[kHeadSize];
#pragma unroll
  for (int key_index = 0; key_index < kHeadSize; ++key_index) {
    const half2* const state_row =
        reinterpret_cast<const half2*>(
            state_base + key_index * kHeadSize) + lane;
    if constexpr (StreamState) {
      state[key_index] = __ldcs(state_row);
    } else {
      state[key_index] = __ldg(state_row);
    }
  }

  __shared__ __align__(128) half2 shared_r[kHalf2HeadSize];
  __shared__ __align__(128) half2 shared_decay[kHalf2HeadSize];
  __shared__ __align__(128) half2 shared_k[kHalf2HeadSize];
  __shared__ __align__(128) half2 shared_a[kHalf2HeadSize];
  __shared__ __align__(128) half2 shared_b[kHalf2HeadSize];
  for (int token_index = token_start; token_index < token_end; ++token_index) {
    const int64_t token_base =
        (static_cast<int64_t>(token_index) * num_heads + head_index) *
        kHeadSize;
    const int64_t pair_base = token_base / 2 + lane;
    shared_r[lane] = __ldg(reinterpret_cast<const half2*>(r_ptr) + pair_base);
    shared_k[lane] = __ldg(reinterpret_cast<const half2*>(k_ptr) + pair_base);
    shared_a[lane] = __ldg(reinterpret_cast<const half2*>(a_ptr) + pair_base);
    shared_b[lane] = __ldg(reinterpret_cast<const half2*>(b_ptr) + pair_base);
    const half2 raw_decay =
        __ldg(reinterpret_cast<const half2*>(decay_ptr) + pair_base);
    float decay0 = __half2float(raw_decay.x);
    float decay1 = __half2float(raw_decay.y);
    if constexpr (AddW0) {
      const half2 bias = __ldg(
          reinterpret_cast<const half2*>(
              decay_bias_ptr + head_index * kHeadSize) + lane);
      decay0 += __half2float(bias.x);
      decay1 += __half2float(bias.y);
    }
    const int token_offset = token_index - token_start;
    const int phase = elapsed_base + head_index * kHeadSize + 2 * lane +
        token_offset;
    shared_decay[lane] = __halves2half2(
        __float2half_rn(recurrent_fp16_delta(decay0, phase)),
        __float2half_rn(recurrent_fp16_delta(decay1, phase + 1)));
    __syncwarp();

    half2 state_dot_a_even = __float2half2_rn(0.0f);
    half2 state_dot_a_odd = __float2half2_rn(0.0f);
#pragma unroll
    for (int key_pair = 0; key_pair < kHalf2HeadSize; ++key_pair) {
      const half2 av = shared_a[key_pair];
      state_dot_a_even = __hfma2(
          state[2 * key_pair], __halves2half2(av.x, av.x),
          state_dot_a_even);
      state_dot_a_odd = __hfma2(
          state[2 * key_pair + 1], __halves2half2(av.y, av.y),
          state_dot_a_odd);
    }
    const half2 state_dot_a =
        __hadd2(state_dot_a_even, state_dot_a_odd);
    const half2 value =
        __ldg(reinterpret_cast<const half2*>(v_ptr) + pair_base);
    half2 output_even = __float2half2_rn(0.0f);
    half2 output_odd = __float2half2_rn(0.0f);
#pragma unroll
    for (int key_pair = 0; key_pair < kHalf2HeadSize; ++key_pair) {
      const half2 rv = shared_r[key_pair];
      const half2 decay = shared_decay[key_pair];
      const half2 kv = shared_k[key_pair];
      const half2 bv = shared_b[key_pair];
      half2& even = state[2 * key_pair];
      half2& odd = state[2 * key_pair + 1];
      even = __hfma2(
          even, __halves2half2(decay.x, decay.x),
          __hfma2(__halves2half2(kv.x, kv.x), value,
                  __hfma2(state_dot_a, __halves2half2(bv.x, bv.x), even)));
      odd = __hfma2(
          odd, __halves2half2(decay.y, decay.y),
          __hfma2(__halves2half2(kv.y, kv.y), value,
                  __hfma2(state_dot_a, __halves2half2(bv.y, bv.y), odd)));
      output_even =
          __hfma2(even, __halves2half2(rv.x, rv.x), output_even);
      output_odd =
          __hfma2(odd, __halves2half2(rv.y, rv.y), output_odd);
    }
    const half2 result = __hadd2(output_even, output_odd);
    reinterpret_cast<half2*>(output_ptr)[pair_base] = __halves2half2(
        __float2half_rn(__half2float(result.x) * scale),
        __float2half_rn(__half2float(result.y) * scale));
    __syncwarp();
  }

#pragma unroll
  for (int key_index = 0; key_index < kHeadSize; ++key_index) {
    half2* const state_row = reinterpret_cast<half2*>(
        state_base + key_index * kHeadSize) + lane;
    if constexpr (StreamState) {
      __stcs(state_row, state[key_index]);
    } else {
      *state_row = state[key_index];
    }
  }
}

// The very-high-concurrency T1 path retains the Albatross int4 [K,V] stream
// and uses shared memory only for the value-owner register transpose.
template <bool AddW0>
__global__ __launch_bounds__(kHeadSize, 2) void wkv_fp16_kv_vector_kernel(
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
  const int head_index = static_cast<int>(blockIdx.x);
  const int sequence_index = static_cast<int>(blockIdx.y);
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

  __shared__ int token_index;
  __shared__ int state_slot;
  __shared__ int elapsed_base;
  if (thread == 0) {
    token_index = query_start_loc[sequence_index];
    state_slot = state_indices[sequence_index];
    elapsed_base = elapsed_state_ptr[state_slot];
  }
  __syncthreads();
  half* const state_base =
      state_ptr +
      (static_cast<int64_t>(state_slot) * num_heads + head_index) *
          kHeadSize * kHeadSize;
  __shared__ __align__(256) half2
      state_shared[kHeadSize][kHalf2HeadSize];

#pragma unroll
  for (int pair_segment = thread;
       pair_segment < kHalf2HeadSize * (kHeadSize / kHalfPerInt4);
       pair_segment += kHeadSize) {
    const int key_pair = pair_segment >> 3;
    const int segment = pair_segment & 7;
    const int4 lo = __ldcs(
        reinterpret_cast<const int4*>(
            state_base + (2 * key_pair) * kHeadSize) + segment);
    const int4 hi = __ldcs(
        reinterpret_cast<const int4*>(
            state_base + (2 * key_pair + 1) * kHeadSize) + segment);
    const int value0 = segment * kHalfPerInt4;
#pragma unroll
    for (int offset = 0; offset < kHalfPerInt4; ++offset) {
      const int value_index = value0 + offset;
      state_shared[value_index][(value_index & 31) ^ key_pair] =
          __halves2half2(
              reinterpret_cast<const half*>(&lo)[offset],
              reinterpret_cast<const half*>(&hi)[offset]);
    }
  }
  __syncthreads();

  half2 state[kHalf2HeadSize];
#pragma unroll
  for (int key_pair = 0; key_pair < kHalf2HeadSize; ++key_pair) {
    state[key_pair] = state_shared[thread][lane ^ key_pair];
  }
  __shared__ __align__(128) half2 shared_r[kHalf2HeadSize];
  __shared__ __align__(128) half2 shared_decay[kHalf2HeadSize];
  __shared__ __align__(128) half2 shared_k[kHalf2HeadSize];
  __shared__ __align__(128) half2 shared_a[kHalf2HeadSize];
  __shared__ __align__(128) half2 shared_b[kHalf2HeadSize];
  if (thread < kHalf2HeadSize) {
    const int64_t token_base =
        (static_cast<int64_t>(token_index) * num_heads + head_index) *
        kHeadSize;
    const int64_t pair_base = token_base / 2 + thread;
    shared_r[thread] =
        __ldg(reinterpret_cast<const half2*>(r_ptr) + pair_base);
    shared_k[thread] =
        __ldg(reinterpret_cast<const half2*>(k_ptr) + pair_base);
    shared_a[thread] =
        __ldg(reinterpret_cast<const half2*>(a_ptr) + pair_base);
    shared_b[thread] =
        __ldg(reinterpret_cast<const half2*>(b_ptr) + pair_base);
    const half2 raw_decay =
        __ldg(reinterpret_cast<const half2*>(decay_ptr) + pair_base);
    float decay0 = __half2float(raw_decay.x);
    float decay1 = __half2float(raw_decay.y);
    if constexpr (AddW0) {
      const half2 bias = __ldg(
          reinterpret_cast<const half2*>(
              decay_bias_ptr + head_index * kHeadSize) + thread);
      decay0 += __half2float(bias.x);
      decay1 += __half2float(bias.y);
    }
    const int phase =
        elapsed_base + head_index * kHeadSize + 2 * thread;
    shared_decay[thread] = __halves2half2(
        __float2half_rn(recurrent_fp16_delta(decay0, phase)),
        __float2half_rn(recurrent_fp16_delta(decay1, phase + 1)));
  }
  __syncthreads();

  half2 state_dot_a = __float2half2_rn(0.0f);
#pragma unroll
  for (int key_pair = 0; key_pair < kHalf2HeadSize; ++key_pair) {
    state_dot_a =
        __hfma2(shared_a[key_pair], state[key_pair], state_dot_a);
  }
  const half state_dot = __hadd(state_dot_a.x, state_dot_a.y);
  const half2 state_dot_pair = __halves2half2(state_dot, state_dot);
  const int64_t token_base =
      (static_cast<int64_t>(token_index) * num_heads + head_index) *
      kHeadSize;
  const half value = __ldg(v_ptr + token_base + thread);
  const half2 value_pair = __halves2half2(value, value);
  half2 result = __float2half2_rn(0.0f);
#pragma unroll
  for (int key_pair = 0; key_pair < kHalf2HeadSize; ++key_pair) {
    half2& state_pair = state[key_pair];
    state_pair = __hfma2(
        state_pair, shared_decay[key_pair],
        __hfma2(shared_k[key_pair], value_pair,
                __hfma2(state_dot_pair, shared_b[key_pair], state_pair)));
    result = __hfma2(state_pair, shared_r[key_pair], result);
  }
  output_ptr[token_base + thread] = __float2half_rn(
      (__half2float(result.x) + __half2float(result.y)) * scale);

#pragma unroll
  for (int key_pair = 0; key_pair < kHalf2HeadSize; ++key_pair) {
    state_shared[thread][lane ^ key_pair] = state[key_pair];
  }
  __syncthreads();
#pragma unroll
  for (int pair_segment = thread;
       pair_segment < kHalf2HeadSize * (kHeadSize / kHalfPerInt4);
       pair_segment += kHeadSize) {
    int4 lo;
    int4 hi;
    const int key_pair = pair_segment >> 3;
    const int segment = pair_segment & 7;
    const int value0 = segment * kHalfPerInt4;
#pragma unroll
    for (int offset = 0; offset < kHalfPerInt4; ++offset) {
      const int value_index = value0 + offset;
      const half2 packed =
          state_shared[value_index][(value_index & 31) ^ key_pair];
      reinterpret_cast<half*>(&lo)[offset] = packed.x;
      reinterpret_cast<half*>(&hi)[offset] = packed.y;
    }
    __stcs(
        reinterpret_cast<int4*>(
            state_base + (2 * key_pair) * kHeadSize) + segment,
        lo);
    __stcs(
        reinterpret_cast<int4*>(
            state_base + (2 * key_pair + 1) * kHeadSize) + segment,
        hi);
  }
}

bool use_kv_vector_auto(int sequence_capacity, int max_seqlen, int num_heads) {
  return max_seqlen == 1 &&
      static_cast<int64_t>(sequence_capacity) * num_heads >= 20000;
}

bool use_kv_warp_auto(int sequence_capacity, int max_seqlen, int num_heads) {
  const int64_t sequence_heads =
      static_cast<int64_t>(sequence_capacity) * num_heads;
  if (max_seqlen == 1) {
    return sequence_heads >= 15000 && sequence_heads < 20000;
  }
  return max_seqlen > 1 && sequence_heads >= 1280;
}

template <bool AddW0>
void launch_kv_warp_pair(
    bool stream_state,
    int num_sequences,
    int num_heads,
    const torch::stable::Tensor& query_start_loc,
    const torch::stable::Tensor& state_indices,
    const torch::stable::Tensor& elapsed_state,
    torch::stable::Tensor& state,
    const torch::stable::Tensor& r,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& decay_bias,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    torch::stable::Tensor& output,
    const torch::stable::Tensor& metadata_status,
    float scale,
    cudaStream_t stream) {
#define LAUNCH_KV_WARP(StreamState) \
  wkv_fp16_kv_warp_pair_kernel<AddW0, StreamState> \
      <<<dim3(num_heads, num_sequences), dim3(32), 0, stream>>>( \
          num_heads, output.numel(), query_start_loc.mutable_data_ptr<int>(), \
          state_indices.mutable_data_ptr<int>(), elapsed_state.mutable_data_ptr<int>(), \
          metadata_status.mutable_data_ptr<int>(), \
          reinterpret_cast<half*>(state.mutable_data_ptr()), \
          reinterpret_cast<const half*>(r.mutable_data_ptr()), \
          reinterpret_cast<const half*>(decay.mutable_data_ptr()), \
          AddW0 ? reinterpret_cast<const half*>(decay_bias.mutable_data_ptr()) : nullptr, \
          reinterpret_cast<const half*>(k.mutable_data_ptr()), \
          reinterpret_cast<const half*>(v.mutable_data_ptr()), \
          reinterpret_cast<const half*>(a.mutable_data_ptr()), \
          reinterpret_cast<const half*>(b.mutable_data_ptr()), \
          reinterpret_cast<half*>(output.mutable_data_ptr()), scale)
  if (stream_state) {
    LAUNCH_KV_WARP(true);
  } else {
    LAUNCH_KV_WARP(false);
  }
#undef LAUNCH_KV_WARP
}

template <bool AddW0>
void launch_kv_vector(
    int num_sequences,
    int num_heads,
    const torch::stable::Tensor& query_start_loc,
    const torch::stable::Tensor& state_indices,
    const torch::stable::Tensor& elapsed_state,
    torch::stable::Tensor& state,
    const torch::stable::Tensor& r,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& decay_bias,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    torch::stable::Tensor& output,
    const torch::stable::Tensor& metadata_status,
    float scale,
    cudaStream_t stream) {
  wkv_fp16_kv_vector_kernel<AddW0>
      <<<dim3(num_heads, num_sequences), dim3(kHeadSize), 0, stream>>>(
          num_heads, output.numel(), query_start_loc.mutable_data_ptr<int>(),
          state_indices.mutable_data_ptr<int>(), elapsed_state.mutable_data_ptr<int>(),
          metadata_status.mutable_data_ptr<int>(),
          reinterpret_cast<half*>(state.mutable_data_ptr()),
          reinterpret_cast<const half*>(r.mutable_data_ptr()),
          reinterpret_cast<const half*>(decay.mutable_data_ptr()),
          AddW0 ? reinterpret_cast<const half*>(decay_bias.mutable_data_ptr()) : nullptr,
          reinterpret_cast<const half*>(k.mutable_data_ptr()),
          reinterpret_cast<const half*>(v.mutable_data_ptr()),
          reinterpret_cast<const half*>(a.mutable_data_ptr()),
          reinterpret_cast<const half*>(b.mutable_data_ptr()),
          reinterpret_cast<half*>(output.mutable_data_ptr()), scale);
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
    const torch::stable::Tensor& query_start_loc,
    const torch::stable::Tensor& state_indices,
    const torch::stable::Tensor& elapsed_state,
    torch::stable::Tensor& state,
    const torch::stable::Tensor& r,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& decay_bias,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    torch::stable::Tensor& output,
    const torch::stable::Tensor& metadata_status,
    float scale,
    cudaStream_t stream) {
  const dim3 grid = Grid2D
      ? dim3(static_cast<unsigned int>(num_heads),
             static_cast<unsigned int>(num_sequences), 1)
      : dim3(static_cast<unsigned int>(num_heads * num_sequences), 1, 1);
  kernel<<<grid, dim3(kHeadSize), 0, stream>>>(
      num_heads, output.numel(), query_start_loc.mutable_data_ptr<int>(),
      state_indices.mutable_data_ptr<int>(), elapsed_state.mutable_data_ptr<int>(),
      metadata_status.mutable_data_ptr<int>(),
      reinterpret_cast<half*>(state.mutable_data_ptr()),
      reinterpret_cast<const half*>(r.mutable_data_ptr()),
      reinterpret_cast<const half*>(decay.mutable_data_ptr()),
      decay_bias.defined()
          ? reinterpret_cast<const half*>(decay_bias.mutable_data_ptr())
          : nullptr,
      reinterpret_cast<const half*>(k.mutable_data_ptr()),
      reinterpret_cast<const half*>(v.mutable_data_ptr()),
      reinterpret_cast<const half*>(a.mutable_data_ptr()),
      reinterpret_cast<const half*>(b.mutable_data_ptr()),
      reinterpret_cast<half*>(output.mutable_data_ptr()), scale);
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
    const torch::stable::Tensor& query_start_loc,
    const torch::stable::Tensor& state_indices,
    const torch::stable::Tensor& elapsed_state,
    torch::stable::Tensor& state,
    const torch::stable::Tensor& r,
    const torch::stable::Tensor& decay,
    const torch::stable::Tensor& decay_bias,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    torch::stable::Tensor& output,
    const torch::stable::Tensor& metadata_status,
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

void recurrent_fp16_advance_i32_varlen_cuda(
    torch::stable::Tensor query_start_loc,
    torch::stable::Tensor state_indices,
    torch::stable::Tensor elapsed_state,
    torch::stable::Tensor metadata_status) {
  const torch::stable::accelerator::DeviceGuard device_guard(elapsed_state.device().index());
  constexpr int threads = 256;
  const int num_sequences = static_cast<int>(state_indices.numel());
  auto stream = flashrwkv2::validation::current_cuda_stream();
  advance_i32_varlen_kernel<<<
      static_cast<int>(ceil_div(num_sequences, threads)), threads, 0, stream>>>(
      query_start_loc.mutable_data_ptr<int>(), state_indices.mutable_data_ptr<int>(),
      metadata_status.mutable_data_ptr<int>(), elapsed_state.mutable_data_ptr<int>(),
      num_sequences);
  FLASHRWKV_CUDA_CHECK(cudaGetLastError());
}

void tmix_wkv7_recurrent_fp16_from_decay_logits_cuda(
    torch::stable::Tensor query_start_loc,
    torch::stable::Tensor state_indices,
    torch::stable::Tensor elapsed_state,
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
  if (max_seqlen <= 0) {
    max_seqlen = INT32_MAX;
  }
  const bool all_t1 = max_seqlen == 1;
  const bool seq_v2 = !all_t1 && use_v2_seq(num_sequences, max_seqlen);
  const bool add_w0 = decay_bias.defined();
  const bool grid2d = use_grid2d(num_sequences, max_seqlen, num_heads);

  if (state.size(2) == kHeadSize) {
    if (use_kv_vector_auto(num_sequences, max_seqlen, num_heads)) {
      if (add_w0) {
        launch_kv_vector<true>(
            num_sequences, num_heads, query_start_loc, state_indices,
            elapsed_state, state, r, decay_logits, decay_bias, k, v, a, b,
            output, metadata_status, static_cast<float>(scale), stream);
      } else {
        launch_kv_vector<false>(
            num_sequences, num_heads, query_start_loc, state_indices,
            elapsed_state, state, r, decay_logits, decay_bias, k, v, a, b,
            output, metadata_status, static_cast<float>(scale), stream);
      }
    } else if (use_kv_warp_auto(num_sequences, max_seqlen, num_heads)) {
      if (add_w0) {
        launch_kv_warp_pair<true>(
            all_t1, num_sequences, num_heads, query_start_loc, state_indices,
            elapsed_state, state, r, decay_logits, decay_bias, k, v, a, b,
            output, metadata_status, static_cast<float>(scale), stream);
      } else {
        launch_kv_warp_pair<false>(
            all_t1, num_sequences, num_heads, query_start_loc, state_indices,
            elapsed_state, state, r, decay_logits, decay_bias, k, v, a, b,
            output, metadata_status, static_cast<float>(scale), stream);
      }
    } else if (all_t1) {
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
    STD_TORCH_CHECK(
        false,
        "FP16 recurrent family supports head sizes 64, 128, and 256");
  }
  FLASHRWKV_CUDA_CHECK(cudaGetLastError());
}
