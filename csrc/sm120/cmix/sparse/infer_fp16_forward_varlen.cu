// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Albatross project
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
//
// Source: BlinkDL/Albatross/faster3a_2607/cuda/rwkv7_fast_ops_fp16.cu,
// revision ee3308f6922e59f2166c7fac3c5a192340a2b48e.
//
// The sparse up-projection, sparse ReLU-square/down projection, split
// accumulators, and T512 kernels are copied from the canonical source.  The
// packed adaptation only changes request-boundary address calculation and
// moves the final shift-state copy into an ordered closure launch.
//
// cmix_sparse_spmv_deterministic_rows_kernel is a FlashRWKV2 extension.  It
// preserves the canonical FP16 tile arithmetic but assigns each output tile
// to one CTA and visits feature tiles in a fixed order, avoiding the canonical
// kernels' cross-CTA FP16 atomicAdd ordering.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

using dtype = at::Half;

constexpr int FFN_SPMV_THREADS = 128;
constexpr int FFN_TILE = 128;

inline int64_t ceil_div(int64_t n, int64_t d) {
  return (n + d - 1) / d;
}

__device__ inline __half2 load_h2(const dtype* ptr) {
  return *reinterpret_cast<const __half2*>(ptr);
}

__device__ inline float load_h1(const dtype* ptr) {
  return __half2float(*reinterpret_cast<const __half*>(ptr));
}

__device__ inline void store_h1(dtype* ptr, float value) {
  *reinterpret_cast<__half*>(ptr) = __float2half_rn(value);
}

__device__ inline void store_h2(dtype* ptr, float x0, float x1) {
  *reinterpret_cast<__half2*>(ptr) = __floats2half2_rn(x0, x1);
}

__device__ inline float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ inline int locate_sequence(
    const int* offsets,
    int batch_size,
    int token) {
  int low = 0;
  int high = batch_size;
  while (low + 1 < high) {
    const int middle = (low + high) >> 1;
    if (offsets[middle] <= token) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return low;
}

template <int THREADS>
__global__ void cmix_sparse_up_one_varlen_kernel(
    int C,
    const dtype* __restrict__ x,
    const dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ key_fc,
    dtype* __restrict__ act,
    const int* __restrict__ state_indices,
    const int* __restrict__ metadata_status) {
  // This body is the Albatross cmix_sparse_up_one_kernel.  state_indices[0]
  // is the only packed address adaptation for the B=1,T=1 family.
  const int f = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (metadata_status[0] != 0) {
    if (tid == 0) {
      store_h1(act + f, __int_as_float(0x7fffffff));
    }
    return;
  }
  if (metadata_status[1] == 0 || metadata_status[2] == 0) {
    return;
  }

  const int slot = state_indices[0];
  float acc = 0.0f;
  const auto x2 = reinterpret_cast<const __half2*>(x);
  const auto p2 = reinterpret_cast<const __half2*>(shift_state + static_cast<int64_t>(slot) * C);
  const auto k2 = reinterpret_cast<const __half2*>(x_k);
  const auto w2 = reinterpret_cast<const __half2*>(key_fc + static_cast<int64_t>(f) * C);
  const int n = C / 2;
  for (int j = tid; j < n; j += THREADS) {
    const float2 xv = __half22float2(x2[j]);
    const float2 pv = __half22float2(p2[j]);
    const float2 kv = __half22float2(k2[j]);
    const float2 wv = __half22float2(w2[j]);
    acc = fmaf(xv.x + (pv.x - xv.x) * kv.x, wv.x, acc);
    acc = fmaf(xv.y + (pv.y - xv.y) * kv.y, wv.y, acc);
  }

  acc = warp_sum(acc);
  __shared__ float warp_sums[THREADS / 32];
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    float total = lane < (THREADS / 32) ? warp_sums[lane] : 0.0f;
    total = warp_sum(total);
    if (lane == 0) {
      const float relu = fmaxf(total, 0.0f);
      store_h1(act + f, relu * relu);
    }
  }
}

template <int THREADS>
__global__ void cmix_sparse_up_rows_varlen_kernel(
    int batch_size,
    int total_tokens,
    int C,
    int F,
    const dtype* __restrict__ x,
    const dtype* __restrict__ shift_state,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ key_fc,
    dtype* __restrict__ act,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ state_indices,
    const int* __restrict__ metadata_status) {
  // This is the Albatross cmix_sparse_up_rows_kernel with token rows packed
  // by cu_seqlens rather than laid out as a rectangular B*T tensor.
  const int f = blockIdx.x;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (row >= total_tokens || f >= F) {
    return;
  }
  if (metadata_status[0] != 0) {
    if (tid == 0) {
      store_h1(act + static_cast<int64_t>(row) * F + f,
               __int_as_float(0x7fffffff));
    }
    return;
  }
  if (row >= metadata_status[1]) {
    return;
  }

  const int sequence = locate_sequence(cu_seqlens, metadata_status[2], row);
  const int token_start = cu_seqlens[sequence];
  const int slot = state_indices[sequence];
  float acc = 0.0f;

  const auto x2 = reinterpret_cast<const __half2*>(
      x + static_cast<int64_t>(row) * C);
  const auto p2 = row == token_start
      ? reinterpret_cast<const __half2*>(shift_state + static_cast<int64_t>(slot) * C)
      : reinterpret_cast<const __half2*>(x + static_cast<int64_t>(row - 1) * C);
  const auto k2 = reinterpret_cast<const __half2*>(x_k);
  const auto w2 = reinterpret_cast<const __half2*>(key_fc + static_cast<int64_t>(f) * C);
  const int n = C / 2;
  for (int j = tid; j < n; j += THREADS) {
    const float2 xv = __half22float2(x2[j]);
    const float2 pv = __half22float2(p2[j]);
    const float2 kv = __half22float2(k2[j]);
    const float2 wv = __half22float2(w2[j]);
    acc = fmaf(xv.x + (pv.x - xv.x) * kv.x, wv.x, acc);
    acc = fmaf(xv.y + (pv.y - xv.y) * kv.y, wv.y, acc);
  }

  acc = warp_sum(acc);
  __shared__ float warp_sums[THREADS / 32];
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    float total = lane < (THREADS / 32) ? warp_sums[lane] : 0.0f;
    total = warp_sum(total);
    if (lane == 0) {
      const float relu = fmaxf(total, 0.0f);
      store_h1(act + static_cast<int64_t>(row) * F + f, relu * relu);
    }
  }
}

__global__ void update_shift_state_varlen_kernel(
    int batch_size,
    int C,
    const dtype* __restrict__ x,
    dtype* __restrict__ shift_state,
    const int* __restrict__ cu_seqlens,
    const int* __restrict__ state_indices,
    const int* __restrict__ metadata_status) {
  const int sequence = blockIdx.x;
  const int c4 = blockIdx.y * blockDim.x + threadIdx.x;
  if (sequence >= batch_size || c4 >= C / 8 || metadata_status[0] != 0 ||
      sequence >= metadata_status[2]) {
    return;
  }
  const int token = cu_seqlens[sequence + 1] - 1;
  const int slot = state_indices[sequence];
  reinterpret_cast<int4*>(shift_state + static_cast<int64_t>(slot) * C)[c4] =
      reinterpret_cast<const int4*>(x + static_cast<int64_t>(token) * C)[c4];
}

__global__ void zero_vec4_kernel(dtype* __restrict__ out, int64_t n_vec4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n_vec4) {
    reinterpret_cast<int4*>(out)[i] = make_int4(0, 0, 0, 0);
  }
}

__global__ __launch_bounds__(FFN_SPMV_THREADS, 4) void cmix_sparse_spmv_one_kernel(
    int C,
    const dtype* __restrict__ act,
    const dtype* __restrict__ value_fc,
    dtype* __restrict__ out) {
  // Canonical Albatross cmix_sparse_spmv_one_kernel.
  __shared__ __align__(256) __half mat_row_smem[2][2 * FFN_SPMV_THREADS];
  __shared__ __align__(256) __half vec_slice[FFN_TILE];
  __shared__ __align__(256) int nnz_ids[FFN_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * FFN_TILE;

  if (tid < FFN_TILE / 2) {
    *reinterpret_cast<__half2*>(vec_slice + tid * 2) =
        *reinterpret_cast<const __half2*>(act + start_f + tid * 2);
  }
  __syncthreads();

  bool nonzero = false;
  int local_pos = 0;
  if (tid < FFN_TILE) {
    nonzero = bool(__half_as_ushort(vec_slice[tid]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    local_pos = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) {
      warp_counts[warp_id] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < FFN_TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

  if (tid < FFN_TILE && nonzero) {
    nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  for (int i = 0; i < nnz_count; ++i) {
    const int actual_f = start_f + nnz_ids[i];
    const __half2 mat = *reinterpret_cast<const __half2*>(
        value_fc + static_cast<int64_t>(actual_f) * C +
        c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
    acc = __hfma2(__half2half2(vec_slice[nnz_ids[i]]), mat, acc);
  }
  atomicAdd(
      reinterpret_cast<__half2*>(out + c_block * (2 * FFN_SPMV_THREADS) + tid * 2),
      acc);
}

__global__ __launch_bounds__(FFN_SPMV_THREADS, 4) void cmix_sparse_spmv_rows_kernel(
    int C,
    int F,
    const dtype* __restrict__ act,
    const dtype* __restrict__ value_fc,
    dtype* __restrict__ out) {
  // Canonical Albatross cmix_sparse_spmv_rows_kernel.  blockIdx.z is already
  // a packed row, so no request metadata is needed in this tokenwise stage.
  __shared__ __align__(256) __half vec_slice[FFN_TILE];
  __shared__ __align__(256) int nnz_ids[FFN_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * FFN_TILE;
  const dtype* act_row = act + static_cast<int64_t>(row) * F;

  if (tid < FFN_TILE / 2) {
    *reinterpret_cast<__half2*>(vec_slice + tid * 2) =
        *reinterpret_cast<const __half2*>(act_row + start_f + tid * 2);
  }
  __syncthreads();

  bool nonzero = false;
  int local_pos = 0;
  if (tid < FFN_TILE) {
    nonzero = bool(__half_as_ushort(vec_slice[tid]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    local_pos = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) {
      warp_counts[warp_id] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < FFN_TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

  if (tid < FFN_TILE && nonzero) {
    nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  for (int i = 0; i < nnz_count; ++i) {
    const int actual_f = start_f + nnz_ids[i];
    const __half2 mat = *reinterpret_cast<const __half2*>(
        value_fc + static_cast<int64_t>(actual_f) * C +
        c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
    acc = __hfma2(__half2half2(vec_slice[nnz_ids[i]]), mat, acc);
  }
  atomicAdd(
      reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C +
                                 c_block * (2 * FFN_SPMV_THREADS) + tid * 2),
      acc);
}

template <bool Split2>
__global__ __launch_bounds__(FFN_SPMV_THREADS, 4) void cmix_sparse_spmv_relu_one_kernel(
    int C,
    const dtype* __restrict__ preact,
    const dtype* __restrict__ value_fc,
    dtype* __restrict__ out) {
  // Canonical Albatross cmix_sparse_spmv_relu_one_kernel.
  __shared__ __align__(256) __half vec_slice[FFN_TILE];
  __shared__ __align__(256) int nnz_ids[FFN_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * FFN_TILE;

  if (tid < FFN_TILE) {
    const float value = fmaxf(load_h1(preact + start_f + tid), 0.0f);
    vec_slice[tid] = __float2half_rn(value * value);
  }
  __syncthreads();

  bool nonzero = false;
  int local_pos = 0;
  if (tid < FFN_TILE) {
    nonzero = bool(__half_as_ushort(vec_slice[tid]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    local_pos = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) {
      warp_counts[warp_id] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < FFN_TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

  if (tid < FFN_TILE && nonzero) {
    nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  if constexpr (Split2) {
    __half2 acc1;
    *reinterpret_cast<int*>(&acc1) = 0;
    for (int i = 0; i < nnz_count; i += 2) {
      const int local_f0 = nnz_ids[i];
      const __half2 mat0 = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(start_f + local_f0) * C +
          c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[local_f0]), mat0, acc);
      if (i + 1 < nnz_count) {
        const int local_f1 = nnz_ids[i + 1];
        const __half2 mat1 = *reinterpret_cast<const __half2*>(
            value_fc + static_cast<int64_t>(start_f + local_f1) * C +
            c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
        acc1 = __hfma2(__half2half2(vec_slice[local_f1]), mat1, acc1);
      }
    }
    acc = __hadd2(acc, acc1);
  } else {
    for (int i = 0; i < nnz_count; ++i) {
      const int actual_f = start_f + nnz_ids[i];
      const __half2 mat = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(actual_f) * C +
          c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[nnz_ids[i]]), mat, acc);
    }
  }
  atomicAdd(
      reinterpret_cast<__half2*>(out + c_block * (2 * FFN_SPMV_THREADS) + tid * 2),
      acc);
}

template <bool Split2>
__global__ __launch_bounds__(FFN_SPMV_THREADS, 4) void cmix_sparse_spmv_relu_rows_kernel(
    int C,
    int F,
    const dtype* __restrict__ preact,
    const dtype* __restrict__ value_fc,
    dtype* __restrict__ out) {
  // Canonical Albatross cmix_sparse_spmv_relu_rows_kernel.
  __shared__ __align__(256) __half vec_slice[FFN_TILE];
  __shared__ __align__(256) int nnz_ids[FFN_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * FFN_TILE;
  const dtype* pre_row = preact + static_cast<int64_t>(row) * F;

  if (tid < FFN_TILE) {
    const float value = fmaxf(load_h1(pre_row + start_f + tid), 0.0f);
    vec_slice[tid] = __float2half_rn(value * value);
  }
  __syncthreads();

  bool nonzero = false;
  int local_pos = 0;
  if (tid < FFN_TILE) {
    nonzero = bool(__half_as_ushort(vec_slice[tid]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    local_pos = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) {
      warp_counts[warp_id] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < FFN_TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

  if (tid < FFN_TILE && nonzero) {
    nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  if constexpr (Split2) {
    __half2 acc1;
    *reinterpret_cast<int*>(&acc1) = 0;
    for (int i = 0; i < nnz_count; i += 2) {
      const int local_f0 = nnz_ids[i];
      const __half2 mat0 = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(start_f + local_f0) * C +
          c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[local_f0]), mat0, acc);
      if (i + 1 < nnz_count) {
        const int local_f1 = nnz_ids[i + 1];
        const __half2 mat1 = *reinterpret_cast<const __half2*>(
            value_fc + static_cast<int64_t>(start_f + local_f1) * C +
            c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
        acc1 = __hfma2(__half2half2(vec_slice[local_f1]), mat1, acc1);
      }
    }
    acc = __hadd2(acc, acc1);
  } else {
    for (int i = 0; i < nnz_count; ++i) {
      const int actual_f = start_f + nnz_ids[i];
      const __half2 mat = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(actual_f) * C +
          c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[nnz_ids[i]]), mat, acc);
    }
  }
  atomicAdd(
      reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C +
                                 c_block * (2 * FFN_SPMV_THREADS) + tid * 2),
      acc);
}

template <int Accumulators>
__global__ __launch_bounds__(256, 2) void cmix_sparse_spmv_relu_rows_t512_kernel(
    int C,
    int F,
    const dtype* __restrict__ preact,
    const dtype* __restrict__ value_fc,
    dtype* __restrict__ out) {
  // Canonical Albatross cmix_sparse_spmv_relu_rows_t512_kernel.
  constexpr int TILE = 512;
  constexpr int THREADS = 256;
  __shared__ __align__(256) __half vec_slice[TILE];
  __shared__ __align__(256) int nnz_ids[TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[TILE / 32];
  __shared__ int warp_prefix[TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * TILE;
  const dtype* pre_row = preact + static_cast<int64_t>(row) * F;

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const float value = fmaxf(load_h1(pre_row + start_f + local_f), 0.0f);
    vec_slice[local_f] = __float2half_rn(value * value);
  }
  __syncthreads();

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const bool nonzero = bool(__half_as_ushort(vec_slice[local_f]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    if (lane == 0) {
      warp_counts[warp_id + u * (THREADS / 32)] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const bool nonzero = bool(__half_as_ushort(vec_slice[local_f]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    const int local_pos = __popc(mask & ((1u << lane) - 1u));
    const int group = warp_id + u * (THREADS / 32);
    if (nonzero) {
      nnz_ids[warp_prefix[group] + local_pos] = local_f;
    }
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  if constexpr (Accumulators == 1) {
    for (int i = 0; i < nnz_count; ++i) {
      const int local_f = nnz_ids[i];
      const int actual_f = start_f + local_f;
      const __half2 mat = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(actual_f) * C +
          c_block * (2 * THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[local_f]), mat, acc);
    }
  } else if constexpr (Accumulators == 2) {
    __half2 acc1;
    *reinterpret_cast<int*>(&acc1) = 0;
    for (int i = 0; i < nnz_count; i += 2) {
      const int local_f0 = nnz_ids[i];
      const __half2 mat0 = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(start_f + local_f0) * C +
          c_block * (2 * THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[local_f0]), mat0, acc);
      if (i + 1 < nnz_count) {
        const int local_f1 = nnz_ids[i + 1];
        const __half2 mat1 = *reinterpret_cast<const __half2*>(
            value_fc + static_cast<int64_t>(start_f + local_f1) * C +
            c_block * (2 * THREADS) + tid * 2);
        acc1 = __hfma2(__half2half2(vec_slice[local_f1]), mat1, acc1);
      }
    }
    acc = __hadd2(acc, acc1);
  } else {
    static_assert(Accumulators == 4, "unsupported t512 accumulator count");
    __half2 acc1;
    __half2 acc2;
    __half2 acc3;
    *reinterpret_cast<int*>(&acc1) = 0;
    *reinterpret_cast<int*>(&acc2) = 0;
    *reinterpret_cast<int*>(&acc3) = 0;
    for (int i = 0; i < nnz_count; i += 4) {
      const int local_f0 = nnz_ids[i];
      const __half2 mat0 = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(start_f + local_f0) * C +
          c_block * (2 * THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[local_f0]), mat0, acc);
      if (i + 1 < nnz_count) {
        const int local_f1 = nnz_ids[i + 1];
        const __half2 mat1 = *reinterpret_cast<const __half2*>(
            value_fc + static_cast<int64_t>(start_f + local_f1) * C +
            c_block * (2 * THREADS) + tid * 2);
        acc1 = __hfma2(__half2half2(vec_slice[local_f1]), mat1, acc1);
      }
      if (i + 2 < nnz_count) {
        const int local_f2 = nnz_ids[i + 2];
        const __half2 mat2 = *reinterpret_cast<const __half2*>(
            value_fc + static_cast<int64_t>(start_f + local_f2) * C +
            c_block * (2 * THREADS) + tid * 2);
        acc2 = __hfma2(__half2half2(vec_slice[local_f2]), mat2, acc2);
      }
      if (i + 3 < nnz_count) {
        const int local_f3 = nnz_ids[i + 3];
        const __half2 mat3 = *reinterpret_cast<const __half2*>(
            value_fc + static_cast<int64_t>(start_f + local_f3) * C +
            c_block * (2 * THREADS) + tid * 2);
        acc3 = __hfma2(__half2half2(vec_slice[local_f3]), mat3, acc3);
      }
    }
    acc = __hadd2(__hadd2(acc, acc1), __hadd2(acc2, acc3));
  }
  atomicAdd(
      reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C +
                                 c_block * (2 * THREADS) + tid * 2),
      acc);
}

template <int Accumulators>
__global__ __launch_bounds__(256, 2) void cmix_sparse_spmv_relu_rows_t512_reuse_kernel(
    int C,
    int F,
    const dtype* __restrict__ preact,
    const dtype* __restrict__ value_fc,
    dtype* __restrict__ out) {
  // Canonical Albatross cmix_sparse_spmv_relu_rows_t512_reuse_kernel.
  static_assert(Accumulators == 1 || Accumulators == 2,
                "unsupported reuse accumulator count");
  constexpr int TILE = 512;
  constexpr int THREADS = 256;
  constexpr int WARPS = THREADS / 32;
  __shared__ __align__(256) __half vec_slice[TILE];
  __shared__ __align__(256) int nnz_ids[TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[TILE / 32];
  __shared__ int warp_prefix[TILE / 32];

  const int f_block = blockIdx.x;
  const int c_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int start_f = f_block * TILE;
  const dtype* pre_row = preact + static_cast<int64_t>(row) * F;

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const float value = fmaxf(load_h1(pre_row + start_f + local_f), 0.0f);
    vec_slice[local_f] = __float2half_rn(value * value);
  }
  __syncthreads();

  unsigned masks[2];
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const bool nonzero = bool(__half_as_ushort(vec_slice[local_f]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    masks[u] = mask;
    if (lane == 0) {
      warp_counts[warp_id + u * WARPS] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int s = 0;
#pragma unroll
    for (int w = 0; w < TILE / 32; ++w) {
      warp_prefix[w] = s;
      s += warp_counts[w];
    }
    nnz_count = s;
  }
  __syncthreads();

#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * THREADS;
    const unsigned mask = masks[u];
    const bool nonzero = (mask & (1u << lane)) != 0;
    const int local_pos = __popc(mask & ((1u << lane) - 1u));
    const int group = warp_id + u * WARPS;
    if (nonzero) {
      nnz_ids[warp_prefix[group] + local_pos] = local_f;
    }
  }
  __syncthreads();

  __half2 acc;
  *reinterpret_cast<int*>(&acc) = 0;
  if constexpr (Accumulators == 1) {
    for (int i = 0; i < nnz_count; ++i) {
      const int local_f = nnz_ids[i];
      const __half2 mat = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(start_f + local_f) * C +
          c_block * (2 * THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[local_f]), mat, acc);
    }
  } else {
    __half2 acc1;
    *reinterpret_cast<int*>(&acc1) = 0;
    for (int i = 0; i < nnz_count; i += 2) {
      const int local_f0 = nnz_ids[i];
      const __half2 mat0 = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(start_f + local_f0) * C +
          c_block * (2 * THREADS) + tid * 2);
      acc = __hfma2(__half2half2(vec_slice[local_f0]), mat0, acc);
      if (i + 1 < nnz_count) {
        const int local_f1 = nnz_ids[i + 1];
        const __half2 mat1 = *reinterpret_cast<const __half2*>(
            value_fc + static_cast<int64_t>(start_f + local_f1) * C +
            c_block * (2 * THREADS) + tid * 2);
        acc1 = __hfma2(__half2half2(vec_slice[local_f1]), mat1, acc1);
      }
    }
    acc = __hadd2(acc, acc1);
  }
  atomicAdd(
      reinterpret_cast<__half2*>(out + static_cast<int64_t>(row) * C +
                                 c_block * (2 * THREADS) + tid * 2),
      acc);
}

template <bool ApplyReluSquare>
__global__ __launch_bounds__(FFN_SPMV_THREADS, 4)
void cmix_sparse_spmv_deterministic_rows_kernel(
    int C,
    int F,
    const dtype* __restrict__ input,
    const dtype* __restrict__ value_fc,
    dtype* __restrict__ out) {
  // FlashRWKV2 deterministic extension: a single CTA owns every half2 in its
  // output tile.  Feature-tile partials retain the canonical FP16 association
  // and are combined in monotonically increasing f_block order.
  __shared__ __align__(256) __half vec_slice[FFN_TILE];
  __shared__ __align__(256) int nnz_ids[FFN_TILE];
  __shared__ int nnz_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];

  const int c_block = blockIdx.x;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const dtype* input_row = input + static_cast<int64_t>(row) * F;

  __half2 total;
  *reinterpret_cast<int*>(&total) = 0;
  for (int start_f = 0; start_f < F; start_f += FFN_TILE) {
    if constexpr (ApplyReluSquare) {
      const float value = fmaxf(load_h1(input_row + start_f + tid), 0.0f);
      vec_slice[tid] = __float2half_rn(value * value);
    } else {
      vec_slice[tid] = *reinterpret_cast<const __half*>(
          input_row + start_f + tid);
    }
    __syncthreads();

    const bool nonzero = bool(__half_as_ushort(vec_slice[tid]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    const int local_pos = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) {
      warp_counts[warp_id] = __popc(mask);
    }
    __syncthreads();

    if (tid == 0) {
      int prefix = 0;
#pragma unroll
      for (int warp = 0; warp < FFN_TILE / 32; ++warp) {
        warp_prefix[warp] = prefix;
        prefix += warp_counts[warp];
      }
      nnz_count = prefix;
    }
    __syncthreads();

    if (nonzero) {
      nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
    }
    __syncthreads();

    __half2 tile;
    *reinterpret_cast<int*>(&tile) = 0;
    for (int i = 0; i < nnz_count; ++i) {
      const int actual_f = start_f + nnz_ids[i];
      const __half2 mat = *reinterpret_cast<const __half2*>(
          value_fc + static_cast<int64_t>(actual_f) * C +
          c_block * (2 * FFN_SPMV_THREADS) + tid * 2);
      tile = __hfma2(__half2half2(vec_slice[nnz_ids[i]]), mat, tile);
    }
    total = __hadd2(total, tile);
    __syncthreads();
  }

  *reinterpret_cast<__half2*>(
      out + static_cast<int64_t>(row) * C +
      c_block * (2 * FFN_SPMV_THREADS) + tid * 2) = total;
}

void launch_sparse_down_deterministic(
    int rows,
    int channels,
    int features,
    torch::Tensor input,
    torch::Tensor value_fc,
    torch::Tensor output,
    bool apply_relu_square) {
  TORCH_CHECK(
      channels % (2 * FFN_SPMV_THREADS) == 0,
      "deterministic CMix sparse output channels must be divisible by 256");
  TORCH_CHECK(
      features % FFN_TILE == 0,
      "deterministic CMix sparse features must be divisible by 128");
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(
      static_cast<unsigned int>(channels / (2 * FFN_SPMV_THREADS)),
      static_cast<unsigned int>(rows));
  if (apply_relu_square) {
    cmix_sparse_spmv_deterministic_rows_kernel<true><<<
        grid, FFN_SPMV_THREADS, 0, stream>>>(
        channels,
        features,
        reinterpret_cast<const dtype*>(input.data_ptr()),
        reinterpret_cast<const dtype*>(value_fc.data_ptr()),
        reinterpret_cast<dtype*>(output.data_ptr()));
  } else {
    cmix_sparse_spmv_deterministic_rows_kernel<false><<<
        grid, FFN_SPMV_THREADS, 0, stream>>>(
        channels,
        features,
        reinterpret_cast<const dtype*>(input.data_ptr()),
        reinterpret_cast<const dtype*>(value_fc.data_ptr()),
        reinterpret_cast<dtype*>(output.data_ptr()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_sparse_up(
    int batch_size,
    int total_tokens,
    int channels,
    int features,
    int max_seqlen,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor key_fc,
    torch::Tensor output,
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor metadata_status) {
  constexpr int up_threads = 64;
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (batch_size == 1 && max_seqlen == 1) {
    cmix_sparse_up_one_varlen_kernel<up_threads><<<features, up_threads, 0, stream>>>(
        channels,
        reinterpret_cast<const dtype*>(x.data_ptr()),
        reinterpret_cast<const dtype*>(shift_state.data_ptr()),
        reinterpret_cast<const dtype*>(x_k.data_ptr()),
        reinterpret_cast<const dtype*>(key_fc.data_ptr()),
        reinterpret_cast<dtype*>(output.data_ptr()),
        state_indices.data_ptr<int>(),
        metadata_status.data_ptr<int>());
  } else {
    cmix_sparse_up_rows_varlen_kernel<up_threads><<<
        dim3(features, total_tokens),
        up_threads,
        0,
        stream>>>(
        batch_size,
        total_tokens,
        channels,
        features,
        reinterpret_cast<const dtype*>(x.data_ptr()),
        reinterpret_cast<const dtype*>(shift_state.data_ptr()),
        reinterpret_cast<const dtype*>(x_k.data_ptr()),
        reinterpret_cast<const dtype*>(key_fc.data_ptr()),
        reinterpret_cast<dtype*>(output.data_ptr()),
        query_start_loc.data_ptr<int>(),
        state_indices.data_ptr<int>(),
        metadata_status.data_ptr<int>());
  }

  const int state_vec4 = channels / 8;
  update_shift_state_varlen_kernel<<<
      dim3(batch_size, state_vec4 == 0 ? 1 : (state_vec4 + 127) / 128),
      128,
      0,
      stream>>>(
      batch_size,
      channels,
      reinterpret_cast<const dtype*>(x.data_ptr()),
      reinterpret_cast<dtype*>(shift_state.data_ptr()),
      query_start_loc.data_ptr<int>(),
      state_indices.data_ptr<int>(),
      metadata_status.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_sparse_down(
    int rows,
    int64_t dispatch_rows,
    int channels,
    int features,
    torch::Tensor preact,
    torch::Tensor value_fc,
    torch::Tensor output,
    int accumulators,
    bool reuse,
    bool split2) {
  TORCH_CHECK(channels % 8 == 0, "CMix sparse output channels must be divisible by 8");
  TORCH_CHECK(features % 128 == 0, "CMix sparse features must be divisible by 128");
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t output_vec4 = static_cast<int64_t>(rows) * (channels / 8);
  zero_vec4_kernel<<<static_cast<int>(ceil_div(output_vec4, 128)), 128, 0, stream>>>(
      reinterpret_cast<dtype*>(output.data_ptr()), output_vec4);

  if (dispatch_rows >= 8 && features % 512 == 0 && channels % 512 == 0) {
    if (reuse) {
      TORCH_CHECK(accumulators == 1 || accumulators == 2,
                  "T512 reuse dispatch requires one or two accumulators");
      if (accumulators == 1) {
        cmix_sparse_spmv_relu_rows_t512_reuse_kernel<1><<<
            dim3(features / 512, channels / 512, rows), 256, 0, stream>>>(
            channels,
            features,
            reinterpret_cast<const dtype*>(preact.data_ptr()),
            reinterpret_cast<const dtype*>(value_fc.data_ptr()),
            reinterpret_cast<dtype*>(output.data_ptr()));
      } else {
        cmix_sparse_spmv_relu_rows_t512_reuse_kernel<2><<<
            dim3(features / 512, channels / 512, rows), 256, 0, stream>>>(
            channels,
            features,
            reinterpret_cast<const dtype*>(preact.data_ptr()),
            reinterpret_cast<const dtype*>(value_fc.data_ptr()),
            reinterpret_cast<dtype*>(output.data_ptr()));
      }
    } else if (accumulators == 1) {
      cmix_sparse_spmv_relu_rows_t512_kernel<1><<<
          dim3(features / 512, channels / 512, rows), 256, 0, stream>>>(
          channels,
          features,
          reinterpret_cast<const dtype*>(preact.data_ptr()),
          reinterpret_cast<const dtype*>(value_fc.data_ptr()),
          reinterpret_cast<dtype*>(output.data_ptr()));
    } else if (accumulators == 2) {
      cmix_sparse_spmv_relu_rows_t512_kernel<2><<<
          dim3(features / 512, channels / 512, rows), 256, 0, stream>>>(
          channels,
          features,
          reinterpret_cast<const dtype*>(preact.data_ptr()),
          reinterpret_cast<const dtype*>(value_fc.data_ptr()),
          reinterpret_cast<dtype*>(output.data_ptr()));
    } else {
      cmix_sparse_spmv_relu_rows_t512_kernel<4><<<
          dim3(features / 512, channels / 512, rows), 256, 0, stream>>>(
          channels,
          features,
          reinterpret_cast<const dtype*>(preact.data_ptr()),
          reinterpret_cast<const dtype*>(value_fc.data_ptr()),
          reinterpret_cast<dtype*>(output.data_ptr()));
    }
  } else if (dispatch_rows == 1) {
    if (split2) {
      cmix_sparse_spmv_relu_one_kernel<true><<<
          dim3(features / FFN_TILE, channels / (2 * FFN_SPMV_THREADS), 1),
          FFN_SPMV_THREADS,
          0,
          stream>>>(
          channels,
          reinterpret_cast<const dtype*>(preact.data_ptr()),
          reinterpret_cast<const dtype*>(value_fc.data_ptr()),
          reinterpret_cast<dtype*>(output.data_ptr()));
    } else {
      cmix_sparse_spmv_relu_one_kernel<false><<<
          dim3(features / FFN_TILE, channels / (2 * FFN_SPMV_THREADS), 1),
          FFN_SPMV_THREADS,
          0,
          stream>>>(
          channels,
          reinterpret_cast<const dtype*>(preact.data_ptr()),
          reinterpret_cast<const dtype*>(value_fc.data_ptr()),
          reinterpret_cast<dtype*>(output.data_ptr()));
    }
  } else {
    if (split2) {
      cmix_sparse_spmv_relu_rows_kernel<true><<<
          dim3(features / FFN_TILE, channels / (2 * FFN_SPMV_THREADS), rows),
          FFN_SPMV_THREADS,
          0,
          stream>>>(
          channels,
          features,
          reinterpret_cast<const dtype*>(preact.data_ptr()),
          reinterpret_cast<const dtype*>(value_fc.data_ptr()),
          reinterpret_cast<dtype*>(output.data_ptr()));
    } else {
      cmix_sparse_spmv_relu_rows_kernel<false><<<
          dim3(features / FFN_TILE, channels / (2 * FFN_SPMV_THREADS), rows),
          FFN_SPMV_THREADS,
          0,
          stream>>>(
          channels,
          features,
          reinterpret_cast<const dtype*>(preact.data_ptr()),
          reinterpret_cast<const dtype*>(value_fc.data_ptr()),
          reinterpret_cast<dtype*>(output.data_ptr()));
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor cmix_sparse_up_forward_varlen_cuda(
    int batch_size,
    int total_tokens,
    int channels,
    int features,
    int max_seqlen,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor key_fc,
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor metadata_status) {
  auto act = torch::empty({total_tokens, features}, x.options());
  launch_sparse_up(
      batch_size,
      total_tokens,
      channels,
      features,
      max_seqlen,
      x,
      shift_state,
      x_k,
      key_fc,
      act,
      query_start_loc,
      state_indices,
      metadata_status);
  return act;
}

torch::Tensor cmix_sparse_down_relu_forward_varlen_cuda(
    torch::Tensor preact,
    torch::Tensor value_fc,
    int64_t batch_size,
    int64_t max_seqlen,
    bool deterministic) {
  const int rows = static_cast<int>(preact.size(0));
  const int features = static_cast<int>(preact.size(1));
  const int channels = static_cast<int>(value_fc.size(1));
  auto output = torch::empty({rows, channels}, preact.options());

  if (deterministic) {
    launch_sparse_down_deterministic(
        rows, channels, features, preact, value_fc, output, true);
    return output;
  }

  // Canonical Albatross caller policy.  The direct public entry point does
  // not expose accumulator/reuse selectors; a caller that knows B/T supplies
  // them as dispatch metadata, while an unannotated low-level launch remains
  // on the canonical one-accumulator body.  The sparse table is intentionally
  // restricted to the canonical C=4096,F=16384 operator shape.
  int accumulators = 1;
  bool reuse = false;
  bool split2 = false;
  const int64_t dispatch_rows =
      batch_size > 0 && max_seqlen > 0
      ? batch_size * max_seqlen
      : static_cast<int64_t>(rows);
  if (batch_size > 0 && max_seqlen > 0 && channels == 4096 && features == 16384) {
    const bool use_t512 = dispatch_rows >= 8;
    if (use_t512) {
      const bool accum2 =
          (batch_size == 8 && max_seqlen == 1) ||
          (batch_size == 4 && max_seqlen == 2) ||
          (batch_size == 3 && max_seqlen == 3) ||
          (batch_size == 19 && max_seqlen == 1);
      if (accum2) {
        accumulators = 2;
      }
      reuse =
          (batch_size == 1 && max_seqlen == 8) ||
          (batch_size == 2 && max_seqlen == 4) ||
          (batch_size == 4 && max_seqlen == 2) ||
          (batch_size == 8 && max_seqlen == 1) ||
          (batch_size == 3 && max_seqlen == 3) ||
          (batch_size == 1 && max_seqlen == 11) ||
          (batch_size == 11 && max_seqlen == 1) ||
          (batch_size == 1 && max_seqlen == 12) ||
          (batch_size == 3 && max_seqlen == 4) ||
          (batch_size == 4 && max_seqlen == 3) ||
          (batch_size == 12 && max_seqlen == 1) ||
          (batch_size == 1 && max_seqlen == 13) ||
          (batch_size == 13 && max_seqlen == 1) ||
          (batch_size == 1 && max_seqlen == 14) ||
          (batch_size == 14 && max_seqlen == 1) ||
          (batch_size == 1 && max_seqlen == 15) ||
          (batch_size == 1 && max_seqlen == 16) ||
          (batch_size == 2 && max_seqlen == 8) ||
          (batch_size == 4 && max_seqlen == 4) ||
          (batch_size == 8 && max_seqlen == 2) ||
          (batch_size == 16 && max_seqlen == 1) ||
          (batch_size == 1 && max_seqlen == 17) ||
          (batch_size == 1 && max_seqlen == 19) ||
          (batch_size == 19 && max_seqlen == 1) ||
          (batch_size == 4 && max_seqlen == 5) ||
          (batch_size == 5 && max_seqlen == 4) ||
          (batch_size == 10 && max_seqlen == 2) ||
          (batch_size == 20 && max_seqlen == 1);
    }
    split2 =
        (batch_size == 1 && max_seqlen == 1) ||
        (batch_size == 1 && max_seqlen == 2) ||
        (batch_size == 2 && max_seqlen == 1) ||
        (batch_size == 1 && max_seqlen == 3) ||
        (batch_size == 3 && max_seqlen == 1) ||
        (batch_size == 1 && max_seqlen == 4) ||
        (batch_size == 2 && max_seqlen == 2) ||
        (batch_size == 1 && max_seqlen == 5) ||
        (batch_size == 5 && max_seqlen == 1) ||
        (batch_size == 3 && max_seqlen == 2) ||
        (batch_size == 1 && max_seqlen == 7) ||
        (batch_size == 7 && max_seqlen == 1);
  }
  launch_sparse_down(
      rows,
      dispatch_rows,
      channels,
      features,
      preact,
      value_fc,
      output,
      accumulators,
      reuse,
      split2);
  return output;
}

torch::Tensor cmix_sparse_forward_varlen_cuda(
    int batch_size,
    int total_tokens,
    int channels,
    int features,
    int max_seqlen,
    torch::Tensor x,
    torch::Tensor shift_state,
    torch::Tensor x_k,
    torch::Tensor key_fc,
    torch::Tensor value_fc,
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor metadata_status,
    bool deterministic) {
  auto act = torch::empty({total_tokens, features}, x.options());
  launch_sparse_up(
      batch_size,
      total_tokens,
      channels,
      features,
      max_seqlen,
      x,
      shift_state,
      x_k,
      key_fc,
      act,
      query_start_loc,
      state_indices,
      metadata_status);

  auto output = torch::empty({total_tokens, channels}, x.options());
  if (deterministic) {
    launch_sparse_down_deterministic(
        total_tokens, channels, features, act, value_fc, output, false);
    return output;
  }
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int c_blocks = channels / (2 * FFN_SPMV_THREADS);
  TORCH_CHECK(c_blocks > 0, "CMix sparse output channels are too small");
  const dim3 grid(
      static_cast<unsigned int>(features / FFN_TILE),
      static_cast<unsigned int>(c_blocks),
      static_cast<unsigned int>(total_tokens));
  if (batch_size == 1 && max_seqlen == 1) {
    // The one-row body is selected only for the exact canonical B=1,T=1
    // dispatch.  The packed output remains [1,C].
    zero_vec4_kernel<<<
        static_cast<int>(ceil_div(channels / 8, 128)), 128, 0, stream>>>(
        reinterpret_cast<dtype*>(output.data_ptr()), channels / 8);
    cmix_sparse_spmv_one_kernel<<<
        dim3(features / FFN_TILE, c_blocks, 1), FFN_SPMV_THREADS, 0, stream>>>(
        channels,
        reinterpret_cast<const dtype*>(act.data_ptr()),
        reinterpret_cast<const dtype*>(value_fc.data_ptr()),
        reinterpret_cast<dtype*>(output.data_ptr()));
  } else {
    const int64_t output_vec4 = static_cast<int64_t>(total_tokens) * (channels / 8);
    zero_vec4_kernel<<<static_cast<int>(ceil_div(output_vec4, 128)), 128, 0, stream>>>(
        reinterpret_cast<dtype*>(output.data_ptr()), output_vec4);
    cmix_sparse_spmv_rows_kernel<<<grid, FFN_SPMV_THREADS, 0, stream>>>(
        channels,
        features,
        reinterpret_cast<const dtype*>(act.data_ptr()),
        reinterpret_cast<const dtype*>(value_fc.data_ptr()),
        reinterpret_cast<dtype*>(output.data_ptr()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
