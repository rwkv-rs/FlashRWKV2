// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Albatross project
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
//
// Internal sparse implementation owned by the complete CMix operator.
// Source: BlinkDL/Albatross/faster3a_2607/cuda/rwkv7_fast_ops_fp16.cu,
// revision ee3308f6922e59f2166c7fac3c5a192340a2b48e.
//
// The reachable sparse ReLU-square/down projection, split accumulators, and
// T512 kernels are copied from the canonical source.  The standalone sparse
// up and full-pipeline launchers are intentionally omitted: the complete CMix
// owner performs its front end and up projection before selecting this path.
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

__global__ void zero_vec4_kernel(dtype* __restrict__ out, int64_t n_vec4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n_vec4) {
    reinterpret_cast<int4*>(out)[i] = make_int4(0, 0, 0, 0);
  }
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
