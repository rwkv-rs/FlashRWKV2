// SPDX-License-Identifier: Apache-2.0
// Native-private SM120 Linear provider. This translation unit has no pybind
// registration; public ownership remains with TMix, CMix, and Head callers.
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Upstream repository: https://github.com/BlinkDL/Albatross
// Albatross source revision: ee3308f6922e59f2166c7fac3c5a192340a2b48e
// Original path: faster3a_2607/cuda/rwkv7_v3a_ops.cu
// Mechanical migration boundary: exact upstream linear kernel/device bodies and
// launch dispatch are retained. Local adaptations are the caller-owned packed
// [total_tokens,C] bindings and the vanilla-LoRA composition/scaled accumulation
// below; rows are already contiguous and no sequence metadata is needed for this
// tokenwise operator family.
// SPDX and provenance are retained from the upstream source below.
#include "backend.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <climits>
#include <optional>
#include <vector>

using dtype = at::Half;
namespace wmma = nvcuda::wmma;

namespace {

constexpr int LN_THREADS = 256;
constexpr int LN_SMALL_THREADS = 1024;
constexpr int LN_SMALL512_THREADS = 512;
constexpr int LN_SMALL_C = 4096;

inline int64_t ceil_div(int64_t n, int64_t d) {
  return (n + d - 1) / d;
}

inline void check_cublas(cublasStatus_t status, const char* what) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, what, " failed with cublas status ", static_cast<int>(status));
}

inline void check_cublaslt(cublasStatus_t status, const char* what) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, what, " failed with cublasLt status ", static_cast<int>(status));
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, offset);
  }
  return x;
}

__device__ __forceinline__ float bf16_bits_to_float_dev(uint16_t bits) {
  union {
    uint32_t u;
    float f;
  } v;
  v.u = static_cast<uint32_t>(bits) << 16;
  return v.f;
}

template <int Threads>
__device__ __forceinline__ float block_sum_t(float x) {
  __shared__ float partial[Threads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  x = warp_sum(x);
  if (lane == 0) {
    partial[warp] = x;
  }
  __syncthreads();
  x = (threadIdx.x < (Threads / 32)) ? partial[lane] : 0.0f;
  if (warp == 0) {
    x = warp_sum(x);
  }
  if (threadIdx.x == 0) {
    partial[0] = x;
  }
  __syncthreads();
  return partial[0];
}
template <int ChunkK, int Warps>
__global__ __launch_bounds__(128, 2) void linear_f16_m1_splitk_partial_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight,
    float* __restrict__ partial) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int pair = (blockIdx.x * Warps + warp) * 32 + lane;
  const int n = pair << 1;
  if (n >= N) {
    return;
  }
  const int k0 = blockIdx.y * ChunkK;
  const int k1 = min(k0 + ChunkK, K);
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (int k = k0; k < k1; ++k) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(x + k));
    const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(weight + static_cast<int64_t>(k) * N + n));
    acc0 = fmaf(xv, wv.x, acc0);
    acc1 = fmaf(xv, wv.y, acc1);
  }
  reinterpret_cast<float2*>(partial + static_cast<int64_t>(blockIdx.y) * N)[pair] = make_float2(acc0, acc1);
}

__global__ void linear_f16_m1_splitk_reduce_kernel(
    int chunks,
    int N,
    const float* __restrict__ partial,
    dtype* __restrict__ y) {
  const int pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int n = pair << 1;
  if (n >= N) {
    return;
  }
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (int c = 0; c < chunks; ++c) {
    const float2 v = reinterpret_cast<const float2*>(partial + static_cast<int64_t>(c) * N)[pair];
    acc0 += v.x;
    acc1 += v.y;
  }
  reinterpret_cast<__half2*>(y)[pair] = __floats2half2_rn(acc0, acc1);
}

__global__ void linear_f16_m1_splitk_reduce_warp_kernel(
    int chunks,
    int N,
    const float* __restrict__ partial,
    dtype* __restrict__ y) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int pair = blockIdx.x * 4 + warp;
  const int n = pair << 1;
  if (n >= N) {
    return;
  }
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (int c = lane; c < chunks; c += 32) {
    const float2 v = reinterpret_cast<const float2*>(partial + static_cast<int64_t>(c) * N)[pair];
    acc0 += v.x;
    acc1 += v.y;
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
    acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
  }
  if (lane == 0) {
    reinterpret_cast<__half2*>(y)[pair] = __floats2half2_rn(acc0, acc1);
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 2) void linear_t_f16_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    dtype* __restrict__ y) {
  const int n = blockIdx.x;
  const int m = blockIdx.y;
  if (m >= M || n >= N) {
    return;
  }
  float acc = 0.0f;
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const dtype* w_row = weight_t + static_cast<int64_t>(n) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x_row + (k2 << 1)));
    const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(w_row + (k2 << 1)));
    acc = fmaf(xv.x, wv.x, acc);
    acc = fmaf(xv.y, wv.y, acc);
  }
  if ((K & 1) && threadIdx.x == 0) {
    acc = fmaf(__half2float(*reinterpret_cast<const __half*>(x_row + K - 1)),
               __half2float(*reinterpret_cast<const __half*>(w_row + K - 1)),
               acc);
  }
  acc = block_sum_t<Threads>(acc);
  if (threadIdx.x == 0) {
    *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(acc);
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_t_f16_ntile_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(weight_t + static_cast<int64_t>(n) * K + k));
        acc[j] = fmaf(xv.x, wv.x, acc[j]);
        acc[j] = fmaf(xv.y, wv.y, acc[j]);
      }
    }
  }
  if ((K & 1) && threadIdx.x == 0) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(x_row + K - 1));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(weight_t + static_cast<int64_t>(n) * K + K - 1)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(sum);
      }
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_t_f16_ntile_scalar_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  for (int k = threadIdx.x; k < K; k += Threads) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(x_row + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(weight_t + static_cast<int64_t>(n) * K + k)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(sum);
      }
    }
  }
}

template <int Threads, int OutTile, bool ScalarInput>
__global__ __launch_bounds__(Threads, 2)
void linear_t_f16_ntile_scaled_accumulate_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    dtype* __restrict__ output,
    float scale) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  if constexpr (ScalarInput) {
    for (int k = threadIdx.x; k < K; k += Threads) {
      const float xv =
          __half2float(*reinterpret_cast<const __half*>(x_row + k));
#pragma unroll
      for (int j = 0; j < OutTile; ++j) {
        const int n = n0 + j;
        if (n < N) {
          const float wv = __half2float(*reinterpret_cast<const __half*>(
              weight_t + static_cast<int64_t>(n) * K + k));
          acc[j] = fmaf(xv, wv, acc[j]);
        }
      }
    }
  } else {
    const int K2 = K >> 1;
    for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
      const int k = k2 << 1;
      const float2 xv = __half22float2(
          *reinterpret_cast<const __half2*>(x_row + k));
#pragma unroll
      for (int j = 0; j < OutTile; ++j) {
        const int n = n0 + j;
        if (n < N) {
          const float2 wv = __half22float2(
              *reinterpret_cast<const __half2*>(
                  weight_t + static_cast<int64_t>(n) * K + k));
          acc[j] = fmaf(xv.x, wv.x, acc[j]);
          acc[j] = fmaf(xv.y, wv.y, acc[j]);
        }
      }
    }
    if ((K & 1) && threadIdx.x == 0) {
      const float xv = __half2float(
          *reinterpret_cast<const __half*>(x_row + K - 1));
#pragma unroll
      for (int j = 0; j < OutTile; ++j) {
        const int n = n0 + j;
        if (n < N) {
          const float wv = __half2float(*reinterpret_cast<const __half*>(
              weight_t + static_cast<int64_t>(n) * K + K - 1));
          acc[j] = fmaf(xv, wv, acc[j]);
        }
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        const int64_t index = static_cast<int64_t>(m) * N + n;
        const float base =
            __half2float(*reinterpret_cast<const __half*>(output + index));
        *reinterpret_cast<__half*>(output + index) =
            __float2half_rn(fmaf(scale, sum, base));
      }
    }
  }
}

template <int Threads, int RowTile, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_orig_rows_f16_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_orig,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m0 = blockIdx.y * RowTile;
  float acc[RowTile][OutTile];
#pragma unroll
  for (int r = 0; r < RowTile; ++r) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      acc[r][j] = 0.0f;
    }
  }
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    float2 wv[OutTile];
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      wv[j] = (n < N)
          ? __half22float2(*reinterpret_cast<const __half2*>(weight_orig + static_cast<int64_t>(n) * K + k))
          : make_float2(0.0f, 0.0f);
    }
#pragma unroll
    for (int r = 0; r < RowTile; ++r) {
      const int m = m0 + r;
      if (m < M) {
        const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x + static_cast<int64_t>(m) * K + k));
#pragma unroll
        for (int j = 0; j < OutTile; ++j) {
          acc[r][j] = fmaf(xv.x, wv[j].x, acc[r][j]);
          acc[r][j] = fmaf(xv.y, wv[j].y, acc[r][j]);
        }
      }
    }
  }
  if ((K & 1) && threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        const float wv = __half2float(*reinterpret_cast<const __half*>(weight_orig + static_cast<int64_t>(n) * K + K - 1));
#pragma unroll
        for (int r = 0; r < RowTile; ++r) {
          const int m = m0 + r;
          if (m < M) {
            const float xv = __half2float(*reinterpret_cast<const __half*>(x + static_cast<int64_t>(m) * K + K - 1));
            acc[r][j] = fmaf(xv, wv, acc[r][j]);
          }
        }
      }
    }
  }
  __shared__ float partial[Threads / 32][RowTile][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int r = 0; r < RowTile; ++r) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const float v = warp_sum(acc[r][j]);
      if (lane == 0) {
        partial[warp][r][j] = v;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int r = 0; r < RowTile; ++r) {
      const int m = m0 + r;
      if (m < M) {
#pragma unroll
        for (int j = 0; j < OutTile; ++j) {
          const int n = n0 + j;
          if (n < N) {
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < Threads / 32; ++w) {
              sum += partial[w][r][j];
            }
            *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(sum);
          }
        }
      }
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_orig_row1_exact_f16_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_orig,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  for (int k2 = threadIdx.x; k2 < (K >> 1); k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(weight_orig + static_cast<int64_t>(n0 + j) * K + k));
      acc[j] = fmaf(xv.x, wv.x, acc[j]);
      acc[j] = fmaf(xv.y, wv.y, acc[j]);
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = v;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      y[n0 + j] = __float2half_rn(sum);
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_orig_row1_exact4_f16_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_orig,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  for (int k = threadIdx.x << 2; k < K; k += Threads << 2) {
    const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 2));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const dtype* wj = weight_orig + static_cast<int64_t>(n0 + j) * K + k;
      const float2 w0 = __half22float2(*reinterpret_cast<const __half2*>(wj));
      const float2 w1 = __half22float2(*reinterpret_cast<const __half2*>(wj + 2));
      acc[j] = fmaf(x0.x, w0.x, acc[j]);
      acc[j] = fmaf(x0.y, w0.y, acc[j]);
      acc[j] = fmaf(x1.x, w1.x, acc[j]);
      acc[j] = fmaf(x1.y, w1.y, acc[j]);
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = v;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      y[n0 + j] = __float2half_rn(sum);
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_orig_row2_exact_f16_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_orig,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  float acc0[OutTile];
  float acc1[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc0[j] = 0.0f;
    acc1[j] = 0.0f;
  }
  for (int k2 = threadIdx.x; k2 < (K >> 1); k2 += Threads) {
    const int k = k2 << 1;
    const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(weight_orig + static_cast<int64_t>(n0 + j) * K + k));
      acc0[j] = fmaf(x0.x, wv.x, acc0[j]);
      acc0[j] = fmaf(x0.y, wv.y, acc0[j]);
      acc1[j] = fmaf(x1.x, wv.x, acc1[j]);
      acc1[j] = fmaf(x1.y, wv.y, acc1[j]);
    }
  }
  __shared__ float partial[Threads / 32][2][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v0 = warp_sum(acc0[j]);
    const float v1 = warp_sum(acc1[j]);
    if (lane == 0) {
      partial[warp][0][j] = v0;
      partial[warp][1][j] = v1;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum0 = 0.0f;
      float sum1 = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum0 += partial[w][0][j];
        sum1 += partial[w][1][j];
      }
      const int n = n0 + j;
      y[n] = __float2half_rn(sum0);
      y[N + n] = __float2half_rn(sum1);
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_orig_row2_exact4_f16_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_orig,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  float acc0[OutTile];
  float acc1[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc0[j] = 0.0f;
    acc1[j] = 0.0f;
  }
  for (int k = threadIdx.x << 2; k < K; k += Threads << 2) {
    const float2 x00 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 x01 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 2));
    const float2 x10 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k));
    const float2 x11 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k + 2));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const dtype* wj = weight_orig + static_cast<int64_t>(n0 + j) * K + k;
      const float2 w0 = __half22float2(*reinterpret_cast<const __half2*>(wj));
      const float2 w1 = __half22float2(*reinterpret_cast<const __half2*>(wj + 2));
      acc0[j] = fmaf(x00.x, w0.x, acc0[j]);
      acc0[j] = fmaf(x00.y, w0.y, acc0[j]);
      acc0[j] = fmaf(x01.x, w1.x, acc0[j]);
      acc0[j] = fmaf(x01.y, w1.y, acc0[j]);
      acc1[j] = fmaf(x10.x, w0.x, acc1[j]);
      acc1[j] = fmaf(x10.y, w0.y, acc1[j]);
      acc1[j] = fmaf(x11.x, w1.x, acc1[j]);
      acc1[j] = fmaf(x11.y, w1.y, acc1[j]);
    }
  }
  __shared__ float partial[Threads / 32][2][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v0 = warp_sum(acc0[j]);
    const float v1 = warp_sum(acc1[j]);
    if (lane == 0) {
      partial[warp][0][j] = v0;
      partial[warp][1][j] = v1;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum0 = 0.0f;
      float sum1 = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum0 += partial[w][0][j];
        sum1 += partial[w][1][j];
      }
      const int n = n0 + j;
      y[n] = __float2half_rn(sum0);
      y[N + n] = __float2half_rn(sum1);
    }
  }
}


} // namespace

at::Tensor linear_f16_cuda(at::Tensor x, at::Tensor weight) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight.size(1);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_f16 K/N too large");
  const int k = static_cast<int>(k64);
  const int n = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_f16 M too large");
  const int m = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (m == 0 || n == 0 || k == 0) {
    return y;
  }

  // Row-major y[M,N] = x[M,K] @ weight[K,N] is column-major
  // y^T[N,M] = weight^T[N,K] @ x^T[K,M].
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  check_cublas(cublasGemmEx(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      n,
      m,
      k,
      &alpha,
      weight.data_ptr<dtype>(),
      CUDA_R_16F,
      n,
      x.data_ptr<dtype>(),
      CUDA_R_16F,
      k,
      &beta,
      y.data_ptr<dtype>(),
      CUDA_R_16F,
      n,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "linear_f16 cublasGemmEx");
  return y;
}

at::Tensor linear_f16_orig_cuda(at::Tensor x, at::Tensor weight_orig) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_orig.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_f16_orig K/N too large");
  const int k = static_cast<int>(k64);
  const int n = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_f16_orig M too large");
  const int m = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (m == 0 || n == 0 || k == 0) {
    return y;
  }

  // weight_orig is row-major [N,K], i.e. column-major [K,N].
  // Row-major y[M,N] = x[M,K] @ weight_orig[N,K]^T becomes
  // column-major y^T[N,M] = opT(weight_orig_col[K,N]) @ x_col[K,M].
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  check_cublas(cublasGemmEx(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n,
      m,
      k,
      &alpha,
      weight_orig.data_ptr<dtype>(),
      CUDA_R_16F,
      k,
      x.data_ptr<dtype>(),
      CUDA_R_16F,
      k,
      &beta,
      y.data_ptr<dtype>(),
      CUDA_R_16F,
      n,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "linear_f16_orig cublasGemmEx");
  return y;
}

void linear_f16_orig_scaled_accumulate_cuda(
    at::Tensor x,
    at::Tensor weight_orig,
    at::Tensor output,
    float scale) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_orig.size(0);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX && m64 <= INT_MAX,
              "linear_f16_orig_scaled_accumulate shape is too large");
  const int k = static_cast<int>(k64);
  const int n = static_cast<int>(n64);
  const int m = static_cast<int>(m64);
  TORCH_CHECK(output.dim() == 2 && output.size(0) == m64 &&
                  output.size(1) == n64,
              "LoRA accumulation output shape mismatch");

  // Row-major output[M,N] is column-major output^T[N,M].  beta=1 reads the
  // FP16 base projection into FP32 accumulation, adds the scaled rank-out,
  // then rounds once back to the same FP16 output buffer.
  const float beta = 1.0f;
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  check_cublas(cublasGemmEx(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n,
      m,
      k,
      &scale,
      weight_orig.data_ptr<dtype>(),
      CUDA_R_16F,
      k,
      x.data_ptr<dtype>(),
      CUDA_R_16F,
      k,
      &beta,
      output.data_ptr<dtype>(),
      CUDA_R_16F,
      n,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "linear_f16_orig_scaled_accumulate cublasGemmEx");
}

template <int RowTile, int OutTile>
at::Tensor linear_orig_rows_f16_cuda_impl(at::Tensor x, at::Tensor weight_orig) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_orig.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_orig_rows_f16 K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_orig_rows_f16 M too large");
  const int M = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (M == 0 || N == 0 || K == 0) {
    return y;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_orig_rows_f16_kernel<128, RowTile, OutTile><<<dim3(ceil_div(N, OutTile), ceil_div(M, RowTile), 1), 128, 0, stream>>>(
      M, K, N, x.data_ptr<dtype>(), weight_orig.data_ptr<dtype>(), y.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

template <int Threads, int RowTile, int OutTile>
at::Tensor linear_orig_rows_cfg_f16_cuda_impl(at::Tensor x, at::Tensor weight_orig) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_orig.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_orig_rows_cfg_f16 K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_orig_rows_cfg_f16 M too large");
  const int M = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (M == 0 || N == 0 || K == 0) {
    return y;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_orig_rows_f16_kernel<Threads, RowTile, OutTile><<<dim3(ceil_div(N, OutTile), ceil_div(M, RowTile), 1), Threads, 0, stream>>>(
      M, K, N, x.data_ptr<dtype>(), weight_orig.data_ptr<dtype>(), y.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor linear_orig_rows_f16_cuda(at::Tensor x, at::Tensor weight_orig, int64_t row_tile, int64_t out_tile) {
  if (row_tile == 1 && out_tile == 2) return linear_orig_rows_f16_cuda_impl<1, 2>(x, weight_orig);
  if (row_tile == 1 && out_tile == 4) return linear_orig_rows_f16_cuda_impl<1, 4>(x, weight_orig);
  if (row_tile == 1 && out_tile == 8) return linear_orig_rows_f16_cuda_impl<1, 8>(x, weight_orig);
  if (row_tile == 1 && out_tile == 16) return linear_orig_rows_f16_cuda_impl<1, 16>(x, weight_orig);
  if (row_tile == 2 && out_tile == 2) return linear_orig_rows_f16_cuda_impl<2, 2>(x, weight_orig);
  if (row_tile == 2 && out_tile == 4) return linear_orig_rows_f16_cuda_impl<2, 4>(x, weight_orig);
  if (row_tile == 2 && out_tile == 8) return linear_orig_rows_f16_cuda_impl<2, 8>(x, weight_orig);
  if (row_tile == 3 && out_tile == 2) return linear_orig_rows_f16_cuda_impl<3, 2>(x, weight_orig);
  if (row_tile == 3 && out_tile == 4) return linear_orig_rows_f16_cuda_impl<3, 4>(x, weight_orig);
  if (row_tile == 3 && out_tile == 8) return linear_orig_rows_f16_cuda_impl<3, 8>(x, weight_orig);
  if (row_tile == 4 && out_tile == 2) return linear_orig_rows_f16_cuda_impl<4, 2>(x, weight_orig);
  if (row_tile == 4 && out_tile == 4) return linear_orig_rows_f16_cuda_impl<4, 4>(x, weight_orig);
  if (row_tile == 4 && out_tile == 8) return linear_orig_rows_f16_cuda_impl<4, 8>(x, weight_orig);
  if (row_tile == 8 && out_tile == 2) return linear_orig_rows_f16_cuda_impl<8, 2>(x, weight_orig);
  if (row_tile == 8 && out_tile == 4) return linear_orig_rows_f16_cuda_impl<8, 4>(x, weight_orig);
  if (row_tile == 16 && out_tile == 1) return linear_orig_rows_f16_cuda_impl<16, 1>(x, weight_orig);
  if (row_tile == 16 && out_tile == 2) return linear_orig_rows_f16_cuda_impl<16, 2>(x, weight_orig);
  if (row_tile == 16 && out_tile == 4) return linear_orig_rows_f16_cuda_impl<16, 4>(x, weight_orig);
  TORCH_CHECK(false, "unsupported linear_orig_rows_f16 row_tile/out_tile");
}

at::Tensor linear_orig_rows_cfg_f16_cuda(at::Tensor x, at::Tensor weight_orig, int64_t threads, int64_t row_tile, int64_t out_tile) {
  if (threads == 64 && row_tile == 1 && out_tile == 4) return linear_orig_rows_cfg_f16_cuda_impl<64, 1, 4>(x, weight_orig);
  if (threads == 64 && row_tile == 1 && out_tile == 8) return linear_orig_rows_cfg_f16_cuda_impl<64, 1, 8>(x, weight_orig);
  if (threads == 128 && row_tile == 1 && out_tile == 8) return linear_orig_rows_cfg_f16_cuda_impl<128, 1, 8>(x, weight_orig);
  if (threads == 256 && row_tile == 1 && out_tile == 1) return linear_orig_rows_cfg_f16_cuda_impl<256, 1, 1>(x, weight_orig);
  if (threads == 32 && row_tile == 4 && out_tile == 4) return linear_orig_rows_cfg_f16_cuda_impl<32, 4, 4>(x, weight_orig);
  if (threads == 64 && row_tile == 4 && out_tile == 4) return linear_orig_rows_cfg_f16_cuda_impl<64, 4, 4>(x, weight_orig);
  if (threads == 96 && row_tile == 4 && out_tile == 4) return linear_orig_rows_cfg_f16_cuda_impl<96, 4, 4>(x, weight_orig);
  if (threads == 32 && row_tile == 4 && out_tile == 8) return linear_orig_rows_cfg_f16_cuda_impl<32, 4, 8>(x, weight_orig);
  if (threads == 64 && row_tile == 4 && out_tile == 8) return linear_orig_rows_cfg_f16_cuda_impl<64, 4, 8>(x, weight_orig);
  if (threads == 32 && row_tile == 8 && out_tile == 4) return linear_orig_rows_cfg_f16_cuda_impl<32, 8, 4>(x, weight_orig);
  if (threads == 64 && row_tile == 8 && out_tile == 4) return linear_orig_rows_cfg_f16_cuda_impl<64, 8, 4>(x, weight_orig);
  if (threads == 32 && row_tile == 2 && out_tile == 4) return linear_orig_rows_cfg_f16_cuda_impl<32, 2, 4>(x, weight_orig);
  if (threads == 64 && row_tile == 2 && out_tile == 2) return linear_orig_rows_cfg_f16_cuda_impl<64, 2, 2>(x, weight_orig);
  if (threads == 64 && row_tile == 2 && out_tile == 4) return linear_orig_rows_cfg_f16_cuda_impl<64, 2, 4>(x, weight_orig);
  if (threads == 32 && row_tile == 3 && out_tile == 4) return linear_orig_rows_cfg_f16_cuda_impl<32, 3, 4>(x, weight_orig);
  if (threads == 64 && row_tile == 3 && out_tile == 4) return linear_orig_rows_cfg_f16_cuda_impl<64, 3, 4>(x, weight_orig);
  if (threads == 96 && row_tile == 3 && out_tile == 4) return linear_orig_rows_cfg_f16_cuda_impl<96, 3, 4>(x, weight_orig);
  if (threads == 32 && row_tile == 3 && out_tile == 8) return linear_orig_rows_cfg_f16_cuda_impl<32, 3, 8>(x, weight_orig);
  if (threads == 64 && row_tile == 3 && out_tile == 8) return linear_orig_rows_cfg_f16_cuda_impl<64, 3, 8>(x, weight_orig);
  TORCH_CHECK(false, "unsupported linear_orig_rows_cfg_f16 threads/row_tile/out_tile");
}

template <int Threads, int OutTile, bool Use4>
at::Tensor linear_orig_row1_exact_f16_cuda_impl(at::Tensor x, at::Tensor weight_orig) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_orig.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_orig_row1_exact_f16 K/N too large");
  TORCH_CHECK((n64 % OutTile) == 0, "linear_orig_row1_exact_f16 requires N divisible by out_tile");
  TORCH_CHECK((k64 % (Use4 ? 4 : 2)) == 0, "linear_orig_row1_exact_f16 unsupported K alignment");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 == 1, "linear_orig_row1_exact_f16 requires one row");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if constexpr (Use4) {
    linear_orig_row1_exact4_f16_kernel<Threads, OutTile><<<N / OutTile, Threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        K, N, x.data_ptr<dtype>(), weight_orig.data_ptr<dtype>(), y.data_ptr<dtype>());
  } else {
    linear_orig_row1_exact_f16_kernel<Threads, OutTile><<<N / OutTile, Threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        K, N, x.data_ptr<dtype>(), weight_orig.data_ptr<dtype>(), y.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

template <int Threads, int OutTile, bool Use4>
at::Tensor linear_orig_row2_exact_f16_cuda_impl(at::Tensor x, at::Tensor weight_orig) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_orig.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_orig_row2_exact_f16 K/N too large");
  TORCH_CHECK((n64 % OutTile) == 0, "linear_orig_row2_exact_f16 requires N divisible by out_tile");
  TORCH_CHECK((k64 % (Use4 ? 4 : 2)) == 0, "linear_orig_row2_exact_f16 unsupported K alignment");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 == 2, "linear_orig_row2_exact_f16 requires two rows");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if constexpr (Use4) {
    linear_orig_row2_exact4_f16_kernel<Threads, OutTile><<<N / OutTile, Threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        K, N, x.data_ptr<dtype>(), weight_orig.data_ptr<dtype>(), y.data_ptr<dtype>());
  } else {
    linear_orig_row2_exact_f16_kernel<Threads, OutTile><<<N / OutTile, Threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        K, N, x.data_ptr<dtype>(), weight_orig.data_ptr<dtype>(), y.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor linear_orig_rows_exact_f16_cuda(at::Tensor x, at::Tensor weight_orig, int64_t threads, int64_t out_tile, bool use4) {
  const int64_t rows = x.numel() / x.size(-1);
  if (rows == 1) {
    if (!use4 && threads == 128 && out_tile == 2) return linear_orig_row1_exact_f16_cuda_impl<128, 2, false>(x, weight_orig);
    if (use4 && threads == 128 && out_tile == 2) return linear_orig_row1_exact_f16_cuda_impl<128, 2, true>(x, weight_orig);
  }
  if (rows == 2) {
    if (use4 && threads == 64 && out_tile == 2) return linear_orig_row2_exact_f16_cuda_impl<64, 2, true>(x, weight_orig);
    if (use4 && threads == 256 && out_tile == 1) return linear_orig_row2_exact_f16_cuda_impl<256, 1, true>(x, weight_orig);
    if (!use4 && threads == 128 && out_tile == 2) return linear_orig_row2_exact_f16_cuda_impl<128, 2, false>(x, weight_orig);
  }
  TORCH_CHECK(false, "unsupported linear_orig_rows_exact_f16 rows/threads/out_tile/use4");
}

at::Tensor linear_f16_orig_lt_cfg_impl(
    at::Tensor x, at::Tensor weight_orig, int64_t workspace_mb, int64_t algo_index, bool strict_algo);

at::Tensor linear_f16_orig_lt_cfg_cuda(at::Tensor x, at::Tensor weight_orig, int64_t workspace_mb, int64_t algo_index) {
  return linear_f16_orig_lt_cfg_impl(x, weight_orig, workspace_mb, algo_index, false);
}

at::Tensor linear_f16_orig_lt_cfg_impl(
    at::Tensor x, at::Tensor weight_orig, int64_t workspace_mb, int64_t algo_index, bool strict_algo) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_orig.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_f16_orig_lt_cfg K/N too large");
  const int k = static_cast<int>(k64);
  const int n = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_f16_orig_lt_cfg M too large");
  const int m = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (m == 0 || n == 0 || k == 0) {
    return y;
  }

  const size_t workspace_size = static_cast<size_t>(workspace_mb) << 20;
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size > 0) {
    workspace = at::empty({static_cast<int64_t>(workspace_size)}, x.options().dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }

  static cublasLtHandle_t lt_handle = nullptr;
  if (lt_handle == nullptr) {
    check_cublaslt(cublasLtCreate(&lt_handle), "cublasLtCreate");
  }

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  check_cublaslt(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "linear_f16_orig_lt desc");
  const cublasOperation_t transa = CUBLAS_OP_T;
  const cublasOperation_t transb = CUBLAS_OP_N;
  check_cublaslt(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)), "linear_f16_orig_lt transa");
  check_cublaslt(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)), "linear_f16_orig_lt transb");
  check_cublaslt(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16F, k, n, k), "linear_f16_orig_lt a layout");
  check_cublaslt(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16F, k, m, k), "linear_f16_orig_lt b layout");
  check_cublaslt(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16F, n, m, n), "linear_f16_orig_lt c layout");
  check_cublaslt(cublasLtMatmulPreferenceCreate(&pref), "linear_f16_orig_lt preference");
  check_cublaslt(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)),
                 "linear_f16_orig_lt workspace");

  std::vector<cublasLtMatmulHeuristicResult_t> heuristics(64);
  int returned = 0;
  check_cublaslt(cublasLtMatmulAlgoGetHeuristic(lt_handle, op_desc, a_desc, b_desc, c_desc, c_desc, pref, static_cast<int>(heuristics.size()), heuristics.data(), &returned),
                 "linear_f16_orig_lt heuristic");
  if (returned <= 0 || (strict_algo && algo_index >= returned)) {
    // Strict tuning intentionally rejects many candidates. Release per-call Lt
    // objects before TORCH_CHECK throws, otherwise a sweep leaks host resources.
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(c_desc);
    cublasLtMatrixLayoutDestroy(b_desc);
    cublasLtMatrixLayoutDestroy(a_desc);
    cublasLtMatmulDescDestroy(op_desc);
    TORCH_CHECK(returned > 0, "linear_f16_orig_lt found no algorithm");
    TORCH_CHECK(algo_index < returned,
                "linear_f16_orig_lt requested algorithm ", algo_index, " but only ", returned, " algorithms are available");
  }
  const int selected_algo = algo_index < returned ? static_cast<int>(algo_index) : 0;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublaslt(cublasLtMatmul(
      lt_handle,
      op_desc,
      &alpha,
      weight_orig.data_ptr<dtype>(),
      a_desc,
      x.data_ptr<dtype>(),
      b_desc,
      &beta,
      y.data_ptr<dtype>(),
      c_desc,
      y.data_ptr<dtype>(),
      c_desc,
      &heuristics[selected_algo].algo,
      workspace_ptr,
      workspace_size,
      at::cuda::getCurrentCUDAStream()),
      "linear_f16_orig_lt matmul");
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(c_desc);
  cublasLtMatrixLayoutDestroy(b_desc);
  cublasLtMatrixLayoutDestroy(a_desc);
  cublasLtMatmulDescDestroy(op_desc);
  return y;
}

at::Tensor linear_f16_lt_cfg_impl(
    at::Tensor x, at::Tensor weight, int64_t workspace_mb, int64_t algo_index, bool strict_algo);

at::Tensor linear_f16_lt_cfg_cuda(at::Tensor x, at::Tensor weight, int64_t workspace_mb, int64_t algo_index) {
  return linear_f16_lt_cfg_impl(x, weight, workspace_mb, algo_index, false);
}

at::Tensor linear_f16_lt_cfg_impl(
    at::Tensor x, at::Tensor weight, int64_t workspace_mb, int64_t algo_index, bool strict_algo) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight.size(1);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_f16_lt K/N too large");
  const int k = static_cast<int>(k64);
  const int n = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_f16_lt M too large");
  const int m = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (m == 0 || n == 0 || k == 0) {
    return y;
  }

  const size_t workspace_size = static_cast<size_t>(workspace_mb) << 20;
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size > 0) {
    workspace = at::empty({static_cast<int64_t>(workspace_size)}, x.options().dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }

  static cublasLtHandle_t lt_handle = nullptr;
  if (lt_handle == nullptr) {
    check_cublaslt(cublasLtCreate(&lt_handle), "cublasLtCreate");
  }

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  check_cublaslt(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "cublasLtMatmulDescCreate");
  const cublasOperation_t trans = CUBLAS_OP_N;
  check_cublaslt(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)), "cublasLt set transa");
  check_cublaslt(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)), "cublasLt set transb");
  check_cublaslt(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16F, n, k, n), "cublasLt a layout");
  check_cublaslt(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16F, k, m, k), "cublasLt b layout");
  check_cublaslt(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16F, n, m, n), "cublasLt c layout");
  check_cublaslt(cublasLtMatmulPreferenceCreate(&pref), "cublasLt preference");
  check_cublaslt(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)),
                 "cublasLt set workspace");

  std::vector<cublasLtMatmulHeuristicResult_t> heuristics(64);
  int returned = 0;
  check_cublaslt(cublasLtMatmulAlgoGetHeuristic(lt_handle, op_desc, a_desc, b_desc, c_desc, c_desc, pref, static_cast<int>(heuristics.size()), heuristics.data(), &returned),
                 "cublasLt heuristic");
  if (returned <= 0 || (strict_algo && algo_index >= returned)) {
    // See the original-layout path above: unavailable strict candidates are an
    // expected tuning result, so they must not leak descriptors on every throw.
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(c_desc);
    cublasLtMatrixLayoutDestroy(b_desc);
    cublasLtMatrixLayoutDestroy(a_desc);
    cublasLtMatmulDescDestroy(op_desc);
    TORCH_CHECK(returned > 0, "cublasLt found no algorithm");
    TORCH_CHECK(algo_index < returned,
                "linear_f16_lt requested algorithm ", algo_index, " but only ", returned, " algorithms are available");
  }
  const int selected_algo = algo_index < returned ? static_cast<int>(algo_index) : 0;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublaslt(cublasLtMatmul(
      lt_handle,
      op_desc,
      &alpha,
      weight.data_ptr<dtype>(),
      a_desc,
      x.data_ptr<dtype>(),
      b_desc,
      &beta,
      y.data_ptr<dtype>(),
      c_desc,
      y.data_ptr<dtype>(),
      c_desc,
      &heuristics[selected_algo].algo,
      workspace_ptr,
      workspace_size,
      at::cuda::getCurrentCUDAStream()),
      "cublasLtMatmul");
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(c_desc);
  cublasLtMatrixLayoutDestroy(b_desc);
  cublasLtMatrixLayoutDestroy(a_desc);
  cublasLtMatmulDescDestroy(op_desc);
  return y;
}

template <int ChunkK, int Warps, bool WarpReduce = false>
at::Tensor linear_f16_m1_splitk_cuda_impl(at::Tensor x, at::Tensor weight) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight.size(1);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_f16_m1_splitk K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  TORCH_CHECK(x.numel() == k64, "linear_f16_m1_splitk requires M=1");
  TORCH_CHECK((N % 64) == 0, "linear_f16_m1_splitk requires N multiple of 64");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (K == 0 || N == 0) {
    return y;
  }
  const int chunks = static_cast<int>(ceil_div(K, ChunkK));
  auto partial = at::empty({chunks, n64}, x.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_f16_m1_splitk_partial_kernel<ChunkK, Warps><<<dim3(ceil_div(N, Warps * 64), chunks, 1), Warps * 32, 0, stream>>>(
      K, N, x.data_ptr<dtype>(), weight.data_ptr<dtype>(), partial.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if constexpr (WarpReduce) {
    linear_f16_m1_splitk_reduce_warp_kernel<<<static_cast<int>(ceil_div(N / 2, 4)), 128, 0, stream>>>(
        chunks, N, partial.data_ptr<float>(), y.data_ptr<dtype>());
  } else {
    linear_f16_m1_splitk_reduce_kernel<<<static_cast<int>(ceil_div(N / 2, 128)), 128, 0, stream>>>(
        chunks, N, partial.data_ptr<float>(), y.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor linear_f16_m1_splitk_cuda(at::Tensor x, at::Tensor weight) {
  const int64_t K = x.size(-1);
  const int64_t N = weight.size(1);
  if (K == 4096 && N == 4096) {
    return linear_f16_m1_splitk_cuda_impl<160, 1, true>(x, weight);
  }
  if (N >= 65536) {
    return linear_f16_m1_splitk_cuda_impl<768, 2>(x, weight);
  }
  if (K == 4096 && N == 16384) {
    return linear_f16_m1_splitk_cuda_impl<512, 2>(x, weight);
  }
  if (K >= 8192) {
    return linear_f16_m1_splitk_cuda_impl<512, 2>(x, weight);
  }
  return linear_f16_m1_splitk_cuda_impl<256, 4>(x, weight);
}

at::Tensor linear_t_f16_cuda(at::Tensor x, at::Tensor weight_t) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_t.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_t_f16 K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_t_f16 M too large");
  const int M = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (M == 0 || N == 0 || K == 0) {
    return y;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  if (K <= 512 && N >= 1024 && M <= 4) {
    if (M == 1) {
      linear_t_f16_ntile_scalar_kernel<128, 2><<<dim3(ceil_div(N, 2), M, 1), 128, 0, stream>>>(
          M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(), y.data_ptr<dtype>());
    } else {
      linear_t_f16_ntile_kernel<128, 4><<<dim3(ceil_div(N, 4), M, 1), 128, 0, stream>>>(
          M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(), y.data_ptr<dtype>());
    }
  } else if (K >= 1024) {
    linear_t_f16_kernel<256><<<dim3(N, M, 1), 256, 0, stream>>>(
        M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(), y.data_ptr<dtype>());
  } else {
    linear_t_f16_kernel<128><<<dim3(N, M, 1), 128, 0, stream>>>(
        M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(), y.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

void linear_t_f16_scaled_accumulate_cuda(
    at::Tensor x,
    at::Tensor weight_t,
    at::Tensor output,
    float scale) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_t.size(0);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(k64 <= 512 && n64 >= 1024 && m64 >= 1 && m64 <= 4,
              "small-row LoRA accumulation shape is unsupported");
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX,
              "small-row LoRA accumulation shape is too large");
  TORCH_CHECK(output.dim() == 2 && output.size(0) == m64 &&
                  output.size(1) == n64,
              "LoRA accumulation output shape mismatch");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int M = static_cast<int>(m64);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (M == 1) {
    linear_t_f16_ntile_scaled_accumulate_kernel<128, 2, true>
        <<<dim3(ceil_div(N, 2), M, 1), 128, 0, stream>>>(
            M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(),
            output.data_ptr<dtype>(), scale);
  } else {
    linear_t_f16_ntile_scaled_accumulate_kernel<128, 4, false>
        <<<dim3(ceil_div(N, 4), M, 1), 128, 0, stream>>>(
            M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(),
            output.data_ptr<dtype>(), scale);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

enum class RankProjectionBackend : uint8_t {
  kRuntimeLt,
  kOriginalLt,
  kOriginalGemmex,
};

struct RankProjectionConfig {
  int rank;
  int rows;
  RankProjectionBackend backend;
  int workspace_mb;
  int algo_index;
};

// Mechanical copy of LOWRANK_IN_GEMM_4096 from the canonical Albatross
// caller.  The table is intentionally kept in the CUDA caller dispatch: it
// selects an existing upstream linear body and never creates a generic GEMM
// or a forced/config public operator.
constexpr RankProjectionConfig kRankProjectionIn4096[] = {
    {128, 8, RankProjectionBackend::kOriginalLt, 32, 2},
    {128, 16, RankProjectionBackend::kOriginalLt, 32, 1},
    {128, 48, RankProjectionBackend::kOriginalLt, 32, 1},
    {128, 64, RankProjectionBackend::kOriginalLt, 128, 2},
    {128, 96, RankProjectionBackend::kOriginalGemmex, 0, 0},
    {128, 128, RankProjectionBackend::kOriginalGemmex, 0, 0},
    {128, 192, RankProjectionBackend::kOriginalLt, 128, 1},
    {128, 256, RankProjectionBackend::kOriginalLt, 128, 6},
    {128, 512, RankProjectionBackend::kOriginalLt, 128, 0},
    {128, 1024, RankProjectionBackend::kOriginalLt, 32, 1},
    {480, 8, RankProjectionBackend::kOriginalLt, 128, 0},
    {480, 16, RankProjectionBackend::kRuntimeLt, 128, 1},
    {480, 24, RankProjectionBackend::kOriginalLt, 128, 5},
    {480, 32, RankProjectionBackend::kOriginalLt, 32, 1},
    {480, 48, RankProjectionBackend::kOriginalLt, 128, 5},
    {480, 96, RankProjectionBackend::kOriginalLt, 128, 4},
    {480, 128, RankProjectionBackend::kOriginalLt, 32, 4},
    {480, 192, RankProjectionBackend::kOriginalLt, 128, 0},
    {480, 256, RankProjectionBackend::kOriginalLt, 32, 1},
    {480, 512, RankProjectionBackend::kOriginalLt, 128, 1},
    {96, 8, RankProjectionBackend::kOriginalLt, 32, 2},
    {96, 16, RankProjectionBackend::kOriginalLt, 128, 1},
    {96, 96, RankProjectionBackend::kOriginalLt, 128, 2},
    {96, 128, RankProjectionBackend::kOriginalLt, 32, 0},
    {96, 192, RankProjectionBackend::kRuntimeLt, 32, 2},
    {96, 256, RankProjectionBackend::kOriginalLt, 32, 1},
    {96, 512, RankProjectionBackend::kOriginalGemmex, 0, 0},
    {96, 1024, RankProjectionBackend::kOriginalLt, 128, 1},
};

// Mechanical copy of LOWRANK_OUT_GEMM_4096 from the canonical Albatross
// caller.  ``rows`` is the packed input row count; the model-width guard is
// applied to the output channel dimension for rank-out.
constexpr RankProjectionConfig kRankProjectionOut4096[] = {
    {128, 8, RankProjectionBackend::kRuntimeLt, 128, 3},
    {128, 16, RankProjectionBackend::kRuntimeLt, 128, 2},
    {128, 24, RankProjectionBackend::kOriginalLt, 128, 0},
    {128, 32, RankProjectionBackend::kOriginalLt, 128, 2},
    {128, 48, RankProjectionBackend::kOriginalLt, 0, 5},
    {128, 64, RankProjectionBackend::kOriginalLt, 0, 2},
    {128, 96, RankProjectionBackend::kOriginalLt, 32, 3},
    {128, 192, RankProjectionBackend::kOriginalLt, 128, 2},
    {128, 256, RankProjectionBackend::kRuntimeLt, 128, 3},
    {128, 512, RankProjectionBackend::kRuntimeLt, 128, 1},
    {128, 1024, RankProjectionBackend::kRuntimeLt, 128, 1},
    {480, 8, RankProjectionBackend::kOriginalLt, 0, 1},
    {480, 16, RankProjectionBackend::kOriginalLt, 0, 1},
    {480, 24, RankProjectionBackend::kOriginalLt, 128, 0},
    {480, 32, RankProjectionBackend::kOriginalGemmex, 0, 0},
    {480, 96, RankProjectionBackend::kRuntimeLt, 128, 1},
    {480, 128, RankProjectionBackend::kRuntimeLt, 32, 0},
    {480, 256, RankProjectionBackend::kRuntimeLt, 32, 2},
    {480, 512, RankProjectionBackend::kRuntimeLt, 128, 1},
    {480, 1024, RankProjectionBackend::kRuntimeLt, 0, 1},
    {96, 8, RankProjectionBackend::kRuntimeLt, 128, 5},
    {96, 16, RankProjectionBackend::kRuntimeLt, 32, 4},
    {96, 24, RankProjectionBackend::kOriginalGemmex, 0, 0},
    {96, 32, RankProjectionBackend::kOriginalLt, 128, 1},
    {96, 48, RankProjectionBackend::kOriginalLt, 128, 3},
    {96, 64, RankProjectionBackend::kOriginalLt, 128, 2},
    {96, 96, RankProjectionBackend::kOriginalLt, 32, 3},
    {96, 128, RankProjectionBackend::kRuntimeLt, 32, 2},
    {96, 256, RankProjectionBackend::kRuntimeLt, 128, 4},
    {96, 512, RankProjectionBackend::kRuntimeLt, 128, 0},
    {96, 1024, RankProjectionBackend::kRuntimeLt, 128, 2},
};

template <size_t Count>
const RankProjectionConfig* find_rank_projection_config(
    const RankProjectionConfig (&configs)[Count], int rank, int rows) {
  for (const auto& config : configs) {
    if (config.rank == rank && config.rows == rows) {
      return &config;
    }
  }
  return nullptr;
}

at::Tensor internal_linear_rank_projection_f16_cuda(
    at::Tensor x,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& weight_orig,
    bool input_projection) {
  TORCH_CHECK(weight.has_value() || weight_orig.has_value(),
              "one of weight or weight_t must be provided");
  const int64_t input_features = x.size(-1);
  const int64_t rows64 = x.numel() / input_features;
  const int64_t channels64 = input_projection
      ? input_features
      : (weight.has_value() ? weight->size(1) : weight_orig->size(0));
  const int64_t rank64 = input_projection
      ? (weight.has_value() ? weight->size(1) : weight_orig->size(0))
      : (weight.has_value() ? weight->size(0) : weight_orig->size(1));
  TORCH_CHECK(rows64 <= INT_MAX && rank64 <= INT_MAX,
              "rank projection shape exceeds int32");

  const RankProjectionConfig* config = nullptr;
  if (channels64 == 4096) {
    config = input_projection
        ? find_rank_projection_config(kRankProjectionIn4096,
                              static_cast<int>(rank64),
                              static_cast<int>(rows64))
        : find_rank_projection_config(kRankProjectionOut4096,
                              static_cast<int>(rank64),
                              static_cast<int>(rows64));
  }
  if (config != nullptr) {
    switch (config->backend) {
      case RankProjectionBackend::kRuntimeLt:
        if (weight.has_value()) {
          return linear_f16_lt_cfg_cuda(
              x, *weight, config->workspace_mb, config->algo_index);
        }
        break;
      case RankProjectionBackend::kOriginalLt:
        if (weight_orig.has_value()) {
          return linear_f16_orig_lt_cfg_cuda(
              x, *weight_orig, config->workspace_mb, config->algo_index);
        }
        break;
      case RankProjectionBackend::kOriginalGemmex:
        if (weight_orig.has_value()) {
          return linear_f16_orig_cuda(x, *weight_orig);
        }
        break;
    }
  }
  if (weight.has_value()) {
    return linear_f16_cuda(x, *weight);
  }
  return linear_f16_orig_cuda(x, *weight_orig);
}



at::Tensor internal_linear_transposed_f16_cuda(
    at::Tensor x, at::Tensor weight) {
  if (x.numel() == x.size(-1) && weight.size(1) % 64 == 0) {
    return linear_f16_m1_splitk_cuda(x, weight);
  }
  return linear_f16_cuda(x, weight);
}

at::Tensor internal_linear_lora_accumulate_f16_cuda(
    at::Tensor x,
    at::Tensor lora_a,
    at::Tensor lora_b,
    at::Tensor output,
    double lora_scale) {
  const int64_t rows = x.size(0);
  const float scale = static_cast<float>(lora_scale);
  auto rank_features = rows <= 7
      ? linear_t_f16_cuda(x, lora_a)
      : internal_linear_rank_projection_f16_cuda(
            x, std::nullopt, lora_a, true);
  if (rows <= 4 && lora_b.size(0) >= 1024) {
    linear_t_f16_scaled_accumulate_cuda(
        rank_features, lora_b, output, scale);
  } else {
    linear_f16_orig_scaled_accumulate_cuda(
        rank_features, lora_b, output, scale);
  }
  return output;
}
