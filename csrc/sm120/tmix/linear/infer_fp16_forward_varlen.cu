// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Upstream repository: https://github.com/BlinkDL/Albatross
// Albatross source revision: ee3308f6922e59f2166c7fac3c5a192340a2b48e
// Original path: faster3a_2607/cuda/rwkv7_v3a_ops.cu
// Mechanical migration boundary: exact upstream linear kernel/device bodies and
// launch dispatch are retained. Local adaptations are the caller-owned packed
// [total_tokens,C] bindings and the vanilla-LoRA composition/scaled accumulation
// below; rows are already contiguous and no sequence metadata is needed for this
// tokenwise operator family.
// The standalone act_tanh/act_sigmoid bodies are copied from
// faster3a_2607/cuda/rwkv7_fast_ops_fp16.cu at the same revision because the
// canonical large-rank lowrank caller reaches those helpers directly.
//
// SPDX and provenance are retained from the upstream source below.
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

void tmix_vres_gate_forward_varlen_cuda(
    int total_tokens,
    int channels,
    at::Tensor v,
    at::Tensor v_first,
    at::Tensor v0,
    at::Tensor v12,
    at::Tensor output);

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

template <int Act>
__device__ __forceinline__ float apply_act(float x) {
  if constexpr (Act == 1) {
    return tanhf(x);
  } else {
    return 1.0f / (1.0f + expf(-x));
  }
}

__global__ void linear_act_tanh_f16_kernel(
    const dtype* __restrict__ x,
    dtype* __restrict__ out,
    int64_t total_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }
  const int64_t idx = pair_idx * 2;
  const float2 value = __half22float2(*reinterpret_cast<const __half2*>(x + idx));
  *reinterpret_cast<__half2*>(out + idx) = __floats2half2_rn(tanhf(value.x), tanhf(value.y));
}

__global__ void linear_act_sigmoid_f16_kernel(
    const dtype* __restrict__ x,
    dtype* __restrict__ out,
    int64_t total_pairs) {
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }
  const int64_t idx = pair_idx * 2;
  const float2 value = __half22float2(*reinterpret_cast<const __half2*>(x + idx));
  const float sigmoid_x = 1.0f / (1.0f + __expf(-value.x));
  const float sigmoid_y = 1.0f / (1.0f + __expf(-value.y));
  *reinterpret_cast<__half2*>(out + idx) = __floats2half2_rn(sigmoid_x, sigmoid_y);
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

template <int ChunkK, int Warps>
__global__ __launch_bounds__(128, 2) void linear_f16_rows_splitk_partial_kernel(
    int K,
    int N,
    int chunks,
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
  const int chunk = blockIdx.y;
  const int m = blockIdx.z;
  const int k0 = chunk * ChunkK;
  const int k1 = min(k0 + ChunkK, K);
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (int k = k0; k < k1; ++k) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(x_row + k));
    const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(weight + static_cast<int64_t>(k) * N + n));
    acc0 = fmaf(xv, wv.x, acc0);
    acc1 = fmaf(xv, wv.y, acc1);
  }
  reinterpret_cast<float2*>(partial + (static_cast<int64_t>(m) * chunks + chunk) * N)[pair] = make_float2(acc0, acc1);
}

__global__ void linear_f16_rows_splitk_reduce_kernel(
    int chunks,
    int N,
    const float* __restrict__ partial,
    dtype* __restrict__ y) {
  const int pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int m = blockIdx.y;
  const int n = pair << 1;
  if (n >= N) {
    return;
  }
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (int c = 0; c < chunks; ++c) {
    const float2 v = reinterpret_cast<const float2*>(partial + (static_cast<int64_t>(m) * chunks + c) * N)[pair];
    acc0 += v.x;
    acc1 += v.y;
  }
  reinterpret_cast<__half2*>(y + static_cast<int64_t>(m) * N)[pair] = __floats2half2_rn(acc0, acc1);
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

__global__ __launch_bounds__(32, 8) void linear_orig_wmma16_f16_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_orig,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * 16;
  const int m0 = blockIdx.y * 16;
  __shared__ __half a_tile[16 * 16];
  __shared__ __half b_tile[16 * 16];
  __shared__ float c_tile[16 * 16];

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int k0 = 0; k0 < K; k0 += 16) {
    for (int idx = threadIdx.x; idx < 16 * 16; idx += 32) {
      const int r = idx >> 4;
      const int kk = idx & 15;
      const int m = m0 + r;
      a_tile[idx] = (m < M && k0 + kk < K)
          ? *reinterpret_cast<const __half*>(x + static_cast<int64_t>(m) * K + k0 + kk)
          : __float2half(0.0f);
      const int n = n0 + r;
      b_tile[r * 16 + kk] = (n < N && k0 + kk < K)
          ? *reinterpret_cast<const __half*>(weight_orig + static_cast<int64_t>(n) * K + k0 + kk)
          : __float2half(0.0f);
    }
    __syncwarp();
    wmma::load_matrix_sync(a_frag, a_tile, 16);
    wmma::load_matrix_sync(b_frag, b_tile, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncwarp();
  }

  wmma::store_matrix_sync(c_tile, c_frag, 16, wmma::mem_row_major);
  __syncwarp();
  for (int idx = threadIdx.x; idx < 16 * 16; idx += 32) {
    const int r = idx >> 4;
    const int j = idx & 15;
    const int m = m0 + r;
    const int n = n0 + j;
    if (m < M && n < N) {
      *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(c_tile[idx]);
    }
  }
}

template <int Threads, int OutTile, int Act>
__global__ __launch_bounds__(Threads, 2) void linear_t_act_f16_ntile_scalar_kernel(
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
    const float xv = apply_act<Act>(__half2float(*reinterpret_cast<const __half*>(x_row + k)));
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

template <int Threads, int OutTile, int Act>
__global__ __launch_bounds__(Threads, 2) void linear_t_act_f16_ntile_kernel(
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
    float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
    xv.x = apply_act<Act>(xv.x);
    xv.y = apply_act<Act>(xv.y);
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
    const float xv = apply_act<Act>(__half2float(*reinterpret_cast<const __half*>(x_row + K - 1)));
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

template <int Threads>
__global__ __launch_bounds__(Threads, 2) void linear_wag_rank_in_f16_kernel(
    int M,
    int K,
    int Rw,
    int Ra,
    int Rg,
    int Rmax,
    const dtype* __restrict__ xw,
    const dtype* __restrict__ xa,
    const dtype* __restrict__ xg,
    const dtype* __restrict__ w1_t,
    const dtype* __restrict__ a1_t,
    const dtype* __restrict__ g1_t,
    dtype* __restrict__ w1,
    dtype* __restrict__ a1,
    dtype* __restrict__ g1) {
  const int r = blockIdx.x;
  const int m = blockIdx.y;
  const int group = blockIdx.z;
  int R = Rw;
  const dtype* x = xw;
  const dtype* wt = w1_t;
  dtype* y = w1;
  if (group == 1) {
    R = Ra;
    x = xa;
    wt = a1_t;
    y = a1;
  } else if (group == 2) {
    R = Rg;
    x = xg;
    wt = g1_t;
    y = g1;
  }
  if (m >= M || r >= R || r >= Rmax) {
    return;
  }
  float acc = 0.0f;
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const dtype* w_row = wt + static_cast<int64_t>(r) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
    const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(w_row + k));
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
    *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * R + r) = __float2half_rn(acc);
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 2) void linear_wagv_rank_in_f16_kernel(
    int M,
    int K,
    int Rw,
    int Ra,
    int Rg,
    int Rv,
    int Rmax,
    const dtype* __restrict__ xw,
    const dtype* __restrict__ xa,
    const dtype* __restrict__ xg,
    const dtype* __restrict__ xv,
    const dtype* __restrict__ w1_t,
    const dtype* __restrict__ a1_t,
    const dtype* __restrict__ g1_t,
    const dtype* __restrict__ v1_t,
    dtype* __restrict__ w1,
    dtype* __restrict__ a1,
    dtype* __restrict__ g1,
    dtype* __restrict__ v1) {
  const int r = blockIdx.x;
  const int m = blockIdx.y;
  const int group = blockIdx.z;
  int R = Rw;
  const dtype* x = xw;
  const dtype* wt = w1_t;
  dtype* y = w1;
  if (group == 1) {
    R = Ra;
    x = xa;
    wt = a1_t;
    y = a1;
  } else if (group == 2) {
    R = Rg;
    x = xg;
    wt = g1_t;
    y = g1;
  } else if (group == 3) {
    R = Rv;
    x = xv;
    wt = v1_t;
    y = v1;
  }
  if (m >= M || r >= R || r >= Rmax) {
    return;
  }
  float acc = 0.0f;
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const dtype* w_row = wt + static_cast<int64_t>(r) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv2 = __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
    const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(w_row + k));
    acc = fmaf(xv2.x, wv.x, acc);
    acc = fmaf(xv2.y, wv.y, acc);
  }
  if ((K & 1) && threadIdx.x == 0) {
    acc = fmaf(__half2float(*reinterpret_cast<const __half*>(x_row + K - 1)),
               __half2float(*reinterpret_cast<const __half*>(w_row + K - 1)),
               acc);
  }
  acc = block_sum_t<Threads>(acc);
  if (threadIdx.x == 0) {
    *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * R + r) = __float2half_rn(acc);
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_wag_rank_out_f16_kernel(
    int M,
    int C,
    int Kw,
    int Ka,
    int Kg,
    const dtype* __restrict__ w1,
    const dtype* __restrict__ a1,
    const dtype* __restrict__ g1,
    const dtype* __restrict__ w2_t,
    const dtype* __restrict__ a2_t,
    const dtype* __restrict__ g2_t,
    dtype* __restrict__ w,
    dtype* __restrict__ a,
    dtype* __restrict__ g) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  const int group = blockIdx.z;
  int K = Kw;
  const dtype* x = w1;
  const dtype* wt = w2_t;
  dtype* y = w;
  if (group == 1) {
    K = Ka;
    x = a1;
    wt = a2_t;
    y = a;
  } else if (group == 2) {
    K = Kg;
    x = g1;
    wt = g2_t;
    y = g;
  }
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
    float xv = __half2float(*reinterpret_cast<const __half*>(x_row + k));
    if (group == 0) {
      xv = tanhf(xv);
    } else if (group == 2) {
      xv = 1.0f / (1.0f + expf(-xv));
    }
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < C) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(wt + static_cast<int64_t>(n) * K + k)), acc[j]);
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
      for (int u = 0; u < Threads / 32; ++u) {
        sum += partial[u][j];
      }
      const int n = n0 + j;
      if (n < C) {
        *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * C + n) = __float2half_rn(sum);
      }
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_wagv_rank_out_f16_kernel(
    int M,
    int C,
    int Kw,
    int Ka,
    int Kg,
    int Kv,
    const dtype* __restrict__ w1,
    const dtype* __restrict__ a1,
    const dtype* __restrict__ g1,
    const dtype* __restrict__ v1,
    const dtype* __restrict__ w2_t,
    const dtype* __restrict__ a2_t,
    const dtype* __restrict__ g2_t,
    const dtype* __restrict__ v2_t,
    const dtype* __restrict__ v,
    const dtype* __restrict__ v_first,
    const dtype* __restrict__ v0,
    dtype* __restrict__ w,
    dtype* __restrict__ a,
    dtype* __restrict__ g,
    dtype* __restrict__ v_out) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  const int group = blockIdx.z;
  int K = Kw;
  const dtype* x = w1;
  const dtype* wt = w2_t;
  dtype* y = w;
  if (group == 1) {
    K = Ka;
    x = a1;
    wt = a2_t;
    y = a;
  } else if (group == 2) {
    K = Kg;
    x = g1;
    wt = g2_t;
    y = g;
  } else if (group == 3) {
    K = Kv;
    x = v1;
    wt = v2_t;
    y = v_out;
  }
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
    float xv = __half2float(*reinterpret_cast<const __half*>(x_row + k));
    if (group == 0) {
      xv = tanhf(xv);
    } else if (group == 2) {
      xv = 1.0f / (1.0f + expf(-xv));
    }
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < C) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(wt + static_cast<int64_t>(n) * K + k)), acc[j]);
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
      for (int u = 0; u < Threads / 32; ++u) {
        sum += partial[u][j];
      }
      const int n = n0 + j;
      if (n < C) {
        if (group == 3) {
          const int64_t idx = static_cast<int64_t>(m) * C + n;
          const float vv = __half2float(*reinterpret_cast<const __half*>(v + idx));
          const float vf = __half2float(*reinterpret_cast<const __half*>(v_first + idx));
          const float gate = 1.0f / (1.0f + expf(-(__half2float(*reinterpret_cast<const __half*>(v0 + n)) + sum)));
          *reinterpret_cast<__half*>(y + idx) = __float2half_rn(vv + (vf - vv) * gate);
        } else {
          *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * C + n) = __float2half_rn(sum);
        }
      }
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_t_vres_f16_ntile_scalar_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    const dtype* __restrict__ v,
    const dtype* __restrict__ v_first,
    const dtype* __restrict__ v0,
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
        const int64_t idx = static_cast<int64_t>(m) * N + n;
        const float vv = __half2float(*reinterpret_cast<const __half*>(v + idx));
        const float vf = __half2float(*reinterpret_cast<const __half*>(v_first + idx));
        const float gate = 1.0f / (1.0f + expf(-(__half2float(*reinterpret_cast<const __half*>(v0 + n)) + sum)));
        *reinterpret_cast<__half*>(y + idx) = __float2half_rn(vv + (vf - vv) * gate);
      }
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_t_vres_f16_ntile_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    const dtype* __restrict__ v,
    const dtype* __restrict__ v_first,
    const dtype* __restrict__ v0,
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
        const int64_t idx = static_cast<int64_t>(m) * N + n;
        const float vv = __half2float(*reinterpret_cast<const __half*>(v + idx));
        const float vf = __half2float(*reinterpret_cast<const __half*>(v_first + idx));
        const float gate = 1.0f / (1.0f + expf(-(__half2float(*reinterpret_cast<const __half*>(v0 + n)) + sum)));
        *reinterpret_cast<__half*>(y + idx) = __float2half_rn(vv + (vf - vv) * gate);
      }
    }
  }
}

} // namespace

at::Tensor tmix_linear_act_tanh_forward_varlen_cuda(at::Tensor x) {
  TORCH_CHECK((x.numel() % 2) == 0,
              "tmix_linear_act_tanh_forward_varlen requires an even number of elements");
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = x.numel() / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_act_tanh_f16_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
      x.data_ptr<dtype>(), out.data_ptr<dtype>(), total_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor tmix_linear_act_sigmoid_forward_varlen_cuda(at::Tensor x) {
  TORCH_CHECK((x.numel() % 2) == 0,
              "tmix_linear_act_sigmoid_forward_varlen requires an even number of elements");
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = x.numel() / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_act_sigmoid_f16_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
      x.data_ptr<dtype>(), out.data_ptr<dtype>(), total_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

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

at::Tensor linear_orig_wmma16_f16_cuda(at::Tensor x, at::Tensor weight_orig) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_orig.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_orig_wmma16_f16 K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_orig_wmma16_f16 M too large");
  const int M = static_cast<int>(m64);
  TORCH_CHECK((K % 16) == 0 && (N % 16) == 0, "linear_orig_wmma16_f16 requires K/N multiple of 16");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (M == 0 || N == 0 || K == 0) {
    return y;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_orig_wmma16_f16_kernel<<<dim3(N / 16, ceil_div(M, 16), 1), 32, 0, stream>>>(
      M, K, N, x.data_ptr<dtype>(), weight_orig.data_ptr<dtype>(), y.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor linear_f16_orig_lt_cfg_impl(
    at::Tensor x, at::Tensor weight_orig, int64_t workspace_mb, int64_t algo_index, bool strict_algo);

at::Tensor linear_f16_orig_lt_cuda(at::Tensor x, at::Tensor weight_orig) {
  return linear_f16_orig_lt_cfg_impl(x, weight_orig, 0, 0, false);
}

at::Tensor linear_f16_orig_lt_cfg_cuda(at::Tensor x, at::Tensor weight_orig, int64_t workspace_mb, int64_t algo_index) {
  return linear_f16_orig_lt_cfg_impl(x, weight_orig, workspace_mb, algo_index, false);
}

at::Tensor linear_f16_orig_lt_cfg_strict_cuda(at::Tensor x, at::Tensor weight_orig, int64_t workspace_mb, int64_t algo_index) {
  return linear_f16_orig_lt_cfg_impl(x, weight_orig, workspace_mb, algo_index, true);
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

at::Tensor linear_f16_lt_cuda(at::Tensor x, at::Tensor weight) {
  return linear_f16_lt_cfg_impl(x, weight, 0, 0, false);
}

at::Tensor linear_f16_lt_cfg_cuda(at::Tensor x, at::Tensor weight, int64_t workspace_mb, int64_t algo_index) {
  return linear_f16_lt_cfg_impl(x, weight, workspace_mb, algo_index, false);
}

at::Tensor linear_f16_lt_cfg_strict_cuda(at::Tensor x, at::Tensor weight, int64_t workspace_mb, int64_t algo_index) {
  return linear_f16_lt_cfg_impl(x, weight, workspace_mb, algo_index, true);
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

at::Tensor linear_f16_m1_splitk_cfg_cuda(at::Tensor x, at::Tensor weight, int64_t chunk_k) {
  switch (chunk_k) {
    case 64:
      return linear_f16_m1_splitk_cuda_impl<64, 4>(x, weight);
      case 96:
        return linear_f16_m1_splitk_cuda_impl<96, 4>(x, weight);
      case 112:
        return linear_f16_m1_splitk_cuda_impl<112, 4>(x, weight);
      case 128:
        return linear_f16_m1_splitk_cuda_impl<128, 4>(x, weight);
      case 144:
        return linear_f16_m1_splitk_cuda_impl<144, 4>(x, weight);
      case 152:
        return linear_f16_m1_splitk_cuda_impl<152, 4>(x, weight);
      case 160:
        return linear_f16_m1_splitk_cuda_impl<160, 4>(x, weight);
      case 168:
        return linear_f16_m1_splitk_cuda_impl<168, 4>(x, weight);
      case 176:
        return linear_f16_m1_splitk_cuda_impl<176, 4>(x, weight);
      case 184:
        return linear_f16_m1_splitk_cuda_impl<184, 4>(x, weight);
      case 192:
        return linear_f16_m1_splitk_cuda_impl<192, 4>(x, weight);
      case 208:
        return linear_f16_m1_splitk_cuda_impl<208, 4>(x, weight);
    case 224:
      return linear_f16_m1_splitk_cuda_impl<224, 4>(x, weight);
    case 256:
      return linear_f16_m1_splitk_cuda_impl<256, 4>(x, weight);
    case 384:
      return linear_f16_m1_splitk_cuda_impl<384, 4>(x, weight);
    case 512:
      return linear_f16_m1_splitk_cuda_impl<512, 4>(x, weight);
    case 640:
      return linear_f16_m1_splitk_cuda_impl<640, 4>(x, weight);
    case 768:
      return linear_f16_m1_splitk_cuda_impl<768, 4>(x, weight);
    case 896:
      return linear_f16_m1_splitk_cuda_impl<896, 4>(x, weight);
    case 1024:
      return linear_f16_m1_splitk_cuda_impl<1024, 4>(x, weight);
    case 2048:
      return linear_f16_m1_splitk_cuda_impl<2048, 4>(x, weight);
    case 4096:
      return linear_f16_m1_splitk_cuda_impl<4096, 4>(x, weight);
    default:
      TORCH_CHECK(false, "unsupported chunk_k");
  }
}

at::Tensor linear_f16_m1_splitk_tile_cuda(at::Tensor x, at::Tensor weight, int64_t chunk_k, int64_t tile_cols) {
  if (tile_cols == 64) {
    switch (chunk_k) {
      case 64:
        return linear_f16_m1_splitk_cuda_impl<64, 1>(x, weight);
      case 96:
        return linear_f16_m1_splitk_cuda_impl<96, 1>(x, weight);
      case 112:
        return linear_f16_m1_splitk_cuda_impl<112, 1>(x, weight);
      case 128:
        return linear_f16_m1_splitk_cuda_impl<128, 1>(x, weight);
      case 144:
        return linear_f16_m1_splitk_cuda_impl<144, 1>(x, weight);
      case 152:
        return linear_f16_m1_splitk_cuda_impl<152, 1>(x, weight);
      case 160:
        return linear_f16_m1_splitk_cuda_impl<160, 1>(x, weight);
      case 168:
        return linear_f16_m1_splitk_cuda_impl<168, 1>(x, weight);
      case 176:
        return linear_f16_m1_splitk_cuda_impl<176, 1>(x, weight);
      case 184:
        return linear_f16_m1_splitk_cuda_impl<184, 1>(x, weight);
      case 192:
        return linear_f16_m1_splitk_cuda_impl<192, 1>(x, weight);
      case 208:
        return linear_f16_m1_splitk_cuda_impl<208, 1>(x, weight);
      case 224:
        return linear_f16_m1_splitk_cuda_impl<224, 1>(x, weight);
      case 256:
        return linear_f16_m1_splitk_cuda_impl<256, 1>(x, weight);
      case 384:
        return linear_f16_m1_splitk_cuda_impl<384, 1>(x, weight);
      case 512:
        return linear_f16_m1_splitk_cuda_impl<512, 1>(x, weight);
      case 640:
        return linear_f16_m1_splitk_cuda_impl<640, 1>(x, weight);
      case 768:
        return linear_f16_m1_splitk_cuda_impl<768, 1>(x, weight);
      case 896:
        return linear_f16_m1_splitk_cuda_impl<896, 1>(x, weight);
      default:
        TORCH_CHECK(false, "unsupported chunk_k");
    }
  }
  if (tile_cols == 128) {
    switch (chunk_k) {
      case 64:
        return linear_f16_m1_splitk_cuda_impl<64, 2>(x, weight);
      case 96:
        return linear_f16_m1_splitk_cuda_impl<96, 2>(x, weight);
      case 112:
        return linear_f16_m1_splitk_cuda_impl<112, 2>(x, weight);
      case 128:
        return linear_f16_m1_splitk_cuda_impl<128, 2>(x, weight);
      case 144:
        return linear_f16_m1_splitk_cuda_impl<144, 2>(x, weight);
      case 152:
        return linear_f16_m1_splitk_cuda_impl<152, 2>(x, weight);
      case 160:
        return linear_f16_m1_splitk_cuda_impl<160, 2>(x, weight);
      case 168:
        return linear_f16_m1_splitk_cuda_impl<168, 2>(x, weight);
      case 176:
        return linear_f16_m1_splitk_cuda_impl<176, 2>(x, weight);
      case 184:
        return linear_f16_m1_splitk_cuda_impl<184, 2>(x, weight);
      case 192:
        return linear_f16_m1_splitk_cuda_impl<192, 2>(x, weight);
      case 208:
        return linear_f16_m1_splitk_cuda_impl<208, 2>(x, weight);
      case 224:
        return linear_f16_m1_splitk_cuda_impl<224, 2>(x, weight);
      case 256:
        return linear_f16_m1_splitk_cuda_impl<256, 2>(x, weight);
      case 384:
        return linear_f16_m1_splitk_cuda_impl<384, 2>(x, weight);
      case 512:
        return linear_f16_m1_splitk_cuda_impl<512, 2>(x, weight);
      case 640:
        return linear_f16_m1_splitk_cuda_impl<640, 2>(x, weight);
      case 768:
        return linear_f16_m1_splitk_cuda_impl<768, 2>(x, weight);
      case 896:
        return linear_f16_m1_splitk_cuda_impl<896, 2>(x, weight);
      case 1024:
        return linear_f16_m1_splitk_cuda_impl<1024, 2>(x, weight);
      default:
        TORCH_CHECK(false, "unsupported chunk_k");
    }
  }
  TORCH_CHECK(tile_cols == 256, "unsupported tile_cols");
  return linear_f16_m1_splitk_cfg_cuda(x, weight, chunk_k);
}

at::Tensor linear_f16_m1_splitk_warpred_tile_cuda(at::Tensor x, at::Tensor weight, int64_t chunk_k, int64_t tile_cols) {
  if (tile_cols == 64) {
    switch (chunk_k) {
      case 64:
        return linear_f16_m1_splitk_cuda_impl<64, 1, true>(x, weight);
      case 96:
        return linear_f16_m1_splitk_cuda_impl<96, 1, true>(x, weight);
      case 112:
        return linear_f16_m1_splitk_cuda_impl<112, 1, true>(x, weight);
      case 128:
        return linear_f16_m1_splitk_cuda_impl<128, 1, true>(x, weight);
      case 144:
        return linear_f16_m1_splitk_cuda_impl<144, 1, true>(x, weight);
      case 152:
        return linear_f16_m1_splitk_cuda_impl<152, 1, true>(x, weight);
      case 160:
        return linear_f16_m1_splitk_cuda_impl<160, 1, true>(x, weight);
      case 168:
        return linear_f16_m1_splitk_cuda_impl<168, 1, true>(x, weight);
      case 176:
        return linear_f16_m1_splitk_cuda_impl<176, 1, true>(x, weight);
      case 184:
        return linear_f16_m1_splitk_cuda_impl<184, 1, true>(x, weight);
      case 192:
        return linear_f16_m1_splitk_cuda_impl<192, 1, true>(x, weight);
      case 208:
        return linear_f16_m1_splitk_cuda_impl<208, 1, true>(x, weight);
      case 224:
        return linear_f16_m1_splitk_cuda_impl<224, 1, true>(x, weight);
      case 256:
        return linear_f16_m1_splitk_cuda_impl<256, 1, true>(x, weight);
      default:
        TORCH_CHECK(false, "unsupported warpred chunk_k");
    }
  }
  if (tile_cols == 128) {
    switch (chunk_k) {
      case 64:
        return linear_f16_m1_splitk_cuda_impl<64, 2, true>(x, weight);
      case 96:
        return linear_f16_m1_splitk_cuda_impl<96, 2, true>(x, weight);
      case 112:
        return linear_f16_m1_splitk_cuda_impl<112, 2, true>(x, weight);
      case 128:
        return linear_f16_m1_splitk_cuda_impl<128, 2, true>(x, weight);
      case 144:
        return linear_f16_m1_splitk_cuda_impl<144, 2, true>(x, weight);
      case 152:
        return linear_f16_m1_splitk_cuda_impl<152, 2, true>(x, weight);
      case 160:
        return linear_f16_m1_splitk_cuda_impl<160, 2, true>(x, weight);
      case 168:
        return linear_f16_m1_splitk_cuda_impl<168, 2, true>(x, weight);
      case 176:
        return linear_f16_m1_splitk_cuda_impl<176, 2, true>(x, weight);
      case 184:
        return linear_f16_m1_splitk_cuda_impl<184, 2, true>(x, weight);
      case 192:
        return linear_f16_m1_splitk_cuda_impl<192, 2, true>(x, weight);
      case 208:
        return linear_f16_m1_splitk_cuda_impl<208, 2, true>(x, weight);
      case 224:
        return linear_f16_m1_splitk_cuda_impl<224, 2, true>(x, weight);
      case 256:
        return linear_f16_m1_splitk_cuda_impl<256, 2, true>(x, weight);
      default:
        TORCH_CHECK(false, "unsupported warpred chunk_k");
    }
  }
  TORCH_CHECK(tile_cols == 256, "unsupported warpred tile_cols");
  switch (chunk_k) {
    case 64:
      return linear_f16_m1_splitk_cuda_impl<64, 4, true>(x, weight);
    case 96:
      return linear_f16_m1_splitk_cuda_impl<96, 4, true>(x, weight);
    case 112:
      return linear_f16_m1_splitk_cuda_impl<112, 4, true>(x, weight);
    case 128:
      return linear_f16_m1_splitk_cuda_impl<128, 4, true>(x, weight);
    case 144:
      return linear_f16_m1_splitk_cuda_impl<144, 4, true>(x, weight);
    case 152:
      return linear_f16_m1_splitk_cuda_impl<152, 4, true>(x, weight);
    case 160:
      return linear_f16_m1_splitk_cuda_impl<160, 4, true>(x, weight);
    case 168:
      return linear_f16_m1_splitk_cuda_impl<168, 4, true>(x, weight);
    case 176:
      return linear_f16_m1_splitk_cuda_impl<176, 4, true>(x, weight);
    case 184:
      return linear_f16_m1_splitk_cuda_impl<184, 4, true>(x, weight);
    case 192:
      return linear_f16_m1_splitk_cuda_impl<192, 4, true>(x, weight);
    case 208:
      return linear_f16_m1_splitk_cuda_impl<208, 4, true>(x, weight);
    case 224:
      return linear_f16_m1_splitk_cuda_impl<224, 4, true>(x, weight);
    case 256:
      return linear_f16_m1_splitk_cuda_impl<256, 4, true>(x, weight);
    default:
      TORCH_CHECK(false, "unsupported warpred chunk_k");
  }
}

template <int ChunkK, int Warps>
at::Tensor linear_f16_rows_splitk_cuda_impl(at::Tensor x, at::Tensor weight) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight.size(1);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_f16_rows_splitk K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_f16_rows_splitk M too large");
  const int M = static_cast<int>(m64);
  TORCH_CHECK((N % 64) == 0, "linear_f16_rows_splitk requires N multiple of 64");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (M == 0 || K == 0 || N == 0) {
    return y;
  }
  const int chunks = static_cast<int>(ceil_div(K, ChunkK));
  auto partial = at::empty({m64, chunks, n64}, x.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_f16_rows_splitk_partial_kernel<ChunkK, Warps><<<dim3(ceil_div(N, Warps * 64), chunks, M), Warps * 32, 0, stream>>>(
      K, N, chunks, x.data_ptr<dtype>(), weight.data_ptr<dtype>(), partial.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  linear_f16_rows_splitk_reduce_kernel<<<dim3(static_cast<int>(ceil_div(N / 2, 128)), M, 1), 128, 0, stream>>>(
      chunks, N, partial.data_ptr<float>(), y.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor linear_f16_rows_splitk_cuda(at::Tensor x, at::Tensor weight, int64_t chunk_k) {
  switch (chunk_k) {
    case 128:
      return linear_f16_rows_splitk_cuda_impl<128, 2>(x, weight);
    case 256:
      return linear_f16_rows_splitk_cuda_impl<256, 2>(x, weight);
    case 512:
      return linear_f16_rows_splitk_cuda_impl<512, 2>(x, weight);
    case 1024:
      return linear_f16_rows_splitk_cuda_impl<1024, 2>(x, weight);
    default:
      TORCH_CHECK(false, "unsupported chunk_k");
  }
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

template <int Act>
at::Tensor linear_t_act_f16_cuda_impl(at::Tensor x, at::Tensor weight_t) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_t.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_t_act_f16 K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_t_act_f16 M too large");
  const int M = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (M == 0 || N == 0 || K == 0) {
    return y;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(K <= 512 && N >= 1024 && M <= 4, "linear_t_act_f16 currently supports only small-rank rank-out");
  if (M == 1) {
    linear_t_act_f16_ntile_scalar_kernel<128, 2, Act><<<dim3(ceil_div(N, 2), M, 1), 128, 0, stream>>>(
        M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(), y.data_ptr<dtype>());
  } else {
    linear_t_act_f16_ntile_kernel<128, 4, Act><<<dim3(ceil_div(N, 4), M, 1), 128, 0, stream>>>(
        M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(), y.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor linear_t_act_f16_cuda(at::Tensor x, at::Tensor weight_t, int64_t act) {
  if (act == 1) {
    return linear_t_act_f16_cuda_impl<1>(x, weight_t);
  }
  return linear_t_act_f16_cuda_impl<2>(x, weight_t);
}

std::vector<at::Tensor> linear_wag_rank_in_f16_cuda(
    at::Tensor xw,
    at::Tensor xa,
    at::Tensor xg,
    at::Tensor w1_t,
    at::Tensor a1_t,
    at::Tensor g1_t) {
  const int64_t k64 = xw.size(-1);
  const int64_t rw64 = w1_t.size(0);
  const int64_t ra64 = a1_t.size(0);
  const int64_t rg64 = g1_t.size(0);
  const int64_t m64 = xw.numel() / k64;
  TORCH_CHECK(k64 <= INT_MAX && rw64 <= INT_MAX && ra64 <= INT_MAX && rg64 <= INT_MAX && m64 <= INT_MAX,
              "linear_wag_rank_in_f16 shape too large");
  const int K = static_cast<int>(k64);
  const int Rw = static_cast<int>(rw64);
  const int Ra = static_cast<int>(ra64);
  const int Rg = static_cast<int>(rg64);
  const int Rmax = std::max(Rw, std::max(Ra, Rg));
  const int M = static_cast<int>(m64);
  TORCH_CHECK(K >= 1024 && Rmax <= 512 && M <= 8, "linear_wag_rank_in_f16 supports only K>=1024,R<=512,M<=8");
  std::vector<int64_t> w_sizes(xw.sizes().begin(), xw.sizes().end());
  std::vector<int64_t> a_sizes = w_sizes;
  std::vector<int64_t> g_sizes = w_sizes;
  w_sizes.back() = rw64;
  a_sizes.back() = ra64;
  g_sizes.back() = rg64;
  auto w1 = at::empty(w_sizes, xw.options());
  auto a1 = at::empty(a_sizes, xw.options());
  auto g1 = at::empty(g_sizes, xw.options());
  if (M == 0 || K == 0 || Rmax == 0) {
    return {w1, a1, g1};
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_wag_rank_in_f16_kernel<256><<<dim3(Rmax, M, 3), 256, 0, stream>>>(
      M, K, Rw, Ra, Rg, Rmax,
      xw.data_ptr<dtype>(), xa.data_ptr<dtype>(), xg.data_ptr<dtype>(),
      w1_t.data_ptr<dtype>(), a1_t.data_ptr<dtype>(), g1_t.data_ptr<dtype>(),
      w1.data_ptr<dtype>(), a1.data_ptr<dtype>(), g1.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {w1, a1, g1};
}

std::vector<at::Tensor> linear_wagv_rank_in_f16_cuda(
    at::Tensor xw,
    at::Tensor xa,
    at::Tensor xg,
    at::Tensor xv,
    at::Tensor w1_t,
    at::Tensor a1_t,
    at::Tensor g1_t,
    at::Tensor v1_t) {
  const int64_t k64 = xw.size(-1);
  const int64_t rw64 = w1_t.size(0);
  const int64_t ra64 = a1_t.size(0);
  const int64_t rg64 = g1_t.size(0);
  const int64_t rv64 = v1_t.size(0);
  const int64_t m64 = xw.numel() / k64;
  TORCH_CHECK(k64 <= INT_MAX && rw64 <= INT_MAX && ra64 <= INT_MAX && rg64 <= INT_MAX && rv64 <= INT_MAX && m64 <= INT_MAX,
              "linear_wagv_rank_in_f16 shape too large");
  const int K = static_cast<int>(k64);
  const int Rw = static_cast<int>(rw64);
  const int Ra = static_cast<int>(ra64);
  const int Rg = static_cast<int>(rg64);
  const int Rv = static_cast<int>(rv64);
  const int Rmax = std::max(std::max(Rw, Ra), std::max(Rg, Rv));
  const int M = static_cast<int>(m64);
  TORCH_CHECK(K >= 1024 && Rmax <= 512 && M <= 8, "linear_wagv_rank_in_f16 supports only K>=1024,R<=512,M<=8");
  std::vector<int64_t> w_sizes(xw.sizes().begin(), xw.sizes().end());
  std::vector<int64_t> a_sizes = w_sizes;
  std::vector<int64_t> g_sizes = w_sizes;
  std::vector<int64_t> v_sizes = w_sizes;
  w_sizes.back() = rw64;
  a_sizes.back() = ra64;
  g_sizes.back() = rg64;
  v_sizes.back() = rv64;
  auto w1 = at::empty(w_sizes, xw.options());
  auto a1 = at::empty(a_sizes, xw.options());
  auto g1 = at::empty(g_sizes, xw.options());
  auto v1 = at::empty(v_sizes, xw.options());
  if (M == 0 || K == 0 || Rmax == 0) {
    return {w1, a1, g1, v1};
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_wagv_rank_in_f16_kernel<256><<<dim3(Rmax, M, 4), 256, 0, stream>>>(
      M, K, Rw, Ra, Rg, Rv, Rmax,
      xw.data_ptr<dtype>(), xa.data_ptr<dtype>(), xg.data_ptr<dtype>(), xv.data_ptr<dtype>(),
      w1_t.data_ptr<dtype>(), a1_t.data_ptr<dtype>(), g1_t.data_ptr<dtype>(), v1_t.data_ptr<dtype>(),
      w1.data_ptr<dtype>(), a1.data_ptr<dtype>(), g1.data_ptr<dtype>(), v1.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {w1, a1, g1, v1};
}

std::vector<at::Tensor> linear_wag_rank_out_f16_cuda(
    at::Tensor w1,
    at::Tensor a1,
    at::Tensor g1,
    at::Tensor w2_t,
    at::Tensor a2_t,
    at::Tensor g2_t) {
  const int64_t kw64 = w1.size(-1);
  const int64_t ka64 = a1.size(-1);
  const int64_t kg64 = g1.size(-1);
  const int64_t c64 = w2_t.size(0);
  const int64_t m64 = w1.numel() / kw64;
  TORCH_CHECK(kw64 <= INT_MAX && ka64 <= INT_MAX && kg64 <= INT_MAX && c64 <= INT_MAX && m64 <= INT_MAX,
              "linear_wag_rank_out_f16 shape too large");
  const int Kw = static_cast<int>(kw64);
  const int Ka = static_cast<int>(ka64);
  const int Kg = static_cast<int>(kg64);
  const int C = static_cast<int>(c64);
  const int M = static_cast<int>(m64);
  TORCH_CHECK(Kw <= 512 && Ka <= 512 && Kg <= 512 && C >= 1024 && M <= 4,
              "linear_wag_rank_out_f16 supports only small-rank M<=4");
  std::vector<int64_t> out_sizes(w1.sizes().begin(), w1.sizes().end());
  out_sizes.back() = c64;
  auto w = at::empty(out_sizes, w1.options());
  auto a = at::empty(out_sizes, w1.options());
  auto g = at::empty(out_sizes, w1.options());
  if (M == 0 || C == 0 || Kw == 0 || Ka == 0 || Kg == 0) {
    return {w, a, g};
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  if (M == 1) {
    linear_wag_rank_out_f16_kernel<128, 4><<<dim3(ceil_div(C, 4), M, 3), 128, 0, stream>>>(
        M, C, Kw, Ka, Kg,
        w1.data_ptr<dtype>(), a1.data_ptr<dtype>(), g1.data_ptr<dtype>(),
        w2_t.data_ptr<dtype>(), a2_t.data_ptr<dtype>(), g2_t.data_ptr<dtype>(),
        w.data_ptr<dtype>(), a.data_ptr<dtype>(), g.data_ptr<dtype>());
  } else {
    linear_wag_rank_out_f16_kernel<128, 4><<<dim3(ceil_div(C, 4), M, 3), 128, 0, stream>>>(
        M, C, Kw, Ka, Kg,
        w1.data_ptr<dtype>(), a1.data_ptr<dtype>(), g1.data_ptr<dtype>(),
        w2_t.data_ptr<dtype>(), a2_t.data_ptr<dtype>(), g2_t.data_ptr<dtype>(),
        w.data_ptr<dtype>(), a.data_ptr<dtype>(), g.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {w, a, g};
}

std::vector<at::Tensor> linear_wagv_rank_out_f16_cuda(
    at::Tensor w1,
    at::Tensor a1,
    at::Tensor g1,
    at::Tensor v1,
    at::Tensor w2_t,
    at::Tensor a2_t,
    at::Tensor g2_t,
    at::Tensor v2_t,
    at::Tensor v,
    at::Tensor v_first,
    at::Tensor v0) {
  const int64_t kw64 = w1.size(-1);
  const int64_t ka64 = a1.size(-1);
  const int64_t kg64 = g1.size(-1);
  const int64_t kv64 = v1.size(-1);
  const int64_t c64 = w2_t.size(0);
  const int64_t m64 = w1.numel() / kw64;
  TORCH_CHECK(kw64 <= INT_MAX && ka64 <= INT_MAX && kg64 <= INT_MAX && kv64 <= INT_MAX && c64 <= INT_MAX && m64 <= INT_MAX,
              "linear_wagv_rank_out_f16 shape too large");
  const int Kw = static_cast<int>(kw64);
  const int Ka = static_cast<int>(ka64);
  const int Kg = static_cast<int>(kg64);
  const int Kv = static_cast<int>(kv64);
  const int C = static_cast<int>(c64);
  const int M = static_cast<int>(m64);
  TORCH_CHECK(Kw <= 512 && Ka <= 512 && Kg <= 512 && Kv <= 512 && C >= 1024 && M <= 4,
              "linear_wagv_rank_out_f16 supports only small-rank M<=4");
  std::vector<int64_t> out_sizes(w1.sizes().begin(), w1.sizes().end());
  out_sizes.back() = c64;
  auto w = at::empty(out_sizes, w1.options());
  auto a = at::empty(out_sizes, w1.options());
  auto g = at::empty(out_sizes, w1.options());
  auto v_out = at::empty(out_sizes, w1.options());
  if (M == 0 || C == 0 || Kw == 0 || Ka == 0 || Kg == 0 || Kv == 0) {
    return {w, a, g, v_out};
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  if (M == 1) {
    linear_wagv_rank_out_f16_kernel<128, 4><<<dim3(ceil_div(C, 4), M, 4), 128, 0, stream>>>(
        M, C, Kw, Ka, Kg, Kv,
        w1.data_ptr<dtype>(), a1.data_ptr<dtype>(), g1.data_ptr<dtype>(), v1.data_ptr<dtype>(),
        w2_t.data_ptr<dtype>(), a2_t.data_ptr<dtype>(), g2_t.data_ptr<dtype>(), v2_t.data_ptr<dtype>(),
        v.data_ptr<dtype>(), v_first.data_ptr<dtype>(), v0.data_ptr<dtype>(),
        w.data_ptr<dtype>(), a.data_ptr<dtype>(), g.data_ptr<dtype>(), v_out.data_ptr<dtype>());
  } else {
    linear_wagv_rank_out_f16_kernel<128, 4><<<dim3(ceil_div(C, 4), M, 4), 128, 0, stream>>>(
        M, C, Kw, Ka, Kg, Kv,
        w1.data_ptr<dtype>(), a1.data_ptr<dtype>(), g1.data_ptr<dtype>(), v1.data_ptr<dtype>(),
        w2_t.data_ptr<dtype>(), a2_t.data_ptr<dtype>(), g2_t.data_ptr<dtype>(), v2_t.data_ptr<dtype>(),
        v.data_ptr<dtype>(), v_first.data_ptr<dtype>(), v0.data_ptr<dtype>(),
        w.data_ptr<dtype>(), a.data_ptr<dtype>(), g.data_ptr<dtype>(), v_out.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {w, a, g, v_out};
}

at::Tensor linear_t_vres_f16_cuda(at::Tensor x, at::Tensor weight_t, at::Tensor v, at::Tensor v_first, at::Tensor v0) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_t.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_t_vres_f16 K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_t_vres_f16 M too large");
  const int M = static_cast<int>(m64);
  auto y = at::empty_like(v);
  if (M == 0 || N == 0 || K == 0) {
    return y;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(K <= 512 && N >= 1024 && M <= 4, "linear_t_vres_f16 currently supports only small-rank rank-out");
  if (M == 1) {
    linear_t_vres_f16_ntile_scalar_kernel<128, 2><<<dim3(ceil_div(N, 2), M, 1), 128, 0, stream>>>(
        M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(), v.data_ptr<dtype>(), v_first.data_ptr<dtype>(), v0.data_ptr<dtype>(), y.data_ptr<dtype>());
  } else {
    linear_t_vres_f16_ntile_kernel<128, 4><<<dim3(ceil_div(N, 4), M, 1), 128, 0, stream>>>(
        M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(), v.data_ptr<dtype>(), v_first.data_ptr<dtype>(), v0.data_ptr<dtype>(), y.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

enum class LowrankBackend : uint8_t {
  kRuntimeLt,
  kOriginalLt,
  kOriginalGemmex,
};

struct LowrankGemmConfig {
  int rank;
  int rows;
  LowrankBackend backend;
  int workspace_mb;
  int algo_index;
};

// Mechanical copy of LOWRANK_IN_GEMM_4096 from the canonical Albatross
// caller.  The table is intentionally kept in the CUDA caller dispatch: it
// selects an existing upstream linear body and never creates a generic GEMM
// or a forced/config public operator.
constexpr LowrankGemmConfig kLowrankInGemm4096[] = {
    {128, 8, LowrankBackend::kOriginalLt, 32, 2},
    {128, 16, LowrankBackend::kOriginalLt, 32, 1},
    {128, 48, LowrankBackend::kOriginalLt, 32, 1},
    {128, 64, LowrankBackend::kOriginalLt, 128, 2},
    {128, 96, LowrankBackend::kOriginalGemmex, 0, 0},
    {128, 128, LowrankBackend::kOriginalGemmex, 0, 0},
    {128, 192, LowrankBackend::kOriginalLt, 128, 1},
    {128, 256, LowrankBackend::kOriginalLt, 128, 6},
    {128, 512, LowrankBackend::kOriginalLt, 128, 0},
    {128, 1024, LowrankBackend::kOriginalLt, 32, 1},
    {480, 8, LowrankBackend::kOriginalLt, 128, 0},
    {480, 16, LowrankBackend::kRuntimeLt, 128, 1},
    {480, 24, LowrankBackend::kOriginalLt, 128, 5},
    {480, 32, LowrankBackend::kOriginalLt, 32, 1},
    {480, 48, LowrankBackend::kOriginalLt, 128, 5},
    {480, 96, LowrankBackend::kOriginalLt, 128, 4},
    {480, 128, LowrankBackend::kOriginalLt, 32, 4},
    {480, 192, LowrankBackend::kOriginalLt, 128, 0},
    {480, 256, LowrankBackend::kOriginalLt, 32, 1},
    {480, 512, LowrankBackend::kOriginalLt, 128, 1},
    {96, 8, LowrankBackend::kOriginalLt, 32, 2},
    {96, 16, LowrankBackend::kOriginalLt, 128, 1},
    {96, 96, LowrankBackend::kOriginalLt, 128, 2},
    {96, 128, LowrankBackend::kOriginalLt, 32, 0},
    {96, 192, LowrankBackend::kRuntimeLt, 32, 2},
    {96, 256, LowrankBackend::kOriginalLt, 32, 1},
    {96, 512, LowrankBackend::kOriginalGemmex, 0, 0},
    {96, 1024, LowrankBackend::kOriginalLt, 128, 1},
};

// Mechanical copy of LOWRANK_OUT_GEMM_4096 from the canonical Albatross
// caller.  ``rows`` is the packed input row count; the model-width guard is
// applied to the output channel dimension for rank-out.
constexpr LowrankGemmConfig kLowrankOutGemm4096[] = {
    {128, 8, LowrankBackend::kRuntimeLt, 128, 3},
    {128, 16, LowrankBackend::kRuntimeLt, 128, 2},
    {128, 24, LowrankBackend::kOriginalLt, 128, 0},
    {128, 32, LowrankBackend::kOriginalLt, 128, 2},
    {128, 48, LowrankBackend::kOriginalLt, 0, 5},
    {128, 64, LowrankBackend::kOriginalLt, 0, 2},
    {128, 96, LowrankBackend::kOriginalLt, 32, 3},
    {128, 192, LowrankBackend::kOriginalLt, 128, 2},
    {128, 256, LowrankBackend::kRuntimeLt, 128, 3},
    {128, 512, LowrankBackend::kRuntimeLt, 128, 1},
    {128, 1024, LowrankBackend::kRuntimeLt, 128, 1},
    {480, 8, LowrankBackend::kOriginalLt, 0, 1},
    {480, 16, LowrankBackend::kOriginalLt, 0, 1},
    {480, 24, LowrankBackend::kOriginalLt, 128, 0},
    {480, 32, LowrankBackend::kOriginalGemmex, 0, 0},
    {480, 96, LowrankBackend::kRuntimeLt, 128, 1},
    {480, 128, LowrankBackend::kRuntimeLt, 32, 0},
    {480, 256, LowrankBackend::kRuntimeLt, 32, 2},
    {480, 512, LowrankBackend::kRuntimeLt, 128, 1},
    {480, 1024, LowrankBackend::kRuntimeLt, 0, 1},
    {96, 8, LowrankBackend::kRuntimeLt, 128, 5},
    {96, 16, LowrankBackend::kRuntimeLt, 32, 4},
    {96, 24, LowrankBackend::kOriginalGemmex, 0, 0},
    {96, 32, LowrankBackend::kOriginalLt, 128, 1},
    {96, 48, LowrankBackend::kOriginalLt, 128, 3},
    {96, 64, LowrankBackend::kOriginalLt, 128, 2},
    {96, 96, LowrankBackend::kOriginalLt, 32, 3},
    {96, 128, LowrankBackend::kRuntimeLt, 32, 2},
    {96, 256, LowrankBackend::kRuntimeLt, 128, 4},
    {96, 512, LowrankBackend::kRuntimeLt, 128, 0},
    {96, 1024, LowrankBackend::kRuntimeLt, 128, 2},
};

template <size_t Count>
const LowrankGemmConfig* find_lowrank_config(
    const LowrankGemmConfig (&configs)[Count], int rank, int rows) {
  for (const auto& config : configs) {
    if (config.rank == rank && config.rows == rows) {
      return &config;
    }
  }
  return nullptr;
}

at::Tensor linear_lowrank_large_dispatch_f16_cuda(
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
              "lowrank dispatch shape exceeds int32");

  const LowrankGemmConfig* config = nullptr;
  if (channels64 == 4096) {
    config = input_projection
        ? find_lowrank_config(kLowrankInGemm4096,
                              static_cast<int>(rank64),
                              static_cast<int>(rows64))
        : find_lowrank_config(kLowrankOutGemm4096,
                              static_cast<int>(rank64),
                              static_cast<int>(rows64));
  }
  if (config != nullptr) {
    switch (config->backend) {
      case LowrankBackend::kRuntimeLt:
        if (weight.has_value()) {
          return linear_f16_lt_cfg_cuda(
              x, *weight, config->workspace_mb, config->algo_index);
        }
        break;
      case LowrankBackend::kOriginalLt:
        if (weight_orig.has_value()) {
          return linear_f16_orig_lt_cfg_cuda(
              x, *weight_orig, config->workspace_mb, config->algo_index);
        }
        break;
      case LowrankBackend::kOriginalGemmex:
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

// Caller ownership is kept in the module-local dispatch instead of exporting
// rwkv7_v3a_ops forced/config symbols.  The integer is passed only by the
// caller-owned binding: 0=ordinary, 1=att_c2c, 2=ffn_key, 3=head.
enum class LinearCallerGroup : int64_t {
  kOrdinary = 0,
  kAttentionC2C = 1,
  kFfnKey = 2,
  kHead = 3,
};

at::Tensor linear_orig_caller_dispatch_f16_cuda(
    at::Tensor x,
    at::Tensor weight_orig,
    LinearCallerGroup group) {
  const int64_t c = x.size(-1);
  const int64_t rows = x.numel() / c;
  const bool attention = group == LinearCallerGroup::kAttentionC2C;
  const bool ffn_key = group == LinearCallerGroup::kFfnKey;
  const bool head = group == LinearCallerGroup::kHead;

  // Exact C=4096 winners admitted by the canonical Albatross caller.  The
  // shape guards are part of the policy; do not broaden these tables to other
  // model widths or callers.
  if (c == 4096) {
    if (attention && weight_orig.size(0) == 4096 &&
        weight_orig.size(1) == 4096) {
      switch (rows) {
        case 24:
          return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
        case 32:
        case 96:
          return linear_f16_orig_cuda(x, weight_orig);
        case 48:
        case 64:
          return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 0);
        case 192:
          return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 2);
        default:
          break;
      }
    }
    if (ffn_key && weight_orig.size(0) == 16384 &&
        weight_orig.size(1) == 4096) {
      switch (rows) {
        case 24:
        case 32:
          return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 2);
        case 128:
          return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 3);
        case 192:
          return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 3);
        case 384:
          return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 2);
        default:
          break;
      }
    }
  }

  if (rows == 1) {
    if (ffn_key && c == 2560) {
      return linear_orig_rows_exact_f16_cuda(x, weight_orig, 128, 2, true);
    }
    return linear_orig_rows_exact_f16_cuda(
        x, weight_orig, 128, 2,
        ffn_key ? c <= 1024 : (!attention || c < 2048));
  }
  if (rows == 2) {
    if (attention) {
      return linear_orig_rows_exact_f16_cuda(x, weight_orig, 64, 2, true);
    }
    if (ffn_key) {
      if (c == 2560) {
        return linear_orig_rows_exact_f16_cuda(x, weight_orig, 128, 2, false);
      }
      if (c < 4096) {
        return linear_orig_rows_exact_f16_cuda(x, weight_orig, 64, 2, true);
      }
      return linear_orig_rows_exact_f16_cuda(x, weight_orig, 128, 2, false);
    }
    if (head && c == 2560) {
      return linear_orig_rows_exact_f16_cuda(x, weight_orig, 128, 2, false);
    }
    return linear_orig_rows_exact_f16_cuda(x, weight_orig, 64, 2, true);
  }
  if (rows == 3) {
    if (head) {
      if (c <= 2560) {
        return linear_f16_orig_cuda(x, weight_orig);
      }
      return linear_orig_rows_f16_cuda(x, weight_orig, 3, 2);
    }
    if (ffn_key) {
      if (c <= 1024) {
        return linear_orig_rows_cfg_f16_cuda(x, weight_orig, 64, 3, 4);
      }
      if (c == 2048 || c == 2560) {
        return linear_f16_orig_cuda(x, weight_orig);
      }
      return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
    }
    if (attention) {
      if (c == 768) return linear_orig_rows_f16_cuda(x, weight_orig, 1, 2);
      if (c == 1024) return linear_orig_rows_f16_cuda(x, weight_orig, 2, 2);
      if (c == 2048) return linear_orig_rows_f16_cuda(x, weight_orig, 3, 4);
      if (c == 2560) return linear_orig_rows_f16_cuda(x, weight_orig, 3, 2);
      return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 2);
    }
    return linear_orig_rows_cfg_f16_cuda(x, weight_orig, 64, 3, 4);
  }
  if (rows == 4) {
    if (ffn_key) {
      if (c <= 1024) {
        return linear_orig_rows_cfg_f16_cuda(x, weight_orig, 64, 2, 4);
      }
      if (c == 2048 || c == 2560) {
        return linear_f16_orig_cuda(x, weight_orig);
      }
      return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
    }
    if (attention) {
      if (c <= 1024) return linear_orig_rows_f16_cuda(x, weight_orig, 2, 2);
      if (c == 2048 || c == 2560) {
        return linear_orig_rows_f16_cuda(x, weight_orig, 4, 2);
      }
      return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 2);
    }
  }

  // The remaining shape windows are the canonical caller-specific Lt policy.
  // It is intentionally explicit; an unlisted shape ends in the upstream
  // original-layout GEMM rather than a new local fallback implementation.
  if (head) {
    if (c == 768) {
      if (rows >= 192 && rows < 256) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 3);
      if (rows >= 96 && rows < 160) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 1);
    }
    if (c == 1024) {
      if (rows >= 256 && rows < 384) return linear_f16_orig_cuda(x, weight_orig);
      if (rows >= 192 && rows < 256) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 2);
      if (rows >= 96 && rows < 160) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 1);
    }
    if (c == 2048) {
      if (rows >= 256 && rows < 384) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 0);
      if (rows >= 192 && rows < 256) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 6);
      if (rows >= 128 && rows < 160) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 1);
      if (rows >= 96 && rows < 112) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
    }
    if (c == 2560) {
      if (rows >= 256) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 0);
      if (rows >= 192) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 5);
      if (rows >= 160) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 5);
      if (rows >= 128) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 1);
      if (rows >= 96) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 0);
      if (rows >= 80) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
      if (rows >= 72) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 1);
    }
    if (rows >= 1024) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 0);
    if (rows >= 512) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 2);
    if (rows >= 384) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 2);
    if (rows >= 256) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 1);
    if (rows >= 192) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 0);
    if (rows >= 160) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 0);
    if (rows >= 128) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 0);
    if (rows >= 112) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 0);
    if (rows >= 96) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 1);
    if (rows >= 80) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 2);
    if (rows >= 72) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 2);
  }
  if (attention) {
    if (c == 2560 && rows >= 17 && rows <= 20) {
      return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
    }
    if (c == 768) {
      if (rows >= 256 && rows < 384) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 1);
      if (rows >= 96 && rows < 112) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 3);
    }
    if (c == 1024) {
      if (rows >= 256 && rows < 384) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 0);
      if (rows >= 96 && rows < 112) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 6);
    }
    if (c == 2048) {
      if (rows >= 256 && rows < 384) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 3);
      if (rows >= 192 && rows < 256) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 0);
      if (rows >= 96 && rows < 112) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 4);
    }
    if (c == 2560) {
      if (rows >= 256) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 1);
      if (rows >= 160) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 2);
      if (rows >= 128) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 2);
      if (rows >= 112) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 3);
      if (rows >= 96) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 2);
      if (rows >= 72) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 2);
      if (rows >= 5) return linear_f16_orig_cuda(x, weight_orig);
    }
    if (rows >= 1024) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 4);
    if (rows >= 768) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 0);
    if (rows >= 512) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 1);
    if (rows >= 384) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 2);
    if (rows >= 256) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 4);
    if (rows >= 192) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
    if (rows >= 160) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 1);
    if (rows >= 112) return linear_f16_orig_cuda(x, weight_orig);
    if (rows >= 96) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 5);
    if (rows >= 72) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 0);
    if (rows >= 48) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 6);
    if (rows >= 32) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
    if (rows >= 24) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 6);
    if (rows >= 12) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
    if (rows >= 5) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 2);
  }
  if (ffn_key) {
    if (c == 2560 && rows >= 17 && rows <= 20) {
      return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
    }
    if (c == 768 && ((rows >= 256 && rows < 384) || (rows >= 96 && rows < 112))) {
      return linear_f16_orig_cuda(x, weight_orig);
    }
    if (c == 1024) {
      if (rows >= 256 && rows < 384) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 2);
      if (rows >= 192 && rows < 256) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
      if (rows >= 96 && rows < 160) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 2);
    }
    if (c == 2048 && rows >= 128 && rows < 160) {
      return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 3);
    }
    if (c == 2560) {
      if (rows >= 192) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 5);
      if (rows >= 160) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 4);
      if (rows >= 128) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 5);
      if (rows >= 112) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 4);
      if (rows >= 96) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 4);
      if (rows >= 80) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 3);
      if (rows >= 72) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 4);
      if (rows >= 3) return linear_f16_orig_cuda(x, weight_orig);
    }
    if (rows >= 1024) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
    if (rows >= 768) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 1);
    if (rows >= 512) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 3);
    if (rows >= 384) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 0);
    if (rows >= 256) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 4);
    if (rows >= 192) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 1);
    if (rows >= 160) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 2);
    if (rows >= 128) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 0);
    if (rows >= 112) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 3);
    if (rows >= 96) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 32, 1);
    if (rows >= 72) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 128, 1);
    if (rows >= 48) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 1);
    if (rows >= 12) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
    if (rows == 5 || rows == 6) return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 1);
  }
  return linear_f16_orig_cuda(x, weight_orig);
}

at::Tensor tmix_linear_original_caller_dispatch_cuda(
    at::Tensor x, at::Tensor weight_orig, int64_t caller_group) {
  TORCH_CHECK(caller_group >= 0 && caller_group <= 3,
              "invalid linear caller group");
  return linear_orig_caller_dispatch_f16_cuda(
      x, weight_orig, static_cast<LinearCallerGroup>(caller_group));
}


at::Tensor tmix_linear_forward_varlen_cuda(
    at::Tensor x,
    at::Tensor weight,
    bool weight_is_transposed,
    int64_t caller_group) {
  // Albatross uses [K,N] for the transposed fast path and [N,K] for the
  // original-layout path.  Both bodies above are upstream implementations.
  if (weight_is_transposed) {
    if (x.numel() == x.size(-1) && weight.size(1) % 64 == 0) {
      return linear_f16_m1_splitk_cuda(x, weight);
    }
    return linear_f16_cuda(x, weight);
  }
  return tmix_linear_original_caller_dispatch_cuda(x, weight, caller_group);
}

at::Tensor tmix_linear_t_forward_varlen_cuda(
    at::Tensor x, at::Tensor weight_t) {
  return linear_t_f16_cuda(x, weight_t);
}

at::Tensor tmix_linear_t_tanh_forward_varlen_cuda(
    at::Tensor x, at::Tensor weight_t) {
  return linear_t_act_f16_cuda(x, weight_t, 1);
}

at::Tensor tmix_linear_t_sigmoid_forward_varlen_cuda(
    at::Tensor x, at::Tensor weight_t) {
  return linear_t_act_f16_cuda(x, weight_t, 2);
}

at::Tensor tmix_linear_t_vres_forward_varlen_cuda(
    at::Tensor x,
    at::Tensor weight_t,
    at::Tensor v,
    at::Tensor v_first,
    at::Tensor v0) {
  return linear_t_vres_f16_cuda(x, weight_t, v, v_first, v0);
}

at::Tensor tmix_linear_rank_in_dispatch_forward_varlen_cuda(
    at::Tensor x,
    std::optional<at::Tensor> weight,
    std::optional<at::Tensor> weight_t) {
  return linear_lowrank_large_dispatch_f16_cuda(
      x, weight, weight_t, true);
}

at::Tensor tmix_linear_rank_out_dispatch_forward_varlen_cuda(
    at::Tensor x,
    std::optional<at::Tensor> weight,
    std::optional<at::Tensor> weight_t) {
  return linear_lowrank_large_dispatch_f16_cuda(
      x, weight, weight_t, false);
}

at::Tensor tmix_linear_attention_c2c_forward_varlen_cuda(
    at::Tensor x,
    at::Tensor weight,
    at::Tensor lora_a,
    at::Tensor lora_b,
    double lora_scale) {
  auto output = tmix_linear_original_caller_dispatch_cuda(
      x, weight, static_cast<int64_t>(LinearCallerGroup::kAttentionC2C));
  const int64_t rows = x.size(0);
  const float scale = static_cast<float>(lora_scale);
  auto rank_features = rows <= 7
      ? linear_t_f16_cuda(x, lora_a)
      : linear_lowrank_large_dispatch_f16_cuda(
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

std::vector<at::Tensor> tmix_lowrank_in_forward_varlen_cuda(
    at::Tensor x_w,
    at::Tensor x_a,
    at::Tensor x_g,
    std::optional<at::Tensor> w1_t,
    std::optional<at::Tensor> a1_t,
    std::optional<at::Tensor> g1_t,
    std::optional<at::Tensor> w1,
    std::optional<at::Tensor> a1,
    std::optional<at::Tensor> g1) {
  const int64_t rows = x_w.size(0);
  if (rows <= 7 && w1_t.has_value() && a1_t.has_value() && g1_t.has_value()) {
    return linear_wag_rank_in_f16_cuda(
        x_w, x_a, x_g, *w1_t, *a1_t, *g1_t);
  }
  return {
      linear_lowrank_large_dispatch_f16_cuda(x_w, w1, w1_t, true),
      linear_lowrank_large_dispatch_f16_cuda(x_a, a1, a1_t, true),
      linear_lowrank_large_dispatch_f16_cuda(x_g, g1, g1_t, true),
  };
}

std::vector<at::Tensor> tmix_lowrank_wagv_in_forward_varlen_cuda(
    at::Tensor x_w,
    at::Tensor x_a,
    at::Tensor x_g,
    at::Tensor x_v,
    std::optional<at::Tensor> w1_t,
    std::optional<at::Tensor> a1_t,
    std::optional<at::Tensor> g1_t,
    std::optional<at::Tensor> v1_t,
    std::optional<at::Tensor> w1,
    std::optional<at::Tensor> a1,
    std::optional<at::Tensor> g1,
    std::optional<at::Tensor> v1) {
  const int64_t rows = x_w.size(0);
  if (rows <= 7 && w1_t.has_value() && a1_t.has_value() &&
      g1_t.has_value() && v1_t.has_value()) {
    return linear_wagv_rank_in_f16_cuda(
        x_w, x_a, x_g, x_v, *w1_t, *a1_t, *g1_t, *v1_t);
  }
  return {
      linear_lowrank_large_dispatch_f16_cuda(x_w, w1, w1_t, true),
      linear_lowrank_large_dispatch_f16_cuda(x_a, a1, a1_t, true),
      linear_lowrank_large_dispatch_f16_cuda(x_g, g1, g1_t, true),
      linear_lowrank_large_dispatch_f16_cuda(x_v, v1, v1_t, true),
  };
}

std::vector<at::Tensor> tmix_lowrank_out_forward_varlen_cuda(
    at::Tensor w1,
    at::Tensor a1,
    at::Tensor g1,
    std::optional<at::Tensor> w2_t,
    std::optional<at::Tensor> a2_t,
    std::optional<at::Tensor> g2_t,
    std::optional<at::Tensor> w2,
    std::optional<at::Tensor> a2,
    std::optional<at::Tensor> g2) {
  const int64_t rows = w1.size(0);
  if (rows <= 4 && w2_t.has_value() && a2_t.has_value() && g2_t.has_value()) {
    return linear_wag_rank_out_f16_cuda(
        w1, a1, g1, *w2_t, *a2_t, *g2_t);
  }
  auto w1_activated = tmix_linear_act_tanh_forward_varlen_cuda(w1);
  auto g1_activated = tmix_linear_act_sigmoid_forward_varlen_cuda(g1);
  return {
      linear_lowrank_large_dispatch_f16_cuda(
          w1_activated, w2, w2_t, false),
      linear_lowrank_large_dispatch_f16_cuda(a1, a2, a2_t, false),
      linear_lowrank_large_dispatch_f16_cuda(
          g1_activated, g2, g2_t, false),
  };
}

std::vector<at::Tensor> tmix_lowrank_vres_forward_varlen_cuda(
    at::Tensor w1,
    at::Tensor a1,
    at::Tensor g1,
    at::Tensor v1,
    std::optional<at::Tensor> w2_t,
    std::optional<at::Tensor> a2_t,
    std::optional<at::Tensor> g2_t,
    std::optional<at::Tensor> v2_t,
    std::optional<at::Tensor> w2,
    std::optional<at::Tensor> a2,
    std::optional<at::Tensor> g2,
    std::optional<at::Tensor> v2,
    at::Tensor v,
    at::Tensor v_first,
    at::Tensor v0) {
  const int64_t rows = w1.size(0);
  if (rows <= 4 && w2_t.has_value() && a2_t.has_value() &&
      g2_t.has_value() && v2_t.has_value()) {
    return linear_wagv_rank_out_f16_cuda(
        w1, a1, g1, v1, *w2_t, *a2_t, *g2_t, *v2_t,
        v, v_first, v0);
  }
  auto wag = tmix_lowrank_out_forward_varlen_cuda(
      w1, a1, g1, w2_t, a2_t, g2_t, w2, a2, g2);
  auto v12 = linear_lowrank_large_dispatch_f16_cuda(v1, v2, v2_t, false);
  auto value = at::empty_like(v);
  tmix_vres_gate_forward_varlen_cuda(
      static_cast<int>(rows), static_cast<int>(v.size(1)),
      v, v_first, v0, v12, value);
  return {wag[0], wag[1], wag[2], value};
}
