// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Upstream repository: https://github.com/BlinkDL/Albatross
// Albatross source revision: ee3308f6922e59f2166c7fac3c5a192340a2b48e
// Original path: faster3a_2607/cuda/rwkv7_v3a_ops.cu
// Mechanical migration boundary: exact upstream add, layer-normalization,
// statistics, and indexed-last-layer-normalization bodies are retained.
// The local wrappers expose packed token rows; simple row-wise operations need
// no request metadata. Packed last-layer adaptation is owned by head/linear.
// No ATen implementation or generic fallback is present here.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <climits>
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

template <int Act>
__device__ __forceinline__ float apply_act(float x) {
  if constexpr (Act == 1) {
    return tanhf(x);
  } else {
    return 1.0f / (1.0f + expf(-x));
  }
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
__global__ void add_f16_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ y,
    dtype* __restrict__ out,
    int64_t n_pairs) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n_pairs) {
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[i]);
    const float2 yv = __half22float2(reinterpret_cast<const __half2*>(y)[i]);
    reinterpret_cast<__half2*>(out)[i] = __floats2half2_rn(xv.x + yv.x, xv.y + yv.y);
  }
}

__global__ void layer_norm_f16_kernel(
    int C,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    dtype* __restrict__ y,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * C;
  float sum = 0.0f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c));
    sum += v;
  }
  sum = block_sum_t<LN_THREADS>(sum);
  const float inv_c = 1.0f / static_cast<float>(C);
  const float mean = sum * inv_c;
  float sum_var = 0.0f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<LN_THREADS>(sum_var);
  const float var = sum_var * inv_c;
  const float rstd = rsqrtf(var + eps);
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c));
    const float w = __half2float(*reinterpret_cast<const __half*>(weight + c));
    const float b = __half2float(*reinterpret_cast<const __half*>(bias + c));
    *reinterpret_cast<__half*>(y + base + c) = __float2half_rn((v - mean) * rstd * w + b);
  }
}

__global__ void add_layer_norm_f16_kernel(
    int C,
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    dtype* __restrict__ x_out,
    dtype* __restrict__ y,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * C;
  float sum = 0.0f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                    __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    sum += v;
  }
  sum = block_sum_t<LN_THREADS>(sum);
  const float inv_c = 1.0f / static_cast<float>(C);
  const float mean = sum * inv_c;
  float sum_var = 0.0f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                    __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<LN_THREADS>(sum_var);
  const float rstd = rsqrtf(sum_var * inv_c + eps);
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                    __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float w = __half2float(*reinterpret_cast<const __half*>(weight + c));
    const float b = __half2float(*reinterpret_cast<const __half*>(bias + c));
    *reinterpret_cast<__half*>(x_out + base + c) = __float2half_rn(v);
    *reinterpret_cast<__half*>(y + base + c) = __float2half_rn((v - mean) * rstd * w + b);
  }
}

template <int Threads, bool VecStats, bool VecOut>
__global__ __launch_bounds__(Threads, 1) void layer_norm_f16_small_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    dtype* __restrict__ y,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * LN_SMALL_C;
  float sum = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 v = __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      sum += v.x + v.y;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c));
      sum += v;
    }
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 v = __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float dx = v.x - mean;
      const float dy = v.y - mean;
      sum_var += dx * dx + dy * dy;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c));
      const float d = v - mean;
      sum_var += d * d;
    }
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
  if constexpr (VecOut) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 v = __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[idx]);
      const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[idx]);
      reinterpret_cast<__half2*>(y + base)[idx] = __floats2half2_rn(
          (v.x - mean) * rstd * w.x + b.x,
          (v.y - mean) * rstd * w.y + b.y);
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c));
      const float w = __half2float(*reinterpret_cast<const __half*>(weight + c));
      const float b = __half2float(*reinterpret_cast<const __half*>(bias + c));
      *reinterpret_cast<__half*>(y + base + c) = __float2half_rn((v - mean) * rstd * w + b);
    }
  }
}

template <int Threads, bool VecStats, bool VecOut>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_f16_small_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    dtype* __restrict__ x_out,
    dtype* __restrict__ y,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * LN_SMALL_C;
  float sum = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + base)[idx]);
      sum += xv.x + rv.x + xv.y + rv.y;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                      __half2float(*reinterpret_cast<const __half*>(residual + base + c));
      sum += v;
    }
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + base)[idx]);
      const float dx = xv.x + rv.x - mean;
      const float dy = xv.y + rv.y - mean;
      sum_var += dx * dx + dy * dy;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                      __half2float(*reinterpret_cast<const __half*>(residual + base + c));
      const float d = v - mean;
      sum_var += d * d;
    }
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
  if constexpr (VecOut) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + base)[idx]);
      const float sx = xv.x + rv.x;
      const float sy = xv.y + rv.y;
      const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[idx]);
      const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[idx]);
      reinterpret_cast<__half2*>(x_out + base)[idx] = __floats2half2_rn(sx, sy);
      reinterpret_cast<__half2*>(y + base)[idx] = __floats2half2_rn(
          (sx - mean) * rstd * w.x + b.x,
          (sy - mean) * rstd * w.y + b.y);
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                      __half2float(*reinterpret_cast<const __half*>(residual + base + c));
      const float w = __half2float(*reinterpret_cast<const __half*>(weight + c));
      const float b = __half2float(*reinterpret_cast<const __half*>(bias + c));
      *reinterpret_cast<__half*>(x_out + base + c) = __float2half_rn(v);
      *reinterpret_cast<__half*>(y + base + c) = __float2half_rn((v - mean) * rstd * w + b);
    }
  }
}

__device__ __forceinline__ void welford_merge_equal(
    float& mean,
    float& m2,
    float other_mean,
    float other_m2,
    float correction_factor) {
  // This simplified merge is valid only for equal-size partials. The local,
  // warp, and block trees below deliberately preserve that invariant.
  const float delta = other_mean - mean;
  mean = fmaf(delta, 0.5f, mean);
  m2 = fmaf(delta * delta, correction_factor, m2 + other_m2);
}

__device__ __forceinline__ void block_welford_256(float& mean, float& m2) {
  __shared__ float warp_mean[8];
  __shared__ float warp_m2[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  float group_count = 16.0f;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other_mean = __shfl_down_sync(0xffffffffu, mean, offset);
    const float other_m2 = __shfl_down_sync(0xffffffffu, m2, offset);
    if (lane < offset) {
      welford_merge_equal(mean, m2, other_mean, other_m2, group_count * 0.5f);
    }
    group_count *= 2.0f;
  }
  if (lane == 0) {
    warp_mean[warp] = mean;
    warp_m2[warp] = m2;
  }
  __syncthreads();

  if (warp == 0) {
    mean = lane < 8 ? warp_mean[lane] : 0.0f;
    m2 = lane < 8 ? warp_m2[lane] : 0.0f;
    group_count = 512.0f;
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
      const float other_mean = __shfl_down_sync(0xffffffffu, mean, offset);
      const float other_m2 = __shfl_down_sync(0xffffffffu, m2, offset);
      if (lane < offset) {
        welford_merge_equal(mean, m2, other_mean, other_m2, group_count * 0.5f);
      }
      group_count *= 2.0f;
    }
    if (lane == 0) {
      warp_mean[0] = mean;
      warp_m2[0] = m2;
    }
  }
  __syncthreads();
  mean = warp_mean[0];
  m2 = warp_m2[0];
}

template <bool CacheRounded>
__device__ __forceinline__ float2 load_add_pair(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    dtype* __restrict__ x_out,
    int64_t pair_index) {
  const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[pair_index]);
  const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[pair_index]);
  float2 sum = make_float2(xv.x + rv.x, xv.y + rv.y);
  if constexpr (CacheRounded) {
    const __half2 rounded = __floats2half2_rn(sum.x, sum.y);
    reinterpret_cast<__half2*>(x_out)[pair_index] = rounded;
    sum = __half22float2(rounded);
  }
  return sum;
}

template <bool CacheRounded>
__global__ __launch_bounds__(256, 1) void add_layer_norm_f16_welford_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    dtype* __restrict__ x_out,
    dtype* __restrict__ y,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  constexpr int Threads = 256;
  constexpr int PairsPerThread = (LN_SMALL_C / 2) / Threads;
  const int64_t base2 = row * (LN_SMALL_C / 2);

  float2 pair = load_add_pair<CacheRounded>(x, residual, x_out, base2 + threadIdx.x);
  float delta = pair.y - pair.x;
  float mean = (pair.x + pair.y) * 0.5f;
  float m2 = delta * delta * 0.5f;
#pragma unroll
  for (int k = 1; k < PairsPerThread; ++k) {
    pair = load_add_pair<CacheRounded>(
        x, residual, x_out, base2 + threadIdx.x + static_cast<int64_t>(k) * Threads);
    delta = pair.y - pair.x;
    const float pair_mean = (pair.x + pair.y) * 0.5f;
    const float pair_m2 = delta * delta * 0.5f;
    const float old_count = static_cast<float>(2 * k);
    const float inv_count = 1.0f / static_cast<float>(2 * (k + 1));
    delta = pair_mean - mean;
    mean = fmaf(delta, 2.0f * inv_count, mean);
    m2 = fmaf(delta * delta, old_count * 2.0f * inv_count, m2 + pair_m2);
  }
  block_welford_256(mean, m2);
  const float rstd = rsqrtf(m2 * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);

  // CacheRounded stores x_out before block_welford_256. Its two CTA barriers
  // are required for visibility here; do not replace them with warp-only sync.
#pragma unroll
  for (int k = 0; k < PairsPerThread; ++k) {
    const int64_t pair_index = base2 + threadIdx.x + static_cast<int64_t>(k) * Threads;
    float2 sum;
    if constexpr (CacheRounded) {
      sum = __half22float2(reinterpret_cast<const __half2*>(x_out)[pair_index]);
    } else {
      sum = load_add_pair<false>(x, residual, x_out, pair_index);
      reinterpret_cast<__half2*>(x_out)[pair_index] = __floats2half2_rn(sum.x, sum.y);
    }
    const int pair_c = threadIdx.x + k * Threads;
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[pair_c]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[pair_c]);
    reinterpret_cast<__half2*>(y)[pair_index] = __floats2half2_rn(
        (sum.x - mean) * rstd * w.x + b.x,
        (sum.y - mean) * rstd * w.y + b.y);
  }
}

__global__ __launch_bounds__(256, 1) void add_layer_norm_f16_centered_cache_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    dtype* __restrict__ x_out,
    dtype* __restrict__ y,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  constexpr int Threads = 256;
  constexpr int PairsPerThread = (LN_SMALL_C / 2) / Threads;
  const int64_t base2 = row * (LN_SMALL_C / 2);
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < PairsPerThread; ++k) {
    const float2 value = load_add_pair<true>(
        x, residual, x_out, base2 + threadIdx.x + static_cast<int64_t>(k) * Threads);
    sum += value.x + value.y;
  }
  const float mean = block_sum_t<Threads>(sum) * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
#pragma unroll
  for (int k = 0; k < PairsPerThread; ++k) {
    const int64_t pair_index = base2 + threadIdx.x + static_cast<int64_t>(k) * Threads;
    const float2 value = __half22float2(reinterpret_cast<const __half2*>(x_out)[pair_index]);
    const float dx = value.x - mean;
    const float dy = value.y - mean;
    sum_var += dx * dx + dy * dy;
  }
  const float rstd = rsqrtf(
      block_sum_t<Threads>(sum_var) * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
#pragma unroll
  for (int k = 0; k < PairsPerThread; ++k) {
    const int64_t pair_index = base2 + threadIdx.x + static_cast<int64_t>(k) * Threads;
    const int pair_c = threadIdx.x + k * Threads;
    const float2 value = __half22float2(reinterpret_cast<const __half2*>(x_out)[pair_index]);
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[pair_c]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[pair_c]);
    reinterpret_cast<__half2*>(y)[pair_index] = __floats2half2_rn(
        (value.x - mean) * rstd * w.x + b.x,
        (value.y - mean) * rstd * w.y + b.y);
  }
}

// Disabled migration references (Albatross revision
// ee3308f6922e59f2166c7fac3c5a192340a2b48e): these non-packed fused CMix/TMix
// bodies have no launch owner in the current FlashRWKV translation unit.  The
// packed families below own the public varlen paths.  Keep the upstream bodies
// for provenance, but do not re-enable them without an owned non-packed API,
// correctness coverage, launch guards and benchmark evidence.
#if 0
template <bool CacheRounded>
__global__ __launch_bounds__(256, 1) void add_layer_norm_cmix_mix_f16_welford_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ x_k,
    dtype* __restrict__ x_out,
    dtype* __restrict__ mixed,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  constexpr int Threads = 256;
  constexpr int PairsPerThread = (LN_SMALL_C / 2) / Threads;
  const int64_t base2 = row * (LN_SMALL_C / 2);

  float2 pair = load_add_pair<CacheRounded>(x, residual, x_out, base2 + threadIdx.x);
  float delta = pair.y - pair.x;
  float mean = (pair.x + pair.y) * 0.5f;
  float m2 = delta * delta * 0.5f;
#pragma unroll
  for (int k = 1; k < PairsPerThread; ++k) {
    pair = load_add_pair<CacheRounded>(
        x, residual, x_out, base2 + threadIdx.x + static_cast<int64_t>(k) * Threads);
    delta = pair.y - pair.x;
    const float pair_mean = (pair.x + pair.y) * 0.5f;
    const float pair_m2 = delta * delta * 0.5f;
    const float old_count = static_cast<float>(2 * k);
    const float inv_count = 1.0f / static_cast<float>(2 * (k + 1));
    delta = pair_mean - mean;
    mean = fmaf(delta, 2.0f * inv_count, mean);
    m2 = fmaf(delta * delta, old_count * 2.0f * inv_count, m2 + pair_m2);
  }
  block_welford_256(mean, m2);
  const float rstd = rsqrtf(m2 * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);

  // CacheRounded relies on the CTA barriers inside block_welford_256 before
  // reloading x_out. The same owner also updates its shift-state half2, so no
  // cross-thread or cross-CTA communication is required in the output pass.
#pragma unroll
  for (int k = 0; k < PairsPerThread; ++k) {
    const int64_t pair_index = base2 + threadIdx.x + static_cast<int64_t>(k) * Threads;
    float2 sum;
    if constexpr (CacheRounded) {
      sum = __half22float2(reinterpret_cast<const __half2*>(x_out)[pair_index]);
    } else {
      sum = load_add_pair<false>(x, residual, x_out, pair_index);
      reinterpret_cast<__half2*>(x_out)[pair_index] = __floats2half2_rn(sum.x, sum.y);
    }
    const int pair_c = threadIdx.x + k * Threads;
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[pair_c]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[pair_c]);
    const float2 previous = __half22float2(reinterpret_cast<const __half2*>(shift_state)[pair_index]);
    const float2 mix = __half22float2(reinterpret_cast<const __half2*>(x_k)[pair_c]);
    const __half2 normalized = __floats2half2_rn(
        (sum.x - mean) * rstd * w.x + b.x,
        (sum.y - mean) * rstd * w.y + b.y);
    const float2 normalized_f = __half22float2(normalized);
    reinterpret_cast<__half2*>(mixed)[pair_index] = __floats2half2_rn(
        normalized_f.x + (previous.x - normalized_f.x) * mix.x,
        normalized_f.y + (previous.y - normalized_f.y) * mix.y);
    reinterpret_cast<__half2*>(shift_state)[pair_index] = normalized;
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_cmix_mix_f16_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ x_k,
    dtype* __restrict__ x_out,
    dtype* __restrict__ mixed,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * LN_SMALL_C;
  float sum = 0.0f;
  const int64_t base2 = base >> 1;
  constexpr int pairs = LN_SMALL_C >> 1;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    sum += xv.x + rv.x + xv.y + rv.y;
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const float d0 = x0 - mean;
    const float d1 = x1 - mean;
    sum_var += d0 * d0 + d1 * d1;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(reinterpret_cast<const __half2*>(shift_state)[base2 + p]);
    const float2 mix = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x, (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(mixed)[base2 + p] =
        __floats2half2_rn(yv.x + (prev.x - yv.x) * mix.x, yv.y + (prev.y - yv.y) * mix.y);
    reinterpret_cast<__half2*>(shift_state)[base2 + p] = y2;
  }
}

// Disabled migration reference (Albatross revision
// ee3308f6922e59f2166c7fac3c5a192340a2b48e): this scalar-statistics CMix
// body has no launch owner or selector in the current FlashRWKV translation
// unit.  Its vectorized active counterpart owns the public contract.  Do not
// re-enable without a real dispatch owner, correctness coverage and benchmark
// evidence.
#if 0
template <int Threads>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_cmix_mix_f16_scalar_stats_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ x_k,
    dtype* __restrict__ x_out,
    dtype* __restrict__ mixed,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * LN_SMALL_C;
  const int64_t base2 = base >> 1;
  constexpr int pairs = LN_SMALL_C >> 1;
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
    const int c = threadIdx.x + k * Threads;
    sum += __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
           __half2float(*reinterpret_cast<const __half*>(residual + base + c));
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
#pragma unroll
  for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
    const int c = threadIdx.x + k * Threads;
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                    __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(reinterpret_cast<const __half2*>(shift_state)[base2 + p]);
    const float2 mix = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x, (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(mixed)[base2 + p] =
        __floats2half2_rn(yv.x + (prev.x - yv.x) * mix.x, yv.y + (prev.y - yv.y) * mix.y);
    reinterpret_cast<__half2*>(shift_state)[base2 + p] = y2;
  }
}

#endif

template <int Threads>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_tmix_mix6_f16_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v,
    const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g,
    dtype* __restrict__ x_out,
    dtype* __restrict__ out_r,
    dtype* __restrict__ out_w,
    dtype* __restrict__ out_k,
    dtype* __restrict__ out_v,
    dtype* __restrict__ out_a,
    dtype* __restrict__ out_g,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base2 = row * (LN_SMALL_C >> 1);
  constexpr int pairs = LN_SMALL_C >> 1;
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    sum += xv.x + rv.x + xv.y + rv.y;
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const float d0 = x0 - mean;
    const float d1 = x1 - mean;
    sum_var += d0 * d0 + d1 * d1;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(reinterpret_cast<const __half2*>(shift_state)[base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x, (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    const float dx0 = prev.x - yv.x;
    const float dx1 = prev.y - yv.y;
    const float2 mr = __half22float2(reinterpret_cast<const __half2*>(x_r)[p]);
    const float2 mw = __half22float2(reinterpret_cast<const __half2*>(x_w)[p]);
    const float2 mk = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float2 mv = __half22float2(reinterpret_cast<const __half2*>(x_v)[p]);
    const float2 ma = __half22float2(reinterpret_cast<const __half2*>(x_a)[p]);
    const float2 mg = __half22float2(reinterpret_cast<const __half2*>(x_g)[p]);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(out_r)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mr.x, yv.y + dx1 * mr.y);
    reinterpret_cast<__half2*>(out_w)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mw.x, yv.y + dx1 * mw.y);
    reinterpret_cast<__half2*>(out_k)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mk.x, yv.y + dx1 * mk.y);
    reinterpret_cast<__half2*>(out_v)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mv.x, yv.y + dx1 * mv.y);
    reinterpret_cast<__half2*>(out_a)[base2 + p] = __floats2half2_rn(yv.x + dx0 * ma.x, yv.y + dx1 * ma.y);
    reinterpret_cast<__half2*>(out_g)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mg.x, yv.y + dx1 * mg.y);
    reinterpret_cast<__half2*>(shift_state)[base2 + p] = y2;
  }
}

#endif

// Exact Albatross add_layer_norm_cmix_mix_f16_welford_kernel with the only
// varlen adaptation being the state-slot address.  The fused caller is
// intentionally restricted to the upstream T==1 path by its module-local
// binding, so packed row `row` is also sequence `row` and the scheduler slot
// comes from state_indices[row].
template <bool CacheRounded>
__global__ __launch_bounds__(256, 1) void add_layer_norm_cmix_mix_f16_packed_welford_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ x_k,
    dtype* __restrict__ x_out,
    dtype* __restrict__ mixed,
    const int* __restrict__ state_indices,
    const int* __restrict__ metadata_status,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  if (metadata_status[0] != 0) {
    const __half invalid = __ushort_as_half(0x7e00);
    for (int c = threadIdx.x; c < LN_SMALL_C; c += blockDim.x) {
      reinterpret_cast<__half*>(x_out + row * LN_SMALL_C)[c] = invalid;
      reinterpret_cast<__half*>(mixed + row * LN_SMALL_C)[c] = invalid;
    }
    return;
  }
  if (row >= metadata_status[2]) {
    return;
  }

  constexpr int Threads = 256;
  constexpr int PairsPerThread = (LN_SMALL_C / 2) / Threads;
  const int64_t base2 = row * (LN_SMALL_C / 2);
  const int64_t state_base2 =
      static_cast<int64_t>(state_indices[row]) * (LN_SMALL_C / 2);

  float2 pair = load_add_pair<CacheRounded>(
      x, residual, x_out, base2 + threadIdx.x);
  float delta = pair.y - pair.x;
  float mean = (pair.x + pair.y) * 0.5f;
  float m2 = delta * delta * 0.5f;
#pragma unroll
  for (int k = 1; k < PairsPerThread; ++k) {
    pair = load_add_pair<CacheRounded>(
        x, residual, x_out,
        base2 + threadIdx.x + static_cast<int64_t>(k) * Threads);
    delta = pair.y - pair.x;
    const float pair_mean = (pair.x + pair.y) * 0.5f;
    const float pair_m2 = delta * delta * 0.5f;
    const float old_count = static_cast<float>(2 * k);
    const float inv_count = 1.0f / static_cast<float>(2 * (k + 1));
    delta = pair_mean - mean;
    mean = fmaf(delta, 2.0f * inv_count, mean);
    m2 = fmaf(delta * delta, old_count * 2.0f * inv_count, m2 + pair_m2);
  }
  block_welford_256(mean, m2);
  const float rstd = rsqrtf(m2 * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);

#pragma unroll
  for (int k = 0; k < PairsPerThread; ++k) {
    const int64_t pair_index =
        base2 + threadIdx.x + static_cast<int64_t>(k) * Threads;
    float2 sum;
    if constexpr (CacheRounded) {
      sum = __half22float2(reinterpret_cast<const __half2*>(x_out)[pair_index]);
    } else {
      sum = load_add_pair<false>(x, residual, x_out, pair_index);
      reinterpret_cast<__half2*>(x_out)[pair_index] =
          __floats2half2_rn(sum.x, sum.y);
    }
    const int pair_c = threadIdx.x + k * Threads;
    const float2 w = __half22float2(
        reinterpret_cast<const __half2*>(weight)[pair_c]);
    const float2 b = __half22float2(
        reinterpret_cast<const __half2*>(bias)[pair_c]);
    const float2 previous = __half22float2(
        reinterpret_cast<const __half2*>(shift_state)[state_base2 +
                                                       threadIdx.x +
                                                       static_cast<int64_t>(k) *
                                                           Threads]);
    const float2 mix = __half22float2(
        reinterpret_cast<const __half2*>(x_k)[pair_c]);
    const __half2 normalized = __floats2half2_rn(
        (sum.x - mean) * rstd * w.x + b.x,
        (sum.y - mean) * rstd * w.y + b.y);
    const float2 normalized_f = __half22float2(normalized);
    reinterpret_cast<__half2*>(mixed)[pair_index] = __floats2half2_rn(
        normalized_f.x + (previous.x - normalized_f.x) * mix.x,
        normalized_f.y + (previous.y - normalized_f.y) * mix.y);
    reinterpret_cast<__half2*>(shift_state)[state_base2 + threadIdx.x +
                                             static_cast<int64_t>(k) * Threads] =
        normalized;
  }
}

// Exact Albatross add_layer_norm_cmix_mix_f16_kernel with packed input rows
// and scheduler-selected shift-state slots.  This is the non-statistics
// branch of the canonical fused CMix caller.
template <int Threads>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_cmix_mix_f16_packed_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ x_k,
    dtype* __restrict__ x_out,
    dtype* __restrict__ mixed,
    const int* __restrict__ state_indices,
    const int* __restrict__ metadata_status,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  if (metadata_status[0] != 0) {
    const __half invalid = __ushort_as_half(0x7e00);
    for (int c = threadIdx.x; c < LN_SMALL_C; c += Threads) {
      reinterpret_cast<__half*>(x_out + row * LN_SMALL_C)[c] = invalid;
      reinterpret_cast<__half*>(mixed + row * LN_SMALL_C)[c] = invalid;
    }
    return;
  }
  if (row >= metadata_status[2]) {
    return;
  }

  const int64_t base = row * LN_SMALL_C;
  const int64_t base2 = base >> 1;
  const int64_t state_base2 =
      static_cast<int64_t>(state_indices[row]) * (LN_SMALL_C / 2);
  constexpr int pairs = LN_SMALL_C >> 1;
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(
        reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(
        reinterpret_cast<const __half2*>(residual)[base2 + p]);
    sum += xv.x + rv.x + xv.y + rv.y;
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(
        reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(
        reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const float d0 = x0 - mean;
    const float d1 = x1 - mean;
    sum_var += d0 * d0 + d1 * d1;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd =
      rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(
        reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(
        reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w = __half22float2(
        reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(
        reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(
        reinterpret_cast<const __half2*>(shift_state)[state_base2 + p]);
    const float2 mix = __half22float2(
        reinterpret_cast<const __half2*>(x_k)[p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn(
        (x0 - mean) * rstd * w.x + b.x,
        (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    reinterpret_cast<__half2*>(x_out)[base2 + p] =
        __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(mixed)[base2 + p] = __floats2half2_rn(
        yv.x + (prev.x - yv.x) * mix.x,
        yv.y + (prev.y - yv.y) * mix.y);
    reinterpret_cast<__half2*>(shift_state)[state_base2 + p] = y2;
  }
}

// Exact Albatross add_layer_norm_tmix_mix6_f16 with the same T==1 packed
// boundary adaptation as the CMix kernels above.  The seven outputs preserve
// the upstream fused caller's x_out plus r/w/k/v/a/g ordering.
template <int Threads>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_tmix_mix6_f16_packed_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v,
    const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g,
    dtype* __restrict__ x_out,
    dtype* __restrict__ out_r,
    dtype* __restrict__ out_w,
    dtype* __restrict__ out_k,
    dtype* __restrict__ out_v,
    dtype* __restrict__ out_a,
    dtype* __restrict__ out_g,
    const int* __restrict__ state_indices,
    const int* __restrict__ metadata_status,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  if (metadata_status[0] != 0) {
    const __half invalid = __ushort_as_half(0x7e00);
    for (int c = threadIdx.x; c < LN_SMALL_C; c += Threads) {
      const int64_t output_index = row * LN_SMALL_C + c;
      reinterpret_cast<__half*>(x_out)[output_index] = invalid;
      reinterpret_cast<__half*>(out_r)[output_index] = invalid;
      reinterpret_cast<__half*>(out_w)[output_index] = invalid;
      reinterpret_cast<__half*>(out_k)[output_index] = invalid;
      reinterpret_cast<__half*>(out_v)[output_index] = invalid;
      reinterpret_cast<__half*>(out_a)[output_index] = invalid;
      reinterpret_cast<__half*>(out_g)[output_index] = invalid;
    }
    return;
  }
  if (row >= metadata_status[2]) {
    return;
  }

  const int64_t base2 = row * (LN_SMALL_C >> 1);
  const int64_t state_base2 =
      static_cast<int64_t>(state_indices[row]) * (LN_SMALL_C >> 1);
  constexpr int pairs = LN_SMALL_C >> 1;
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(
        reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(
        reinterpret_cast<const __half2*>(residual)[base2 + p]);
    sum += xv.x + rv.x + xv.y + rv.y;
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(
        reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(
        reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const float d0 = x0 - mean;
    const float d1 = x1 - mean;
    sum_var += d0 * d0 + d1 * d1;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd =
      rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(
        reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(
        reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w = __half22float2(
        reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(
        reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(
        reinterpret_cast<const __half2*>(shift_state)[state_base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn(
        (x0 - mean) * rstd * w.x + b.x,
        (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    const float dx0 = prev.x - yv.x;
    const float dx1 = prev.y - yv.y;
    const float2 mr = __half22float2(
        reinterpret_cast<const __half2*>(x_r)[p]);
    const float2 mw = __half22float2(
        reinterpret_cast<const __half2*>(x_w)[p]);
    const float2 mk = __half22float2(
        reinterpret_cast<const __half2*>(x_k)[p]);
    const float2 mv = __half22float2(
        reinterpret_cast<const __half2*>(x_v)[p]);
    const float2 ma = __half22float2(
        reinterpret_cast<const __half2*>(x_a)[p]);
    const float2 mg = __half22float2(
        reinterpret_cast<const __half2*>(x_g)[p]);
    reinterpret_cast<__half2*>(x_out)[base2 + p] =
        __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(out_r)[base2 + p] = __floats2half2_rn(
        yv.x + dx0 * mr.x, yv.y + dx1 * mr.y);
    reinterpret_cast<__half2*>(out_w)[base2 + p] = __floats2half2_rn(
        yv.x + dx0 * mw.x, yv.y + dx1 * mw.y);
    reinterpret_cast<__half2*>(out_k)[base2 + p] = __floats2half2_rn(
        yv.x + dx0 * mk.x, yv.y + dx1 * mk.y);
    reinterpret_cast<__half2*>(out_v)[base2 + p] = __floats2half2_rn(
        yv.x + dx0 * mv.x, yv.y + dx1 * mv.y);
    reinterpret_cast<__half2*>(out_a)[base2 + p] = __floats2half2_rn(
        yv.x + dx0 * ma.x, yv.y + dx1 * ma.y);
    reinterpret_cast<__half2*>(out_g)[base2 + p] = __floats2half2_rn(
        yv.x + dx0 * mg.x, yv.y + dx1 * mg.y);
    reinterpret_cast<__half2*>(shift_state)[state_base2 + p] = y2;
  }
}

// Disabled migration reference (same fixed Albatross revision): this
// scalar-statistics TMix body has no local launch owner or selector.  The
// active vectorized family above owns the public path.  Re-enabling requires
// an explicit dispatch contract, correctness coverage and benchmark evidence.
#if 0
template <int Threads>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_tmix_mix6_f16_scalar_stats_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v,
    const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g,
    dtype* __restrict__ x_out,
    dtype* __restrict__ out_r,
    dtype* __restrict__ out_w,
    dtype* __restrict__ out_k,
    dtype* __restrict__ out_v,
    dtype* __restrict__ out_a,
    dtype* __restrict__ out_g,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * LN_SMALL_C;
  const int64_t base2 = row * (LN_SMALL_C >> 1);
  constexpr int pairs = LN_SMALL_C >> 1;
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
    const int c = threadIdx.x + k * Threads;
    sum += __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
           __half2float(*reinterpret_cast<const __half*>(residual + base + c));
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
#pragma unroll
  for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
    const int c = threadIdx.x + k * Threads;
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                    __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(reinterpret_cast<const __half2*>(shift_state)[base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x, (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    const float dx0 = prev.x - yv.x;
    const float dx1 = prev.y - yv.y;
    const float2 mr = __half22float2(reinterpret_cast<const __half2*>(x_r)[p]);
    const float2 mw = __half22float2(reinterpret_cast<const __half2*>(x_w)[p]);
    const float2 mk = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float2 mv = __half22float2(reinterpret_cast<const __half2*>(x_v)[p]);
    const float2 ma = __half22float2(reinterpret_cast<const __half2*>(x_a)[p]);
    const float2 mg = __half22float2(reinterpret_cast<const __half2*>(x_g)[p]);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(out_r)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mr.x, yv.y + dx1 * mr.y);
    reinterpret_cast<__half2*>(out_w)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mw.x, yv.y + dx1 * mw.y);
    reinterpret_cast<__half2*>(out_k)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mk.x, yv.y + dx1 * mk.y);
    reinterpret_cast<__half2*>(out_v)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mv.x, yv.y + dx1 * mv.y);
    reinterpret_cast<__half2*>(out_a)[base2 + p] = __floats2half2_rn(yv.x + dx0 * ma.x, yv.y + dx1 * ma.y);
    reinterpret_cast<__half2*>(out_g)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mg.x, yv.y + dx1 * mg.y);
    reinterpret_cast<__half2*>(shift_state)[base2 + p] = y2;
  }
}

#endif

template <int Threads, bool VecStats, bool VecOut, bool Indexed, bool Packed = false>
__global__ __launch_bounds__(Threads, 1) void add_last_layer_norm_f16_small_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const int64_t* __restrict__ last_indices,
    dtype* __restrict__ y,
    int64_t B,
    int64_t T,
    float eps) {
  const int64_t bidx = blockIdx.x;
  if (bidx >= B) {
    return;
  }
  const int64_t dst = bidx * LN_SMALL_C;
  int64_t selected_t = T - 1;
  if constexpr (Indexed) {
    selected_t = last_indices[bidx];
    // Bounds are checked on device to preserve graph capture. Invalid indices
    // must never form an OOB pointer; a NaN row makes the violation visible.
    if (selected_t < 0 || selected_t >= T) {
      const __half invalid = __ushort_as_half(0x7e00);
      for (int c = threadIdx.x; c < LN_SMALL_C; c += Threads) {
        reinterpret_cast<__half*>(y + dst)[c] = invalid;
      }
      return;
    }
  }
  const int64_t src = [&] {
    if constexpr (Packed) {
      return selected_t * LN_SMALL_C;
    } else {
      return (bidx * T + selected_t) * LN_SMALL_C;
    }
  }();
  float sum = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + src)[idx]);
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + src)[idx]);
      sum += xv.x + rv.x + xv.y + rv.y;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
                      __half2float(*reinterpret_cast<const __half*>(residual + src + c));
      sum += v;
    }
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + src)[idx]);
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + src)[idx]);
      const float dx = xv.x + rv.x - mean;
      const float dy = xv.y + rv.y - mean;
      sum_var += dx * dx + dy * dy;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
                      __half2float(*reinterpret_cast<const __half*>(residual + src + c));
      const float d = v - mean;
      sum_var += d * d;
    }
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
  if constexpr (VecOut) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + src)[idx]);
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + src)[idx]);
      const float sx = xv.x + rv.x;
      const float sy = xv.y + rv.y;
      const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[idx]);
      const float2 bb = __half22float2(reinterpret_cast<const __half2*>(bias)[idx]);
      reinterpret_cast<__half2*>(y + dst)[idx] = __floats2half2_rn(
          (sx - mean) * rstd * w.x + bb.x,
          (sy - mean) * rstd * w.y + bb.y);
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
                      __half2float(*reinterpret_cast<const __half*>(residual + src + c));
      const float w = __half2float(*reinterpret_cast<const __half*>(weight + c));
      const float bb = __half2float(*reinterpret_cast<const __half*>(bias + c));
      *reinterpret_cast<__half*>(y + dst + c) = __float2half_rn((v - mean) * rstd * w + bb);
    }
  }
}

template <int Threads, bool Indexed, bool Packed = false>
__global__ __launch_bounds__(Threads, 1) void add_last_layer_norm_f16_generic_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const int64_t* __restrict__ last_indices,
    dtype* __restrict__ y,
    int64_t B,
    int64_t T,
    int C,
    float eps) {
  const int64_t bidx = blockIdx.x;
  if (bidx >= B) {
    return;
  }
  const int64_t dst = bidx * static_cast<int64_t>(C);
  int64_t selected_t = T - 1;
  if constexpr (Indexed) {
    selected_t = last_indices[bidx];
    if (selected_t < 0 || selected_t >= T) {
      const __half invalid = __ushort_as_half(0x7e00);
      for (int c = threadIdx.x; c < C; c += Threads) {
        reinterpret_cast<__half*>(y + dst)[c] = invalid;
      }
      return;
    }
  }
  const int64_t src = [&] {
    if constexpr (Packed) {
      return selected_t * static_cast<int64_t>(C);
    } else {
      return (bidx * T + selected_t) * static_cast<int64_t>(C);
    }
  }();
  float sum = 0.0f;
  for (int c = threadIdx.x; c < C; c += Threads) {
    sum += __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
           __half2float(*reinterpret_cast<const __half*>(residual + src + c));
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum / static_cast<float>(C);
  float sum_var = 0.0f;
  for (int c = threadIdx.x; c < C; c += Threads) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
                    __half2float(*reinterpret_cast<const __half*>(residual + src + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var / static_cast<float>(C) + eps);
  const int pairs = C >> 1;
  for (int p = threadIdx.x; p < pairs; p += Threads) {
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + src)[p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + src)[p]);
    const float sx = xv.x + rv.x;
    const float sy = xv.y + rv.y;
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 bb = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    reinterpret_cast<__half2*>(y + dst)[p] = __floats2half2_rn(
        (sx - mean) * rstd * w.x + bb.x,
        (sy - mean) * rstd * w.y + bb.y);
  }
}

// Disabled migration references (Albatross revision
// ee3308f6922e59f2166c7fac3c5a192340a2b48e): these generic fused CMix/TMix
// bodies have no launch owner or selector in the current local translation
// unit.  They are retained for provenance but excluded from the active binary.
// Re-enable only with an owned selector, shape/launch guards, correctness tests
// and benchmark evidence.
#if 0
template <int Threads>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_cmix_mix_f16_generic_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ x_k,
    dtype* __restrict__ x_out,
    dtype* __restrict__ mixed,
    int64_t rows,
    int C,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * static_cast<int64_t>(C);
  float sum = 0.0f;
  for (int c = threadIdx.x; c < C; c += Threads) {
    sum += __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
           __half2float(*reinterpret_cast<const __half*>(residual + base + c));
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum / static_cast<float>(C);
  float sum_var = 0.0f;
  for (int c = threadIdx.x; c < C; c += Threads) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                    __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var / static_cast<float>(C) + eps);
  const int pairs = C >> 1;
  const int64_t base2 = base >> 1;
  for (int p = threadIdx.x; p < pairs; p += Threads) {
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(reinterpret_cast<const __half2*>(shift_state)[base2 + p]);
    const float2 mix = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x, (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(mixed)[base2 + p] =
        __floats2half2_rn(yv.x + (prev.x - yv.x) * mix.x, yv.y + (prev.y - yv.y) * mix.y);
    reinterpret_cast<__half2*>(shift_state)[base2 + p] = y2;
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_tmix_mix6_f16_generic_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v,
    const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g,
    dtype* __restrict__ x_out,
    dtype* __restrict__ out_r,
    dtype* __restrict__ out_w,
    dtype* __restrict__ out_k,
    dtype* __restrict__ out_v,
    dtype* __restrict__ out_a,
    dtype* __restrict__ out_g,
    int64_t rows,
    int C,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * static_cast<int64_t>(C);
  float sum = 0.0f;
  for (int c = threadIdx.x; c < C; c += Threads) {
    sum += __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
           __half2float(*reinterpret_cast<const __half*>(residual + base + c));
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum / static_cast<float>(C);
  float sum_var = 0.0f;
  for (int c = threadIdx.x; c < C; c += Threads) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                    __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var / static_cast<float>(C) + eps);
  const int pairs = C >> 1;
  const int64_t base2 = base >> 1;
  for (int p = threadIdx.x; p < pairs; p += Threads) {
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(reinterpret_cast<const __half2*>(shift_state)[base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x, (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    const float dx0 = prev.x - yv.x;
    const float dx1 = prev.y - yv.y;
    const float2 mr = __half22float2(reinterpret_cast<const __half2*>(x_r)[p]);
    const float2 mw = __half22float2(reinterpret_cast<const __half2*>(x_w)[p]);
    const float2 mk = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float2 mv = __half22float2(reinterpret_cast<const __half2*>(x_v)[p]);
    const float2 ma = __half22float2(reinterpret_cast<const __half2*>(x_a)[p]);
    const float2 mg = __half22float2(reinterpret_cast<const __half2*>(x_g)[p]);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(out_r)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mr.x, yv.y + dx1 * mr.y);
    reinterpret_cast<__half2*>(out_w)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mw.x, yv.y + dx1 * mw.y);
    reinterpret_cast<__half2*>(out_k)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mk.x, yv.y + dx1 * mk.y);
    reinterpret_cast<__half2*>(out_v)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mv.x, yv.y + dx1 * mv.y);
    reinterpret_cast<__half2*>(out_a)[base2 + p] = __floats2half2_rn(yv.x + dx0 * ma.x, yv.y + dx1 * ma.y);
    reinterpret_cast<__half2*>(out_g)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mg.x, yv.y + dx1 * mg.y);
    reinterpret_cast<__half2*>(shift_state)[base2 + p] = y2;
  }
}

#endif

} // namespace

at::Tensor add_f16_cuda(at::Tensor x, at::Tensor y) {
  TORCH_CHECK((x.numel() % 2) == 0, "add_f16 requires even numel");
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t n_pairs = x.numel() / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  add_f16_kernel<<<static_cast<int>(ceil_div(n_pairs, threads)), threads, 0, stream>>>(
      x.data_ptr<dtype>(), y.data_ptr<dtype>(), out.data_ptr<dtype>(), n_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor layer_norm_f16_cuda(at::Tensor x, at::Tensor weight, at::Tensor bias, double eps) {
  auto y = at::empty_like(x);
  const int64_t c64 = x.size(-1);
  TORCH_CHECK(c64 <= INT_MAX, "C too large");
  const int C = static_cast<int>(c64);
  const int64_t rows = x.numel() / C;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (C == LN_SMALL_C) {
    if (rows >= 1024) {
      layer_norm_f16_small_kernel<LN_SMALL512_THREADS, true, true><<<static_cast<int>(rows), LN_SMALL512_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(),
          weight.data_ptr<dtype>(),
          bias.data_ptr<dtype>(),
          y.data_ptr<dtype>(),
          rows,
          static_cast<float>(eps));
    } else if (rows >= 512) {
      layer_norm_f16_small_kernel<LN_SMALL512_THREADS, false, false><<<static_cast<int>(rows), LN_SMALL512_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(),
          weight.data_ptr<dtype>(),
          bias.data_ptr<dtype>(),
          y.data_ptr<dtype>(),
          rows,
          static_cast<float>(eps));
    } else {
      layer_norm_f16_small_kernel<LN_SMALL_THREADS, false, false><<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(),
          weight.data_ptr<dtype>(),
          bias.data_ptr<dtype>(),
          y.data_ptr<dtype>(),
          rows,
          static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
  }
  layer_norm_f16_kernel<<<static_cast<int>(rows), LN_THREADS, 0, stream>>>(
      C,
      x.data_ptr<dtype>(),
      weight.data_ptr<dtype>(),
      bias.data_ptr<dtype>(),
      y.data_ptr<dtype>(),
      rows,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

template <int Threads>
void launch_layer_norm_f16_cfg(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& y,
    int64_t rows,
    float eps,
    bool vectorized,
    cudaStream_t stream) {
  if (vectorized) {
    layer_norm_f16_small_kernel<Threads, true, true><<<static_cast<int>(rows), Threads, 0, stream>>>(
        x.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
        y.data_ptr<dtype>(), rows, eps);
  } else {
    layer_norm_f16_small_kernel<Threads, false, false><<<static_cast<int>(rows), Threads, 0, stream>>>(
        x.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
        y.data_ptr<dtype>(), rows, eps);
  }
}

at::Tensor layer_norm_f16_cfg_cuda(
    at::Tensor x,
    at::Tensor weight,
    at::Tensor bias,
    double eps,
    int threads,
    bool vectorized) {
  auto y = at::empty_like(x);
  const int64_t rows = x.numel() / LN_SMALL_C;
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (threads) {
    case 128:
      launch_layer_norm_f16_cfg<128>(x, weight, bias, y, rows, static_cast<float>(eps), vectorized, stream);
      break;
    case 256:
      launch_layer_norm_f16_cfg<256>(x, weight, bias, y, rows, static_cast<float>(eps), vectorized, stream);
      break;
    case 512:
      launch_layer_norm_f16_cfg<512>(x, weight, bias, y, rows, static_cast<float>(eps), vectorized, stream);
      break;
    default:
      launch_layer_norm_f16_cfg<1024>(x, weight, bias, y, rows, static_cast<float>(eps), vectorized, stream);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor layer_norm_f16_small_cuda(at::Tensor x, at::Tensor weight, at::Tensor bias, double eps) {
  auto y = at::empty_like(x);
  const int64_t rows = x.numel() / LN_SMALL_C;
  auto stream = at::cuda::getCurrentCUDAStream();
  layer_norm_f16_small_kernel<LN_SMALL_THREADS, false, false><<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
      x.data_ptr<dtype>(),
      weight.data_ptr<dtype>(),
      bias.data_ptr<dtype>(),
      y.data_ptr<dtype>(),
      rows,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor layer_norm_f16_small512_cuda(at::Tensor x, at::Tensor weight, at::Tensor bias, double eps) {
  auto y = at::empty_like(x);
  const int64_t rows = x.numel() / LN_SMALL_C;
  auto stream = at::cuda::getCurrentCUDAStream();
  layer_norm_f16_small_kernel<LN_SMALL512_THREADS, false, false><<<static_cast<int>(rows), LN_SMALL512_THREADS, 0, stream>>>(
      x.data_ptr<dtype>(),
      weight.data_ptr<dtype>(),
      bias.data_ptr<dtype>(),
      y.data_ptr<dtype>(),
      rows,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

std::vector<at::Tensor> add_layer_norm_f16_cuda(at::Tensor x, at::Tensor residual, at::Tensor weight, at::Tensor bias, double eps) {
  auto x_out = at::empty_like(x);
  auto y = at::empty_like(x);
  const int64_t c64 = x.size(-1);
  TORCH_CHECK(c64 <= INT_MAX, "C too large");
  const int C = static_cast<int>(c64);
  const int64_t rows = x.numel() / C;
  auto stream = at::cuda::getCurrentCUDAStream();
  // Canonical Albatross add_ln owner policy: C=4096 and rows<1024 use the
  // 256-thread vectorized owner.  This is the automatic caller policy, not a
  // public forced/config entry point.
  if (C == LN_SMALL_C && rows < 1024) {
    add_layer_norm_f16_small_kernel<LN_SMALL512_THREADS / 2, true, true>
        <<<static_cast<int>(rows), LN_SMALL512_THREADS / 2, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
            x_out.data_ptr<dtype>(), y.data_ptr<dtype>(), rows,
            static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {x_out, y};
  }
  if (C == LN_SMALL_C) {
    if (rows >= 1024) {
      add_layer_norm_f16_small_kernel<LN_SMALL512_THREADS, true, true><<<static_cast<int>(rows), LN_SMALL512_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
          x_out.data_ptr<dtype>(), y.data_ptr<dtype>(), rows, static_cast<float>(eps));
    } else if (rows >= 512) {
      add_layer_norm_f16_small_kernel<LN_SMALL512_THREADS, false, false><<<static_cast<int>(rows), LN_SMALL512_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
          x_out.data_ptr<dtype>(), y.data_ptr<dtype>(), rows, static_cast<float>(eps));
    } else {
      add_layer_norm_f16_small_kernel<LN_SMALL_THREADS, false, false><<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
          x_out.data_ptr<dtype>(), y.data_ptr<dtype>(), rows, static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {x_out, y};
  }
  add_layer_norm_f16_kernel<<<static_cast<int>(rows), LN_THREADS, 0, stream>>>(
      C,
      x.data_ptr<dtype>(),
      residual.data_ptr<dtype>(),
      weight.data_ptr<dtype>(),
      bias.data_ptr<dtype>(),
      x_out.data_ptr<dtype>(),
      y.data_ptr<dtype>(),
      rows,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, y};
}

template <int Threads>
void launch_add_layer_norm_f16_cfg(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& x_out,
    at::Tensor& y,
    int64_t rows,
    float eps,
    bool vectorized,
    cudaStream_t stream) {
  if (vectorized) {
    add_layer_norm_f16_small_kernel<Threads, true, true><<<static_cast<int>(rows), Threads, 0, stream>>>(
        x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
        x_out.data_ptr<dtype>(), y.data_ptr<dtype>(), rows, eps);
  } else {
    add_layer_norm_f16_small_kernel<Threads, false, false><<<static_cast<int>(rows), Threads, 0, stream>>>(
        x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
        x_out.data_ptr<dtype>(), y.data_ptr<dtype>(), rows, eps);
  }
}

std::vector<at::Tensor> add_layer_norm_f16_cfg_cuda(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor weight,
    at::Tensor bias,
    double eps,
    int threads,
    bool vectorized) {
  auto x_out = at::empty_like(x);
  auto y = at::empty_like(x);
  const int64_t rows = x.numel() / LN_SMALL_C;
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (threads) {
    case 128:
      launch_add_layer_norm_f16_cfg<128>(
          x, residual, weight, bias, x_out, y, rows, static_cast<float>(eps), vectorized, stream);
      break;
    case 256:
      launch_add_layer_norm_f16_cfg<256>(
          x, residual, weight, bias, x_out, y, rows, static_cast<float>(eps), vectorized, stream);
      break;
    case 512:
      launch_add_layer_norm_f16_cfg<512>(
          x, residual, weight, bias, x_out, y, rows, static_cast<float>(eps), vectorized, stream);
      break;
    default:
      launch_add_layer_norm_f16_cfg<1024>(
          x, residual, weight, bias, x_out, y, rows, static_cast<float>(eps), vectorized, stream);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, y};
}

void add_layer_norm_f16_cfg_out_cuda(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor x_out,
    at::Tensor y,
    double eps,
    int threads,
    bool vectorized) {
  const int64_t rows = x.numel() / LN_SMALL_C;
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (threads) {
    case 128:
      launch_add_layer_norm_f16_cfg<128>(
          x, residual, weight, bias, x_out, y, rows, static_cast<float>(eps), vectorized, stream);
      break;
    case 256:
      launch_add_layer_norm_f16_cfg<256>(
          x, residual, weight, bias, x_out, y, rows, static_cast<float>(eps), vectorized, stream);
      break;
    case 512:
      launch_add_layer_norm_f16_cfg<512>(
          x, residual, weight, bias, x_out, y, rows, static_cast<float>(eps), vectorized, stream);
      break;
    default:
      launch_add_layer_norm_f16_cfg<1024>(
          x, residual, weight, bias, x_out, y, rows, static_cast<float>(eps), vectorized, stream);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_add_layer_norm_f16_stats_cfg(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& x_out,
    at::Tensor& y,
    int64_t rows,
    float eps,
    int mode,
    cudaStream_t stream) {
  if (mode == 0) {
    add_layer_norm_f16_welford_kernel<false><<<static_cast<int>(rows), 256, 0, stream>>>(
        x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
        x_out.data_ptr<dtype>(), y.data_ptr<dtype>(), rows, eps);
  } else if (mode == 1) {
    add_layer_norm_f16_welford_kernel<true><<<static_cast<int>(rows), 256, 0, stream>>>(
        x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
        x_out.data_ptr<dtype>(), y.data_ptr<dtype>(), rows, eps);
  } else {
    add_layer_norm_f16_centered_cache_kernel<<<static_cast<int>(rows), 256, 0, stream>>>(
        x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
        x_out.data_ptr<dtype>(), y.data_ptr<dtype>(), rows, eps);
  }
}

std::vector<at::Tensor> add_layer_norm_f16_stats_cfg_cuda(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor weight,
    at::Tensor bias,
    double eps,
    int mode) {
  auto x_out = at::empty_like(x);
  auto y = at::empty_like(x);
  const int64_t rows = x.numel() / LN_SMALL_C;
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_add_layer_norm_f16_stats_cfg(
      x, residual, weight, bias, x_out, y, rows, static_cast<float>(eps), mode, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, y};
}

void add_layer_norm_f16_stats_cfg_out_cuda(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor x_out,
    at::Tensor y,
    double eps,
    int mode) {
  const int64_t rows = x.numel() / LN_SMALL_C;
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_add_layer_norm_f16_stats_cfg(
      x, residual, weight, bias, x_out, y, rows, static_cast<float>(eps), mode, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor add_last_layer_norm_f16_cuda_impl(
    at::Tensor x,
    at::Tensor residual,
    const int64_t* last_indices,
    at::Tensor weight,
    at::Tensor bias,
    double eps) {
  const int64_t B = x.size(0);
  const int64_t T = x.size(1);
  const int64_t C = x.size(2);
  TORCH_CHECK((C % 2) == 0, "add_last_layer_norm_f16 requires even C");
  auto y = at::empty({B, C}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool indexed = last_indices != nullptr;
  if (C != LN_SMALL_C) {
    if (indexed) {
      add_last_layer_norm_f16_generic_kernel<LN_THREADS, true><<<static_cast<int>(B), LN_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
          last_indices, y.data_ptr<dtype>(), B, T, static_cast<int>(C), static_cast<float>(eps));
    } else {
      add_last_layer_norm_f16_generic_kernel<LN_THREADS, false><<<static_cast<int>(B), LN_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
          nullptr, y.data_ptr<dtype>(), B, T, static_cast<int>(C), static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
  }
  if (B >= 1024) {
    if (indexed) {
      add_last_layer_norm_f16_small_kernel<LN_SMALL512_THREADS, true, true, true><<<static_cast<int>(B), LN_SMALL512_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
          last_indices, y.data_ptr<dtype>(), B, T, static_cast<float>(eps));
    } else {
      add_last_layer_norm_f16_small_kernel<LN_SMALL512_THREADS, true, true, false><<<static_cast<int>(B), LN_SMALL512_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
          nullptr, y.data_ptr<dtype>(), B, T, static_cast<float>(eps));
    }
  } else if (B >= 512) {
    if (indexed) {
      add_last_layer_norm_f16_small_kernel<LN_SMALL512_THREADS, false, false, true><<<static_cast<int>(B), LN_SMALL512_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
          last_indices, y.data_ptr<dtype>(), B, T, static_cast<float>(eps));
    } else {
      add_last_layer_norm_f16_small_kernel<LN_SMALL512_THREADS, false, false, false><<<static_cast<int>(B), LN_SMALL512_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
          nullptr, y.data_ptr<dtype>(), B, T, static_cast<float>(eps));
    }
  } else {
    if (indexed) {
      add_last_layer_norm_f16_small_kernel<LN_SMALL_THREADS, false, false, true><<<static_cast<int>(B), LN_SMALL_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
          last_indices, y.data_ptr<dtype>(), B, T, static_cast<float>(eps));
    } else {
      add_last_layer_norm_f16_small_kernel<LN_SMALL_THREADS, false, false, false><<<static_cast<int>(B), LN_SMALL_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
          nullptr, y.data_ptr<dtype>(), B, T, static_cast<float>(eps));
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor add_last_layer_norm_f16_cuda(at::Tensor x, at::Tensor residual, at::Tensor weight, at::Tensor bias, double eps) {
  return add_last_layer_norm_f16_cuda_impl(x, residual, nullptr, weight, bias, eps);
}

at::Tensor add_last_layer_norm_indexed_f16_cuda(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor last_indices,
    at::Tensor weight,
    at::Tensor bias,
    double eps) {
  return add_last_layer_norm_f16_cuda_impl(
      x, residual, last_indices.data_ptr<int64_t>(), weight, bias, eps);
}

at::Tensor add_last_layer_norm_indexed_packed_f16_cuda(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor last_indices,
    at::Tensor weight,
    at::Tensor bias,
    double eps) {
  TORCH_CHECK(x.dim() == 2 && residual.sizes() == x.sizes(),
              "packed indexed last layer norm expects x/residual [total_tokens,C]");
  TORCH_CHECK(last_indices.dim() == 1 && last_indices.numel() > 0,
              "packed indexed last layer norm expects last_indices [batch]");
  TORCH_CHECK((x.size(1) % 2) == 0,
              "packed indexed last layer norm requires even C");
  const int64_t B = last_indices.size(0);
  const int64_t total_rows = x.size(0);
  const int64_t C = x.size(1);
  auto y = at::empty({B, C}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (C != LN_SMALL_C) {
    add_last_layer_norm_f16_generic_kernel<LN_THREADS, true, true>
        <<<static_cast<int>(B), LN_THREADS, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
            last_indices.data_ptr<int64_t>(), y.data_ptr<dtype>(), B,
            total_rows, static_cast<int>(C), static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
  }
  if (B >= 1024) {
    add_last_layer_norm_f16_small_kernel<LN_SMALL512_THREADS, true, true, true, true>
        <<<static_cast<int>(B), LN_SMALL512_THREADS, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
            last_indices.data_ptr<int64_t>(), y.data_ptr<dtype>(), B,
            total_rows, static_cast<float>(eps));
  } else if (B >= 512) {
    add_last_layer_norm_f16_small_kernel<LN_SMALL512_THREADS, false, false, true, true>
        <<<static_cast<int>(B), LN_SMALL512_THREADS, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
            last_indices.data_ptr<int64_t>(), y.data_ptr<dtype>(), B,
            total_rows, static_cast<float>(eps));
  } else {
    add_last_layer_norm_f16_small_kernel<LN_SMALL_THREADS, false, false, true, true>
        <<<static_cast<int>(B), LN_SMALL_THREADS, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
            last_indices.data_ptr<int64_t>(), y.data_ptr<dtype>(), B,
            total_rows, static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

std::vector<at::Tensor> cmix_add_layer_norm_mix_forward_varlen_cuda(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor shift_state,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor x_k,
    at::Tensor state_indices,
    at::Tensor metadata_status,
    double eps) {
  TORCH_CHECK(
      x.dim() == 2 && x.size(1) == LN_SMALL_C,
      "canonical Albatross fused CMix layer norm requires packed [B,4096]");
  const int64_t rows = x.size(0);
  auto x_out = at::empty_like(x);
  auto mixed = at::empty_like(x);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (rows >= 192 && rows <= 1024) {
    // Exact tuned_cmix_ln_stats_mode(): cache-rounded Welford is selected
    // only for the upstream B interval [192, 1024].
    add_layer_norm_cmix_mix_f16_packed_welford_kernel<true>
        <<<static_cast<int>(rows), 256, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            shift_state.data_ptr<dtype>(), weight.data_ptr<dtype>(),
            bias.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
            x_out.data_ptr<dtype>(), mixed.data_ptr<dtype>(),
            state_indices.data_ptr<int>(), metadata_status.data_ptr<int>(),
            rows, static_cast<float>(eps));
  } else {
    add_layer_norm_cmix_mix_f16_packed_kernel<256>
        <<<static_cast<int>(rows), 256, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            shift_state.data_ptr<dtype>(), weight.data_ptr<dtype>(),
            bias.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
            x_out.data_ptr<dtype>(), mixed.data_ptr<dtype>(),
            state_indices.data_ptr<int>(), metadata_status.data_ptr<int>(),
            rows, static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, mixed};
}

std::vector<at::Tensor> tmix_mix6_add_layer_norm_forward_varlen_cuda(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor shift_state,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor x_r,
    at::Tensor x_w,
    at::Tensor x_k,
    at::Tensor x_v,
    at::Tensor x_a,
    at::Tensor x_g,
    at::Tensor state_indices,
    at::Tensor metadata_status,
    double eps) {
  TORCH_CHECK(
      x.dim() == 2 && x.size(1) == LN_SMALL_C,
      "canonical Albatross fused TMix layer norm requires packed [B,4096]");
  const int64_t rows = x.size(0);
  std::vector<at::Tensor> outputs;
  outputs.reserve(7);
  for (int index = 0; index < 7; ++index) {
    outputs.push_back(at::empty_like(x));
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  add_layer_norm_tmix_mix6_f16_packed_kernel<256>
      <<<static_cast<int>(rows), 256, 0, stream>>>(
          x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
          shift_state.data_ptr<dtype>(), weight.data_ptr<dtype>(),
          bias.data_ptr<dtype>(), x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(),
          x_k.data_ptr<dtype>(), x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(),
          x_g.data_ptr<dtype>(), outputs[0].data_ptr<dtype>(),
          outputs[1].data_ptr<dtype>(), outputs[2].data_ptr<dtype>(),
          outputs[3].data_ptr<dtype>(), outputs[4].data_ptr<dtype>(),
          outputs[5].data_ptr<dtype>(), outputs[6].data_ptr<dtype>(),
          state_indices.data_ptr<int>(), metadata_status.data_ptr<int>(),
          rows, static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outputs;
}


at::Tensor tmix_layer_norm_forward_varlen_cuda(
    at::Tensor x, at::Tensor weight, at::Tensor bias, double eps) {
  return layer_norm_f16_cuda(x, weight, bias, eps);
}

std::vector<at::Tensor> tmix_add_layer_norm_forward_varlen_cuda(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor weight,
    at::Tensor bias,
    double eps,
    int64_t batch_size) {
  const int64_t C = x.size(-1);
  const int64_t rows = x.numel() / C;
  if (C == LN_SMALL_C && batch_size >= 2 && batch_size <= rows && rows <= 1024) {
    // Mechanical copy of tuned_ln_stats_mode(): rows < 192 selects direct
    // Welford and rows >= 192 selects cache-rounded Welford.  The caller must
    // provide the real packed batch size; guessing it from total_tokens would
    // change the Albatross policy for ragged requests.
    const int mode = rows < 192 ? 0 : 1;
    return add_layer_norm_f16_stats_cfg_cuda(
        x, residual, weight, bias, eps, mode);
  }
  return add_layer_norm_f16_cuda(x, residual, weight, bias, eps);
}

at::Tensor tmix_add_last_layer_norm_forward_varlen_cuda(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor weight,
    at::Tensor bias,
    double eps) {
  // This existing packed-row entry point historically returned one normalized
  // row per packed token. Keep that contract while using the exact upstream
  // fused add+LN body; request-level indexed selection belongs to head/linear.
  auto outputs = add_layer_norm_f16_cuda(x, residual, weight, bias, eps);
  return outputs[1];
}

at::Tensor tmix_add_forward_varlen_cuda(
    at::Tensor x, at::Tensor residual) {
  return add_f16_cuda(x, residual);
}
