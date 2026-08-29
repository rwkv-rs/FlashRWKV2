// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Upstream repository: https://github.com/BlinkDL/Albatross
// Albatross source revision: ee3308f6922e59f2166c7fac3c5a192340a2b48e
// Original path: faster3a_2607/cuda/rwkv7_v3a_ops.cu
// Mechanical migration boundary: the active upstream Res, LN, and statistics
// bodies are retained. Simple row-wise operations need no request metadata.
// No ATen fallback is present.
#include "validation.h"

#include <cuda_fp16.h>

#include <climits>
#include <vector>

using dtype = torch::headeronly::Half;


namespace {

constexpr int LN_THREADS = 256;
constexpr int LN_SMALL_THREADS = 1024;
constexpr int LN_SMALL512_THREADS = 512;
constexpr int LN_SMALL_C = 4096;

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, offset);
  }
  return x;
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
  const float result = partial[0];
  __syncthreads();
  return result;
}
__global__ void res_f16_kernel(
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

__global__ void ln_f16_kernel(
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

__global__ void res_ln_f16_kernel(
    int C,
    const dtype* __restrict__ x,
    const dtype* __restrict__ res,
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
                    __half2float(*reinterpret_cast<const __half*>(res + base + c));
    sum += v;
  }
  sum = block_sum_t<LN_THREADS>(sum);
  const float inv_c = 1.0f / static_cast<float>(C);
  const float mean = sum * inv_c;
  float sum_var = 0.0f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                    __half2float(*reinterpret_cast<const __half*>(res + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<LN_THREADS>(sum_var);
  const float rstd = rsqrtf(sum_var * inv_c + eps);
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                    __half2float(*reinterpret_cast<const __half*>(res + base + c));
    const float w = __half2float(*reinterpret_cast<const __half*>(weight + c));
    const float b = __half2float(*reinterpret_cast<const __half*>(bias + c));
    *reinterpret_cast<__half*>(x_out + base + c) = __float2half_rn(v);
    *reinterpret_cast<__half*>(y + base + c) = __float2half_rn((v - mean) * rstd * w + b);
  }
}

template <int Threads, bool VecStats, bool VecOut>
__global__ __launch_bounds__(Threads, 1) void ln_f16_small_kernel(
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
__global__ __launch_bounds__(Threads, 1) void res_ln_f16_small_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ res,
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
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(res + base)[idx]);
      sum += xv.x + rv.x + xv.y + rv.y;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                      __half2float(*reinterpret_cast<const __half*>(res + base + c));
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
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(res + base)[idx]);
      const float dx = xv.x + rv.x - mean;
      const float dy = xv.y + rv.y - mean;
      sum_var += dx * dx + dy * dy;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                      __half2float(*reinterpret_cast<const __half*>(res + base + c));
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
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(res + base)[idx]);
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
                      __half2float(*reinterpret_cast<const __half*>(res + base + c));
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
__device__ __forceinline__ float2 load_res_pair(
    const dtype* __restrict__ x,
    const dtype* __restrict__ res,
    dtype* __restrict__ x_out,
    int64_t pair_index) {
  const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[pair_index]);
  const float2 rv = __half22float2(reinterpret_cast<const __half2*>(res)[pair_index]);
  float2 sum = make_float2(xv.x + rv.x, xv.y + rv.y);
  if constexpr (CacheRounded) {
    const __half2 rounded = __floats2half2_rn(sum.x, sum.y);
    reinterpret_cast<__half2*>(x_out)[pair_index] = rounded;
    sum = __half22float2(rounded);
  }
  return sum;
}

template <bool CacheRounded>
__global__ __launch_bounds__(256, 1) void res_ln_f16_welford_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ res,
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

  float2 pair = load_res_pair<CacheRounded>(x, res, x_out, base2 + threadIdx.x);
  float delta = pair.y - pair.x;
  float mean = (pair.x + pair.y) * 0.5f;
  float m2 = delta * delta * 0.5f;
#pragma unroll
  for (int k = 1; k < PairsPerThread; ++k) {
    pair = load_res_pair<CacheRounded>(
        x, res, x_out, base2 + threadIdx.x + static_cast<int64_t>(k) * Threads);
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
      sum = load_res_pair<false>(x, res, x_out, pair_index);
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

} // namespace

torch::stable::Tensor res_forward_varlen_cuda(torch::stable::Tensor x, torch::stable::Tensor res) {
  STD_TORCH_CHECK((x.numel() % 2) == 0, "res_f16 requires even numel");
  auto out = torch::stable::empty_like(x);
  constexpr int threads = 256;
  const int64_t n_pairs = x.numel() / 2;
  auto stream = flashrwkv2::validation::current_cuda_stream();
  res_f16_kernel<<<static_cast<int>((n_pairs + threads - 1) / threads), threads, 0, stream>>>(
      x.mutable_data_ptr<dtype>(), res.mutable_data_ptr<dtype>(), out.mutable_data_ptr<dtype>(), n_pairs);
  FLASHRWKV_CUDA_CHECK(cudaGetLastError());
  return out;
}

torch::stable::Tensor ln_forward_varlen_cuda(torch::stable::Tensor x, torch::stable::Tensor weight, torch::stable::Tensor bias, double eps) {
  auto y = torch::stable::empty_like(x);
  const int64_t c64 = x.size(-1);
  STD_TORCH_CHECK(c64 <= INT_MAX, "C too large");
  const int C = static_cast<int>(c64);
  const int64_t rows = x.numel() / C;
  auto stream = flashrwkv2::validation::current_cuda_stream();
  if (C == LN_SMALL_C) {
    if (rows >= 1024) {
      ln_f16_small_kernel<LN_SMALL512_THREADS, true, true><<<static_cast<int>(rows), LN_SMALL512_THREADS, 0, stream>>>(
          x.mutable_data_ptr<dtype>(),
          weight.mutable_data_ptr<dtype>(),
          bias.mutable_data_ptr<dtype>(),
          y.mutable_data_ptr<dtype>(),
          rows,
          static_cast<float>(eps));
    } else if (rows >= 512) {
      ln_f16_small_kernel<LN_SMALL512_THREADS, false, false><<<static_cast<int>(rows), LN_SMALL512_THREADS, 0, stream>>>(
          x.mutable_data_ptr<dtype>(),
          weight.mutable_data_ptr<dtype>(),
          bias.mutable_data_ptr<dtype>(),
          y.mutable_data_ptr<dtype>(),
          rows,
          static_cast<float>(eps));
    } else {
      ln_f16_small_kernel<LN_SMALL_THREADS, false, false><<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
          x.mutable_data_ptr<dtype>(),
          weight.mutable_data_ptr<dtype>(),
          bias.mutable_data_ptr<dtype>(),
          y.mutable_data_ptr<dtype>(),
          rows,
          static_cast<float>(eps));
    }
    FLASHRWKV_CUDA_CHECK(cudaGetLastError());
    return y;
  }
  ln_f16_kernel<<<static_cast<int>(rows), LN_THREADS, 0, stream>>>(
      C,
      x.mutable_data_ptr<dtype>(),
      weight.mutable_data_ptr<dtype>(),
      bias.mutable_data_ptr<dtype>(),
      y.mutable_data_ptr<dtype>(),
      rows,
      static_cast<float>(eps));
  FLASHRWKV_CUDA_CHECK(cudaGetLastError());
  return y;
}

std::vector<torch::stable::Tensor> res_ln_f16_cuda(torch::stable::Tensor x, torch::stable::Tensor res, torch::stable::Tensor weight, torch::stable::Tensor bias, double eps) {
  auto x_out = torch::stable::empty_like(x);
  auto y = torch::stable::empty_like(x);
  const int64_t c64 = x.size(-1);
  STD_TORCH_CHECK(c64 <= INT_MAX, "C too large");
  const int C = static_cast<int>(c64);
  const int64_t rows = x.numel() / C;
  auto stream = flashrwkv2::validation::current_cuda_stream();
  // Canonical Albatross add_ln owner policy: C=4096 and rows<1024 use the
  // 256-thread vectorized owner.  This is the automatic caller policy, not a
  // public forced/config entry point.
  if (C == LN_SMALL_C && rows < 1024) {
    res_ln_f16_small_kernel<LN_SMALL512_THREADS / 2, true, true>
        <<<static_cast<int>(rows), LN_SMALL512_THREADS / 2, 0, stream>>>(
            x.mutable_data_ptr<dtype>(), res.mutable_data_ptr<dtype>(),
            weight.mutable_data_ptr<dtype>(), bias.mutable_data_ptr<dtype>(),
            x_out.mutable_data_ptr<dtype>(), y.mutable_data_ptr<dtype>(), rows,
            static_cast<float>(eps));
    FLASHRWKV_CUDA_CHECK(cudaGetLastError());
    return {x_out, y};
  }
  if (C == LN_SMALL_C) {
    res_ln_f16_small_kernel<LN_SMALL512_THREADS, true, true>
        <<<static_cast<int>(rows), LN_SMALL512_THREADS, 0, stream>>>(
            x.mutable_data_ptr<dtype>(), res.mutable_data_ptr<dtype>(),
            weight.mutable_data_ptr<dtype>(), bias.mutable_data_ptr<dtype>(),
            x_out.mutable_data_ptr<dtype>(), y.mutable_data_ptr<dtype>(), rows,
            static_cast<float>(eps));
    FLASHRWKV_CUDA_CHECK(cudaGetLastError());
    return {x_out, y};
  }
  res_ln_f16_kernel<<<static_cast<int>(rows), LN_THREADS, 0, stream>>>(
      C,
      x.mutable_data_ptr<dtype>(),
      res.mutable_data_ptr<dtype>(),
      weight.mutable_data_ptr<dtype>(),
      bias.mutable_data_ptr<dtype>(),
      x_out.mutable_data_ptr<dtype>(),
      y.mutable_data_ptr<dtype>(),
      rows,
      static_cast<float>(eps));
  FLASHRWKV_CUDA_CHECK(cudaGetLastError());
  return {x_out, y};
}

std::vector<torch::stable::Tensor> post_norm_forward_varlen_cuda(
    torch::stable::Tensor x,
    torch::stable::Tensor res,
    torch::stable::Tensor weight,
    torch::stable::Tensor bias,
    double eps,
    int64_t batch_size) {
  const int64_t C = x.size(-1);
  const int64_t rows = x.numel() / C;
  if (C == LN_SMALL_C && batch_size >= 2 && batch_size <= rows && rows <= 1024) {
    // Mechanical copy of tuned_ln_stats_mode(): rows < 192 selects direct
    // Welford and rows >= 192 selects cache-rounded Welford.  The caller must
    // provide the real packed batch size; guessing it from total_tokens would
    // change the Albatross policy for ragged requests.
    auto x_out = torch::stable::empty_like(x);
    auto y = torch::stable::empty_like(x);
    auto stream = flashrwkv2::validation::current_cuda_stream();
    if (rows < 192) {
      res_ln_f16_welford_kernel<false>
          <<<static_cast<int>(rows), 256, 0, stream>>>(
              x.mutable_data_ptr<dtype>(), res.mutable_data_ptr<dtype>(),
              weight.mutable_data_ptr<dtype>(), bias.mutable_data_ptr<dtype>(),
              x_out.mutable_data_ptr<dtype>(), y.mutable_data_ptr<dtype>(), rows,
              static_cast<float>(eps));
    } else {
      res_ln_f16_welford_kernel<true>
          <<<static_cast<int>(rows), 256, 0, stream>>>(
              x.mutable_data_ptr<dtype>(), res.mutable_data_ptr<dtype>(),
              weight.mutable_data_ptr<dtype>(), bias.mutable_data_ptr<dtype>(),
              x_out.mutable_data_ptr<dtype>(), y.mutable_data_ptr<dtype>(), rows,
              static_cast<float>(eps));
    }
    FLASHRWKV_CUDA_CHECK(cudaGetLastError());
    return {x_out, y};
  }
  return res_ln_f16_cuda(x, res, weight, bias, eps);
}

torch::stable::Tensor post_norm_output_forward_varlen_cuda(
    torch::stable::Tensor x,
    torch::stable::Tensor res,
    torch::stable::Tensor weight,
    torch::stable::Tensor bias,
    double eps) {
  // The final output contract returns one normalized row per packed token.
  auto outputs = res_ln_f16_cuda(x, res, weight, bias, eps);
  return outputs[1];
}
