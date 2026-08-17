// SPDX-License-Identifier: Apache-2.0
// Native-private WAG/WAGV/VRes implementation owned by TMix WKV Prepare.
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
#include "../../../../internal/linear/backend.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <climits>
#include <optional>
#include <vector>

using dtype = at::Half;

void wkv_prepare_vres_forward_varlen_cuda(
    int total_tokens,
    int channels,
    at::Tensor v,
    at::Tensor v_first,
    at::Tensor v0,
    at::Tensor v12,
    at::Tensor output);

namespace {

inline int64_t ceil_div(int64_t n, int64_t d) {
  return (n + d - 1) / d;
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

} // namespace

at::Tensor lowrank_tanh_f16_cuda(at::Tensor x) {
  TORCH_CHECK((x.numel() % 2) == 0,
              "lowrank tanh requires an even number of elements");
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = x.numel() / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_act_tanh_f16_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
      x.data_ptr<dtype>(), out.data_ptr<dtype>(), total_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor lowrank_sigmoid_f16_cuda(at::Tensor x) {
  TORCH_CHECK((x.numel() % 2) == 0,
              "lowrank sigmoid requires an even number of elements");
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = x.numel() / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_act_sigmoid_f16_kernel<<<static_cast<int>(ceil_div(total_pairs, threads)), threads, 0, stream>>>(
      x.data_ptr<dtype>(), out.data_ptr<dtype>(), total_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
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

enum class WkvPrepareRankBackend : uint8_t {
  kRuntimeLt,
  kOriginalLt,
  kOriginalGemmex,
};

struct WkvPrepareRankConfig {
  int rank;
  int rows;
  WkvPrepareRankBackend backend;
  int workspace_mb;
  int algo_index;
};

// Mechanical copy of LOWRANK_IN_GEMM_4096 from the canonical Albatross
// caller.  The table is intentionally kept in the CUDA caller dispatch: it
// selects an existing upstream linear body and never creates a generic GEMM
// or a forced/config public operator.
constexpr WkvPrepareRankConfig kWkvPrepareRankIn4096[] = {
    {128, 8, WkvPrepareRankBackend::kOriginalLt, 32, 2},
    {128, 16, WkvPrepareRankBackend::kOriginalLt, 32, 1},
    {128, 48, WkvPrepareRankBackend::kOriginalLt, 32, 1},
    {128, 64, WkvPrepareRankBackend::kOriginalLt, 128, 2},
    {128, 96, WkvPrepareRankBackend::kOriginalGemmex, 0, 0},
    {128, 128, WkvPrepareRankBackend::kOriginalGemmex, 0, 0},
    {128, 192, WkvPrepareRankBackend::kOriginalLt, 128, 1},
    {128, 256, WkvPrepareRankBackend::kOriginalLt, 128, 6},
    {128, 512, WkvPrepareRankBackend::kOriginalLt, 128, 0},
    {128, 1024, WkvPrepareRankBackend::kOriginalLt, 32, 1},
    {480, 8, WkvPrepareRankBackend::kOriginalLt, 128, 0},
    {480, 16, WkvPrepareRankBackend::kRuntimeLt, 128, 1},
    {480, 24, WkvPrepareRankBackend::kOriginalLt, 128, 5},
    {480, 32, WkvPrepareRankBackend::kOriginalLt, 32, 1},
    {480, 48, WkvPrepareRankBackend::kOriginalLt, 128, 5},
    {480, 96, WkvPrepareRankBackend::kOriginalLt, 128, 4},
    {480, 128, WkvPrepareRankBackend::kOriginalLt, 32, 4},
    {480, 192, WkvPrepareRankBackend::kOriginalLt, 128, 0},
    {480, 256, WkvPrepareRankBackend::kOriginalLt, 32, 1},
    {480, 512, WkvPrepareRankBackend::kOriginalLt, 128, 1},
    {96, 8, WkvPrepareRankBackend::kOriginalLt, 32, 2},
    {96, 16, WkvPrepareRankBackend::kOriginalLt, 128, 1},
    {96, 96, WkvPrepareRankBackend::kOriginalLt, 128, 2},
    {96, 128, WkvPrepareRankBackend::kOriginalLt, 32, 0},
    {96, 192, WkvPrepareRankBackend::kRuntimeLt, 32, 2},
    {96, 256, WkvPrepareRankBackend::kOriginalLt, 32, 1},
    {96, 512, WkvPrepareRankBackend::kOriginalGemmex, 0, 0},
    {96, 1024, WkvPrepareRankBackend::kOriginalLt, 128, 1},
};

// Mechanical copy of LOWRANK_OUT_GEMM_4096 from the canonical Albatross
// caller.  ``rows`` is the packed input row count; the model-width guard is
// applied to the output channel dimension for rank-out.
constexpr WkvPrepareRankConfig kWkvPrepareRankOut4096[] = {
    {128, 8, WkvPrepareRankBackend::kRuntimeLt, 128, 3},
    {128, 16, WkvPrepareRankBackend::kRuntimeLt, 128, 2},
    {128, 24, WkvPrepareRankBackend::kOriginalLt, 128, 0},
    {128, 32, WkvPrepareRankBackend::kOriginalLt, 128, 2},
    {128, 48, WkvPrepareRankBackend::kOriginalLt, 0, 5},
    {128, 64, WkvPrepareRankBackend::kOriginalLt, 0, 2},
    {128, 96, WkvPrepareRankBackend::kOriginalLt, 32, 3},
    {128, 192, WkvPrepareRankBackend::kOriginalLt, 128, 2},
    {128, 256, WkvPrepareRankBackend::kRuntimeLt, 128, 3},
    {128, 512, WkvPrepareRankBackend::kRuntimeLt, 128, 1},
    {128, 1024, WkvPrepareRankBackend::kRuntimeLt, 128, 1},
    {480, 8, WkvPrepareRankBackend::kOriginalLt, 0, 1},
    {480, 16, WkvPrepareRankBackend::kOriginalLt, 0, 1},
    {480, 24, WkvPrepareRankBackend::kOriginalLt, 128, 0},
    {480, 32, WkvPrepareRankBackend::kOriginalGemmex, 0, 0},
    {480, 96, WkvPrepareRankBackend::kRuntimeLt, 128, 1},
    {480, 128, WkvPrepareRankBackend::kRuntimeLt, 32, 0},
    {480, 256, WkvPrepareRankBackend::kRuntimeLt, 32, 2},
    {480, 512, WkvPrepareRankBackend::kRuntimeLt, 128, 1},
    {480, 1024, WkvPrepareRankBackend::kRuntimeLt, 0, 1},
    {96, 8, WkvPrepareRankBackend::kRuntimeLt, 128, 5},
    {96, 16, WkvPrepareRankBackend::kRuntimeLt, 32, 4},
    {96, 24, WkvPrepareRankBackend::kOriginalGemmex, 0, 0},
    {96, 32, WkvPrepareRankBackend::kOriginalLt, 128, 1},
    {96, 48, WkvPrepareRankBackend::kOriginalLt, 128, 3},
    {96, 64, WkvPrepareRankBackend::kOriginalLt, 128, 2},
    {96, 96, WkvPrepareRankBackend::kOriginalLt, 32, 3},
    {96, 128, WkvPrepareRankBackend::kRuntimeLt, 32, 2},
    {96, 256, WkvPrepareRankBackend::kRuntimeLt, 128, 4},
    {96, 512, WkvPrepareRankBackend::kRuntimeLt, 128, 0},
    {96, 1024, WkvPrepareRankBackend::kRuntimeLt, 128, 2},
};

template <size_t Count>
const WkvPrepareRankConfig* find_wkv_prepare_rank_config(
    const WkvPrepareRankConfig (&configs)[Count], int rank, int rows) {
  for (const auto& config : configs) {
    if (config.rank == rank && config.rows == rows) {
      return &config;
    }
  }
  return nullptr;
}

at::Tensor wkv_prepare_rank_dispatch_f16_cuda(
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

  const WkvPrepareRankConfig* config = nullptr;
  if (channels64 == 4096) {
    config = input_projection
        ? find_wkv_prepare_rank_config(kWkvPrepareRankIn4096,
                              static_cast<int>(rank64),
                              static_cast<int>(rows64))
        : find_wkv_prepare_rank_config(kWkvPrepareRankOut4096,
                              static_cast<int>(rank64),
                              static_cast<int>(rows64));
  }
  if (config != nullptr) {
    switch (config->backend) {
      case WkvPrepareRankBackend::kRuntimeLt:
        if (weight.has_value()) {
          return linear_f16_lt_cfg_cuda(
              x, *weight, config->workspace_mb, config->algo_index);
        }
        break;
      case WkvPrepareRankBackend::kOriginalLt:
        if (weight_orig.has_value()) {
          return linear_f16_orig_lt_cfg_cuda(
              x, *weight_orig, config->workspace_mb, config->algo_index);
        }
        break;
      case WkvPrepareRankBackend::kOriginalGemmex:
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



std::vector<at::Tensor> lowrank_wag_rank_in_f16_cuda(
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
      wkv_prepare_rank_dispatch_f16_cuda(x_w, w1, w1_t, true),
      wkv_prepare_rank_dispatch_f16_cuda(x_a, a1, a1_t, true),
      wkv_prepare_rank_dispatch_f16_cuda(x_g, g1, g1_t, true),
  };
}

std::vector<at::Tensor> lowrank_wagv_rank_in_f16_cuda(
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
      wkv_prepare_rank_dispatch_f16_cuda(x_w, w1, w1_t, true),
      wkv_prepare_rank_dispatch_f16_cuda(x_a, a1, a1_t, true),
      wkv_prepare_rank_dispatch_f16_cuda(x_g, g1, g1_t, true),
      wkv_prepare_rank_dispatch_f16_cuda(x_v, v1, v1_t, true),
  };
}

std::vector<at::Tensor> lowrank_wag_rank_out_f16_cuda(
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
  auto w1_activated = lowrank_tanh_f16_cuda(w1);
  auto g1_activated = lowrank_sigmoid_f16_cuda(g1);
  return {
      wkv_prepare_rank_dispatch_f16_cuda(
          w1_activated, w2, w2_t, false),
      wkv_prepare_rank_dispatch_f16_cuda(a1, a2, a2_t, false),
      wkv_prepare_rank_dispatch_f16_cuda(
          g1_activated, g2, g2_t, false),
  };
}

std::vector<at::Tensor> lowrank_wagv_vres_f16_cuda(
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
  auto wag = lowrank_wag_rank_out_f16_cuda(
      w1, a1, g1, w2_t, a2_t, g2_t, w2, a2, g2);
  auto v12 = wkv_prepare_rank_dispatch_f16_cuda(v1, v2, v2_t, false);
  auto value = at::empty_like(v);
  wkv_prepare_vres_forward_varlen_cuda(
      static_cast<int>(rows), static_cast<int>(v.size(1)),
      v, v_first, v0, v12, value);
  return {wag[0], wag[1], wag[2], value};
}
