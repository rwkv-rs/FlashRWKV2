// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Albatross project
//
// Source: BlinkDL/Albatross/faster3a_2607/cuda/rwkv7_fast_ops_fp16.cu,
// revision ee3308f6922e59f2166c7fac3c5a192340a2b48e.
//
// The half2 normalization/gating bodies below are the upstream
// tmix_kk_a_gate and tmix_kk_a_gate_2d implementations.  This file changes
// only their row addressing from [B,T,C] to packed [total_tokens,C].

#include <assert.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

using dtype = at::Half;

namespace {

constexpr int HEAD_SIZE = 64;
constexpr int WARPS_PER_BLOCK = 4;
constexpr unsigned int kMaxGridDimYZ = 65535;
constexpr float KK_NORMALIZE_EPS = 1.0e-12f;

inline int64_t ceil_div(int64_t n, int64_t d) {
  return (n + d - 1) / d;
}

__device__ inline __half2 load_h2(const dtype* ptr) {
  return *reinterpret_cast<const __half2*>(ptr);
}

__device__ inline void store_h2(dtype* ptr, float x0, float x1) {
  *reinterpret_cast<__half2*>(ptr) = __floats2half2_rn(x0, x1);
}

__device__ inline float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

__device__ inline float sigmoid_fast(float x);

template <int HeadSize>
__global__ void tmix_kk_a_gate_generic_kernel(
    int H,
    const dtype* __restrict__ k,
    const dtype* __restrict__ k_k,
    const dtype* __restrict__ a0,
    const dtype* __restrict__ a12,
    const dtype* __restrict__ k_a,
    dtype* __restrict__ new_k,
    dtype* __restrict__ neg_kk,
    dtype* __restrict__ kka,
    int64_t bth_size) {
  constexpr int kPairs = HeadSize / 2;
  constexpr int kWarps = kPairs / 32;
  __shared__ float partial[kWarps];
  const int64_t bth = static_cast<int64_t>(blockIdx.x);
  if (bth >= bth_size) {
    return;
  }
  const int pair = threadIdx.x;
  const int lane = pair & 31;
  const int warp = pair >> 5;
  const int h = static_cast<int>(bth % H);
  const int64_t idx = bth * HeadSize + static_cast<int64_t>(pair) * 2;
  const int64_t c = static_cast<int64_t>(h) * HeadSize + pair * 2;

  const float2 kv = __half22float2(load_h2(k + idx));
  const float2 kk_scale = __half22float2(load_h2(k_k + c));
  const float u0 = kv.x * kk_scale.x;
  const float u1 = kv.y * kk_scale.y;
  float sum_sq = warp_sum(u0 * u0 + u1 * u1);
  if (lane == 0) {
    partial[warp] = sum_sq;
  }
  __syncthreads();
  if (warp == 0) {
    float total = lane < kWarps ? partial[lane] : 0.0f;
    total = warp_sum(total);
    if (lane == 0) {
      partial[0] = total;
    }
  }
  __syncthreads();
  const float inv_d = 1.0f / fmaxf(sqrtf(partial[0]), KK_NORMALIZE_EPS);
  const float kk0 = u0 * inv_d;
  const float kk1 = u1 * inv_d;
  const float2 a0v = __half22float2(load_h2(a0 + c));
  const float2 a12v = __half22float2(load_h2(a12 + idx));
  const float av0 = sigmoid_fast(a0v.x + a12v.x);
  const float av1 = sigmoid_fast(a0v.y + a12v.y);
  const float2 ka = __half22float2(load_h2(k_a + c));
  store_h2(new_k + idx,
           kv.x * fmaf(av0, ka.x, 1.0f - ka.x),
           kv.y * fmaf(av1, ka.y, 1.0f - ka.y));
  store_h2(neg_kk + idx, -kk0, -kk1);
  store_h2(kka + idx, kk0 * av0, kk1 * av1);
}

__device__ inline float sigmoid_fast(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

// Exact Albatross tmix_kk_a_gate_kernel body.  Packed rows are represented by
// bth = token * H + head, so every address below remains the upstream address
// expression after replacing B*T with total_tokens.
__global__ void tmix_kk_a_gate_kernel(
    int H,
    const dtype* __restrict__ k,
    const dtype* __restrict__ k_k,
    const dtype* __restrict__ a0,
    const dtype* __restrict__ a12,
    const dtype* __restrict__ k_a,
    dtype* __restrict__ new_k,
    dtype* __restrict__ neg_kk,
    dtype* __restrict__ kka,
    int64_t bth_size) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t bth = static_cast<int64_t>(blockIdx.x) * WARPS_PER_BLOCK + warp;
  if (bth >= bth_size) {
    return;
  }

  const int64_t h = bth % H;
  const int64_t base = bth * HEAD_SIZE;
  const int64_t c = h * HEAD_SIZE + static_cast<int64_t>(lane) * 2;
  const int64_t idx = base + static_cast<int64_t>(lane) * 2;

  const float2 kv = __half22float2(load_h2(k + idx));
  const float2 kk_scale = __half22float2(load_h2(k_k + c));
  const float u0 = kv.x * kk_scale.x;
  const float u1 = kv.y * kk_scale.y;

  float sum_sq = u0 * u0 + u1 * u1;
  sum_sq = warp_sum(sum_sq);
  const float total = __shfl_sync(0xffffffffu, sum_sq, 0);
  const float inv_d = 1.0f / fmaxf(sqrtf(total), KK_NORMALIZE_EPS);
  const float kk0 = u0 * inv_d;
  const float kk1 = u1 * inv_d;

  const float2 a0v = __half22float2(load_h2(a0 + c));
  const float2 a12v = __half22float2(load_h2(a12 + idx));
  const float av0 = sigmoid_fast(a0v.x + a12v.x);
  const float av1 = sigmoid_fast(a0v.y + a12v.y);
  const float2 ka = __half22float2(load_h2(k_a + c));
  store_h2(new_k + idx,
           kv.x * fmaf(av0, ka.x, 1.0f - ka.x),
           kv.y * fmaf(av1, ka.y, 1.0f - ka.y));
  store_h2(neg_kk + idx, -kk0, -kk1);
  store_h2(kka + idx, kk0 * av0, kk1 * av1);
}

// Exact Albatross tmix_kk_a_gate_2d_kernel body with packed row addressing.
__global__ void tmix_kk_a_gate_2d_kernel(
    int H,
    const dtype* __restrict__ k,
    const dtype* __restrict__ k_k,
    const dtype* __restrict__ a0,
    const dtype* __restrict__ a12,
    const dtype* __restrict__ k_a,
    dtype* __restrict__ new_k,
    dtype* __restrict__ neg_kk,
    dtype* __restrict__ kka) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int h = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp;
  if (h >= H) {
    return;
  }

  const int64_t row = blockIdx.y;
  const int64_t bth = row * H + h;
  const int64_t base = bth * HEAD_SIZE;
  const int64_t c = static_cast<int64_t>(h) * HEAD_SIZE +
      static_cast<int64_t>(lane) * 2;
  const int64_t idx = base + static_cast<int64_t>(lane) * 2;

  const float2 kv = __half22float2(load_h2(k + idx));
  const float2 kk_scale = __half22float2(load_h2(k_k + c));
  const float u0 = kv.x * kk_scale.x;
  const float u1 = kv.y * kk_scale.y;

  float sum_sq = u0 * u0 + u1 * u1;
  sum_sq = warp_sum(sum_sq);
  const float total = __shfl_sync(0xffffffffu, sum_sq, 0);
  const float inv_d = 1.0f / fmaxf(sqrtf(total), KK_NORMALIZE_EPS);
  const float kk0 = u0 * inv_d;
  const float kk1 = u1 * inv_d;

  const float2 a0v = __half22float2(load_h2(a0 + c));
  const float2 a12v = __half22float2(load_h2(a12 + idx));
  const float av0 = sigmoid_fast(a0v.x + a12v.x);
  const float av1 = sigmoid_fast(a0v.y + a12v.y);
  const float2 ka = __half22float2(load_h2(k_a + c));
  store_h2(new_k + idx,
           kv.x * fmaf(av0, ka.x, 1.0f - ka.x),
           kv.y * fmaf(av1, ka.y, 1.0f - ka.y));
  store_h2(neg_kk + idx, -kk0, -kk1);
  store_h2(kka + idx, kk0 * av0, kk1 * av1);
}

}  // namespace

void tmix_kk_a_gate_forward_varlen_cuda(
    int batch_size,
    int max_seqlen,
    int total_tokens,
    int channels,
    int heads,
    int head_size,
    torch::Tensor k,
    torch::Tensor k_k,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a,
    torch::Tensor new_k,
    torch::Tensor neg_kk,
    torch::Tensor kka) {
  assert(channels == heads * head_size);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t bth_size = static_cast<int64_t>(total_tokens) * heads;
  if (head_size == 128) {
    tmix_kk_a_gate_generic_kernel<128><<<bth_size, 64, 0, stream>>>(
        heads, k.data_ptr<dtype>(), k_k.data_ptr<dtype>(), a0.data_ptr<dtype>(),
        a12.data_ptr<dtype>(), k_a.data_ptr<dtype>(), new_k.data_ptr<dtype>(),
        neg_kk.data_ptr<dtype>(), kka.data_ptr<dtype>(), bth_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (head_size == 256) {
    tmix_kk_a_gate_generic_kernel<256><<<bth_size, 128, 0, stream>>>(
        heads, k.data_ptr<dtype>(), k_k.data_ptr<dtype>(), a0.data_ptr<dtype>(),
        a12.data_ptr<dtype>(), k_a.data_ptr<dtype>(), new_k.data_ptr<dtype>(),
        neg_kk.data_ptr<dtype>(), kka.data_ptr<dtype>(), bth_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  assert(head_size == HEAD_SIZE);
  // Canonical Albatross use_kk_head_grid_2d() in its tuned mode: only the
  // C=4096/H=64 production family uses the head-grid launch, and the packed
  // metadata supplies the original B/T dispatch coordinates.
  const bool use_2d = channels == 4096 && heads == 64 && batch_size > 0 &&
      max_seqlen > 0 &&
      static_cast<int64_t>(batch_size) * max_seqlen <= kMaxGridDimYZ &&
      total_tokens <= kMaxGridDimYZ;
  if (use_2d) {
    const dim3 grid(
        static_cast<unsigned int>(ceil_div(heads, WARPS_PER_BLOCK)),
        static_cast<unsigned int>(total_tokens));
    tmix_kk_a_gate_2d_kernel<<<grid, WARPS_PER_BLOCK * 32, 0, stream>>>(
        heads, k.data_ptr<dtype>(), k_k.data_ptr<dtype>(), a0.data_ptr<dtype>(),
        a12.data_ptr<dtype>(), k_a.data_ptr<dtype>(),
        new_k.data_ptr<dtype>(), neg_kk.data_ptr<dtype>(), kka.data_ptr<dtype>());
  } else {
    const int blocks = static_cast<int>(ceil_div(
        bth_size, static_cast<int64_t>(WARPS_PER_BLOCK)));
    tmix_kk_a_gate_kernel<<<blocks, WARPS_PER_BLOCK * 32, 0, stream>>>(
        heads, k.data_ptr<dtype>(), k_k.data_ptr<dtype>(), a0.data_ptr<dtype>(),
        a12.data_ptr<dtype>(), k_a.data_ptr<dtype>(),
        new_k.data_ptr<dtype>(), neg_kk.data_ptr<dtype>(), kka.data_ptr<dtype>(),
        bth_size);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
