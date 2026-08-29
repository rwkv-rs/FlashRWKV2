// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Albatross project
//
// Source: BlinkDL/Albatross/faster3a_2607/cuda/rwkv7_fast_ops_fp16.cu,
// revision ee3308f6922e59f2166c7fac3c5a192340a2b48e.
//
// The three lnx/rkv-residual/gate bodies are mechanically carried over from
// Albatross.  Packed varlen changes only replace bth=B*T indexing by the
// packed row/head index; this operator has no request state of its own.

#include <assert.h>

#include <cuda_fp16.h>
#include "validation.h"

#include <cstdint>

using dtype = torch::headeronly::Half;

namespace {

constexpr int HEAD_SIZE = 64;
constexpr float TMIX_LN_X_EPS = 64.0e-5f;

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

__device__ inline float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

template <int HeadSize>
__global__ void tmix_lnx_rkvres_xg_generic_kernel(
    int H,
    const dtype* __restrict__ x,
    const dtype* __restrict__ r,
    const dtype* __restrict__ k,
    const dtype* __restrict__ v,
    const dtype* __restrict__ r_k,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ g,
    dtype* __restrict__ out,
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
  const int64_t offset = static_cast<int64_t>(pair) * 2;
  const int64_t idx = bth * HeadSize + offset;
  const int64_t c = static_cast<int64_t>(h) * HeadSize + offset;
  const float2 xv = __half22float2(load_h2(x + idx));

  float sum = warp_sum(xv.x + xv.y);
  if (lane == 0) partial[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    float total = lane < kWarps ? partial[lane] : 0.0f;
    total = warp_sum(total);
    if (lane == 0) partial[0] = total;
  }
  __syncthreads();
  const float mean = partial[0] * (1.0f / HeadSize);
  __syncthreads();
  const float d0 = xv.x - mean;
  const float d1 = xv.y - mean;

  float ss = warp_sum(d0 * d0 + d1 * d1);
  if (lane == 0) partial[warp] = ss;
  __syncthreads();
  if (warp == 0) {
    float total = lane < kWarps ? partial[lane] : 0.0f;
    total = warp_sum(total);
    if (lane == 0) partial[0] = total;
  }
  __syncthreads();
  const float rstd = rsqrtf(partial[0] * (1.0f / HeadSize) + TMIX_LN_X_EPS);
  __syncthreads();

  const float2 rv = __half22float2(load_h2(r + idx));
  const float2 kv = __half22float2(load_h2(k + idx));
  const float2 vv = __half22float2(load_h2(v + idx));
  const float2 rkv_weight = __half22float2(load_h2(r_k + c));
  float dot = warp_sum(
      rv.x * kv.x * rkv_weight.x + rv.y * kv.y * rkv_weight.y);
  if (lane == 0) partial[warp] = dot;
  __syncthreads();
  if (warp == 0) {
    float total = lane < kWarps ? partial[lane] : 0.0f;
    total = warp_sum(total);
    if (lane == 0) partial[0] = total;
  }
  __syncthreads();
  const float rkv = partial[0];
  const float2 ln_weight = __half22float2(load_h2(weight + c));
  const float2 ln_bias = __half22float2(load_h2(bias + c));
  const float2 gate = __half22float2(load_h2(g + idx));
  store_h2(
      out + idx,
      (d0 * rstd * ln_weight.x + ln_bias.x + rkv * vv.x) * gate.x,
      (d1 * rstd * ln_weight.y + ln_bias.y + rkv * vv.y) * gate.y);
}

// Exact Albatross basic lnx body; bth_size is total_tokens * heads.
__global__ void tmix_lnx_rkvres_xg_kernel(
    int H,
    const dtype* __restrict__ x,
    const dtype* __restrict__ r,
    const dtype* __restrict__ k,
    const dtype* __restrict__ v,
    const dtype* __restrict__ r_k,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ g,
    dtype* __restrict__ out,
    int64_t bth_size) {
  __shared__ float partial[2];
  const int bth = static_cast<int>(blockIdx.x);
  if (bth >= bth_size) {
    return;
  }
  const int lane = threadIdx.x;
  const int warp = lane >> 5;
  const int warp_lane = lane & 31;
  const int h = bth % H;
  const int64_t base = static_cast<int64_t>(bth) * HEAD_SIZE;
  const int64_t cbase = static_cast<int64_t>(h) * HEAD_SIZE;
  const int64_t idx = base + lane;
  const int64_t c = cbase + lane;

  const float xv = load_h1(x + idx);
  float sum = warp_sum(xv);
  if (warp_lane == 0) {
    partial[warp] = sum;
  }
  __syncthreads();
  const float mean = (partial[0] + partial[1]) * (1.0f / 64.0f);
  __syncthreads();

  const float d = xv - mean;
  float ss = warp_sum(d * d);
  if (warp_lane == 0) {
    partial[warp] = ss;
  }
  __syncthreads();
  const float var = (partial[0] + partial[1]) * (1.0f / 64.0f);
  const float rstd = rsqrtf(var + TMIX_LN_X_EPS);
  __syncthreads();

  const float rv = load_h1(r + idx);
  const float kv = load_h1(k + idx);
  const float vv = load_h1(v + idx);
  float dot = rv * kv * load_h1(r_k + c);
  dot = warp_sum(dot);
  if (warp_lane == 0) {
    partial[warp] = dot;
  }
  __syncthreads();
  const float rkv = partial[0] + partial[1];
  __syncthreads();

  const float y =
      (d * rstd * load_h1(weight + c) + load_h1(bias + c) + rkv * vv) *
      load_h1(g + idx);
  store_h1(out + idx, y);
}

// Exact Albatross warp body with packed row/head indexing.
__global__ __launch_bounds__(32) void tmix_lnx_rkvres_xg_warp_kernel(
    int H,
    const dtype* __restrict__ x,
    const dtype* __restrict__ r,
    const dtype* __restrict__ k,
    const dtype* __restrict__ v,
    const dtype* __restrict__ r_k,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ g,
    dtype* __restrict__ out,
    int64_t bth_size) {
  const int64_t bth = static_cast<int64_t>(blockIdx.x);
  if (bth >= bth_size) {
    return;
  }
  const int lane = threadIdx.x;
  const int h = static_cast<int>(bth % H);
  const int64_t base = bth * HEAD_SIZE;
  const int64_t cbase = static_cast<int64_t>(h) * HEAD_SIZE;
  const int64_t pair = static_cast<int64_t>(lane) * 2;
  const int64_t idx = base + pair;
  const int64_t c = cbase + pair;

  const float2 xv = __half22float2(load_h2(x + idx));
  float sum = warp_sum(xv.x + xv.y);
  const float mean = __shfl_sync(0xffffffffu, sum, 0) * (1.0f / 64.0f);
  const float d0 = xv.x - mean;
  const float d1 = xv.y - mean;
  float ss = warp_sum(d0 * d0 + d1 * d1);
  const float var = __shfl_sync(0xffffffffu, ss, 0) * (1.0f / 64.0f);
  const float rstd = rsqrtf(var + TMIX_LN_X_EPS);

  const float2 rv = __half22float2(load_h2(r + idx));
  const float2 kv = __half22float2(load_h2(k + idx));
  const float2 vv = __half22float2(load_h2(v + idx));
  const float2 rkv_weight = __half22float2(load_h2(r_k + c));
  float dot = warp_sum(rv.x * kv.x * rkv_weight.x + rv.y * kv.y * rkv_weight.y);
  const float rkv = __shfl_sync(0xffffffffu, dot, 0);

  const float2 ln_weight = __half22float2(load_h2(weight + c));
  const float2 ln_bias = __half22float2(load_h2(bias + c));
  const float2 gate = __half22float2(load_h2(g + idx));
  store_h2(
      out + idx,
      (d0 * rstd * ln_weight.x + ln_bias.x + rkv * vv.x) * gate.x,
      (d1 * rstd * ln_weight.y + ln_bias.y + rkv * vv.y) * gate.y);
}

// Upstream status: this exact Albatross body is reachable through the forced
// head-grid lnx2d/grid2d operator at revision
// ee3308f6922e59f2166c7fac3c5a192340a2b48e, but the canonical tuned policy
// does not select it.  Local status: FlashRWKV's public packed-varlen API does
// not expose those forced head-grid modes.  Preserve the upstream body as
// disabled reference code; do not re-enable it without a real selector,
// packed-varlen correctness coverage, an actual grid.y extent guard, and
// benchmark evidence.
#if 0
__global__ __launch_bounds__(32) void tmix_lnx_rkvres_xg_warp_2d_kernel(
    int H,
    const dtype* __restrict__ x,
    const dtype* __restrict__ r,
    const dtype* __restrict__ k,
    const dtype* __restrict__ v,
    const dtype* __restrict__ r_k,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ g,
    dtype* __restrict__ out) {
  const int lane = threadIdx.x;
  const int h = static_cast<int>(blockIdx.x);
  const int64_t row = blockIdx.y;
  const int64_t bth = row * H + h;
  const int64_t base = bth * HEAD_SIZE;
  const int64_t cbase = static_cast<int64_t>(h) * HEAD_SIZE;
  const int64_t pair = static_cast<int64_t>(lane) * 2;
  const int64_t idx = base + pair;
  const int64_t c = cbase + pair;

  const float2 xv = __half22float2(load_h2(x + idx));
  float sum = warp_sum(xv.x + xv.y);
  const float mean = __shfl_sync(0xffffffffu, sum, 0) * (1.0f / 64.0f);
  const float d0 = xv.x - mean;
  const float d1 = xv.y - mean;
  float ss = warp_sum(d0 * d0 + d1 * d1);
  const float var = __shfl_sync(0xffffffffu, ss, 0) * (1.0f / 64.0f);
  const float rstd = rsqrtf(var + TMIX_LN_X_EPS);

  const float2 rv = __half22float2(load_h2(r + idx));
  const float2 kv = __half22float2(load_h2(k + idx));
  const float2 vv = __half22float2(load_h2(v + idx));
  const float2 rkv_weight = __half22float2(load_h2(r_k + c));
  float dot = warp_sum(rv.x * kv.x * rkv_weight.x + rv.y * kv.y * rkv_weight.y);
  const float rkv = __shfl_sync(0xffffffffu, dot, 0);

  const float2 ln_weight = __half22float2(load_h2(weight + c));
  const float2 ln_bias = __half22float2(load_h2(bias + c));
  const float2 gate = __half22float2(load_h2(g + idx));
  store_h2(
      out + idx,
      (d0 * rstd * ln_weight.x + ln_bias.x + rkv * vv.x) * gate.x,
      (d1 * rstd * ln_weight.y + ln_bias.y + rkv * vv.y) * gate.y);
}
#endif

}  // namespace

void tmix_lnx_rkvres_xg_forward_varlen_cuda(
    int batch_size,
    int max_seqlen,
    int total_tokens,
    int channels,
    int heads,
    int head_size,
    torch::stable::Tensor x,
    torch::stable::Tensor r,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor r_k,
    torch::stable::Tensor weight,
    torch::stable::Tensor bias,
    torch::stable::Tensor g,
    torch::stable::Tensor output) {
  assert(channels == heads * head_size);
  const auto stream = flashrwkv2::validation::current_cuda_stream();
  const int64_t rows = static_cast<int64_t>(total_tokens) * heads;
  if (head_size == 128) {
    tmix_lnx_rkvres_xg_generic_kernel<128><<<rows, 64, 0, stream>>>(
        heads, x.mutable_data_ptr<dtype>(), r.mutable_data_ptr<dtype>(), k.mutable_data_ptr<dtype>(),
        v.mutable_data_ptr<dtype>(), r_k.mutable_data_ptr<dtype>(), weight.mutable_data_ptr<dtype>(),
        bias.mutable_data_ptr<dtype>(), g.mutable_data_ptr<dtype>(), output.mutable_data_ptr<dtype>(), rows);
    FLASHRWKV_CUDA_CHECK(cudaGetLastError());
    return;
  }
  if (head_size == 256) {
    tmix_lnx_rkvres_xg_generic_kernel<256><<<rows, 128, 0, stream>>>(
        heads, x.mutable_data_ptr<dtype>(), r.mutable_data_ptr<dtype>(), k.mutable_data_ptr<dtype>(),
        v.mutable_data_ptr<dtype>(), r_k.mutable_data_ptr<dtype>(), weight.mutable_data_ptr<dtype>(),
        bias.mutable_data_ptr<dtype>(), g.mutable_data_ptr<dtype>(), output.mutable_data_ptr<dtype>(), rows);
    FLASHRWKV_CUDA_CHECK(cudaGetLastError());
    return;
  }
  assert(head_size == HEAD_SIZE);
  const int64_t head_tasks = rows;

  // Exact tuned predicates from rwkv7_fast_v3a.py.  In the canonical tuned
  // mode the lnx head-grid policy is disabled.  Packed admission uses the
  // real token/head task supply; B1 remains an exact uniform anchor because a
  // single packed sequence necessarily has total_tokens == max_seqlen.
  constexpr int kB1T4096[] = {64, 96, 128, 160, 192, 240, 248, 264, 512};
  bool use_warp = channels == 4096 && heads == 64 && head_tasks >= 4096;
  if (use_warp && batch_size == 1) {
    use_warp = false;
    for (const int candidate : kB1T4096) {
      if (max_seqlen == candidate) {
        use_warp = true;
        break;
      }
    }
  }
  // Disabled upstream forced-mode launch retained beside its original owner.
  // See the #if 0 kernel body above for provenance and re-enable conditions.
#if 0
  if (use_lnx_head_grid_2d) {
    tmix_lnx_rkvres_xg_warp_2d_kernel<<<
        dim3(static_cast<unsigned int>(heads), static_cast<unsigned int>(total_tokens)),
        dim3(32), 0, stream>>>(
        heads, x.mutable_data_ptr<dtype>(), r.mutable_data_ptr<dtype>(), k.mutable_data_ptr<dtype>(),
        v.mutable_data_ptr<dtype>(), r_k.mutable_data_ptr<dtype>(), weight.mutable_data_ptr<dtype>(),
        bias.mutable_data_ptr<dtype>(), g.mutable_data_ptr<dtype>(), output.mutable_data_ptr<dtype>());
  }
#endif
  if (use_warp) {
    tmix_lnx_rkvres_xg_warp_kernel<<<
        static_cast<unsigned int>(rows), dim3(32), 0, stream>>>(
        heads, x.mutable_data_ptr<dtype>(), r.mutable_data_ptr<dtype>(), k.mutable_data_ptr<dtype>(),
        v.mutable_data_ptr<dtype>(), r_k.mutable_data_ptr<dtype>(), weight.mutable_data_ptr<dtype>(),
        bias.mutable_data_ptr<dtype>(), g.mutable_data_ptr<dtype>(), output.mutable_data_ptr<dtype>(), rows);
  } else {
    tmix_lnx_rkvres_xg_kernel<<<
        static_cast<unsigned int>(rows), dim3(64), 0, stream>>>(
        heads, x.mutable_data_ptr<dtype>(), r.mutable_data_ptr<dtype>(), k.mutable_data_ptr<dtype>(),
        v.mutable_data_ptr<dtype>(), r_k.mutable_data_ptr<dtype>(), weight.mutable_data_ptr<dtype>(),
        bias.mutable_data_ptr<dtype>(), g.mutable_data_ptr<dtype>(), output.mutable_data_ptr<dtype>(), rows);
  }
  FLASHRWKV_CUDA_CHECK(cudaGetLastError());
}
