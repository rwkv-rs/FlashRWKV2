// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Upstream repository: https://github.com/BlinkDL/Albatross
// Albatross source revision: ee3308f6922e59f2166c7fac3c5a192340a2b48e
// Original path: faster3a_2607/cuda/rwkv7_v3a_ops.cu
// Migration boundary: retain the upstream BF16 embedding initial-layer-norm
// math and thread layout. Local adaptations remove monolithic-file helpers and
// redundant forwarding; the packed-row operator takes no sequence metadata.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

#include <climits>

using dtype = at::Half;

namespace {

// Reduce 32 register values inside one warp; lane 0 receives the sum.
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, offset);
  }
  return x;
}

// BF16 stores the high 16 bits of FP32, so restoring those bits is exact.
__device__ __forceinline__ float bf16_bits_to_float_dev(uint16_t bits) {
  union {
    uint32_t u;
    float f;
  } v;
  v.u = static_cast<uint32_t>(bits) << 16;
  return v.f;
}

// First reduce inside each warp, then let warp 0 combine the warp sums.
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

__global__ void emb_ln0_bf16_to_f16_kernel(
    int C,
    const uint16_t* __restrict__ emb,
    const uint16_t* __restrict__ weight,
    const uint16_t* __restrict__ bias,
    dtype* __restrict__ out,
    float eps) {
  // One block owns one packed token row; its threads stride across C channels.
  // Rows are independent, so no sequence-length metadata is needed here.
  // Precision path: BF16 inputs -> FP32 statistics/affine -> FP16 output.
  const int tok = blockIdx.x;
  const int tid = threadIdx.x;
  const uint16_t* er = emb + static_cast<int64_t>(tok) * C;

  // Pass 1: mean over this token's channels.
  float sum = 0.0f;
  for (int c = tid; c < C; c += blockDim.x) {
    sum += bf16_bits_to_float_dev(er[c]);
  }
  const float mean = block_sum_t<256>(sum) / static_cast<float>(C);

  // Pass 2: variance over the same channels.
  float var = 0.0f;
  for (int c = tid; c < C; c += blockDim.x) {
    const float d = bf16_bits_to_float_dev(er[c]) - mean;
    var += d * d;
  }
  const float rstd = rsqrtf(block_sum_t<256>(var) / static_cast<float>(C) + eps);

  // Pass 3: y = (x - mean) * rstd * weight + bias.
  dtype* yr = out + static_cast<int64_t>(tok) * C;
  for (int c = tid; c < C; c += blockDim.x) {
    const float x = bf16_bits_to_float_dev(er[c]);
    const float w = bf16_bits_to_float_dev(weight[c]);
    const float b = bf16_bits_to_float_dev(bias[c]);
    yr[c] = static_cast<dtype>((x - mean) * rstd * w + b);
  }
}

} // namespace

at::Tensor embedding_ln0_forward_varlen_cuda(
    at::Tensor embedding,
    at::Tensor weight,
    at::Tensor bias,
    double eps) {
  auto out = at::empty(embedding.sizes(), embedding.options().dtype(at::kHalf));
  const int64_t v64 = embedding.size(0);
  const int64_t c64 = embedding.size(1);
  TORCH_CHECK(v64 <= INT_MAX && c64 <= INT_MAX, "emb shape too large");
  const int V = static_cast<int>(v64);
  const int C = static_cast<int>(c64);
  auto stream = at::cuda::getCurrentCUDAStream();
  // Grid x equals V: launch one 256-thread block per packed token row.
  emb_ln0_bf16_to_f16_kernel<<<V, 256, 0, stream>>>(
      C,
      reinterpret_cast<const uint16_t*>(embedding.data_ptr<at::BFloat16>()),
      reinterpret_cast<const uint16_t*>(weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<const uint16_t*>(bias.data_ptr<at::BFloat16>()),
      out.data_ptr<dtype>(),
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
