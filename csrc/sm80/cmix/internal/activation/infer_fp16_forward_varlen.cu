// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Albatross project
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
// Internal activation implementation owned by the complete CMix operator.
// Albatross source revision: ee3308f6922e59f2166c7fac3c5a192340a2b48e.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

using dtype = at::Half;

__global__ void cmix_relu_square_varlen_kernel(
    const dtype* __restrict__ x,
    dtype* __restrict__ out,
    int64_t total_pairs) {
  const int64_t pair_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair_idx >= total_pairs) {
    return;
  }
  const int64_t idx = pair_idx * 2;
  const float2 value = __half22float2(
      *reinterpret_cast<const __half2*>(x + idx));
  const float relu_x = fmaxf(value.x, 0.0f);
  const float relu_y = fmaxf(value.y, 0.0f);
  *reinterpret_cast<__half2*>(out + idx) =
      __floats2half2_rn(relu_x * relu_x, relu_y * relu_y);
}

}  // namespace

torch::Tensor cmix_relu_square_forward_varlen_cuda(torch::Tensor x) {
  auto out = torch::empty_like(x);
  constexpr int threads = 256;
  const int64_t total_pairs = x.numel() / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  cmix_relu_square_varlen_kernel<<<
      static_cast<int>((total_pairs + threads - 1) / threads),
      threads,
      0,
      stream>>>(
      reinterpret_cast<const dtype*>(x.data_ptr()),
      reinterpret_cast<dtype*>(out.data_ptr()),
      total_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
