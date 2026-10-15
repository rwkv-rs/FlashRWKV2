// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Canonical source: RWKV-LM RWKV-v7/train_temp/cuda/rwkv7_cmix_bf16_v5.cu
// Source revision: 952102498e9ed367ea0a59ee64106916d474d30f.
// Local adaptation: full ChannelMix with a nonzero initial shift and returned final input.

#include <torch/extension.h>
#include "../internal/linear/backend.cuh"
#include <optional>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <vector>

namespace {


__device__ inline __nv_bfloat162 load_bf16x2(const at::BFloat16* ptr) {
  return *reinterpret_cast<const __nv_bfloat162*>(ptr);
}
__device__ inline void store_bf16x2(at::BFloat16* ptr,
                                    __nv_bfloat162 value) {
  *reinterpret_cast<__nv_bfloat162*>(ptr) = value;
}
inline int64_t ceil_div(int64_t n, int64_t d) { return (n + d - 1) / d; }
constexpr int kThreads = 256;

__global__ void statetune_cmix_tokenshift_forward_kernel(
    const at::BFloat16* __restrict__ x,
    const at::BFloat16* __restrict__ initial_shift,
    const at::BFloat16* __restrict__ x_k,
    at::BFloat16* __restrict__ mixed, int64_t bt_size, int64_t t_size,
    int64_t c_size) {
  const int64_t pair =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t pairs_per_row = c_size / 2;
  if (pair >= bt_size * pairs_per_row) return;
  const int64_t bt = pair / pairs_per_row;
  const int64_t c = (pair % pairs_per_row) * 2;
  const int64_t idx = bt * c_size + c;
  const int64_t t = bt % t_size;
  const int64_t b = bt / t_size;
  const __nv_bfloat162 current = load_bf16x2(x + idx);
  const __nv_bfloat162 previous =
      t == 0 ? load_bf16x2(initial_shift + b * c_size + c)
             : load_bf16x2(x + idx - c_size);
  store_bf16x2(mixed + idx,
               __hadd2(current, __hmul2(__hsub2(previous, current),
                                        load_bf16x2(x_k + c))));
}

__global__ void relu_square_inplace_kernel(at::BFloat16* data,
                                            int64_t pairs) {
  const int64_t pair =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= pairs) return;
  const int64_t idx = pair * 2;
  const float2 value = __bfloat1622float2(load_bf16x2(data + idx));
  const float x0 = fmaxf(value.x, 0.0f);
  const float x1 = fmaxf(value.y, 0.0f);
  store_bf16x2(data + idx, __floats2bfloat162_rn(x0 * x0, x1 * x1));
}

}  // namespace

std::vector<torch::Tensor> statetune_cmix_forward_cuda(
    torch::Tensor x, torch::Tensor initial_shift, torch::Tensor x_k,
    torch::Tensor key_weight, torch::Tensor value_weight,
    std::optional<torch::Tensor> key_lora_a,
    std::optional<torch::Tensor> key_lora_b,
    std::optional<torch::Tensor> value_lora_a,
    std::optional<torch::Tensor> value_lora_b,
    double key_lora_scale, double value_lora_scale,
    bool key_use_nora, bool value_use_nora) {
  auto mixed = torch::empty_like(x);
  const int64_t bt_size = x.size(0) * x.size(1);
  const int64_t mix_pairs = bt_size * (x.size(2) / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  statetune_cmix_tokenshift_forward_kernel<<<
      static_cast<int>(ceil_div(mix_pairs, kThreads)), kThreads, 0, stream>>>(
      x.data_ptr<at::BFloat16>(), initial_shift.data_ptr<at::BFloat16>(),
      x_k.data_ptr<at::BFloat16>(), mixed.data_ptr<at::BFloat16>(), bt_size,
      x.size(1), x.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto key_result = internal_linear_train_forward(
      mixed, key_weight, key_lora_a.value_or(torch::Tensor()),
      key_lora_b.value_or(torch::Tensor()), key_lora_scale, key_use_nora);
  auto activation = key_result[0];
  const int64_t activation_pairs = activation.numel() / 2;
  relu_square_inplace_kernel<<<static_cast<int>(ceil_div(activation_pairs, kThreads)), kThreads, 0, stream>>>(
      activation.data_ptr<at::BFloat16>(), activation_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto value_result = internal_linear_train_forward(
      activation, value_weight, value_lora_a.value_or(torch::Tensor()),
      value_lora_b.value_or(torch::Tensor()), value_lora_scale, value_use_nora);
  auto output = value_result[0].view(x.sizes());
  auto next_shift = x.select(1, x.size(1) - 1).contiguous();
  return {output, next_shift, mixed, activation,
           key_result[1], key_result[2], value_result[1], value_result[2]};
}
