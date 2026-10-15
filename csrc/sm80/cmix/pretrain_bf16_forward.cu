// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Canonical source: RWKV-LM RWKV-v7/train_temp/cuda/rwkv7_cmix_bf16_v5.cu
// Source revision: 952102498e9ed367ea0a59ee64106916d474d30f.
// Local adaptation: ChannelMix-owned FlashRWKV2 binding names; tensor contract
// remains the canonical train_temp [B,T,C] BF16 contract.

#include <torch/extension.h>
#include "../internal/linear/backend.cuh"
#include <optional>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

__device__ inline __nv_bfloat162 load_bf16x2(const at::BFloat16* ptr) {
    return *reinterpret_cast<const __nv_bfloat162*>(ptr);
}

__device__ inline void store_bf16x2(at::BFloat16* ptr, __nv_bfloat162 value) {
    *reinterpret_cast<__nv_bfloat162*>(ptr) = value;
}

inline int64_t ceil_div(int64_t n, int64_t d) {
    return (n + d - 1) / d;
}

__global__ void cmix_tokenshift_forward_vec2_kernel(
    const at::BFloat16* __restrict__ x,
    const at::BFloat16* __restrict__ x_k,
    at::BFloat16* __restrict__ out,
    int64_t bt_size,
    int64_t t_size,
    int64_t c_size) {
    int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_pairs = bt_size * (c_size / 2);
    if (pair_idx >= total_pairs) {
        return;
    }

    int64_t bt = pair_idx / (c_size / 2);
    int64_t c_pair = pair_idx % (c_size / 2);
    int64_t c = c_pair * 2;
    int64_t idx = bt * c_size + c;
    int64_t t = bt % t_size;

    __nv_bfloat162 x_now = load_bf16x2(x + idx);
    __nv_bfloat162 x_prev = __floats2bfloat162_rn(0.0f, 0.0f);
    if (t > 0) {
        x_prev = load_bf16x2(x + idx - c_size);
    }
    __nv_bfloat162 mix = load_bf16x2(x_k + c);
    store_bf16x2(out + idx, __hadd2(x_now, __hmul2(__hsub2(x_prev, x_now), mix)));
}

constexpr int CMIX_THREADS = 256;

__global__ void relu_sq_inplace_vec2_kernel(at::BFloat16* __restrict__ x, int64_t total_pairs) {
    int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair_idx >= total_pairs) {
        return;
    }

    int64_t idx = pair_idx * 2;
    float2 value = __bfloat1622float2(load_bf16x2(x + idx));
    float x0 = value.x > 0.0f ? value.x : 0.0f;
    float x1 = value.y > 0.0f ? value.y : 0.0f;
    store_bf16x2(x + idx, __floats2bfloat162_rn(x0 * x0, x1 * x1));
}

torch::Tensor cmix_tokenshift_forward_cuda(torch::Tensor x, torch::Tensor x_k) {
    auto out = torch::empty_like(x);
    const int64_t bt_size = x.size(0) * x.size(1);
    const int64_t c_size = x.size(2);
    const int64_t total_pairs = bt_size * (c_size / 2);
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = static_cast<int>(ceil_div(total_pairs, static_cast<int64_t>(CMIX_THREADS)));
    cmix_tokenshift_forward_vec2_kernel<<<blocks, CMIX_THREADS, 0, stream>>>(
        x.data_ptr<at::BFloat16>(),
        x_k.data_ptr<at::BFloat16>(),
        out.data_ptr<at::BFloat16>(),
        bt_size,
        x.size(1),
        c_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

void relu_sq_inplace_cuda(torch::Tensor x) {
    const int64_t total_pairs = x.numel() / 2;
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = static_cast<int>(ceil_div(total_pairs, static_cast<int64_t>(CMIX_THREADS)));
    relu_sq_inplace_vec2_kernel<<<blocks, CMIX_THREADS, 0, stream>>>(x.data_ptr<at::BFloat16>(), total_pairs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}
std::vector<torch::Tensor> pretrain_cmix_forward_cuda(
    torch::Tensor x,
    torch::Tensor x_k,
    torch::Tensor key_weight,
    torch::Tensor value_weight,
    std::optional<torch::Tensor> key_lora_a,
    std::optional<torch::Tensor> key_lora_b,
    std::optional<torch::Tensor> value_lora_a,
    std::optional<torch::Tensor> value_lora_b,
    double key_lora_scale, double value_lora_scale,
    bool key_use_nora, bool value_use_nora) {
    auto mixed = cmix_tokenshift_forward_cuda(x, x_k);
  auto key_result = internal_linear_train_forward(
      mixed, key_weight, key_lora_a.value_or(torch::Tensor()),
      key_lora_b.value_or(torch::Tensor()), key_lora_scale, key_use_nora);
  auto act = key_result[0];
  relu_sq_inplace_cuda(act);
  auto value_result = internal_linear_train_forward(
      act, value_weight, value_lora_a.value_or(torch::Tensor()),
      value_lora_b.value_or(torch::Tensor()), value_lora_scale, value_use_nora);
  auto output = value_result[0].view(x.sizes());
  return {output, mixed, act,
           key_result[1], key_result[2], value_result[1], value_result[2]};
}
