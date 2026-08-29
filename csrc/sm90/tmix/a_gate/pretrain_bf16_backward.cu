// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Canonical source: RWKV-LM RWKV-v7/train_temp/cuda/rwkv7_tmix_a_gate_bf16.cu
// Source revision: 952102498e9ed367ea0a59ee64106916d474d30f.
// Local adaptation: module-local FlashRWKV2 binding names only; tensor contract
// remains the canonical train_temp [B,T,C] BF16 contract.

#include "validation.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <vector>

namespace {

inline int64_t ceil_div(int64_t n, int64_t d) {
    return (n + d - 1) / d;
}

inline bool is_power_of_two(int64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

__device__ inline float bf16_to_float(const torch::headeronly::BFloat16* ptr) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(ptr));
}

__device__ inline void store_bf16(torch::headeronly::BFloat16* ptr, float value) {
    *reinterpret_cast<__nv_bfloat16*>(ptr) = __float2bfloat16_rn(value);
}

__device__ inline float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void a_gate_forward_kernel(
    const torch::headeronly::BFloat16* __restrict__ a0,
    const torch::headeronly::BFloat16* __restrict__ a12,
    torch::headeronly::BFloat16* __restrict__ out,
    int64_t total,
    int64_t c_size) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int64_t c = idx % c_size;
    float pre = bf16_to_float(a0 + c) + bf16_to_float(a12 + idx);
    store_bf16(out + idx, sigmoidf_fast(pre));
}

__global__ void a_gate_forward_pow2c_kernel(
    const torch::headeronly::BFloat16* __restrict__ a0,
    const torch::headeronly::BFloat16* __restrict__ a12,
    torch::headeronly::BFloat16* __restrict__ out,
    int64_t total,
    int64_t c_mask) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int64_t c = idx & c_mask;
    float pre = bf16_to_float(a0 + c) + bf16_to_float(a12 + idx);
    store_bf16(out + idx, sigmoidf_fast(pre));
}

constexpr int kBackwardTileM = 16;
constexpr int kDefaultThreads = 256;

__global__ void a_gate_backward_full_kernel(
    const torch::headeronly::BFloat16* __restrict__ grad_out,
    const torch::headeronly::BFloat16* __restrict__ a0,
    const torch::headeronly::BFloat16* __restrict__ a12,
    torch::headeronly::BFloat16* __restrict__ grad_a12,
    float* __restrict__ partial_a0,
    int64_t rows,
    int64_t c_size) {
    int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= c_size) {
        return;
    }

    int64_t row_base = static_cast<int64_t>(blockIdx.y) * kBackwardTileM;
    float a0v = bf16_to_float(a0 + c);
    float sum = 0.0f;
    for (int i = 0; i < kBackwardTileM; ++i) {
        int64_t row = row_base + i;
        if (row >= rows) {
            break;
        }
        int64_t idx = row * c_size + c;
        float go = bf16_to_float(grad_out + idx);
        float a = sigmoidf_fast(a0v + bf16_to_float(a12 + idx));
        float gp = go * a * (1.0f - a);
        store_bf16(grad_a12 + idx, gp);
        sum += gp;
    }

    partial_a0[static_cast<int64_t>(blockIdx.y) * c_size + c] = sum;
}

__global__ void a_gate_reduce_a0_kernel(
    const float* __restrict__ partial_a0,
    torch::headeronly::BFloat16* __restrict__ grad_a0,
    int64_t num_tiles,
    int64_t c_size) {
    int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= c_size) {
        return;
    }

    float sum = 0.0f;
    for (int64_t tile = 0; tile < num_tiles; ++tile) {
        sum += partial_a0[tile * c_size + c];
    }
    store_bf16(grad_a0 + c, sum);
}

}
std::vector<torch::stable::Tensor> pretrain_tmix_a_gate_backward_cuda(torch::stable::Tensor grad_out, torch::stable::Tensor a0, torch::stable::Tensor a12) {
    auto grad_a0 = torch::stable::empty_like(a0);
    auto grad_a12 = torch::stable::empty_like(a12);
    int64_t total = a12.numel();
    int64_t c_size = a12.size(2);
    int64_t rows = total / c_size;
    int64_t num_tiles = ceil_div(rows, static_cast<int64_t>(kBackwardTileM));
    auto partial_a0 = torch::stable::new_empty(a12, {num_tiles, c_size}, torch::headeronly::ScalarType::Float);
    int threads = kDefaultThreads;
    if (const char* env = std::getenv("A_GATE_THREADS")) {
        threads = std::atoi(env);
    }
    dim3 blocks(
        static_cast<unsigned int>(ceil_div(c_size, static_cast<int64_t>(threads))),
        static_cast<unsigned int>(num_tiles));
    auto stream = flashrwkv2::validation::current_cuda_stream();
    a_gate_backward_full_kernel<<<blocks, threads, 0, stream>>>(
        grad_out.mutable_data_ptr<torch::headeronly::BFloat16>(),
        a0.mutable_data_ptr<torch::headeronly::BFloat16>(),
        a12.mutable_data_ptr<torch::headeronly::BFloat16>(),
        grad_a12.mutable_data_ptr<torch::headeronly::BFloat16>(),
        partial_a0.mutable_data_ptr<float>(),
        rows,
        c_size);
    FLASHRWKV_CUDA_CHECK(cudaGetLastError());
    int reduce_blocks = static_cast<int>(ceil_div(c_size, static_cast<int64_t>(threads)));
    a_gate_reduce_a0_kernel<<<reduce_blocks, threads, 0, stream>>>(
        partial_a0.mutable_data_ptr<float>(),
        grad_a0.mutable_data_ptr<torch::headeronly::BFloat16>(),
        num_tiles,
        c_size);
    FLASHRWKV_CUDA_CHECK(cudaGetLastError());
    return {grad_a0, grad_a12};
}

