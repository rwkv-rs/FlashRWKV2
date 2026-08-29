// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Canonical source: RWKV-LM RWKV-v7/train_temp/cuda/rwkv7_cmix_bf16_v5.cu
// Source revision: 952102498e9ed367ea0a59ee64106916d474d30f.
// Local adaptation: ChannelMix-owned FlashRWKV2 binding names; tensor contract
// remains the canonical train_temp [B,T,C] BF16 contract.

#include "validation.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <vector>

namespace {

__device__ inline __nv_bfloat162 load_bf16x2(const torch::headeronly::BFloat16* ptr) {
    return *reinterpret_cast<const __nv_bfloat162*>(ptr);
}

__device__ inline void store_bf16x2(torch::headeronly::BFloat16* ptr, __nv_bfloat162 value) {
    *reinterpret_cast<__nv_bfloat162*>(ptr) = value;
}

__device__ inline void atomic_add_float2(float* ptr, float x0, float x1) {
    atomicAdd(reinterpret_cast<float2*>(ptr), make_float2(x0, x1));
}

inline int64_t ceil_div(int64_t n, int64_t d) {
    return (n + d - 1) / d;
}

__global__ void cmix_tokenshift_forward_vec2_kernel(
    const torch::headeronly::BFloat16* __restrict__ x,
    const torch::headeronly::BFloat16* __restrict__ x_k,
    torch::headeronly::BFloat16* __restrict__ out,
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

template<int BT_TILE>
__global__ void cmix_tokenshift_backward_fused_vec2_kernel(
    const torch::headeronly::BFloat16* __restrict__ grad_out,
    const torch::headeronly::BFloat16* __restrict__ x,
    const torch::headeronly::BFloat16* __restrict__ x_k,
    torch::headeronly::BFloat16* __restrict__ grad_x,
    float* __restrict__ grad_x_k,
    int64_t bt_size,
    int64_t t_size,
    int64_t c_size) {
    int64_t c_pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_pairs = c_size / 2;
    if (c_pair >= total_pairs) {
        return;
    }

    int64_t c = c_pair * 2;
    int64_t bt_start = static_cast<int64_t>(blockIdx.y) * BT_TILE;
    int64_t bt_end = min(bt_start + static_cast<int64_t>(BT_TILE), bt_size);

    __nv_bfloat162 mix = load_bf16x2(x_k + c);
    __nv_bfloat162 one_minus_mix = __hsub2(__floats2bfloat162_rn(1.0f, 1.0f), mix);

    __nv_bfloat162 grad = __floats2bfloat162_rn(0.0f, 0.0f);
    bool have_grad = false;
    float2 prev_x = make_float2(0.0f, 0.0f);
    bool have_prev_x = false;
    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (int64_t bt = bt_start; bt < bt_end; ++bt) {
        int64_t idx = bt * c_size + c;
        int64_t t = bt % t_size;
        if (!have_grad) {
            grad = load_bf16x2(grad_out + idx);
        }

        __nv_bfloat162 next_grad = __floats2bfloat162_rn(0.0f, 0.0f);
        bool have_next_grad = false;
        if (t + 1 < t_size) {
            next_grad = load_bf16x2(grad_out + idx + c_size);
            have_next_grad = true;
        }

        __nv_bfloat162 dx = __hmul2(grad, one_minus_mix);
        if (have_next_grad) {
            dx = __hadd2(dx, __hmul2(next_grad, mix));
        }
        store_bf16x2(grad_x + idx, dx);

        float2 x_now = __bfloat1622float2(load_bf16x2(x + idx));
        float2 x_prev = make_float2(0.0f, 0.0f);
        if (t > 0) {
            x_prev = have_prev_x ? prev_x : __bfloat1622float2(load_bf16x2(x + idx - c_size));
        }
        float2 g = __bfloat1622float2(grad);
        sum0 += g.x * (x_prev.x - x_now.x);
        sum1 += g.y * (x_prev.y - x_now.y);

        prev_x = x_now;
        have_prev_x = (t + 1 < t_size);
        grad = next_grad;
        have_grad = have_next_grad;
    }

    atomic_add_float2(grad_x_k + c, sum0, sum1);
}

template<int BT_TILE>
void launch_cmix_tokenshift_backward_fused_vec2_kernel(
    const torch::headeronly::BFloat16* grad_out,
    const torch::headeronly::BFloat16* x,
    const torch::headeronly::BFloat16* x_k,
    torch::headeronly::BFloat16* grad_x,
    float* grad_x_k,
    int64_t bt_size,
    int64_t t_size,
    int64_t c_size,
    cudaStream_t stream) {
    dim3 blocks(
        static_cast<unsigned int>(ceil_div(c_size / 2, static_cast<int64_t>(CMIX_THREADS))),
        static_cast<unsigned int>(ceil_div(bt_size, static_cast<int64_t>(BT_TILE))),
        1);
    cmix_tokenshift_backward_fused_vec2_kernel<BT_TILE><<<blocks, CMIX_THREADS, 0, stream>>>(
        grad_out,
        x,
        x_k,
        grad_x,
        grad_x_k,
        bt_size,
        t_size,
        c_size);
}

__global__ void cast_float_to_bf16_vec2_kernel(
    const float* __restrict__ src,
    torch::headeronly::BFloat16* __restrict__ dst,
    int64_t total_pairs) {
    int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair_idx >= total_pairs) {
        return;
    }
    int64_t idx = pair_idx * 2;
    store_bf16x2(dst + idx, __floats2bfloat162_rn(src[idx], src[idx + 1]));
}

__global__ void relu_sq_inplace_vec2_kernel(torch::headeronly::BFloat16* __restrict__ x, int64_t total_pairs) {
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

__global__ void relu_sq_backward_from_output_inplace_vec2_kernel(
    torch::headeronly::BFloat16* __restrict__ grad_out,
    const torch::headeronly::BFloat16* __restrict__ out,
    int64_t total_pairs) {
    int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair_idx >= total_pairs) {
        return;
    }

    int64_t idx = pair_idx * 2;
    float2 grad_v = __bfloat1622float2(load_bf16x2(grad_out + idx));
    float2 act_v = __bfloat1622float2(load_bf16x2(out + idx));
    float g0 = grad_v.x * (2.0f * sqrtf(fmaxf(act_v.x, 0.0f)));
    float g1 = grad_v.y * (2.0f * sqrtf(fmaxf(act_v.y, 0.0f)));
    store_bf16x2(grad_out + idx, __floats2bfloat162_rn(g0, g1));
}

torch::stable::Tensor cmix_tokenshift_forward_cuda(torch::stable::Tensor x, torch::stable::Tensor x_k) {
    auto out = torch::stable::empty_like(x);
    const int64_t bt_size = x.size(0) * x.size(1);
    const int64_t c_size = x.size(2);
    const int64_t total_pairs = bt_size * (c_size / 2);
    auto stream = flashrwkv2::validation::current_cuda_stream();
    const int blocks = static_cast<int>(ceil_div(total_pairs, static_cast<int64_t>(CMIX_THREADS)));
    cmix_tokenshift_forward_vec2_kernel<<<blocks, CMIX_THREADS, 0, stream>>>(
        x.mutable_data_ptr<torch::headeronly::BFloat16>(),
        x_k.mutable_data_ptr<torch::headeronly::BFloat16>(),
        out.mutable_data_ptr<torch::headeronly::BFloat16>(),
        bt_size,
        x.size(1),
        c_size);
    FLASHRWKV_CUDA_CHECK(cudaGetLastError());
    return out;
}

std::vector<torch::stable::Tensor> cmix_tokenshift_backward_cuda(torch::stable::Tensor grad_out, torch::stable::Tensor x, torch::stable::Tensor x_k) {
    auto grad_x = torch::stable::empty_like(x);
    auto grad_x_k_fp32 = torch::stable::new_zeros(x, {x.size(2)}, torch::headeronly::ScalarType::Float);
    auto grad_x_k = torch::stable::new_empty(x, {x.size(2)});

    const int64_t bt_size = x.size(0) * x.size(1);
    const int64_t c_size = x.size(2);
    auto stream = flashrwkv2::validation::current_cuda_stream();

    int bt_tile = 16;
    if (const char* env = std::getenv("CMIX_V5_BT_TILE")) {
        bt_tile = std::atoi(env);
    }
    if (bt_tile == 16) {
        launch_cmix_tokenshift_backward_fused_vec2_kernel<16>(
            grad_out.mutable_data_ptr<torch::headeronly::BFloat16>(), x.mutable_data_ptr<torch::headeronly::BFloat16>(), x_k.mutable_data_ptr<torch::headeronly::BFloat16>(),
            grad_x.mutable_data_ptr<torch::headeronly::BFloat16>(), grad_x_k_fp32.mutable_data_ptr<float>(), bt_size, x.size(1), c_size, stream);
    } else if (bt_tile == 64) {
        launch_cmix_tokenshift_backward_fused_vec2_kernel<64>(
            grad_out.mutable_data_ptr<torch::headeronly::BFloat16>(), x.mutable_data_ptr<torch::headeronly::BFloat16>(), x_k.mutable_data_ptr<torch::headeronly::BFloat16>(),
            grad_x.mutable_data_ptr<torch::headeronly::BFloat16>(), grad_x_k_fp32.mutable_data_ptr<float>(), bt_size, x.size(1), c_size, stream);
    } else if (bt_tile == 128) {
        launch_cmix_tokenshift_backward_fused_vec2_kernel<128>(
            grad_out.mutable_data_ptr<torch::headeronly::BFloat16>(), x.mutable_data_ptr<torch::headeronly::BFloat16>(), x_k.mutable_data_ptr<torch::headeronly::BFloat16>(),
            grad_x.mutable_data_ptr<torch::headeronly::BFloat16>(), grad_x_k_fp32.mutable_data_ptr<float>(), bt_size, x.size(1), c_size, stream);
    } else {
        launch_cmix_tokenshift_backward_fused_vec2_kernel<32>(
            grad_out.mutable_data_ptr<torch::headeronly::BFloat16>(), x.mutable_data_ptr<torch::headeronly::BFloat16>(), x_k.mutable_data_ptr<torch::headeronly::BFloat16>(),
            grad_x.mutable_data_ptr<torch::headeronly::BFloat16>(), grad_x_k_fp32.mutable_data_ptr<float>(), bt_size, x.size(1), c_size, stream);
    }
    FLASHRWKV_CUDA_CHECK(cudaGetLastError());

    const int64_t total_pairs = c_size / 2;
    const int cast_blocks = static_cast<int>(ceil_div(total_pairs, static_cast<int64_t>(CMIX_THREADS)));
    cast_float_to_bf16_vec2_kernel<<<cast_blocks, CMIX_THREADS, 0, stream>>>(
        grad_x_k_fp32.mutable_data_ptr<float>(),
        grad_x_k.mutable_data_ptr<torch::headeronly::BFloat16>(),
        total_pairs);
    FLASHRWKV_CUDA_CHECK(cudaGetLastError());

    return {grad_x, grad_x_k};
}

void relu_sq_inplace_cuda(torch::stable::Tensor x) {
    const int64_t total_pairs = x.numel() / 2;
    auto stream = flashrwkv2::validation::current_cuda_stream();
    const int blocks = static_cast<int>(ceil_div(total_pairs, static_cast<int64_t>(CMIX_THREADS)));
    relu_sq_inplace_vec2_kernel<<<blocks, CMIX_THREADS, 0, stream>>>(x.mutable_data_ptr<torch::headeronly::BFloat16>(), total_pairs);
    FLASHRWKV_CUDA_CHECK(cudaGetLastError());
}

void relu_sq_backward_from_output_inplace_cuda(torch::stable::Tensor grad_out, torch::stable::Tensor out) {
    const int64_t total_pairs = out.numel() / 2;
    auto stream = flashrwkv2::validation::current_cuda_stream();
    const int blocks = static_cast<int>(ceil_div(total_pairs, static_cast<int64_t>(CMIX_THREADS)));
    relu_sq_backward_from_output_inplace_vec2_kernel<<<blocks, CMIX_THREADS, 0, stream>>>(
        grad_out.mutable_data_ptr<torch::headeronly::BFloat16>(),
        out.mutable_data_ptr<torch::headeronly::BFloat16>(),
        total_pairs);
    FLASHRWKV_CUDA_CHECK(cudaGetLastError());
}

}
std::vector<torch::stable::Tensor> pretrain_cmix_backward_cuda(
    torch::stable::Tensor grad_out,
    torch::stable::Tensor x,
    torch::stable::Tensor x_k,
    torch::stable::Tensor key_weight,
    torch::stable::Tensor value_weight,
    torch::stable::Tensor mixed,
    torch::stable::Tensor act) {
    auto grad_out_2d = torch::stable::contiguous(
        torch::stable::view(grad_out, {-1, grad_out.size(2)}));
    auto mixed_2d = torch::stable::contiguous(
        torch::stable::view(mixed, {-1, mixed.size(2)}));

    auto grad_out_t = torch::stable::transpose(grad_out_2d, 0, 1);
    auto grad_value_weight = torch::stable::matmul(grad_out_t, act);
    auto grad_hidden = torch::stable::contiguous(
        torch::stable::matmul(grad_out_2d, value_weight));
    relu_sq_backward_from_output_inplace_cuda(grad_hidden, act);
    auto grad_hidden_t = torch::stable::transpose(grad_hidden, 0, 1);
    auto grad_key_weight = torch::stable::matmul(grad_hidden_t, mixed_2d);
    auto grad_mixed_2d = torch::stable::contiguous(
        torch::stable::matmul(grad_hidden, key_weight));
    auto grad_mixed = torch::stable::view(grad_mixed_2d, x.sizes());
    auto mix_grads = cmix_tokenshift_backward_cuda(grad_mixed, x, x_k);

    return {
        mix_grads[0],
        mix_grads[1],
        torch::stable::contiguous(grad_key_weight),
        torch::stable::contiguous(grad_value_weight),
    };
}
