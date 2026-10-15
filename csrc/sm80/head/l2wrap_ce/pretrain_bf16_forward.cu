// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Canonical source: RWKV-LM RWKV-v7/train_temp/cuda/rwkv7_head_l2wrap_ce_bf16_v4.cu
// Source revision: 952102498e9ed367ea0a59ee64106916d474d30f.
// Local adaptation: the canonical row-chunk/reduction kernels are retained;
// FlashRWKV2 owns the module-local memory-bounded wrapper below.

#include <torch/extension.h>
#include "../../internal/linear/backend.cuh"
#include <optional>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int64_t HEAD_L2WRAP_CE_VOCAB = 65536;
constexpr float NEG_INF_F = -3.4028234663852886e+38F;

template <typename scalar_t>
__device__ inline float scalar_to_float(scalar_t x) {
    return static_cast<float>(x);
}

template <>
__device__ inline float scalar_to_float<at::BFloat16>(at::BFloat16 x) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&x));
}

template <typename scalar_t>
__device__ inline scalar_t float_to_scalar(float x) {
    return static_cast<scalar_t>(x);
}

template <>
__device__ inline at::BFloat16 float_to_scalar<at::BFloat16>(float x) {
    __nv_bfloat16 v = __float2bfloat16_rn(x);
    return *reinterpret_cast<at::BFloat16*>(&v);
}

__device__ inline void reduce_max_first(float& value, int& index, float other_value, int other_index) {
    if ((other_value > value) || (other_value == value && other_index < index)) {
        value = other_value;
        index = other_index;
    }
}

__device__ inline void warp_reduce_max_first(float& value, int& index) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_value = __shfl_down_sync(mask, value, offset);
        int other_index = __shfl_down_sync(mask, index, offset);
        reduce_max_first(value, index, other_value, other_index);
    }
}

__device__ inline float warp_reduce_sum(float value) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}

template <typename scalar_t, int BLOCK_SIZE>
__global__ void row_chunk_loss_and_grad_kernel(
    scalar_t* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ loss_rows,
    int64_t row_start,
    int64_t chunk_rows,
    int64_t total_rows) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= chunk_rows) {
        return;
    }

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int kWarps = BLOCK_SIZE / 32;
    const int64_t base = row * HEAD_L2WRAP_CE_VOCAB;
    const int target = static_cast<int>(targets[row_start + row]);

    float local_max = NEG_INF_F;
    int local_idx = 0;
    float local_target = 0.0f;
    for (int64_t col = tid; col < HEAD_L2WRAP_CE_VOCAB; col += BLOCK_SIZE) {
        const float v = scalar_to_float(logits[base + col]);
        reduce_max_first(local_max, local_idx, v, static_cast<int>(col));
        if (static_cast<int>(col) == target) {
            local_target = v;
        }
    }

    warp_reduce_max_first(local_max, local_idx);
    local_target = warp_reduce_sum(local_target);

    __shared__ float warp_val[16];
    __shared__ int warp_idx[16];
    __shared__ float warp_target[16];
    __shared__ float shared_max;
    __shared__ float shared_lse;
    __shared__ int shared_argmax;
    if (lane == 0) {
        warp_val[warp] = local_max;
        warp_idx[warp] = local_idx;
        warp_target[warp] = local_target;
    }
    __syncthreads();

    if (warp == 0) {
        float block_max = lane < kWarps ? warp_val[lane] : NEG_INF_F;
        int block_idx = lane < kWarps ? warp_idx[lane] : 0;
        float target_val = lane < kWarps ? warp_target[lane] : 0.0f;
        warp_reduce_max_first(block_max, block_idx);
        target_val = warp_reduce_sum(target_val);
        if (lane == 0) {
            shared_max = block_max;
            shared_argmax = block_idx;
            warp_target[0] = target_val;
        }
    }
    __syncthreads();

    const float block_max = shared_max;
    float local_sum = 0.0f;
    for (int64_t col = tid; col < HEAD_L2WRAP_CE_VOCAB; col += BLOCK_SIZE) {
        local_sum += __expf(scalar_to_float(logits[base + col]) - block_max);
    }
    local_sum = warp_reduce_sum(local_sum);

    if (lane == 0) {
        warp_val[warp] = local_sum;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum = lane < kWarps ? warp_val[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            const float row_lse = logf(block_sum) + block_max;
            shared_lse = row_lse;
            loss_rows[row_start + row] = row_lse - warp_target[0];
        }
    }
    __syncthreads();

    const float row_lse = shared_lse;
    const int max_idx = shared_argmax;
    const float inv_rows = 1.0f / static_cast<float>(total_rows);
    const float l2_val = block_max * (1.0e-4f * inv_rows);
    for (int64_t col = tid; col < HEAD_L2WRAP_CE_VOCAB; col += BLOCK_SIZE) {
        float g = __expf(scalar_to_float(logits[base + col]) - row_lse) * inv_rows;
        if (static_cast<int>(col) == target) {
            g -= inv_rows;
        }
        if (static_cast<int>(col) == max_idx) {
            g += l2_val;
        }
        logits[base + col] = float_to_scalar<scalar_t>(g);
    }
}

__global__ void reduce_loss_kernel(
    const float* __restrict__ loss_rows,
    float* __restrict__ loss,
    int64_t rows) {
    constexpr int BLOCK_SIZE = 256;
    const int tid = threadIdx.x;
    float sum = 0.0f;
    for (int64_t idx = tid; idx < rows; idx += BLOCK_SIZE) {
        sum += loss_rows[idx];
    }
    sum = warp_reduce_sum(sum);

    __shared__ float warp_sum[8];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
        warp_sum[warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
        float block_sum = lane < 8 ? warp_sum[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            loss[0] = block_sum / static_cast<float>(rows);
        }
    }
}

template <typename scalar_t, int BLOCK_SIZE>
void launch_row_chunk_loss_and_grad(
    scalar_t* logits,
    const int64_t* targets,
    float* loss_rows,
    int64_t row_start,
    int64_t chunk_rows,
    int64_t total_rows,
    cudaStream_t stream) {
    dim3 blocks(static_cast<unsigned int>(chunk_rows));
    row_chunk_loss_and_grad_kernel<scalar_t, BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
        logits,
        targets,
        loss_rows,
        row_start,
        chunk_rows,
        total_rows);
}

} // namespace

void head_l2wrap_ce_row_chunk_loss_and_grad_v4_cuda(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor loss_rows,
    int64_t row_start,
    int64_t total_rows) {
    const int64_t chunk_rows = logits.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int threads = 512;
    if (logits.scalar_type() == torch::kBFloat16) {
        launch_row_chunk_loss_and_grad<at::BFloat16, threads>(
            logits.data_ptr<at::BFloat16>(),
            targets.data_ptr<int64_t>(),
            loss_rows.data_ptr<float>(),
            row_start,
            chunk_rows,
            total_rows,
            stream);
    } else {
        launch_row_chunk_loss_and_grad<float, threads>(
            logits.data_ptr<float>(),
            targets.data_ptr<int64_t>(),
            loss_rows.data_ptr<float>(),
            row_start,
            chunk_rows,
            total_rows,
            stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void head_l2wrap_ce_reduce_loss_v4_cuda(torch::Tensor loss_rows, torch::Tensor loss) {
    auto stream = at::cuda::getCurrentCUDAStream();
    reduce_loss_kernel<<<1, 256, 0, stream>>>(
        loss_rows.data_ptr<float>(),
        loss.data_ptr<float>(),
        loss_rows.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#include <ATen/ATen.h>

#include <algorithm>

std::vector<torch::Tensor> pretrain_head_l2wrap_ce_cuda(
    torch::Tensor hidden, torch::Tensor weight, torch::Tensor targets,
    int64_t chunk_rows, std::optional<torch::Tensor> lora_a,
    std::optional<torch::Tensor> lora_b, double lora_scale, bool use_nora,
    std::vector<bool> needs) {
  const c10::cuda::CUDAGuard guard(hidden.device());
  const int64_t rows = hidden.size(0) * hidden.size(1);
  const int64_t channels = hidden.size(2);
  const int64_t chunk = std::min(chunk_rows, rows);
  auto h2d = hidden.view({rows, channels});
  auto flat_targets = targets.view({rows});
  auto loss_rows = torch::empty({rows}, hidden.options().dtype(torch::kFloat32));
  auto loss = torch::empty({}, hidden.options().dtype(torch::kFloat32));
  torch::Tensor grad_hidden, grad_weight, grad_a, grad_b, grad_normalized;
  if (needs[0]) grad_hidden = torch::empty_like(hidden);
  if (needs[1]) grad_weight = torch::zeros(weight.sizes(), weight.options().dtype(torch::kFloat32));
  if (lora_a.has_value() && needs[2])
    grad_normalized = torch::zeros(lora_a->sizes(), hidden.options().dtype(torch::kFloat32));
  if (lora_b.has_value() && needs[3])
    grad_b = torch::zeros(lora_b->sizes(), hidden.options().dtype(torch::kFloat32));
  auto a = lora_a.value_or(torch::Tensor());
  auto b = lora_b.value_or(torch::Tensor());
  auto b_compute = b.defined() ? b.to(at::kBFloat16) : torch::Tensor();
  for (int64_t start = 0; start < rows; start += chunk) {
    const int64_t length = std::min(chunk, rows - start);
    auto h_chunk = h2d.narrow(0, start, length);
    auto projection = internal_linear_train_forward(h_chunk, weight, a, b, lora_scale, use_nora);
    auto logits = projection[0];
    head_l2wrap_ce_row_chunk_loss_and_grad_v4_cuda(logits, flat_targets, loss_rows, start, rows);
    if (needs[0]) {
      auto grads = internal_linear_train_backward(logits, h_chunk, weight, a, b,
          projection[1], projection[2], lora_scale, use_nora, true, false, false, false);
      grad_hidden.view({rows, channels}).narrow(0, start, length).copy_(grads[0]);
    }
    if (needs[1]) internal_linear_train_gemm(logits, h_chunk, true, false, 1.0, true, grad_weight, 1.0);
    if (grad_b.defined())
      internal_linear_train_gemm(logits, projection[1], true, false, lora_scale, true, grad_b, 1.0);
    if (grad_normalized.defined()) {
      auto dz = internal_linear_train_gemm(logits, b_compute, false, false, lora_scale);
      internal_linear_train_gemm(dz, h_chunk, true, false, 1.0, true, grad_normalized, 1.0);
    }
  }
  if (grad_normalized.defined()) grad_a = use_nora
      ? internal_linear_train_normalize_backward(a, grad_normalized)
      : grad_normalized.to(a.scalar_type());
  if (grad_b.defined()) grad_b = grad_b.to(b.scalar_type());
  if (grad_weight.defined()) grad_weight = grad_weight.to(weight.scalar_type());
  head_l2wrap_ce_reduce_loss_v4_cuda(loss_rows, loss);
  return {loss, grad_hidden, grad_weight, grad_a, grad_b};
}
