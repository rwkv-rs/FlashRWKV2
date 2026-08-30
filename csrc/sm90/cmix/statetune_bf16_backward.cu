// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Canonical source: RWKV-LM RWKV-v7/train_temp/cuda/rwkv7_cmix_bf16_v5.cu
// Source revision: 952102498e9ed367ea0a59ee64106916d474d30f.
// Local adaptation: propagate full ChannelMix nonzero initial and returned shift gradients.

#include "validation.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

void gemm_bf16(
    const torch::stable::Tensor& left,
    const torch::stable::Tensor& right,
    torch::stable::Tensor& output,
    int64_t m,
    int64_t n,
    int64_t k,
    bool transpose_left,
    bool transpose_right) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const cublasStatus_t status = cublasGemmEx(
      flashrwkv2::validation::current_cuda_blas_handle(),
      transpose_right ? CUBLAS_OP_T : CUBLAS_OP_N,
      transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N,
      static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
      &alpha,
      right.const_data_ptr<torch::headeronly::BFloat16>(), CUDA_R_16BF,
      static_cast<int>(right.size(-1)),
      left.const_data_ptr<torch::headeronly::BFloat16>(), CUDA_R_16BF,
      static_cast<int>(left.size(-1)),
      &beta,
      output.mutable_data_ptr<torch::headeronly::BFloat16>(), CUDA_R_16BF,
      static_cast<int>(n), CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  STD_TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "CMix BF16 GEMM failed");
}


__device__ inline __nv_bfloat162 load_bf16x2(const torch::headeronly::BFloat16* ptr) {
  return *reinterpret_cast<const __nv_bfloat162*>(ptr);
}
__device__ inline void store_bf16x2(torch::headeronly::BFloat16* ptr,
                                    __nv_bfloat162 value) {
  *reinterpret_cast<__nv_bfloat162*>(ptr) = value;
}
inline int64_t ceil_div(int64_t n, int64_t d) { return (n + d - 1) / d; }
constexpr int kThreads = 256;

__global__ void relu_square_backward_from_output_kernel(
    torch::headeronly::BFloat16* __restrict__ grad,
    const torch::headeronly::BFloat16* __restrict__ activation, int64_t pairs) {
  const int64_t pair =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= pairs) return;
  const int64_t idx = pair * 2;
  const float2 g = __bfloat1622float2(load_bf16x2(grad + idx));
  const float2 out = __bfloat1622float2(load_bf16x2(activation + idx));
  store_bf16x2(
      grad + idx,
      __floats2bfloat162_rn(g.x * 2.0f * sqrtf(fmaxf(out.x, 0.0f)),
                            g.y * 2.0f * sqrtf(fmaxf(out.y, 0.0f))));
}

__global__ void statetune_cmix_tokenshift_backward_kernel(
    const torch::headeronly::BFloat16* __restrict__ grad_mixed,
    const torch::headeronly::BFloat16* __restrict__ grad_next,
    const torch::headeronly::BFloat16* __restrict__ x,
    const torch::headeronly::BFloat16* __restrict__ initial_shift,
    const torch::headeronly::BFloat16* __restrict__ x_k,
    torch::headeronly::BFloat16* __restrict__ grad_x,
    torch::headeronly::BFloat16* __restrict__ grad_initial,
    float* __restrict__ grad_x_k, int64_t b_size, int64_t t_size,
    int64_t c_size) {
  const int64_t bc_pair =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t pairs_per_row = c_size / 2;
  if (bc_pair >= b_size * pairs_per_row) return;
  const int64_t b = bc_pair / pairs_per_row;
  const int64_t c = (bc_pair % pairs_per_row) * 2;
  const int64_t base = b * t_size * c_size + c;
  const __nv_bfloat162 coefficient = load_bf16x2(x_k + c);
  const __nv_bfloat162 one_minus =
      __hsub2(__floats2bfloat162_rn(1.0f, 1.0f), coefficient);
  float2 coefficient_sum = make_float2(0.0f, 0.0f);

  for (int64_t t = 0; t < t_size; ++t) {
    const int64_t idx = base + t * c_size;
    __nv_bfloat162 dx =
        __hmul2(load_bf16x2(grad_mixed + idx), one_minus);
    if (t + 1 < t_size) {
      dx = __hadd2(dx, __hmul2(load_bf16x2(grad_mixed + idx + c_size),
                               coefficient));
    } else {
      dx = __hadd2(dx, load_bf16x2(grad_next + b * c_size + c));
    }
    store_bf16x2(grad_x + idx, dx);

    const __nv_bfloat162 previous =
        t == 0 ? load_bf16x2(initial_shift + b * c_size + c)
               : load_bf16x2(x + idx - c_size);
    const float2 delta =
        __bfloat1622float2(__hsub2(previous, load_bf16x2(x + idx)));
    const float2 grad = __bfloat1622float2(load_bf16x2(grad_mixed + idx));
    coefficient_sum.x += grad.x * delta.x;
    coefficient_sum.y += grad.y * delta.y;
  }
  store_bf16x2(grad_initial + b * c_size + c,
               __hmul2(load_bf16x2(grad_mixed + base), coefficient));
  atomicAdd(reinterpret_cast<float2*>(grad_x_k + c), coefficient_sum);
}

__global__ void cast_float_to_bf16_vec2_kernel(
    const float* __restrict__ source, torch::headeronly::BFloat16* __restrict__ target,
    int64_t pairs) {
  const int64_t pair =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= pairs) return;
  const int64_t c = pair * 2;
  store_bf16x2(target + c,
               __floats2bfloat162_rn(source[c], source[c + 1]));
}

}  // namespace

std::vector<torch::stable::Tensor> statetune_cmix_backward_cuda(
    torch::stable::Tensor grad_output, torch::stable::Tensor grad_next_shift,
    torch::stable::Tensor x, torch::stable::Tensor initial_shift, torch::stable::Tensor x_k,
    torch::stable::Tensor key_weight, torch::stable::Tensor value_weight,
    torch::stable::Tensor mixed, torch::stable::Tensor activation) {
  const int64_t bt_size = x.size(0) * x.size(1);
  auto grad_value_weight = torch::stable::empty_like(value_weight);
  gemm_bf16(
      grad_output, activation, grad_value_weight, grad_output.size(2),
      activation.size(1), bt_size, true, false);
  auto grad_hidden = torch::stable::new_empty(
      grad_output, {bt_size, value_weight.size(1)});
  gemm_bf16(
      grad_output, value_weight, grad_hidden, bt_size, value_weight.size(1),
      grad_output.size(2), false, false);
  auto stream = flashrwkv2::validation::current_cuda_stream();
  const int64_t activation_pairs = activation.numel() / 2;
  relu_square_backward_from_output_kernel<<<
      static_cast<int>(ceil_div(activation_pairs, kThreads)), kThreads, 0,
      stream>>>(grad_hidden.mutable_data_ptr<torch::headeronly::BFloat16>(),
                activation.mutable_data_ptr<torch::headeronly::BFloat16>(), activation_pairs);
  FLASHRWKV_CUDA_CHECK(cudaGetLastError());
  auto grad_key_weight = torch::stable::empty_like(key_weight);
  gemm_bf16(
      grad_hidden, mixed, grad_key_weight, grad_hidden.size(1), x.size(2),
      bt_size, true, false);
  auto grad_mixed = torch::stable::empty_like(x);
  gemm_bf16(
      grad_hidden, key_weight, grad_mixed, bt_size, x.size(2),
      grad_hidden.size(1), false, false);

  auto grad_x = torch::stable::empty_like(x);
  auto grad_initial = torch::stable::empty_like(initial_shift);
  auto grad_x_k_fp32 =
      torch::stable::new_zeros(x, {x.size(2)}, torch::headeronly::ScalarType::Float);
  auto grad_x_k = torch::stable::empty_like(x_k);
  const int64_t bc_pairs = x.size(0) * (x.size(2) / 2);
  statetune_cmix_tokenshift_backward_kernel<<<
      static_cast<int>(ceil_div(bc_pairs, kThreads)), kThreads, 0, stream>>>(
      grad_mixed.mutable_data_ptr<torch::headeronly::BFloat16>(),
      grad_next_shift.mutable_data_ptr<torch::headeronly::BFloat16>(), x.mutable_data_ptr<torch::headeronly::BFloat16>(),
      initial_shift.mutable_data_ptr<torch::headeronly::BFloat16>(), x_k.mutable_data_ptr<torch::headeronly::BFloat16>(),
      grad_x.mutable_data_ptr<torch::headeronly::BFloat16>(),
      grad_initial.mutable_data_ptr<torch::headeronly::BFloat16>(), grad_x_k_fp32.mutable_data_ptr<float>(),
      x.size(0), x.size(1), x.size(2));
  FLASHRWKV_CUDA_CHECK(cudaGetLastError());
  const int64_t channel_pairs = x.size(2) / 2;
  cast_float_to_bf16_vec2_kernel<<<
      static_cast<int>(ceil_div(channel_pairs, kThreads)), kThreads, 0,
      stream>>>(grad_x_k_fp32.mutable_data_ptr<float>(),
                grad_x_k.mutable_data_ptr<torch::headeronly::BFloat16>(), channel_pairs);
  FLASHRWKV_CUDA_CHECK(cudaGetLastError());
  return {grad_x, grad_initial, grad_x_k, grad_key_weight, grad_value_weight};
}
