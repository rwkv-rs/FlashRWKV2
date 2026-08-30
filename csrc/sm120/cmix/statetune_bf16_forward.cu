// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Canonical source: RWKV-LM RWKV-v7/train_temp/cuda/rwkv7_cmix_bf16_v5.cu
// Source revision: 952102498e9ed367ea0a59ee64106916d474d30f.
// Local adaptation: full ChannelMix with a nonzero initial shift and returned final input.

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

__global__ void statetune_cmix_tokenshift_forward_kernel(
    const torch::headeronly::BFloat16* __restrict__ x,
    const torch::headeronly::BFloat16* __restrict__ initial_shift,
    const torch::headeronly::BFloat16* __restrict__ x_k,
    torch::headeronly::BFloat16* __restrict__ mixed, int64_t bt_size, int64_t t_size,
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

__global__ void relu_square_inplace_kernel(torch::headeronly::BFloat16* data,
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

std::vector<torch::stable::Tensor> statetune_cmix_forward_cuda(
    torch::stable::Tensor x, torch::stable::Tensor initial_shift, torch::stable::Tensor x_k,
    torch::stable::Tensor key_weight, torch::stable::Tensor value_weight) {
  auto mixed = torch::stable::empty_like(x);
  const int64_t bt_size = x.size(0) * x.size(1);
  const int64_t mix_pairs = bt_size * (x.size(2) / 2);
  auto stream = flashrwkv2::validation::current_cuda_stream();
  statetune_cmix_tokenshift_forward_kernel<<<
      static_cast<int>(ceil_div(mix_pairs, kThreads)), kThreads, 0, stream>>>(
      x.mutable_data_ptr<torch::headeronly::BFloat16>(), initial_shift.mutable_data_ptr<torch::headeronly::BFloat16>(),
      x_k.mutable_data_ptr<torch::headeronly::BFloat16>(), mixed.mutable_data_ptr<torch::headeronly::BFloat16>(), bt_size,
      x.size(1), x.size(2));
  FLASHRWKV_CUDA_CHECK(cudaGetLastError());

  auto activation = torch::stable::new_empty(
      x, {bt_size, key_weight.size(0)});
  gemm_bf16(
      mixed, key_weight, activation, bt_size, key_weight.size(0), x.size(2),
      false, true);
  const int64_t activation_pairs = activation.numel() / 2;
  relu_square_inplace_kernel<<<
      static_cast<int>(ceil_div(activation_pairs, kThreads)), kThreads, 0,
      stream>>>(activation.mutable_data_ptr<torch::headeronly::BFloat16>(), activation_pairs);
  FLASHRWKV_CUDA_CHECK(cudaGetLastError());
  auto output = torch::stable::empty_like(x);
  gemm_bf16(
      activation, value_weight, output, bt_size, value_weight.size(0),
      activation.size(1), false, true);
  auto next_shift = torch::stable::contiguous(
      torch::stable::select(x, 1, x.size(1) - 1));
  return {output, next_shift, mixed, activation};
}
