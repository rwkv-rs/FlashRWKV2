// SPDX-License-Identifier: MIT
// NoRA semantics: Joluck/NoRA 4669c35ce0ffd027456272ee5050c0f58ac8d22a,
// peft/src/peft/tuners/lora/layer.py (Apache-2.0). Independent CUDA implementation.
#include "backend.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

namespace {
template <typename scalar_t>
__global__ void normalize_columns(
    const scalar_t* a, at::BFloat16* normalized, int64_t rank, int64_t columns) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= columns) return;
  float sum = 0.0f;
  for (int64_t row = 0; row < rank; ++row) {
    const float value = static_cast<float>(a[row * columns + col]);
    sum += value * value;
  }
  const float inverse = 1.0f / (sqrtf(sum) + 1.0e-6f);
  for (int64_t row = 0; row < rank; ++row) {
    normalized[row * columns + col] = static_cast<at::BFloat16>(
        static_cast<float>(a[row * columns + col]) * inverse);
  }
}
}  // namespace

at::Tensor internal_linear_train_gemm(
    const at::Tensor& left, const at::Tensor& right,
    bool transpose_left, bool transpose_right, double scale,
    bool fp32_output, at::Tensor output, double beta) {
  const int64_t m = transpose_left ? left.size(1) : left.size(0);
  const int64_t k = transpose_left ? left.size(0) : left.size(1);
  const int64_t n = transpose_right ? right.size(0) : right.size(1);
  if (!output.defined()) {
    output = at::empty({m, n}, left.options().dtype(fp32_output ? at::kFloat : at::kBFloat16));
  }
  const float alpha_f = static_cast<float>(scale), beta_f = static_cast<float>(beta);
  const auto status = cublasGemmEx(
      at::cuda::getCurrentCUDABlasHandle(),
      transpose_right ? CUBLAS_OP_T : CUBLAS_OP_N,
      transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N,
      static_cast<int>(n), static_cast<int>(m), static_cast<int>(k), &alpha_f,
      right.data_ptr<at::BFloat16>(), CUDA_R_16BF, static_cast<int>(right.size(1)),
      left.data_ptr<at::BFloat16>(), CUDA_R_16BF, static_cast<int>(left.size(1)),
      &beta_f, output.data_ptr(), fp32_output ? CUDA_R_32F : CUDA_R_16BF,
      static_cast<int>(n), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "training projection GEMM failed: ", status);
  return output;
}

std::vector<at::Tensor> internal_linear_train_forward(
    at::Tensor x, at::Tensor weight, at::Tensor a, at::Tensor b,
    double scale, bool use_nora) {
  const c10::cuda::CUDAGuard guard(x.device());
  auto rows = x.view({-1, x.size(-1)});
  auto output = internal_linear_train_gemm(rows, weight, false, true);
  auto features = at::empty({0}, x.options());
  auto normalized = at::empty({0}, x.options());
  if (a.defined()) {
    if (use_nora) {
      normalized = at::empty(a.sizes(), x.options());
      const int blocks = static_cast<int>((a.size(1) + 255) / 256);
      auto stream = at::cuda::getCurrentCUDAStream();
      if (a.scalar_type() == at::kFloat) {
        normalize_columns<<<blocks, 256, 0, stream>>>(
            a.data_ptr<float>(), normalized.data_ptr<at::BFloat16>(), a.size(0), a.size(1));
      } else {
        normalize_columns<<<blocks, 256, 0, stream>>>(
            a.data_ptr<at::BFloat16>(), normalized.data_ptr<at::BFloat16>(), a.size(0), a.size(1));
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      normalized = a.to(at::kBFloat16);
    }
    features = internal_linear_train_gemm(rows, normalized, false, true);
    internal_linear_train_gemm(features, b.to(at::kBFloat16), false, true,
                               scale, false, output, 1.0);
  }
  return {output, features, normalized};
}
