// SPDX-License-Identifier: MIT
#include "backend.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {
template <typename scalar_t>
__global__ void normalize_columns_backward(
    const scalar_t* a, const float* grad_normalized, scalar_t* grad_a,
    int64_t rank, int64_t columns) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= columns) return;
  float squared = 0.0f, dot = 0.0f;
  for (int64_t row = 0; row < rank; ++row) {
    const int64_t index = row * columns + col;
    const float value = static_cast<float>(a[index]);
    squared += value * value;
    dot += value * grad_normalized[index];
  }
  const float norm = sqrtf(squared), inverse = 1.0f / (norm + 1.0e-6f);
  const float correction = norm > 0.0f ? dot * inverse * inverse / norm : 0.0f;
  for (int64_t row = 0; row < rank; ++row) {
    const int64_t index = row * columns + col;
    grad_a[index] = static_cast<scalar_t>(
        grad_normalized[index] * inverse - static_cast<float>(a[index]) * correction);
  }
}
}  // namespace

at::Tensor internal_linear_train_normalize_backward(at::Tensor a, at::Tensor grad) {
  auto da = at::empty_like(a);
  const int blocks = static_cast<int>((a.size(1) + 255) / 256);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (a.scalar_type() == at::kFloat) {
    normalize_columns_backward<<<blocks, 256, 0, stream>>>(
        a.data_ptr<float>(), grad.data_ptr<float>(), da.data_ptr<float>(), a.size(0), a.size(1));
  } else {
    normalize_columns_backward<<<blocks, 256, 0, stream>>>(
        a.data_ptr<at::BFloat16>(), grad.data_ptr<float>(), da.data_ptr<at::BFloat16>(), a.size(0), a.size(1));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return da;
}

std::vector<at::Tensor> internal_linear_train_backward(
    at::Tensor grad, at::Tensor x, at::Tensor weight,
    at::Tensor a, at::Tensor b, at::Tensor features, at::Tensor normalized_a,
    double scale, bool use_nora, bool need_x, bool need_weight,
    bool need_a, bool need_b) {
  const c10::cuda::CUDAGuard guard(x.device());
  auto rows = x.view({-1, x.size(-1)});
  auto dy = grad.view({rows.size(0), weight.size(0)});
  at::Tensor dx, dw, da, db;
  if (need_x) dx = internal_linear_train_gemm(dy, weight);
  if (need_weight) dw = internal_linear_train_gemm(dy, rows, true, false);
  if (a.defined()) {
    if (need_b) {
      db = internal_linear_train_gemm(dy, features, true, false, scale, true).to(b.scalar_type());
    }
    if (need_x || need_a) {
      auto dz = internal_linear_train_gemm(dy, b.to(at::kBFloat16), false, false, scale);
      if (need_x) internal_linear_train_gemm(dz, normalized_a, false, false, 1.0, false, dx, 1.0);
      if (need_a) {
        auto grad_normalized = internal_linear_train_gemm(dz, rows, true, false, 1.0, true);
        if (use_nora) {
          da = internal_linear_train_normalize_backward(a, grad_normalized);
        } else {
          da = grad_normalized.to(a.scalar_type());
        }
      }
    }
  }
  if (need_x) dx = dx.view(x.sizes());
  return {dx, dw, da, db};
}
