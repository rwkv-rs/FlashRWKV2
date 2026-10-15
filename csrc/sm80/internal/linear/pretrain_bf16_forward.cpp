// SPDX-License-Identifier: MIT
#include "backend.cuh"

#include <cmath>
#include <limits>

void internal_linear_train_check(
    const at::Tensor& x, const at::Tensor& weight,
    const at::Tensor& a, const at::Tensor& b, double scale) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() >= 2 &&
                  x.numel() > 0 && x.scalar_type() == at::kBFloat16,
              "projection input must be non-empty contiguous CUDA bfloat16");
  TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.dim() == 2 &&
                  weight.scalar_type() == x.scalar_type() &&
                  weight.device() == x.device() && weight.size(1) == x.size(-1),
              "projection weight must be CUDA bfloat16 [O,K] matching input");
  internal_linear_train_check_adapter(x, weight, a, b, scale);
}

void internal_linear_train_check_adapter(
    const at::Tensor& x, const at::Tensor& weight,
    const at::Tensor& a, const at::Tensor& b, double scale) {
  TORCH_CHECK(a.defined() == b.defined(), "lora_a and lora_b must be provided together");
  TORCH_CHECK(std::isfinite(scale) && std::abs(scale) <= std::numeric_limits<float>::max(),
              "lora_scale must be finite and representable as float32");
  if (!a.defined()) return;
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.is_contiguous() && b.is_contiguous() &&
                  a.device() == x.device() && b.device() == x.device() &&
                  a.scalar_type() == b.scalar_type() &&
                  (a.scalar_type() == at::kBFloat16 || a.scalar_type() == at::kFloat),
              "adapter tensors must be contiguous CUDA bfloat16 or float32 on the input device");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(0) > 0 &&
                  a.size(1) == weight.size(1) && b.size(0) == weight.size(0) &&
                  b.size(1) == a.size(0),
              "adapter shapes must be [R,K] and [O,R]");
}
