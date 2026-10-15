// SPDX-License-Identifier: MIT
#include "backend.cuh"

void internal_linear_train_check_backward(
    const at::Tensor& grad, const at::Tensor& x, const at::Tensor& weight) {
  TORCH_CHECK(grad.is_cuda() && grad.is_contiguous() &&
                  grad.scalar_type() == at::kBFloat16 && grad.device() == x.device() &&
                  grad.numel() == x.numel() / x.size(-1) * weight.size(0) &&
                  grad.size(-1) == weight.size(0),
              "projection gradient must be contiguous CUDA bfloat16 matching output");
}
