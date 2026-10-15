// SPDX-License-Identifier: MIT
#include "../../internal/linear/backend.cuh"
#include <optional>

std::vector<at::Tensor> pretrain_head_linear_forward_cuda(
    at::Tensor x, at::Tensor weight, std::optional<at::Tensor> a,
    std::optional<at::Tensor> b, double scale, bool use_nora) {
  auto result = internal_linear_train_forward(
      x, weight, a.value_or(at::Tensor()), b.value_or(at::Tensor()), scale, use_nora);
  auto shape = x.sizes().vec();
  shape.back() = weight.size(0);
  result[0] = result[0].view(shape);
  return result;
}
