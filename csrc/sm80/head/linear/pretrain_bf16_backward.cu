// SPDX-License-Identifier: MIT
#include "../../internal/linear/backend.cuh"
#include <optional>

std::vector<at::Tensor> pretrain_head_linear_backward_cuda(
    at::Tensor grad, at::Tensor x, at::Tensor weight,
    std::optional<at::Tensor> a, std::optional<at::Tensor> b,
    at::Tensor features, at::Tensor normalized_a, double scale, bool use_nora,
    std::vector<bool> needs) {
  return internal_linear_train_backward(
      grad, x, weight, a.value_or(at::Tensor()), b.value_or(at::Tensor()),
      features, normalized_a, scale, use_nora, needs[0], needs[1], needs[2], needs[3]);
}
