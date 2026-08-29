// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Adapted from BlinkDL/Albatross commit ee3308f6922e59f2166c7fac3c5a192340a2b48e.

#include "validation.h"

torch::stable::Tensor head_linear_all_forward_varlen_cuda(
    torch::stable::Tensor x, torch::stable::Tensor weight);
torch::stable::Tensor head_linear_last_forward_varlen_cuda(
    torch::stable::Tensor x, torch::stable::Tensor weight, int64_t tokens_count);

namespace {

void check_half(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  STD_TORCH_CHECK(tensor.scalar_type() == torch::headeronly::ScalarType::Half, name, " must be float16");
}

}  // namespace

torch::stable::Tensor head_linear_all_forward_varlen(
    torch::stable::Tensor x, torch::stable::Tensor weight) {
  check_half(x, "x");
  check_half(weight, "weight");
  STD_TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0 &&
                  weight.dim() == 2 && weight.size(0) > 0 &&
                  x.size(1) == weight.size(1),
              "head all-logits linear expects x [rows,C] and weight [V,C]");
  STD_TORCH_CHECK(weight.device() == x.device(), "weight must share x's device");
  return head_linear_all_forward_varlen_cuda(x, weight);
}

torch::stable::Tensor head_linear_last_forward_varlen(
    torch::stable::Tensor x, torch::stable::Tensor weight, int64_t tokens_count) {
  check_half(x, "x");
  check_half(weight, "weight");
  STD_TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0 &&
                  weight.dim() == 2 && weight.size(0) > 0 &&
                  x.size(1) == weight.size(1),
              "head last-logits linear expects x [batch,C] and weight [V,C]");
  STD_TORCH_CHECK(tokens_count > 0, "tokens_count must be positive");
  STD_TORCH_CHECK(weight.device() == x.device(), "weight must share x's device");
  return head_linear_last_forward_varlen_cuda(x, weight, tokens_count);
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("head_linear_all_forward_varlen(Tensor x, Tensor weight) -> Tensor");
  module.def("head_linear_last_forward_varlen(Tensor x, Tensor weight, int tokens_count) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("head_linear_all_forward_varlen", TORCH_BOX(&head_linear_all_forward_varlen));
  module.impl("head_linear_last_forward_varlen", TORCH_BOX(&head_linear_last_forward_varlen));
}
