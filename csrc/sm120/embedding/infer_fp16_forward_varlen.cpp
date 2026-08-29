// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Adapted from BlinkDL/Albatross commit ee3308f6922e59f2166c7fac3c5a192340a2b48e.

#include "validation.h"

#include <cmath>

torch::stable::Tensor embedding_ln0_forward_varlen_cuda(
    torch::stable::Tensor embedding,
    torch::stable::Tensor weight,
    torch::stable::Tensor bias,
    double eps);

namespace {

void check_bf16(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  STD_TORCH_CHECK(tensor.scalar_type() == torch::headeronly::ScalarType::BFloat16, name, " must be bfloat16");
}

}  // namespace

torch::stable::Tensor embedding_ln0_forward_varlen(
    torch::stable::Tensor embedding,
    torch::stable::Tensor weight,
    torch::stable::Tensor bias,
    double eps) {
  check_bf16(embedding, "embedding");
  check_bf16(weight, "weight");
  check_bf16(bias, "bias");
  STD_TORCH_CHECK(embedding.dim() == 2 && embedding.size(0) > 0 && embedding.size(1) > 0,
              "embedding must have packed shape [rows,C]");
  STD_TORCH_CHECK(weight.dim() == 1 && bias.dim() == 1 &&
                  weight.size(0) == embedding.size(1) && bias.size(0) == embedding.size(1),
              "weight and bias must have shape [C]");
  STD_TORCH_CHECK(weight.device() == embedding.device() && bias.device() == embedding.device(),
              "embedding parameters must share the input device");
  STD_TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be positive");
  return embedding_ln0_forward_varlen_cuda(embedding, weight, bias, eps);
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("embedding_ln0_forward_varlen(Tensor embedding, Tensor weight, Tensor bias, float eps) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("embedding_ln0_forward_varlen", TORCH_BOX(&embedding_ln0_forward_varlen));
}
