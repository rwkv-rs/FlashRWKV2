// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Adapted from BlinkDL/Albatross commit ee3308f6922e59f2166c7fac3c5a192340a2b48e.

#include <torch/extension.h>

torch::Tensor head_linear_all_forward_varlen_cuda(
    torch::Tensor x, torch::Tensor weight);
torch::Tensor head_linear_last_forward_varlen_cuda(
    torch::Tensor x, torch::Tensor weight, int64_t tokens_count);

namespace {

void check_half(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, name, " must be float16");
}

}  // namespace

torch::Tensor head_linear_all_forward_varlen(
    torch::Tensor x, torch::Tensor weight) {
  check_half(x, "x");
  check_half(weight, "weight");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0 &&
                  weight.dim() == 2 && weight.size(0) > 0 &&
                  x.size(1) == weight.size(1),
              "head all-logits linear expects x [rows,C] and weight [V,C]");
  TORCH_CHECK(weight.device() == x.device(), "weight must share x's device");
  return head_linear_all_forward_varlen_cuda(x, weight);
}

torch::Tensor head_linear_last_forward_varlen(
    torch::Tensor x, torch::Tensor weight, int64_t tokens_count) {
  check_half(x, "x");
  check_half(weight, "weight");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0 &&
                  weight.dim() == 2 && weight.size(0) > 0 &&
                  x.size(1) == weight.size(1),
              "head last-logits linear expects x [batch,C] and weight [V,C]");
  TORCH_CHECK(tokens_count > 0, "tokens_count must be positive");
  TORCH_CHECK(weight.device() == x.device(), "weight must share x's device");
  return head_linear_last_forward_varlen_cuda(x, weight, tokens_count);
}

void register_head_linear_bindings(py::module_& module) {
  module.def("head_linear_all_forward_varlen", &head_linear_all_forward_varlen,
             py::arg("x"), py::arg("weight"));
  module.def("head_linear_last_forward_varlen",
             &head_linear_last_forward_varlen,
             py::arg("x"), py::arg("weight"), py::arg("tokens_count"));
}
