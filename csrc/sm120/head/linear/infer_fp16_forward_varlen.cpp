// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Adapted from BlinkDL/Albatross commit ee3308f6922e59f2166c7fac3c5a192340a2b48e.

#include <torch/extension.h>

#include <cmath>

torch::Tensor head_linear_forward_varlen_cuda(torch::Tensor x, torch::Tensor weight);
torch::Tensor head_linear_all_forward_varlen_cuda(
    torch::Tensor x, torch::Tensor weight);
torch::Tensor head_linear_last_forward_varlen_cuda(
    torch::Tensor x, torch::Tensor weight, int64_t tokens_count);
torch::Tensor head_last_norm_forward_varlen_cuda(
    torch::Tensor x, torch::Tensor residual, torch::Tensor indices,
    torch::Tensor weight, torch::Tensor bias, double eps);

namespace {

void check_half(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, name, " must be float16");
}

}  // namespace

torch::Tensor head_linear_forward_varlen(torch::Tensor x, torch::Tensor weight) {
  check_half(x, "x");
  check_half(weight, "weight");
  TORCH_CHECK(x.dim() == 2 && weight.dim() == 2 && x.size(1) == weight.size(1),
              "head linear expects x [rows,C] and weight [V,C]");
  TORCH_CHECK(weight.device() == x.device(), "weight must share x's device");
  return head_linear_forward_varlen_cuda(x, weight);
}

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

torch::Tensor head_last_norm_forward_varlen(
    torch::Tensor x, torch::Tensor residual, torch::Tensor last_indices,
    torch::Tensor weight, torch::Tensor bias, double eps) {
  check_half(x, "x");
  check_half(residual, "residual");
  check_half(weight, "weight");
  check_half(bias, "bias");
  TORCH_CHECK(x.dim() == 2 && residual.sizes() == x.sizes(),
              "x and residual must have packed shape [rows,C]");
  TORCH_CHECK(last_indices.is_cuda() && last_indices.is_contiguous() &&
                  last_indices.scalar_type() == torch::kInt64 && last_indices.dim() == 1,
              "last_indices must be contiguous CUDA int64 [batch]");
  TORCH_CHECK(weight.dim() == 1 && bias.dim() == 1 &&
                  weight.size(0) == x.size(1) && bias.size(0) == x.size(1),
              "weight and bias must have shape [C]");
  TORCH_CHECK(last_indices.size(0) > 0, "last_indices must not be empty");
  TORCH_CHECK((x.size(1) % 2) == 0,
              "head last-layer norm requires an even channel count");
  TORCH_CHECK(residual.device() == x.device() && weight.device() == x.device() &&
                  bias.device() == x.device() && last_indices.device() == x.device(),
              "head tensors must share x's device");
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be positive");
  return head_last_norm_forward_varlen_cuda(
      x, residual, last_indices, weight, bias, eps);
}

void register_head_linear_bindings(py::module_& module) {
  module.def("head_linear_forward_varlen", &head_linear_forward_varlen,
             py::arg("x"), py::arg("weight"));
  module.def("head_linear_all_forward_varlen", &head_linear_all_forward_varlen,
             py::arg("x"), py::arg("weight"));
  module.def("head_linear_last_forward_varlen",
             &head_linear_last_forward_varlen,
             py::arg("x"), py::arg("weight"), py::arg("tokens_count"));
  module.def("head_last_norm_forward_varlen", &head_last_norm_forward_varlen,
             py::arg("x"), py::arg("residual"), py::arg("last_indices"),
             py::arg("weight"), py::arg("bias"), py::arg("eps") = 1.0e-5);
}
