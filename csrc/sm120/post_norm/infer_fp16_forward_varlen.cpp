// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Adapted from BlinkDL/Albatross commit ee3308f6922e59f2166c7fac3c5a192340a2b48e.
// This binding file exposes only the final-output PostNorm island.  Internal
// PostNorm launchers are linked directly by their TMix and CMix owners.

#include <torch/extension.h>

#include <cmath>
torch::Tensor post_norm_output_forward_varlen_cuda(
    torch::Tensor x,
    torch::Tensor res,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps);
namespace {

void check_half(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, name, " must be float16");
}

void check_rows(const torch::Tensor& tensor, const torch::Tensor& reference, const char* name) {
  check_half(tensor, name);
  TORCH_CHECK(tensor.device() == reference.device(), name, " must share the input device");
  TORCH_CHECK(tensor.sizes() == reference.sizes(), name, " shape mismatch");
}

void check_affine(const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& bias) {
  check_half(weight, "weight");
  check_half(bias, "bias");
  TORCH_CHECK(weight.device() == x.device() && bias.device() == x.device(),
              "LN parameters must share x's device");
  TORCH_CHECK(weight.dim() == 1 && bias.dim() == 1 &&
                  weight.size(0) == x.size(-1) && bias.size(0) == x.size(-1),
              "weight and bias must have shape [C]");
}

void check_eps(double eps) {
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be finite and positive");
}

}  // namespace

torch::Tensor post_norm_output_forward_varlen(
    torch::Tensor x,
    torch::Tensor res,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps) {
  check_half(x, "x");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0,
              "x must have packed shape [total_tokens,C]");
  check_rows(res, x, "res");
  check_affine(x, weight, bias);
  check_eps(eps);
  return post_norm_output_forward_varlen_cuda(x, res, weight, bias, eps);
}

void register_post_norm_bindings(py::module_& module) {
  module.def("post_norm_output_forward_varlen",
             &post_norm_output_forward_varlen,
             py::arg("x"), py::arg("res"), py::arg("weight"), py::arg("bias"),
             py::arg("eps") = 1.0e-5);
}
