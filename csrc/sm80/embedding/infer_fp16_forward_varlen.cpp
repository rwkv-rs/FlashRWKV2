// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/Albatross
// Adapted from BlinkDL/Albatross commit ee3308f6922e59f2166c7fac3c5a192340a2b48e.

#include <torch/extension.h>

#include <cmath>

torch::Tensor embedding_ln0_forward_varlen_cuda(
    torch::Tensor embedding,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps);

namespace {

void check_bf16(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16, name, " must be bfloat16");
}

}  // namespace

torch::Tensor embedding_ln0_forward_varlen(
    torch::Tensor embedding,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps) {
  check_bf16(embedding, "embedding");
  check_bf16(weight, "weight");
  check_bf16(bias, "bias");
  TORCH_CHECK(embedding.dim() == 2 && embedding.size(0) > 0 && embedding.size(1) > 0,
              "embedding must have packed shape [rows,C]");
  TORCH_CHECK(weight.dim() == 1 && bias.dim() == 1 &&
                  weight.size(0) == embedding.size(1) && bias.size(0) == embedding.size(1),
              "weight and bias must have shape [C]");
  TORCH_CHECK(weight.device() == embedding.device() && bias.device() == embedding.device(),
              "embedding parameters must share the input device");
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "eps must be positive");
  return embedding_ln0_forward_varlen_cuda(embedding, weight, bias, eps);
}

void register_embedding_bindings(py::module_& module) {
  module.def("embedding_ln0_forward_varlen", &embedding_ln0_forward_varlen,
             py::arg("embedding"), py::arg("weight"), py::arg("bias"),
             py::arg("eps") = 1.0e-5);
}
