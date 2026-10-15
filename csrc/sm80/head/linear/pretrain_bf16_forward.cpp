// SPDX-License-Identifier: MIT
#include <torch/extension.h>
#include "../../internal/linear/backend.cuh"

std::vector<at::Tensor> pretrain_head_linear_forward_cuda(
    at::Tensor x, at::Tensor weight, std::optional<at::Tensor> a,
    std::optional<at::Tensor> b, double scale, bool use_nora);

std::vector<at::Tensor> pretrain_head_linear_forward(
    at::Tensor x, at::Tensor weight, std::optional<at::Tensor> a,
    std::optional<at::Tensor> b, double scale, bool use_nora) {
  internal_linear_train_check(x, weight, a.value_or(at::Tensor()), b.value_or(at::Tensor()), scale);
  return pretrain_head_linear_forward_cuda(x, weight, a, b, scale, use_nora);
}

void register_pretrain_head_linear_forward_bindings(py::module_& module) {
  module.def("pretrain_head_linear_forward", &pretrain_head_linear_forward,
      py::arg("x"), py::arg("weight"), py::arg("lora_a") = py::none(),
      py::arg("lora_b") = py::none(), py::arg("lora_scale") = 1.0,
      py::arg("use_nora") = false);
}
