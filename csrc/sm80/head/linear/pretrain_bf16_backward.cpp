// SPDX-License-Identifier: MIT
#include <torch/extension.h>
#include "../../internal/linear/backend.cuh"

std::vector<at::Tensor> pretrain_head_linear_backward_cuda(
    at::Tensor grad, at::Tensor x, at::Tensor weight,
    std::optional<at::Tensor> a, std::optional<at::Tensor> b,
    at::Tensor features, at::Tensor normalized_a, double scale, bool use_nora,
    std::vector<bool> needs);

std::vector<at::Tensor> pretrain_head_linear_backward(
    at::Tensor grad, at::Tensor x, at::Tensor weight,
    std::optional<at::Tensor> a, std::optional<at::Tensor> b,
    at::Tensor features, at::Tensor normalized_a, double scale, bool use_nora,
    std::vector<bool> needs) {
  internal_linear_train_check_backward(grad, x, weight);
  TORCH_CHECK(needs.size() == 4, "Head projection requires four gradient flags");
  return pretrain_head_linear_backward_cuda(
      grad, x, weight, a, b, features, normalized_a, scale, use_nora, needs);
}

void register_pretrain_head_linear_backward_bindings(py::module_& module) {
  module.def("pretrain_head_linear_backward", &pretrain_head_linear_backward,
      py::arg("grad_output"), py::arg("x"), py::arg("weight"),
      py::arg("lora_a"), py::arg("lora_b"), py::arg("features"),
      py::arg("normalized_a"), py::arg("lora_scale"), py::arg("use_nora"), py::arg("needs"));
}
