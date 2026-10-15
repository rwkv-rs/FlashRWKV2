// SPDX-License-Identifier: MIT
#include <torch/extension.h>
#include "../../internal/linear/backend.cuh"

std::vector<at::Tensor> pretrain_tmix_wkv_prepare_backward_cuda(std::vector<at::Tensor> gradients, std::vector<at::Tensor> inputs, std::vector<at::Tensor> weights,
    std::vector<std::optional<at::Tensor>> adapters,
    std::vector<double> scales, std::vector<bool> use_nora,
    std::vector<at::Tensor> saved, std::vector<bool> needs);

std::vector<at::Tensor> pretrain_tmix_wkv_prepare_backward(std::vector<at::Tensor> gradients, std::vector<at::Tensor> inputs, std::vector<at::Tensor> weights,
    std::vector<std::optional<at::Tensor>> adapters,
    std::vector<double> scales, std::vector<bool> use_nora,
    std::vector<at::Tensor> saved, std::vector<bool> needs) {
  TORCH_CHECK(inputs.size() == 3 && weights.size() == 3 && adapters.size() == 6 &&
                  scales.size() == 3 && use_nora.size() == 3,
              "WKV Prepare requires three projection specifications");
  TORCH_CHECK(gradients.size() == 3 && saved.size() == 6 && needs.size() == 12,
              "invalid WKV Prepare backward specification");
  for (int i = 0; i < 3; ++i) {
    internal_linear_train_check(inputs[i], weights[i], adapters[2*i].value_or(at::Tensor()),
                                adapters[2*i+1].value_or(at::Tensor()), scales[i]);
    internal_linear_train_check_backward(gradients[i], inputs[i], weights[i]);
  }
  return pretrain_tmix_wkv_prepare_backward_cuda(gradients, inputs, weights, adapters, scales, use_nora, saved, needs);
}

void register_pretrain_tmix_wkv_prepare_backward_bindings(py::module_& module) {
  module.def("pretrain_tmix_wkv_prepare_backward", &pretrain_tmix_wkv_prepare_backward);
}
