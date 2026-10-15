// SPDX-License-Identifier: MIT
#include <torch/extension.h>
#include "../../internal/linear/backend.cuh"

std::vector<at::Tensor> pretrain_tmix_wkv_prepare_forward_cuda(std::vector<at::Tensor> inputs, std::vector<at::Tensor> weights,
    std::vector<std::optional<at::Tensor>> adapters,
    std::vector<double> scales, std::vector<bool> use_nora);

std::vector<at::Tensor> pretrain_tmix_wkv_prepare_forward(std::vector<at::Tensor> inputs, std::vector<at::Tensor> weights,
    std::vector<std::optional<at::Tensor>> adapters,
    std::vector<double> scales, std::vector<bool> use_nora) {
  TORCH_CHECK(inputs.size() == 3 && weights.size() == 3 && adapters.size() == 6 &&
                  scales.size() == 3 && use_nora.size() == 3,
              "WKV Prepare requires three projection specifications");
  for (int i = 0; i < 3; ++i) {
    internal_linear_train_check(inputs[i], weights[i], adapters[2*i].value_or(at::Tensor()),
                                adapters[2*i+1].value_or(at::Tensor()), scales[i]);
  }
  return pretrain_tmix_wkv_prepare_forward_cuda(inputs, weights, adapters, scales, use_nora);
}

void register_pretrain_tmix_wkv_prepare_forward_bindings(py::module_& module) {
  module.def("pretrain_tmix_wkv_prepare_forward", &pretrain_tmix_wkv_prepare_forward);
}
