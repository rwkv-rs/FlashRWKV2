// SPDX-License-Identifier: MIT
#include "../../internal/linear/backend.cuh"
#include <optional>

std::vector<at::Tensor> pretrain_tmix_wkv_prepare_forward_cuda(std::vector<at::Tensor> inputs, std::vector<at::Tensor> weights,
    std::vector<std::optional<at::Tensor>> adapters,
    std::vector<double> scales, std::vector<bool> use_nora) {
  std::vector<at::Tensor> outputs, saved;
  for (int i = 0; i < 3; ++i) {
    auto projection = internal_linear_train_forward(
        inputs[i], weights[i], adapters[2*i].value_or(at::Tensor()),
        adapters[2*i+1].value_or(at::Tensor()), scales[i], use_nora[i]);
    outputs.push_back(projection[0].view(inputs[i].sizes()));
    saved.insert(saved.end(), {projection[1], projection[2]});
  }
  outputs.insert(outputs.end(), saved.begin(), saved.end());
  return outputs;
}
