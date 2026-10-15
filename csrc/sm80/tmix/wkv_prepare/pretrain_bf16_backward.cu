// SPDX-License-Identifier: MIT
#include "../../internal/linear/backend.cuh"
#include <optional>

std::vector<at::Tensor> pretrain_tmix_wkv_prepare_backward_cuda(std::vector<at::Tensor> gradients, std::vector<at::Tensor> inputs, std::vector<at::Tensor> weights,
    std::vector<std::optional<at::Tensor>> adapters,
    std::vector<double> scales, std::vector<bool> use_nora,
    std::vector<at::Tensor> saved, std::vector<bool> needs) {
  std::vector<at::Tensor> outputs(12);
  for (int i = 0; i < 3; ++i) {
    auto projection = internal_linear_train_backward(
        gradients[i], inputs[i], weights[i], adapters[2*i].value_or(at::Tensor()),
        adapters[2*i+1].value_or(at::Tensor()), saved[2*i], saved[2*i+1], scales[i], use_nora[i],
        needs[i], needs[3+i], needs[6+2*i], needs[7+2*i]);
    outputs[i] = projection[0];
    outputs[3+i] = projection[1];
    outputs[6+2*i] = projection[2];
    outputs[7+2*i] = projection[3];
  }
  return outputs;
}
