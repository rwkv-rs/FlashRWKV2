// SPDX-License-Identifier: Apache-2.0
// RWKV-LM train_temp revision: 952102498e9ed367ea0a59ee64106916d474d30f.

#include "validation.h"

#include <utility>
#include <vector>

std::vector<torch::stable::Tensor> pretrain_tmix_vres_gate_backward_cuda(
    torch::stable::Tensor grad_output, torch::stable::Tensor value, torch::stable::Tensor first_value,
    torch::stable::Tensor v0, torch::stable::Tensor v12);

namespace {
void check_bf16_cuda(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                  tensor.scalar_type() == torch::headeronly::ScalarType::BFloat16,
              name, " must be contiguous CUDA bfloat16");
}
}  // namespace

auto pretrain_tmix_vres_gate_backward(
    torch::stable::Tensor grad_output, torch::stable::Tensor value, torch::stable::Tensor first_value,
    torch::stable::Tensor v0, torch::stable::Tensor v12) {
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{&grad_output, "grad_output"},
           {&value, "value"}, {&first_value, "first_value"}, {&v0, "v0"},
           {&v12, "v12"}}) {
    check_bf16_cuda(*item.first, item.second);
  }
  STD_TORCH_CHECK(value.dim() == 3 && first_value.sizes() == value.sizes() &&
                  v12.sizes() == value.sizes() && v0.dim() == 1 &&
                  v0.size(0) == value.size(2) &&
                  grad_output.sizes() == value.sizes(),
              "invalid v-residual gate shapes");
  STD_TORCH_CHECK(grad_output.device() == value.device() && first_value.device() == value.device() &&
                  v0.device() == value.device() && v12.device() == value.device(),
              "v-residual gate tensors must share a device");
  return flashrwkv2::validation::tensor_tuple<4>(pretrain_tmix_vres_gate_backward_cuda(
      grad_output, value, first_value, v0, v12));
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("pretrain_tmix_vres_gate_backward(Tensor grad_output, Tensor value, Tensor first_value, Tensor v0, Tensor v12) -> (Tensor, Tensor, Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("pretrain_tmix_vres_gate_backward", TORCH_BOX(&pretrain_tmix_vres_gate_backward));
}
