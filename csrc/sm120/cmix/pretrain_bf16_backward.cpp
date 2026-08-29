// SPDX-License-Identifier: Apache-2.0
// Full ChannelMix operator from RWKV-LM train_temp revision 952102498e9ed367ea0a59ee64106916d474d30f.

#include "validation.h"

#include <utility>
#include <vector>

std::vector<torch::stable::Tensor> pretrain_cmix_backward_cuda(
    torch::stable::Tensor grad_output, torch::stable::Tensor x, torch::stable::Tensor x_k,
    torch::stable::Tensor key_weight, torch::stable::Tensor value_weight,
    torch::stable::Tensor mixed, torch::stable::Tensor preact);

namespace {
void check_bf16_cuda(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                  (tensor.scalar_type() == torch::headeronly::ScalarType::BFloat16 || tensor.scalar_type() == torch::headeronly::ScalarType::Float),
              name, " must be contiguous CUDA bf16/fp32");
}
}  // namespace

auto pretrain_cmix_backward(
    torch::stable::Tensor grad_output, torch::stable::Tensor x, torch::stable::Tensor x_k,
    torch::stable::Tensor key_weight, torch::stable::Tensor value_weight,
    torch::stable::Tensor mixed, torch::stable::Tensor preact) {
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{&grad_output, "grad_output"},
           {&x, "x"}, {&x_k, "x_k"}, {&key_weight, "key_weight"},
           {&value_weight, "value_weight"}, {&mixed, "mixed"}, {&preact, "preact"}}) {
    check_bf16_cuda(*item.first, item.second);
  }
  STD_TORCH_CHECK(grad_output.scalar_type() == torch::headeronly::ScalarType::BFloat16 && x.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
                  x_k.scalar_type() == torch::headeronly::ScalarType::BFloat16 && key_weight.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
                  value_weight.scalar_type() == torch::headeronly::ScalarType::BFloat16 && mixed.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
                  preact.scalar_type() == torch::headeronly::ScalarType::BFloat16,
              "CMix training inputs must be bf16");
  STD_TORCH_CHECK(grad_output.sizes() == x.sizes() && mixed.sizes() == x.sizes() &&
                  preact.dim() == 2 && preact.size(0) == x.size(0) * x.size(1) &&
                  preact.size(1) == 4 * x.size(2),
              "invalid CMix training gradient shapes");
  return flashrwkv2::validation::tensor_tuple<4>(pretrain_cmix_backward_cuda(grad_output, x, x_k, key_weight,
                                     value_weight, mixed, preact));
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("pretrain_cmix_backward(Tensor grad_output, Tensor x, Tensor x_k, Tensor key_weight, Tensor value_weight, Tensor mixed, Tensor preact) -> (Tensor, Tensor, Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("pretrain_cmix_backward", TORCH_BOX(&pretrain_cmix_backward));
}
