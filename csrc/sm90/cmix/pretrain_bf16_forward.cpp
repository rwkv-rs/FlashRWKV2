// SPDX-License-Identifier: Apache-2.0
// Full ChannelMix operator from RWKV-LM train_temp revision 952102498e9ed367ea0a59ee64106916d474d30f.

#include "validation.h"

#include <utility>
#include <vector>

std::vector<torch::stable::Tensor> pretrain_cmix_forward_cuda(
    torch::stable::Tensor x, torch::stable::Tensor x_k, torch::stable::Tensor key_weight,
    torch::stable::Tensor value_weight);

namespace {
void check_bf16_cuda(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                  tensor.scalar_type() == torch::headeronly::ScalarType::BFloat16,
              name, " must be contiguous CUDA bfloat16");
}
void check_inputs(const torch::stable::Tensor& x, const torch::stable::Tensor& x_k,
                  const torch::stable::Tensor& key_weight, const torch::stable::Tensor& value_weight) {
  check_bf16_cuda(x, "x"); check_bf16_cuda(x_k, "x_k");
  check_bf16_cuda(key_weight, "key_weight"); check_bf16_cuda(value_weight, "value_weight");
  STD_TORCH_CHECK(x.dim() == 3 && x.numel() > 0 && x_k.dim() == 1 &&
                  x_k.size(0) == x.size(2) && key_weight.dim() == 2 &&
                  key_weight.size(0) == 4 * x.size(2) &&
                  key_weight.size(1) == x.size(2) && value_weight.dim() == 2 &&
                  value_weight.size(0) == x.size(2) &&
                  value_weight.size(1) == 4 * x.size(2),
              "invalid CMix training shapes");
  STD_TORCH_CHECK(x.device() == x_k.device() && x.device() == key_weight.device() &&
                  x.device() == value_weight.device(),
              "CMix training tensors must share a device");
}
}  // namespace

auto pretrain_cmix_forward(
    torch::stable::Tensor x, torch::stable::Tensor x_k, torch::stable::Tensor key_weight,
    torch::stable::Tensor value_weight) {
  check_inputs(x, x_k, key_weight, value_weight);
  return flashrwkv2::validation::tensor_tuple<3>(pretrain_cmix_forward_cuda(x, x_k, key_weight, value_weight));
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("pretrain_cmix_forward(Tensor x, Tensor x_k, Tensor key_weight, Tensor value_weight) -> (Tensor, Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("pretrain_cmix_forward", TORCH_BOX(&pretrain_cmix_forward));
}
