// SPDX-License-Identifier: Apache-2.0
// RWKV-LM train_temp revision: 952102498e9ed367ea0a59ee64106916d474d30f.

#include "validation.h"

#include <utility>
#include <vector>

std::vector<torch::stable::Tensor> pretrain_tmix_tokenshift_forward_cuda(
    torch::stable::Tensor x, torch::stable::Tensor x_r, torch::stable::Tensor x_w,
    torch::stable::Tensor x_k, torch::stable::Tensor x_v, torch::stable::Tensor x_a,
    torch::stable::Tensor x_g);

namespace {
void check_bf16_cuda(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                  tensor.scalar_type() == torch::headeronly::ScalarType::BFloat16,
              name, " must be contiguous CUDA bfloat16");
}
void check_inputs(const torch::stable::Tensor& x, const std::vector<std::pair<const torch::stable::Tensor*, const char*>>& mixes) {
  check_bf16_cuda(x, "x");
  STD_TORCH_CHECK(x.dim() == 3 && x.numel() > 0, "x must have shape [B,T,C]");
  for (const auto& [tensor, name] : mixes) {
    check_bf16_cuda(*tensor, name);
    STD_TORCH_CHECK(tensor->dim() == 1 && tensor->size(0) == x.size(2), name,
                " must have shape [C]");
    STD_TORCH_CHECK(tensor->device() == x.device(), name, " must share x's device");
  }
}
}  // namespace

auto pretrain_tmix_tokenshift_forward(
    torch::stable::Tensor x, torch::stable::Tensor x_r, torch::stable::Tensor x_w, torch::stable::Tensor x_k,
    torch::stable::Tensor x_v, torch::stable::Tensor x_a, torch::stable::Tensor x_g) {
  check_inputs(x, {{&x_r, "x_r"}, {&x_w, "x_w"}, {&x_k, "x_k"},
                   {&x_v, "x_v"}, {&x_a, "x_a"}, {&x_g, "x_g"}});
  return flashrwkv2::validation::tensor_tuple<6>(pretrain_tmix_tokenshift_forward_cuda(x, x_r, x_w, x_k, x_v, x_a, x_g));
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("pretrain_tmix_tokenshift_forward(Tensor x, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("pretrain_tmix_tokenshift_forward", TORCH_BOX(&pretrain_tmix_tokenshift_forward));
}
