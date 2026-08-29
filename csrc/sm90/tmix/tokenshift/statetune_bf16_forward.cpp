// SPDX-License-Identifier: Apache-2.0
// RWKV-LM train_temp revision: 952102498e9ed367ea0a59ee64106916d474d30f.
// Local adaptation: accept a nonzero chunk-boundary shift and return next shift.

#include "validation.h"

#include <cstdint>
#include <utility>
#include <vector>

std::vector<torch::stable::Tensor> statetune_tmix_tokenshift_forward_cuda(
    torch::stable::Tensor x, torch::stable::Tensor initial_shift, torch::stable::Tensor x_r,
    torch::stable::Tensor x_w, torch::stable::Tensor x_k, torch::stable::Tensor x_v,
    torch::stable::Tensor x_a, torch::stable::Tensor x_g);

namespace {
void check_bf16_cuda(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  STD_TORCH_CHECK(tensor.scalar_type() == torch::headeronly::ScalarType::BFloat16,
              name, " must have dtype torch.bfloat16");
  STD_TORCH_CHECK(reinterpret_cast<std::uintptr_t>(tensor.mutable_data_ptr()) % 4 == 0,
              name, " must be 4-byte aligned for BF16 vec2 access");
}
}  // namespace

auto statetune_tmix_tokenshift_forward(
    torch::stable::Tensor x, torch::stable::Tensor initial_shift, torch::stable::Tensor x_r,
    torch::stable::Tensor x_w, torch::stable::Tensor x_k, torch::stable::Tensor x_v,
    torch::stable::Tensor x_a, torch::stable::Tensor x_g) {
  check_bf16_cuda(x, "x");
  check_bf16_cuda(initial_shift, "initial_shift");
  STD_TORCH_CHECK(x.dim() == 3 && x.size(0) > 0 && x.size(1) > 0 &&
                  x.size(2) > 0,
              "x must have non-empty shape [B,T,C]");
  STD_TORCH_CHECK(x.size(2) % 2 == 0,
              "x channel dimension C must be divisible by 2");
  STD_TORCH_CHECK(initial_shift.sizes() ==
                  torch::headeronly::IntHeaderOnlyArrayRef({x.size(0), x.size(2)}),
              "initial_shift must have shape [B,C]");
  STD_TORCH_CHECK(initial_shift.device() == x.device(),
              "initial_shift must share x's device");
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{&x_r, "x_r"},
           {&x_w, "x_w"}, {&x_k, "x_k"}, {&x_v, "x_v"},
           {&x_a, "x_a"}, {&x_g, "x_g"}}) {
    check_bf16_cuda(*item.first, item.second);
    STD_TORCH_CHECK(item.first->sizes() == torch::headeronly::IntHeaderOnlyArrayRef({x.size(2)}),
                item.second, " must have shape [C]");
    STD_TORCH_CHECK(item.first->device() == x.device(), item.second,
                " must share x's device");
  }
  return flashrwkv2::validation::tensor_tuple<7>(statetune_tmix_tokenshift_forward_cuda(
      x, initial_shift, x_r, x_w, x_k, x_v, x_a, x_g));
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("statetune_tmix_tokenshift_forward(Tensor x, Tensor initial_shift, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("statetune_tmix_tokenshift_forward", TORCH_BOX(&statetune_tmix_tokenshift_forward));
}
