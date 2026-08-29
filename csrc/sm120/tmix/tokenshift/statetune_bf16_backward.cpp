// SPDX-License-Identifier: Apache-2.0
// RWKV-LM train_temp revision: 952102498e9ed367ea0a59ee64106916d474d30f.
// Local adaptation: propagate initial/next shift gradients across chunks.

#include "validation.h"

#include <cstdint>
#include <utility>
#include <vector>

std::vector<torch::stable::Tensor> statetune_tmix_tokenshift_backward_cuda(
    torch::stable::Tensor grad_r, torch::stable::Tensor grad_w, torch::stable::Tensor grad_k,
    torch::stable::Tensor grad_v, torch::stable::Tensor grad_a, torch::stable::Tensor grad_g,
    torch::stable::Tensor grad_next_shift, torch::stable::Tensor x,
    torch::stable::Tensor initial_shift, torch::stable::Tensor x_r, torch::stable::Tensor x_w,
    torch::stable::Tensor x_k, torch::stable::Tensor x_v, torch::stable::Tensor x_a,
    torch::stable::Tensor x_g);

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

auto statetune_tmix_tokenshift_backward(
    torch::stable::Tensor grad_r, torch::stable::Tensor grad_w, torch::stable::Tensor grad_k,
    torch::stable::Tensor grad_v, torch::stable::Tensor grad_a, torch::stable::Tensor grad_g,
    torch::stable::Tensor grad_next_shift, torch::stable::Tensor x,
    torch::stable::Tensor initial_shift, torch::stable::Tensor x_r, torch::stable::Tensor x_w,
    torch::stable::Tensor x_k, torch::stable::Tensor x_v, torch::stable::Tensor x_a,
    torch::stable::Tensor x_g) {
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{&grad_r, "grad_r"},
           {&grad_w, "grad_w"}, {&grad_k, "grad_k"},
           {&grad_v, "grad_v"}, {&grad_a, "grad_a"},
           {&grad_g, "grad_g"}, {&grad_next_shift, "grad_next_shift"},
           {&x, "x"}, {&initial_shift, "initial_shift"}, {&x_r, "x_r"},
           {&x_w, "x_w"}, {&x_k, "x_k"}, {&x_v, "x_v"},
           {&x_a, "x_a"}, {&x_g, "x_g"}}) {
    check_bf16_cuda(*item.first, item.second);
    STD_TORCH_CHECK(item.first->device() == x.device(), item.second,
                " must share x's device");
  }
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{&grad_r, "grad_r"},
           {&grad_w, "grad_w"}, {&grad_k, "grad_k"},
           {&grad_v, "grad_v"}, {&grad_a, "grad_a"},
           {&grad_g, "grad_g"}}) {
    STD_TORCH_CHECK(item.first->sizes() == x.sizes(), item.second,
                " must have the same shape as x");
  }
  STD_TORCH_CHECK(x.dim() == 3 && x.size(0) > 0 && x.size(1) > 0 &&
                  x.size(2) > 0 && x.size(2) % 2 == 0,
              "x must have non-empty shape [B,T,C] with even C");
  STD_TORCH_CHECK(initial_shift.sizes() ==
                  torch::headeronly::IntHeaderOnlyArrayRef({x.size(0), x.size(2)}),
              "initial_shift must have shape [B,C]");
  STD_TORCH_CHECK(grad_next_shift.sizes() == initial_shift.sizes(),
              "grad_next_shift must have shape [B,C]");
  for (const auto* tensor : {&x_r, &x_w, &x_k, &x_v, &x_a, &x_g}) {
    STD_TORCH_CHECK(tensor->sizes() == torch::headeronly::IntHeaderOnlyArrayRef({x.size(2)}),
                "tokenshift coefficients must have shape [C]");
  }
  return flashrwkv2::validation::tensor_tuple<8>(statetune_tmix_tokenshift_backward_cuda(
      grad_r, grad_w, grad_k, grad_v, grad_a, grad_g, grad_next_shift, x,
      initial_shift, x_r, x_w, x_k, x_v, x_a, x_g));
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("statetune_tmix_tokenshift_backward(Tensor grad_r, Tensor grad_w, Tensor grad_k, Tensor grad_v, Tensor grad_a, Tensor grad_g, Tensor grad_next_shift, Tensor x, Tensor initial_shift, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("statetune_tmix_tokenshift_backward", TORCH_BOX(&statetune_tmix_tokenshift_backward));
}
