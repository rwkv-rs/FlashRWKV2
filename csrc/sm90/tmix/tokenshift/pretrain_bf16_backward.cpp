// SPDX-License-Identifier: Apache-2.0
// RWKV-LM train_temp revision: 952102498e9ed367ea0a59ee64106916d474d30f.

#include "validation.h"

#include <utility>
#include <vector>

std::vector<torch::stable::Tensor> pretrain_tmix_tokenshift_backward_cuda(
    torch::stable::Tensor grad_r, torch::stable::Tensor grad_w, torch::stable::Tensor grad_k,
    torch::stable::Tensor grad_v, torch::stable::Tensor grad_a, torch::stable::Tensor grad_g,
    torch::stable::Tensor x, torch::stable::Tensor x_r, torch::stable::Tensor x_w, torch::stable::Tensor x_k,
    torch::stable::Tensor x_v, torch::stable::Tensor x_a, torch::stable::Tensor x_g);

namespace {
void check_bf16_cuda(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                  tensor.scalar_type() == torch::headeronly::ScalarType::BFloat16,
              name, " must be contiguous CUDA bfloat16");
}
}  // namespace

auto pretrain_tmix_tokenshift_backward(
    torch::stable::Tensor grad_r, torch::stable::Tensor grad_w, torch::stable::Tensor grad_k,
    torch::stable::Tensor grad_v, torch::stable::Tensor grad_a, torch::stable::Tensor grad_g,
    torch::stable::Tensor x, torch::stable::Tensor x_r, torch::stable::Tensor x_w, torch::stable::Tensor x_k,
    torch::stable::Tensor x_v, torch::stable::Tensor x_a, torch::stable::Tensor x_g) {
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{&grad_r, "grad_r"},
           {&grad_w, "grad_w"}, {&grad_k, "grad_k"}, {&grad_v, "grad_v"},
           {&grad_a, "grad_a"}, {&grad_g, "grad_g"}, {&x, "x"},
           {&x_r, "x_r"}, {&x_w, "x_w"}, {&x_k, "x_k"}, {&x_v, "x_v"},
           {&x_a, "x_a"}, {&x_g, "x_g"}}) {
    check_bf16_cuda(*item.first, item.second);
  }
  STD_TORCH_CHECK(x.dim() == 3 && grad_r.sizes() == x.sizes() &&
                  grad_w.sizes() == x.sizes() && grad_k.sizes() == x.sizes() &&
                  grad_v.sizes() == x.sizes() && grad_a.sizes() == x.sizes() &&
                  grad_g.sizes() == x.sizes(),
              "tokenshift gradient shapes must match x");
  STD_TORCH_CHECK(x.device() == grad_r.device() && x_r.device() == x.device() &&
                  x_w.device() == x.device() && x_k.device() == x.device() &&
                  x_v.device() == x.device() && x_a.device() == x.device() &&
                  x_g.device() == x.device(),
              "tokenshift tensors must share a device");
  return flashrwkv2::validation::tensor_tuple<7>(pretrain_tmix_tokenshift_backward_cuda(
      grad_r, grad_w, grad_k, grad_v, grad_a, grad_g, x, x_r, x_w, x_k, x_v,
      x_a, x_g));
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("pretrain_tmix_tokenshift_backward(Tensor grad_r, Tensor grad_w, Tensor grad_k, Tensor grad_v, Tensor grad_a, Tensor grad_g, Tensor x, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("pretrain_tmix_tokenshift_backward", TORCH_BOX(&pretrain_tmix_tokenshift_backward));
}
