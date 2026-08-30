// SPDX-License-Identifier: Apache-2.0
// RWKV-LM train_temp revision: 952102498e9ed367ea0a59ee64106916d474d30f.
// Local adaptation: propagate initial/next shift gradients across chunks.

#include <torch/extension.h>

#include <cstdint>
#include <utility>
#include <vector>

std::vector<torch::Tensor> statetune_tmix_tokenshift_backward_cuda(
    torch::Tensor grad_r, torch::Tensor grad_w, torch::Tensor grad_k,
    torch::Tensor grad_v, torch::Tensor grad_a, torch::Tensor grad_g,
    torch::Tensor grad_next_shift, torch::Tensor x,
    torch::Tensor initial_shift, torch::Tensor x_r, torch::Tensor x_w,
    torch::Tensor x_k, torch::Tensor x_v, torch::Tensor x_a,
    torch::Tensor x_g);

namespace {
void check_bf16_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
  TORCH_CHECK(reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) % 4 == 0,
              name, " must be 4-byte aligned for BF16 vec2 access");
}
}  // namespace

std::vector<torch::Tensor> statetune_tmix_tokenshift_backward(
    torch::Tensor grad_r, torch::Tensor grad_w, torch::Tensor grad_k,
    torch::Tensor grad_v, torch::Tensor grad_a, torch::Tensor grad_g,
    torch::Tensor grad_next_shift, torch::Tensor x,
    torch::Tensor initial_shift, torch::Tensor x_r, torch::Tensor x_w,
    torch::Tensor x_k, torch::Tensor x_v, torch::Tensor x_a,
    torch::Tensor x_g) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&grad_r, "grad_r"},
           {&grad_w, "grad_w"}, {&grad_k, "grad_k"},
           {&grad_v, "grad_v"}, {&grad_a, "grad_a"},
           {&grad_g, "grad_g"}, {&grad_next_shift, "grad_next_shift"},
           {&x, "x"}, {&initial_shift, "initial_shift"}, {&x_r, "x_r"},
           {&x_w, "x_w"}, {&x_k, "x_k"}, {&x_v, "x_v"},
           {&x_a, "x_a"}, {&x_g, "x_g"}}) {
    check_bf16_cuda(*item.first, item.second);
    TORCH_CHECK(item.first->device() == x.device(), item.second,
                " must share x's device");
  }
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&grad_r, "grad_r"},
           {&grad_w, "grad_w"}, {&grad_k, "grad_k"},
           {&grad_v, "grad_v"}, {&grad_a, "grad_a"},
           {&grad_g, "grad_g"}}) {
    TORCH_CHECK(item.first->sizes() == x.sizes(), item.second,
                " must have the same shape as x");
  }
  TORCH_CHECK(x.dim() == 3 && x.size(0) > 0 && x.size(1) > 0 &&
                  x.size(2) > 0 && x.size(2) % 2 == 0,
              "x must have non-empty shape [B,T,C] with even C");
  TORCH_CHECK(initial_shift.sizes() ==
                  torch::IntArrayRef({x.size(0), x.size(2)}),
              "initial_shift must have shape [B,C]");
  TORCH_CHECK(grad_next_shift.sizes() == initial_shift.sizes(),
              "grad_next_shift must have shape [B,C]");
  for (const auto* tensor : {&x_r, &x_w, &x_k, &x_v, &x_a, &x_g}) {
    TORCH_CHECK(tensor->sizes() == torch::IntArrayRef({x.size(2)}),
                "tokenshift coefficients must have shape [C]");
  }
  return statetune_tmix_tokenshift_backward_cuda(
      grad_r, grad_w, grad_k, grad_v, grad_a, grad_g, grad_next_shift, x,
      initial_shift, x_r, x_w, x_k, x_v, x_a, x_g);
}

void register_statetune_tmix_tokenshift_backward_bindings(py::module_& module) {
  module.def(
      "statetune_tmix_tokenshift_backward", &statetune_tmix_tokenshift_backward,
      py::arg("grad_r"), py::arg("grad_w"), py::arg("grad_k"),
      py::arg("grad_v"), py::arg("grad_a"), py::arg("grad_g"),
      py::arg("grad_next_shift"), py::arg("x"),
      py::arg("initial_shift"), py::arg("x_r"), py::arg("x_w"),
      py::arg("x_k"), py::arg("x_v"), py::arg("x_a"), py::arg("x_g"));
}
