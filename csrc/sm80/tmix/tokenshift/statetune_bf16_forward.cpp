// SPDX-License-Identifier: Apache-2.0
// RWKV-LM train_temp revision: 952102498e9ed367ea0a59ee64106916d474d30f.
// Local adaptation: accept a nonzero chunk-boundary shift and return next shift.

#include <torch/extension.h>

#include <cstdint>
#include <utility>
#include <vector>

std::vector<torch::Tensor> statetune_tmix_tokenshift_forward_cuda(
    torch::Tensor x, torch::Tensor initial_shift, torch::Tensor x_r,
    torch::Tensor x_w, torch::Tensor x_k, torch::Tensor x_v,
    torch::Tensor x_a, torch::Tensor x_g);

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

std::vector<torch::Tensor> statetune_tmix_tokenshift_forward(
    torch::Tensor x, torch::Tensor initial_shift, torch::Tensor x_r,
    torch::Tensor x_w, torch::Tensor x_k, torch::Tensor x_v,
    torch::Tensor x_a, torch::Tensor x_g) {
  check_bf16_cuda(x, "x");
  check_bf16_cuda(initial_shift, "initial_shift");
  TORCH_CHECK(x.dim() == 3 && x.size(0) > 0 && x.size(1) > 0 &&
                  x.size(2) > 0,
              "x must have non-empty shape [B,T,C]");
  TORCH_CHECK(x.size(2) % 2 == 0,
              "x channel dimension C must be divisible by 2");
  TORCH_CHECK(initial_shift.sizes() ==
                  torch::IntArrayRef({x.size(0), x.size(2)}),
              "initial_shift must have shape [B,C]");
  TORCH_CHECK(initial_shift.device() == x.device(),
              "initial_shift must share x's device");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&x_r, "x_r"},
           {&x_w, "x_w"}, {&x_k, "x_k"}, {&x_v, "x_v"},
           {&x_a, "x_a"}, {&x_g, "x_g"}}) {
    check_bf16_cuda(*item.first, item.second);
    TORCH_CHECK(item.first->sizes() == torch::IntArrayRef({x.size(2)}),
                item.second, " must have shape [C]");
    TORCH_CHECK(item.first->device() == x.device(), item.second,
                " must share x's device");
  }
  return statetune_tmix_tokenshift_forward_cuda(
      x, initial_shift, x_r, x_w, x_k, x_v, x_a, x_g);
}

void register_statetune_tmix_tokenshift_forward_bindings(py::module_& module) {
  module.def(
      "statetune_tmix_tokenshift_forward", &statetune_tmix_tokenshift_forward,
      py::arg("x"), py::arg("initial_shift"), py::arg("x_r"),
      py::arg("x_w"), py::arg("x_k"), py::arg("x_v"), py::arg("x_a"),
      py::arg("x_g"));
}
