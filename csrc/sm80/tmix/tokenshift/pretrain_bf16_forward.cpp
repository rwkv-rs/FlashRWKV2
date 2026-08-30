// SPDX-License-Identifier: Apache-2.0
// RWKV-LM train_temp revision: 952102498e9ed367ea0a59ee64106916d474d30f.

#include <torch/extension.h>

#include <utility>
#include <vector>

std::vector<torch::Tensor> pretrain_tmix_tokenshift_forward_cuda(
    torch::Tensor x, torch::Tensor x_r, torch::Tensor x_w,
    torch::Tensor x_k, torch::Tensor x_v, torch::Tensor x_a,
    torch::Tensor x_g);

namespace {
void check_bf16_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                  tensor.scalar_type() == torch::kBFloat16,
              name, " must be contiguous CUDA bfloat16");
}
void check_inputs(const torch::Tensor& x, const std::vector<std::pair<const torch::Tensor*, const char*>>& mixes) {
  check_bf16_cuda(x, "x");
  TORCH_CHECK(x.dim() == 3 && x.numel() > 0, "x must have shape [B,T,C]");
  for (const auto& [tensor, name] : mixes) {
    check_bf16_cuda(*tensor, name);
    TORCH_CHECK(tensor->dim() == 1 && tensor->size(0) == x.size(2), name,
                " must have shape [C]");
    TORCH_CHECK(tensor->device() == x.device(), name, " must share x's device");
  }
}
}  // namespace

std::vector<torch::Tensor> pretrain_tmix_tokenshift_forward(
    torch::Tensor x, torch::Tensor x_r, torch::Tensor x_w, torch::Tensor x_k,
    torch::Tensor x_v, torch::Tensor x_a, torch::Tensor x_g) {
  check_inputs(x, {{&x_r, "x_r"}, {&x_w, "x_w"}, {&x_k, "x_k"},
                   {&x_v, "x_v"}, {&x_a, "x_a"}, {&x_g, "x_g"}});
  return pretrain_tmix_tokenshift_forward_cuda(x, x_r, x_w, x_k, x_v, x_a, x_g);
}

void register_pretrain_tmix_tokenshift_forward_bindings(py::module_& module) {
  module.def("pretrain_tmix_tokenshift_forward", &pretrain_tmix_tokenshift_forward,
             py::arg("x"), py::arg("x_r"), py::arg("x_w"), py::arg("x_k"),
             py::arg("x_v"), py::arg("x_a"), py::arg("x_g"));
}
