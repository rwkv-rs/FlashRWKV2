// SPDX-License-Identifier: Apache-2.0
// Full ChannelMix operator from RWKV-LM train_temp revision 952102498e9ed367ea0a59ee64106916d474d30f.

#include <torch/extension.h>

#include <utility>
#include <vector>

std::vector<torch::Tensor> pretrain_cmix_backward_cuda(
    torch::Tensor grad_output, torch::Tensor x, torch::Tensor x_k,
    torch::Tensor key_weight, torch::Tensor value_weight,
    torch::Tensor mixed, torch::Tensor preact);

namespace {
void check_bf16_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                  (tensor.scalar_type() == torch::kBFloat16 || tensor.scalar_type() == torch::kFloat32),
              name, " must be contiguous CUDA bf16/fp32");
}
}  // namespace

std::vector<torch::Tensor> pretrain_cmix_backward(
    torch::Tensor grad_output, torch::Tensor x, torch::Tensor x_k,
    torch::Tensor key_weight, torch::Tensor value_weight,
    torch::Tensor mixed, torch::Tensor preact) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&grad_output, "grad_output"},
           {&x, "x"}, {&x_k, "x_k"}, {&key_weight, "key_weight"},
           {&value_weight, "value_weight"}, {&mixed, "mixed"}, {&preact, "preact"}}) {
    check_bf16_cuda(*item.first, item.second);
  }
  TORCH_CHECK(grad_output.scalar_type() == torch::kBFloat16 && x.scalar_type() == torch::kBFloat16 &&
                  x_k.scalar_type() == torch::kBFloat16 && key_weight.scalar_type() == torch::kBFloat16 &&
                  value_weight.scalar_type() == torch::kBFloat16 && mixed.scalar_type() == torch::kBFloat16 &&
                  preact.scalar_type() == torch::kBFloat16,
              "CMix training inputs must be bf16");
  TORCH_CHECK(grad_output.sizes() == x.sizes() && mixed.sizes() == x.sizes() &&
                  preact.dim() == 2 && preact.size(0) == x.size(0) * x.size(1) &&
                  preact.size(1) == 4 * x.size(2),
              "invalid CMix training gradient shapes");
  return pretrain_cmix_backward_cuda(grad_output, x, x_k, key_weight,
                                     value_weight, mixed, preact);
}

void register_pretrain_cmix_backward_bindings(py::module_& module) {
  module.def("pretrain_cmix_backward", &pretrain_cmix_backward,
             py::arg("grad_output"), py::arg("x"), py::arg("x_k"),
             py::arg("key_weight"), py::arg("value_weight"), py::arg("mixed"),
             py::arg("preact"));
}
