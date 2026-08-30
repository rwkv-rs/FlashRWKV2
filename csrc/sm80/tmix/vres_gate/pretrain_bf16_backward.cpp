// SPDX-License-Identifier: Apache-2.0
// RWKV-LM train_temp revision: 952102498e9ed367ea0a59ee64106916d474d30f.

#include <torch/extension.h>

#include <utility>
#include <vector>

std::vector<torch::Tensor> pretrain_tmix_vres_gate_backward_cuda(
    torch::Tensor grad_output, torch::Tensor value, torch::Tensor first_value,
    torch::Tensor v0, torch::Tensor v12);

namespace {
void check_bf16_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                  tensor.scalar_type() == torch::kBFloat16,
              name, " must be contiguous CUDA bfloat16");
}
}  // namespace

std::vector<torch::Tensor> pretrain_tmix_vres_gate_backward(
    torch::Tensor grad_output, torch::Tensor value, torch::Tensor first_value,
    torch::Tensor v0, torch::Tensor v12) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&grad_output, "grad_output"},
           {&value, "value"}, {&first_value, "first_value"}, {&v0, "v0"},
           {&v12, "v12"}}) {
    check_bf16_cuda(*item.first, item.second);
  }
  TORCH_CHECK(value.dim() == 3 && first_value.sizes() == value.sizes() &&
                  v12.sizes() == value.sizes() && v0.dim() == 1 &&
                  v0.size(0) == value.size(2) &&
                  grad_output.sizes() == value.sizes(),
              "invalid v-residual gate shapes");
  TORCH_CHECK(grad_output.device() == value.device() && first_value.device() == value.device() &&
                  v0.device() == value.device() && v12.device() == value.device(),
              "v-residual gate tensors must share a device");
  return pretrain_tmix_vres_gate_backward_cuda(
      grad_output, value, first_value, v0, v12);
}

void register_pretrain_tmix_vres_gate_backward_bindings(py::module_& module) {
  module.def("pretrain_tmix_vres_gate_backward",
             &pretrain_tmix_vres_gate_backward, py::arg("grad_output"),
             py::arg("value"), py::arg("first_value"), py::arg("v0"),
             py::arg("v12"));
}
