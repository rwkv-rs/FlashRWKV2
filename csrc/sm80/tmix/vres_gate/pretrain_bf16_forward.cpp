// SPDX-License-Identifier: Apache-2.0
// RWKV-LM train_temp revision: 952102498e9ed367ea0a59ee64106916d474d30f.

#include <torch/extension.h>

torch::Tensor pretrain_tmix_vres_gate_forward_cuda(
    torch::Tensor value,
    torch::Tensor first_value,
    torch::Tensor v0,
    torch::Tensor v12);

namespace {
void check_bf16_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                  tensor.scalar_type() == torch::kBFloat16,
              name, " must be contiguous CUDA bfloat16");
}
void check_inputs(const torch::Tensor& value, const torch::Tensor& first_value,
                 const torch::Tensor& v0, const torch::Tensor& v12) {
  check_bf16_cuda(value, "value");
  check_bf16_cuda(first_value, "first_value");
  check_bf16_cuda(v0, "v0");
  check_bf16_cuda(v12, "v12");
  TORCH_CHECK(value.dim() == 3 && value.numel() > 0,
              "value must have shape [B,T,C]");
  TORCH_CHECK(first_value.sizes() == value.sizes() && v12.sizes() == value.sizes() &&
                  v0.dim() == 1 && v0.size(0) == value.size(2),
              "invalid v-residual gate shapes");
  TORCH_CHECK(first_value.device() == value.device() && v0.device() == value.device() &&
                  v12.device() == value.device(),
              "v-residual gate tensors must share a device");
}
}  // namespace

torch::Tensor pretrain_tmix_vres_gate_forward(
    torch::Tensor value, torch::Tensor first_value, torch::Tensor v0,
    torch::Tensor v12) {
  check_inputs(value, first_value, v0, v12);
  return pretrain_tmix_vres_gate_forward_cuda(value, first_value, v0, v12);
}

void register_pretrain_tmix_vres_gate_forward_bindings(py::module_& module) {
  module.def("pretrain_tmix_vres_gate_forward",
             &pretrain_tmix_vres_gate_forward, py::arg("value"),
             py::arg("first_value"), py::arg("v0"), py::arg("v12"));
}
