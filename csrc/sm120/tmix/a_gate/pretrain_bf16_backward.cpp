// SPDX-License-Identifier: Apache-2.0
// RWKV-LM train_temp revision: 952102498e9ed367ea0a59ee64106916d474d30f.

#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> pretrain_tmix_a_gate_backward_cuda(
    torch::Tensor grad_output, torch::Tensor a0, torch::Tensor a12);

namespace {

void check_bf16_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(
      tensor.is_cuda() && tensor.is_contiguous() &&
          tensor.scalar_type() == torch::kBFloat16,
      name, " must be contiguous CUDA bfloat16");
}

}  // namespace

std::vector<torch::Tensor> pretrain_tmix_a_gate_backward(
    torch::Tensor grad_output, torch::Tensor a0, torch::Tensor a12) {
  check_bf16_cuda(grad_output, "grad_output");
  check_bf16_cuda(a0, "a0");
  check_bf16_cuda(a12, "a12");
  TORCH_CHECK(a0.dim() == 1 && a12.dim() == 3 &&
                  a12.size(2) == a0.size(0),
              "invalid a-gate shapes");
  TORCH_CHECK(grad_output.sizes() == a12.sizes(), "grad_output shape mismatch");
  TORCH_CHECK(
      grad_output.device() == a0.device() && a12.device() == a0.device(),
      "a-gate tensors must share a device");
  return pretrain_tmix_a_gate_backward_cuda(grad_output, a0, a12);
}

void register_pretrain_tmix_a_gate_backward_bindings(py::module_& module) {
  module.def("pretrain_tmix_a_gate_backward", &pretrain_tmix_a_gate_backward,
             py::arg("grad_output"), py::arg("a0"), py::arg("a12"));
}
