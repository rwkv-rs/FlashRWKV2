// SPDX-License-Identifier: Apache-2.0
// RWKV-LM train_temp revision: 952102498e9ed367ea0a59ee64106916d474d30f.
// Module-local binding for the canonical BF16 TMix a-gate family.

#include <torch/extension.h>

torch::Tensor pretrain_tmix_a_gate_forward_cuda(
    torch::Tensor a0, torch::Tensor a12);

namespace {

void check_bf16_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(
      tensor.is_cuda() && tensor.is_contiguous() &&
          tensor.scalar_type() == torch::kBFloat16,
      name, " must be contiguous CUDA bfloat16");
}

void check_inputs(const torch::Tensor& a0, const torch::Tensor& a12) {
  check_bf16_cuda(a0, "a0");
  check_bf16_cuda(a12, "a12");
  TORCH_CHECK(a0.dim() == 1 && a0.numel() > 0, "a0 must have shape [C]");
  TORCH_CHECK(
      a12.dim() == 3 && a12.size(0) > 0 && a12.size(1) > 0 &&
          a12.size(2) == a0.size(0),
      "a12 must have shape [B,T,C]");
  TORCH_CHECK(a0.device() == a12.device(), "a0 and a12 must share a device");
}

}  // namespace

torch::Tensor pretrain_tmix_a_gate_forward(
    torch::Tensor a0, torch::Tensor a12) {
  check_inputs(a0, a12);
  return pretrain_tmix_a_gate_forward_cuda(a0, a12);
}

void register_pretrain_tmix_a_gate_forward_bindings(py::module_& module) {
  module.def("pretrain_tmix_a_gate_forward", &pretrain_tmix_a_gate_forward,
             py::arg("a0"), py::arg("a12"));
}
