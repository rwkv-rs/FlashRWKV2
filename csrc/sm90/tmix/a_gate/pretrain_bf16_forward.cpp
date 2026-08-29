// SPDX-License-Identifier: Apache-2.0
// RWKV-LM train_temp revision: 952102498e9ed367ea0a59ee64106916d474d30f.
// Module-local binding for the canonical BF16 TMix a-gate family.

#include "validation.h"

torch::stable::Tensor pretrain_tmix_a_gate_forward_cuda(
    torch::stable::Tensor a0, torch::stable::Tensor a12);

namespace {

void check_bf16_cuda(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(
      tensor.is_cuda() && tensor.is_contiguous() &&
          tensor.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      name, " must be contiguous CUDA bfloat16");
}

void check_inputs(const torch::stable::Tensor& a0, const torch::stable::Tensor& a12) {
  check_bf16_cuda(a0, "a0");
  check_bf16_cuda(a12, "a12");
  STD_TORCH_CHECK(a0.dim() == 1 && a0.numel() > 0, "a0 must have shape [C]");
  STD_TORCH_CHECK(
      a12.dim() == 3 && a12.size(0) > 0 && a12.size(1) > 0 &&
          a12.size(2) == a0.size(0),
      "a12 must have shape [B,T,C]");
  STD_TORCH_CHECK(a0.device() == a12.device(), "a0 and a12 must share a device");
}

}  // namespace

torch::stable::Tensor pretrain_tmix_a_gate_forward(
    torch::stable::Tensor a0, torch::stable::Tensor a12) {
  check_inputs(a0, a12);
  return pretrain_tmix_a_gate_forward_cuda(a0, a12);
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("pretrain_tmix_a_gate_forward(Tensor a0, Tensor a12) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("pretrain_tmix_a_gate_forward", TORCH_BOX(&pretrain_tmix_a_gate_forward));
}
