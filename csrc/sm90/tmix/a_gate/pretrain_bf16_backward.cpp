// SPDX-License-Identifier: Apache-2.0
// RWKV-LM train_temp revision: 952102498e9ed367ea0a59ee64106916d474d30f.

#include "validation.h"

#include <vector>

std::vector<torch::stable::Tensor> pretrain_tmix_a_gate_backward_cuda(
    torch::stable::Tensor grad_output, torch::stable::Tensor a0, torch::stable::Tensor a12);

namespace {

void check_bf16_cuda(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(
      tensor.is_cuda() && tensor.is_contiguous() &&
          tensor.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      name, " must be contiguous CUDA bfloat16");
}

}  // namespace

auto pretrain_tmix_a_gate_backward(
    torch::stable::Tensor grad_output, torch::stable::Tensor a0, torch::stable::Tensor a12) {
  check_bf16_cuda(grad_output, "grad_output");
  check_bf16_cuda(a0, "a0");
  check_bf16_cuda(a12, "a12");
  STD_TORCH_CHECK(a0.dim() == 1 && a12.dim() == 3 &&
                  a12.size(2) == a0.size(0),
              "invalid a-gate shapes");
  STD_TORCH_CHECK(grad_output.sizes() == a12.sizes(), "grad_output shape mismatch");
  STD_TORCH_CHECK(
      grad_output.device() == a0.device() && a12.device() == a0.device(),
      "a-gate tensors must share a device");
  return flashrwkv2::validation::tensor_tuple<2>(pretrain_tmix_a_gate_backward_cuda(grad_output, a0, a12));
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("pretrain_tmix_a_gate_backward(Tensor grad_output, Tensor a0, Tensor a12) -> (Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("pretrain_tmix_a_gate_backward", TORCH_BOX(&pretrain_tmix_a_gate_backward));
}
