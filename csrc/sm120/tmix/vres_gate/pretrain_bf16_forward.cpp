// SPDX-License-Identifier: Apache-2.0
// RWKV-LM train_temp revision: 952102498e9ed367ea0a59ee64106916d474d30f.

#include "validation.h"

torch::stable::Tensor pretrain_tmix_vres_gate_forward_cuda(
    torch::stable::Tensor value,
    torch::stable::Tensor first_value,
    torch::stable::Tensor v0,
    torch::stable::Tensor v12);

namespace {
void check_bf16_cuda(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                  tensor.scalar_type() == torch::headeronly::ScalarType::BFloat16,
              name, " must be contiguous CUDA bfloat16");
}
void check_inputs(const torch::stable::Tensor& value, const torch::stable::Tensor& first_value,
                 const torch::stable::Tensor& v0, const torch::stable::Tensor& v12) {
  check_bf16_cuda(value, "value");
  check_bf16_cuda(first_value, "first_value");
  check_bf16_cuda(v0, "v0");
  check_bf16_cuda(v12, "v12");
  STD_TORCH_CHECK(value.dim() == 3 && value.numel() > 0,
              "value must have shape [B,T,C]");
  STD_TORCH_CHECK(first_value.sizes() == value.sizes() && v12.sizes() == value.sizes() &&
                  v0.dim() == 1 && v0.size(0) == value.size(2),
              "invalid v-residual gate shapes");
  STD_TORCH_CHECK(first_value.device() == value.device() && v0.device() == value.device() &&
                  v12.device() == value.device(),
              "v-residual gate tensors must share a device");
}
}  // namespace

torch::stable::Tensor pretrain_tmix_vres_gate_forward(
    torch::stable::Tensor value, torch::stable::Tensor first_value, torch::stable::Tensor v0,
    torch::stable::Tensor v12) {
  check_inputs(value, first_value, v0, v12);
  return pretrain_tmix_vres_gate_forward_cuda(value, first_value, v0, v12);
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("pretrain_tmix_vres_gate_forward(Tensor value, Tensor first_value, Tensor v0, Tensor v12) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("pretrain_tmix_vres_gate_forward", TORCH_BOX(&pretrain_tmix_vres_gate_forward));
}
