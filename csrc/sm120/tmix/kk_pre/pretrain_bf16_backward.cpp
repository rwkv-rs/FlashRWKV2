// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Adapted from RWKV-LM train_temp revision
// 952102498e9ed367ea0a59ee64106916d474d30f.

#include "validation.h"

#include <utility>
#include <vector>

std::vector<torch::stable::Tensor> pretrain_tmix_kk_pre_backward_cuda(
    torch::stable::Tensor grad_new_key,
    torch::stable::Tensor grad_negative_direction,
    torch::stable::Tensor grad_scaled_direction,
    torch::stable::Tensor key,
    torch::stable::Tensor key_scale,
    torch::stable::Tensor learning_rate,
    torch::stable::Tensor learning_rate_scale,
    torch::stable::Tensor inverse_norm,
    int64_t head_size);

namespace {

void check_bf16_cuda(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(
      tensor.is_cuda() && tensor.is_contiguous() &&
          tensor.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      name, " must be contiguous CUDA bfloat16");
}

void check_inputs(
    const torch::stable::Tensor& grad_new_key,
    const torch::stable::Tensor& grad_negative_direction,
    const torch::stable::Tensor& grad_scaled_direction,
    const torch::stable::Tensor& key,
    const torch::stable::Tensor& key_scale,
    const torch::stable::Tensor& learning_rate,
    const torch::stable::Tensor& learning_rate_scale,
    const torch::stable::Tensor& inverse_norm,
    int64_t head_size) {
  STD_TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256,
              "head_size must be one of 64, 128, or 256");
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{
               &grad_new_key, "grad_new_key"},
           {&grad_negative_direction, "grad_negative_direction"},
           {&grad_scaled_direction, "grad_scaled_direction"},
           {&key, "key"},
           {&key_scale, "key_scale"},
           {&learning_rate, "learning_rate"},
           {&learning_rate_scale, "learning_rate_scale"},
       }) {
    check_bf16_cuda(*item.first, item.second);
  }
  STD_TORCH_CHECK(
      key.dim() == 3 && key.size(2) > 0 && key.size(2) % head_size == 0,
      "key must have shape [B,T,C] with C divisible by head_size");
  STD_TORCH_CHECK(grad_new_key.sizes() == key.sizes() &&
                  grad_negative_direction.sizes() == key.sizes() &&
                  grad_scaled_direction.sizes() == key.sizes() &&
                  learning_rate.sizes() == key.sizes(),
              "kk-pre gradients and inputs must match key");
  STD_TORCH_CHECK(key_scale.sizes() == learning_rate_scale.sizes() &&
                  key_scale.dim() == 1 && key_scale.size(0) == key.size(2),
              "kk-pre scale vectors must have shape [C]");
  STD_TORCH_CHECK(
      inverse_norm.dim() == 3 && inverse_norm.size(0) == key.size(0) &&
          inverse_norm.size(1) == key.size(1) &&
          inverse_norm.size(2) == key.size(2) / head_size &&
          inverse_norm.scalar_type() == torch::headeronly::ScalarType::Float &&
          inverse_norm.is_cuda() && inverse_norm.is_contiguous(),
      "inverse_norm must be contiguous CUDA float32 [B,T,C/head_size]");
  STD_TORCH_CHECK(
      key.device() == key_scale.device() &&
          key.device() == learning_rate.device() &&
          key.device() == learning_rate_scale.device() &&
          key.device() == inverse_norm.device(),
      "kk-pre tensors must share a device");
}

}  // namespace

auto pretrain_tmix_kk_pre_backward(
    torch::stable::Tensor grad_new_key,
    torch::stable::Tensor grad_negative_direction,
    torch::stable::Tensor grad_scaled_direction,
    torch::stable::Tensor key,
    torch::stable::Tensor key_scale,
    torch::stable::Tensor learning_rate,
    torch::stable::Tensor learning_rate_scale,
    torch::stable::Tensor inverse_norm,
    int64_t head_size) {
  check_inputs(
      grad_new_key, grad_negative_direction, grad_scaled_direction, key,
      key_scale, learning_rate, learning_rate_scale, inverse_norm, head_size);
  return flashrwkv2::validation::tensor_tuple<4>(pretrain_tmix_kk_pre_backward_cuda(
      grad_new_key, grad_negative_direction, grad_scaled_direction, key,
      key_scale, learning_rate, learning_rate_scale, inverse_norm, head_size));
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("pretrain_tmix_kk_pre_backward(Tensor grad_new_key, Tensor grad_negative_direction, Tensor grad_scaled_direction, Tensor key, Tensor key_scale, Tensor learning_rate, Tensor learning_rate_scale, Tensor inverse_norm, int head_size) -> (Tensor, Tensor, Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("pretrain_tmix_kk_pre_backward", TORCH_BOX(&pretrain_tmix_kk_pre_backward));
}
