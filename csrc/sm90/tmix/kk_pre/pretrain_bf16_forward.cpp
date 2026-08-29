// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Adapted from RWKV-LM train_temp revision
// 952102498e9ed367ea0a59ee64106916d474d30f.
// The module-local binding preserves the per-head key normalization contract.

#include "validation.h"

#include <utility>
#include <vector>

std::vector<torch::stable::Tensor> pretrain_tmix_kk_pre_cuda(
    torch::stable::Tensor key,
    torch::stable::Tensor key_scale,
    torch::stable::Tensor learning_rate,
    torch::stable::Tensor learning_rate_scale,
    int64_t head_size);

namespace {

void check_bf16_cuda(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(
      tensor.is_cuda() && tensor.is_contiguous() &&
          tensor.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      name, " must be contiguous CUDA bfloat16");
}

void check_inputs(
    const torch::stable::Tensor& key,
    const torch::stable::Tensor& key_scale,
    const torch::stable::Tensor& learning_rate,
    const torch::stable::Tensor& learning_rate_scale,
    int64_t head_size) {
  STD_TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256,
              "head_size must be one of 64, 128, or 256");
  check_bf16_cuda(key, "key");
  check_bf16_cuda(key_scale, "key_scale");
  check_bf16_cuda(learning_rate, "learning_rate");
  check_bf16_cuda(learning_rate_scale, "learning_rate_scale");
  STD_TORCH_CHECK(
      key.dim() == 3 && key.size(0) > 0 && key.size(1) > 0 &&
          key.size(2) > 0 && key.size(2) % head_size == 0,
      "key must have shape [B,T,C] with C divisible by head_size");
  STD_TORCH_CHECK(learning_rate.sizes() == key.sizes(),
              "learning_rate must match key");
  STD_TORCH_CHECK(key_scale.dim() == 1 && key_scale.size(0) == key.size(2),
              "key_scale must have shape [C]");
  STD_TORCH_CHECK(
      learning_rate_scale.dim() == 1 &&
          learning_rate_scale.size(0) == key.size(2),
      "learning_rate_scale must have shape [C]");
  STD_TORCH_CHECK(
      key.device() == key_scale.device() &&
          key.device() == learning_rate.device() &&
          key.device() == learning_rate_scale.device(),
      "kk-pre inputs must share a device");
}

}  // namespace

auto pretrain_tmix_kk_pre(
    torch::stable::Tensor key,
    torch::stable::Tensor key_scale,
    torch::stable::Tensor learning_rate,
    torch::stable::Tensor learning_rate_scale,
    int64_t head_size) {
  check_inputs(key, key_scale, learning_rate, learning_rate_scale, head_size);
  return flashrwkv2::validation::tensor_tuple<4>(pretrain_tmix_kk_pre_cuda(
      key, key_scale, learning_rate, learning_rate_scale, head_size));
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("pretrain_tmix_kk_pre_forward(Tensor key, Tensor key_scale, Tensor learning_rate, Tensor learning_rate_scale, int head_size) -> (Tensor, Tensor, Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("pretrain_tmix_kk_pre_forward", TORCH_BOX(&pretrain_tmix_kk_pre));
}
