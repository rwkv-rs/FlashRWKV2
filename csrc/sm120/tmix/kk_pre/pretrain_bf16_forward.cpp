// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Adapted from RWKV-LM train_temp revision
// 952102498e9ed367ea0a59ee64106916d474d30f.
// The module-local binding preserves the per-head key normalization contract.

#include <torch/extension.h>

#include <utility>
#include <vector>

std::vector<torch::Tensor> pretrain_tmix_kk_pre_cuda(
    torch::Tensor key,
    torch::Tensor key_scale,
    torch::Tensor learning_rate,
    torch::Tensor learning_rate_scale,
    int64_t head_size);

namespace {

void check_bf16_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(
      tensor.is_cuda() && tensor.is_contiguous() &&
          tensor.scalar_type() == torch::kBFloat16,
      name, " must be contiguous CUDA bfloat16");
}

void check_inputs(
    const torch::Tensor& key,
    const torch::Tensor& key_scale,
    const torch::Tensor& learning_rate,
    const torch::Tensor& learning_rate_scale,
    int64_t head_size) {
  TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256,
              "head_size must be one of 64, 128, or 256");
  check_bf16_cuda(key, "key");
  check_bf16_cuda(key_scale, "key_scale");
  check_bf16_cuda(learning_rate, "learning_rate");
  check_bf16_cuda(learning_rate_scale, "learning_rate_scale");
  TORCH_CHECK(
      key.dim() == 3 && key.size(0) > 0 && key.size(1) > 0 &&
          key.size(2) > 0 && key.size(2) % head_size == 0,
      "key must have shape [B,T,C] with C divisible by head_size");
  TORCH_CHECK(learning_rate.sizes() == key.sizes(),
              "learning_rate must match key");
  TORCH_CHECK(key_scale.dim() == 1 && key_scale.size(0) == key.size(2),
              "key_scale must have shape [C]");
  TORCH_CHECK(
      learning_rate_scale.dim() == 1 &&
          learning_rate_scale.size(0) == key.size(2),
      "learning_rate_scale must have shape [C]");
  TORCH_CHECK(
      key.device() == key_scale.device() &&
          key.device() == learning_rate.device() &&
          key.device() == learning_rate_scale.device(),
      "kk-pre inputs must share a device");
}

}  // namespace

std::vector<torch::Tensor> pretrain_tmix_kk_pre(
    torch::Tensor key,
    torch::Tensor key_scale,
    torch::Tensor learning_rate,
    torch::Tensor learning_rate_scale,
    int64_t head_size) {
  check_inputs(key, key_scale, learning_rate, learning_rate_scale, head_size);
  return pretrain_tmix_kk_pre_cuda(
      key, key_scale, learning_rate, learning_rate_scale, head_size);
}

void register_pretrain_tmix_kk_pre_bindings(py::module_& module) {
  module.def(
      "pretrain_tmix_kk_pre_forward", &pretrain_tmix_kk_pre,
      "RWKV-7 train_temp per-head key preparation",
      py::arg("key"), py::arg("key_scale"), py::arg("learning_rate"),
      py::arg("learning_rate_scale"), py::arg("head_size") = 64);
}
