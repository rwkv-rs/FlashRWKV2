// SPDX-License-Identifier: Apache-2.0
// Full ChannelMix operator from RWKV-LM train_temp revision 952102498e9ed367ea0a59ee64106916d474d30f.

#include <torch/extension.h>
#include "../internal/linear/backend.cuh"
#include <optional>

#include <utility>
#include <vector>

std::vector<torch::Tensor> pretrain_cmix_forward_cuda(
    torch::Tensor x, torch::Tensor x_k, torch::Tensor key_weight,
    torch::Tensor value_weight,
    std::optional<torch::Tensor> key_lora_a,
    std::optional<torch::Tensor> key_lora_b,
    std::optional<torch::Tensor> value_lora_a,
    std::optional<torch::Tensor> value_lora_b,
    double key_lora_scale, double value_lora_scale,
    bool key_use_nora, bool value_use_nora);

namespace {
void check_bf16_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                  tensor.scalar_type() == torch::kBFloat16,
              name, " must be contiguous CUDA bfloat16");
}
void check_inputs(const torch::Tensor& x, const torch::Tensor& x_k,
                  const torch::Tensor& key_weight, const torch::Tensor& value_weight) {
  check_bf16_cuda(x, "x"); check_bf16_cuda(x_k, "x_k");
  check_bf16_cuda(key_weight, "key_weight"); check_bf16_cuda(value_weight, "value_weight");
  TORCH_CHECK(x.dim() == 3 && x.numel() > 0 && x.size(2) % 2 == 0 && x_k.dim() == 1 &&
                  x_k.size(0) == x.size(2) && key_weight.dim() == 2 &&
                  key_weight.size(0) == 4 * x.size(2) &&
                  key_weight.size(1) == x.size(2) && value_weight.dim() == 2 &&
                  value_weight.size(0) == x.size(2) &&
                  value_weight.size(1) == 4 * x.size(2),
              "invalid CMix training shapes");
  TORCH_CHECK(x.device() == x_k.device() && x.device() == key_weight.device() &&
                  x.device() == value_weight.device(),
              "CMix training tensors must share a device");
}
}  // namespace

std::vector<torch::Tensor> pretrain_cmix_forward(
    torch::Tensor x, torch::Tensor x_k, torch::Tensor key_weight,
    torch::Tensor value_weight,
    std::optional<torch::Tensor> key_lora_a,
    std::optional<torch::Tensor> key_lora_b,
    std::optional<torch::Tensor> value_lora_a,
    std::optional<torch::Tensor> value_lora_b,
    double key_lora_scale, double value_lora_scale,
    bool key_use_nora, bool value_use_nora) {
  check_inputs(x, x_k, key_weight, value_weight);
  auto ka = key_lora_a.value_or(torch::Tensor());
  auto kb = key_lora_b.value_or(torch::Tensor());
  auto va = value_lora_a.value_or(torch::Tensor());
  auto vb = value_lora_b.value_or(torch::Tensor());
  internal_linear_train_check(x, key_weight, ka, kb, key_lora_scale);
  internal_linear_train_check_adapter(x, value_weight, va, vb, value_lora_scale);
  return pretrain_cmix_forward_cuda(x, x_k, key_weight, value_weight,
      key_lora_a, key_lora_b, value_lora_a, value_lora_b, key_lora_scale, value_lora_scale, key_use_nora, value_use_nora);
}

void register_pretrain_cmix_forward_bindings(py::module_& module) {
  module.def("pretrain_cmix_forward", &pretrain_cmix_forward, py::arg("x"),
             py::arg("x_k"), py::arg("key_weight"), py::arg("value_weight"),
      py::arg("key_lora_a") = py::none(), py::arg("key_lora_b") = py::none(),
      py::arg("value_lora_a") = py::none(), py::arg("value_lora_b") = py::none(),
      py::arg("key_lora_scale") = 1.0, py::arg("value_lora_scale") = 1.0,
      py::arg("key_use_nora") = false, py::arg("value_use_nora") = false);
}
