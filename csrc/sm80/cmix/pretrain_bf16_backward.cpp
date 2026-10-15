// SPDX-License-Identifier: Apache-2.0
// Full ChannelMix operator from RWKV-LM train_temp revision 952102498e9ed367ea0a59ee64106916d474d30f.

#include <torch/extension.h>
#include "../internal/linear/backend.cuh"
#include <optional>

#include <utility>
#include <vector>

std::vector<torch::Tensor> pretrain_cmix_backward_cuda(
    torch::Tensor grad_output, torch::Tensor x, torch::Tensor x_k,
    torch::Tensor key_weight, torch::Tensor value_weight,
    torch::Tensor mixed, torch::Tensor preact,
    std::optional<torch::Tensor> key_lora_a,
    std::optional<torch::Tensor> key_lora_b,
    std::optional<torch::Tensor> value_lora_a,
    std::optional<torch::Tensor> value_lora_b,
    double key_lora_scale, double value_lora_scale,
    bool key_use_nora, bool value_use_nora,
    std::vector<torch::Tensor> adapter_state, std::vector<bool> needs);

namespace {
void check_bf16_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                  (tensor.scalar_type() == torch::kBFloat16 || tensor.scalar_type() == torch::kFloat32),
              name, " must be contiguous CUDA bf16/fp32");
}
}  // namespace

std::vector<torch::Tensor> pretrain_cmix_backward(
    torch::Tensor grad_output, torch::Tensor x, torch::Tensor x_k,
    torch::Tensor key_weight, torch::Tensor value_weight,
    torch::Tensor mixed, torch::Tensor preact,
    std::optional<torch::Tensor> key_lora_a,
    std::optional<torch::Tensor> key_lora_b,
    std::optional<torch::Tensor> value_lora_a,
    std::optional<torch::Tensor> value_lora_b,
    double key_lora_scale, double value_lora_scale,
    bool key_use_nora, bool value_use_nora,
    std::vector<torch::Tensor> adapter_state, std::vector<bool> needs) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&grad_output, "grad_output"},
           {&x, "x"}, {&x_k, "x_k"}, {&key_weight, "key_weight"},
           {&value_weight, "value_weight"}, {&mixed, "mixed"}, {&preact, "preact"}}) {
    check_bf16_cuda(*item.first, item.second);
  }
  TORCH_CHECK(grad_output.scalar_type() == torch::kBFloat16 && x.scalar_type() == torch::kBFloat16 &&
                  x_k.scalar_type() == torch::kBFloat16 && key_weight.scalar_type() == torch::kBFloat16 &&
                  value_weight.scalar_type() == torch::kBFloat16 && mixed.scalar_type() == torch::kBFloat16 &&
                  preact.scalar_type() == torch::kBFloat16,
              "CMix training inputs must be bf16");
  TORCH_CHECK(grad_output.sizes() == x.sizes() && mixed.sizes() == x.sizes() &&
                  preact.dim() == 2 && preact.size(0) == x.size(0) * x.size(1) &&
                  preact.size(1) == 4 * x.size(2),
              "invalid CMix training gradient shapes");
  auto ka = key_lora_a.value_or(torch::Tensor());
  auto kb = key_lora_b.value_or(torch::Tensor());
  auto va = value_lora_a.value_or(torch::Tensor());
  auto vb = value_lora_b.value_or(torch::Tensor());
  internal_linear_train_check(x, key_weight, ka, kb, key_lora_scale);
  internal_linear_train_check_adapter(x, value_weight, va, vb, value_lora_scale);
  TORCH_CHECK((needs.empty() || needs.size() == 9) &&
                  (adapter_state.empty() || adapter_state.size() == 4),
              "invalid CMix adapter backward specification");
  return pretrain_cmix_backward_cuda(grad_output, x, x_k, key_weight,
                                     value_weight, mixed, preact,
      key_lora_a, key_lora_b, value_lora_a, value_lora_b, key_lora_scale, value_lora_scale, key_use_nora, value_use_nora, adapter_state, needs);
}

void register_pretrain_cmix_backward_bindings(py::module_& module) {
  module.def("pretrain_cmix_backward", &pretrain_cmix_backward,
             py::arg("grad_output"), py::arg("x"), py::arg("x_k"),
             py::arg("key_weight"), py::arg("value_weight"), py::arg("mixed"),
             py::arg("preact"),
      py::arg("key_lora_a") = py::none(), py::arg("key_lora_b") = py::none(),
      py::arg("value_lora_a") = py::none(), py::arg("value_lora_b") = py::none(),
      py::arg("key_lora_scale") = 1.0, py::arg("value_lora_scale") = 1.0,
      py::arg("key_use_nora") = false, py::arg("value_use_nora") = false, py::arg("adapter_state") = std::vector<torch::Tensor>{},
      py::arg("needs") = std::vector<bool>{});
}
