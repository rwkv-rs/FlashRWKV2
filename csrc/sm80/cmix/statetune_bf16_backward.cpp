// SPDX-License-Identifier: Apache-2.0
// Full ChannelMix operator from RWKV-LM train_temp revision 952102498e9ed367ea0a59ee64106916d474d30f.
// Local adaptation: propagate initial/next shift gradients across chunks.

#include <torch/extension.h>
#include "../internal/linear/backend.cuh"
#include <optional>

#include <cstdint>
#include <utility>
#include <vector>

std::vector<torch::Tensor> statetune_cmix_backward_cuda(
    torch::Tensor grad_output, torch::Tensor grad_next_shift,
    torch::Tensor x, torch::Tensor initial_shift, torch::Tensor x_k,
    torch::Tensor key_weight, torch::Tensor value_weight,
    torch::Tensor mixed, torch::Tensor activation,
    std::optional<torch::Tensor> key_lora_a,
    std::optional<torch::Tensor> key_lora_b,
    std::optional<torch::Tensor> value_lora_a,
    std::optional<torch::Tensor> value_lora_b,
    double key_lora_scale, double value_lora_scale,
    bool key_use_nora, bool value_use_nora,
    std::vector<torch::Tensor> adapter_state, std::vector<bool> needs);

namespace {
void check_bf16_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
  TORCH_CHECK(reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) % 4 == 0,
              name, " must be 4-byte aligned for BF16 vec2 access");
}
}  // namespace

std::vector<torch::Tensor> statetune_cmix_backward(
    torch::Tensor grad_output, torch::Tensor grad_next_shift,
    torch::Tensor x, torch::Tensor initial_shift, torch::Tensor x_k,
    torch::Tensor key_weight, torch::Tensor value_weight,
    torch::Tensor mixed, torch::Tensor activation,
    std::optional<torch::Tensor> key_lora_a,
    std::optional<torch::Tensor> key_lora_b,
    std::optional<torch::Tensor> value_lora_a,
    std::optional<torch::Tensor> value_lora_b,
    double key_lora_scale, double value_lora_scale,
    bool key_use_nora, bool value_use_nora,
    std::vector<torch::Tensor> adapter_state, std::vector<bool> needs) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&grad_output,
                                                        "grad_output"},
           {&grad_next_shift, "grad_next_shift"}, {&x, "x"},
           {&initial_shift, "initial_shift"}, {&x_k, "x_k"},
           {&key_weight, "key_weight"}, {&value_weight, "value_weight"},
           {&mixed, "mixed"}, {&activation, "activation"}}) {
    check_bf16_cuda(*item.first, item.second);
    TORCH_CHECK(item.first->device() == x.device(), item.second,
                " must share x's device");
  }
  const int64_t b = x.size(0), t = x.size(1), c = x.size(2);
  TORCH_CHECK(x.dim() == 3 && b > 0 && t > 0 && c > 0 && c % 2 == 0,
              "x must have non-empty shape [B,T,C] with even C");
  TORCH_CHECK(grad_output.sizes() == x.sizes(),
              "grad_output must have the same shape as x");
  TORCH_CHECK(initial_shift.sizes() == torch::IntArrayRef({b, c}) &&
                  grad_next_shift.sizes() == initial_shift.sizes(),
              "initial_shift and grad_next_shift must have shape [B,C]");
  TORCH_CHECK(x_k.sizes() == torch::IntArrayRef({c}),
              "x_k must have shape [C]");
  TORCH_CHECK(key_weight.sizes() == torch::IntArrayRef({4 * c, c}) &&
                  value_weight.sizes() == torch::IntArrayRef({c, 4 * c}),
              "CMix weights must have shapes [4C,C] and [C,4C]");
  TORCH_CHECK(mixed.sizes() == x.sizes() && activation.dim() == 2 &&
                  activation.size(0) == b * t && activation.size(1) == 4 * c,
              "invalid saved CMix activation shapes");
  auto ka = key_lora_a.value_or(torch::Tensor());
  auto kb = key_lora_b.value_or(torch::Tensor());
  auto va = value_lora_a.value_or(torch::Tensor());
  auto vb = value_lora_b.value_or(torch::Tensor());
  internal_linear_train_check(x, key_weight, ka, kb, key_lora_scale);
  internal_linear_train_check_adapter(x, value_weight, va, vb, value_lora_scale);
  TORCH_CHECK((needs.empty() || needs.size() == 9) &&
                  (adapter_state.empty() || adapter_state.size() == 4),
              "invalid CMix adapter backward specification");
  return statetune_cmix_backward_cuda(
      grad_output, grad_next_shift, x, initial_shift, x_k, key_weight,
      value_weight, mixed, activation,
      key_lora_a, key_lora_b, value_lora_a, value_lora_b, key_lora_scale, value_lora_scale, key_use_nora, value_use_nora, adapter_state, needs);
}

void register_statetune_cmix_backward_bindings(py::module_& module) {
  module.def(
      "statetune_cmix_backward", &statetune_cmix_backward,
      py::arg("grad_output"), py::arg("grad_next_shift"), py::arg("x"),
      py::arg("initial_shift"), py::arg("x_k"), py::arg("key_weight"),
      py::arg("value_weight"), py::arg("mixed"), py::arg("activation"),
      py::arg("key_lora_a") = py::none(), py::arg("key_lora_b") = py::none(),
      py::arg("value_lora_a") = py::none(), py::arg("value_lora_b") = py::none(),
      py::arg("key_lora_scale") = 1.0, py::arg("value_lora_scale") = 1.0,
      py::arg("key_use_nora") = false, py::arg("value_use_nora") = false, py::arg("adapter_state") = std::vector<torch::Tensor>{},
      py::arg("needs") = std::vector<bool>{});
}
