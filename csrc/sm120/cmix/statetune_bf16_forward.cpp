// SPDX-License-Identifier: Apache-2.0
// Full ChannelMix operator from RWKV-LM train_temp revision 952102498e9ed367ea0a59ee64106916d474d30f.
// Local adaptation: accept a nonzero chunk-boundary shift and return next shift.

#include <torch/extension.h>

#include <cstdint>
#include <vector>

std::vector<torch::Tensor> statetune_cmix_forward_cuda(
    torch::Tensor x, torch::Tensor initial_shift, torch::Tensor x_k,
    torch::Tensor key_weight, torch::Tensor value_weight);

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

std::vector<torch::Tensor> statetune_cmix_forward(
    torch::Tensor x, torch::Tensor initial_shift, torch::Tensor x_k,
    torch::Tensor key_weight, torch::Tensor value_weight) {
  check_bf16_cuda(x, "x");
  check_bf16_cuda(initial_shift, "initial_shift");
  check_bf16_cuda(x_k, "x_k");
  check_bf16_cuda(key_weight, "key_weight");
  check_bf16_cuda(value_weight, "value_weight");
  TORCH_CHECK(x.dim() == 3 && x.size(0) > 0 && x.size(1) > 0 &&
                  x.size(2) > 0,
              "x must have non-empty shape [B,T,C]");
  TORCH_CHECK(x.size(2) % 2 == 0,
              "x channel dimension C must be divisible by 2");
  const int64_t b = x.size(0), c = x.size(2);
  TORCH_CHECK(initial_shift.sizes() == torch::IntArrayRef({b, c}),
              "initial_shift must have shape [B,C]");
  TORCH_CHECK(x_k.sizes() == torch::IntArrayRef({c}),
              "x_k must have shape [C]");
  TORCH_CHECK(key_weight.sizes() == torch::IntArrayRef({4 * c, c}),
              "key_weight must have shape [4C,C]");
  TORCH_CHECK(value_weight.sizes() == torch::IntArrayRef({c, 4 * c}),
              "value_weight must have shape [C,4C]");
  for (const auto* tensor :
       {&initial_shift, &x_k, &key_weight, &value_weight}) {
    TORCH_CHECK(tensor->device() == x.device(),
                "CMix tensors must share x's device");
  }
  return statetune_cmix_forward_cuda(
      x, initial_shift, x_k, key_weight, value_weight);
}

void register_statetune_cmix_forward_bindings(py::module_& module) {
  module.def(
      "statetune_cmix_forward", &statetune_cmix_forward, py::arg("x"),
      py::arg("initial_shift"), py::arg("x_k"), py::arg("key_weight"),
      py::arg("value_weight"));
}
