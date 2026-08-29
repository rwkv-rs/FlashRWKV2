// SPDX-License-Identifier: Apache-2.0
// Full ChannelMix operator from RWKV-LM train_temp revision 952102498e9ed367ea0a59ee64106916d474d30f.
// Local adaptation: propagate initial/next shift gradients across chunks.

#include "validation.h"

#include <cstdint>
#include <utility>
#include <vector>

std::vector<torch::stable::Tensor> statetune_cmix_backward_cuda(
    torch::stable::Tensor grad_output, torch::stable::Tensor grad_next_shift,
    torch::stable::Tensor x, torch::stable::Tensor initial_shift, torch::stable::Tensor x_k,
    torch::stable::Tensor key_weight, torch::stable::Tensor value_weight,
    torch::stable::Tensor mixed, torch::stable::Tensor activation);

namespace {
void check_bf16_cuda(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  STD_TORCH_CHECK(tensor.scalar_type() == torch::headeronly::ScalarType::BFloat16,
              name, " must have dtype torch.bfloat16");
  STD_TORCH_CHECK(reinterpret_cast<std::uintptr_t>(tensor.mutable_data_ptr()) % 4 == 0,
              name, " must be 4-byte aligned for BF16 vec2 access");
}
}  // namespace

auto statetune_cmix_backward(
    torch::stable::Tensor grad_output, torch::stable::Tensor grad_next_shift,
    torch::stable::Tensor x, torch::stable::Tensor initial_shift, torch::stable::Tensor x_k,
    torch::stable::Tensor key_weight, torch::stable::Tensor value_weight,
    torch::stable::Tensor mixed, torch::stable::Tensor activation) {
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{&grad_output,
                                                        "grad_output"},
           {&grad_next_shift, "grad_next_shift"}, {&x, "x"},
           {&initial_shift, "initial_shift"}, {&x_k, "x_k"},
           {&key_weight, "key_weight"}, {&value_weight, "value_weight"},
           {&mixed, "mixed"}, {&activation, "activation"}}) {
    check_bf16_cuda(*item.first, item.second);
    STD_TORCH_CHECK(item.first->device() == x.device(), item.second,
                " must share x's device");
  }
  const int64_t b = x.size(0), t = x.size(1), c = x.size(2);
  STD_TORCH_CHECK(x.dim() == 3 && b > 0 && t > 0 && c > 0 && c % 2 == 0,
              "x must have non-empty shape [B,T,C] with even C");
  STD_TORCH_CHECK(grad_output.sizes() == x.sizes(),
              "grad_output must have the same shape as x");
  STD_TORCH_CHECK(initial_shift.sizes() == torch::headeronly::IntHeaderOnlyArrayRef({b, c}) &&
                  grad_next_shift.sizes() == initial_shift.sizes(),
              "initial_shift and grad_next_shift must have shape [B,C]");
  STD_TORCH_CHECK(x_k.sizes() == torch::headeronly::IntHeaderOnlyArrayRef({c}),
              "x_k must have shape [C]");
  STD_TORCH_CHECK(key_weight.sizes() == torch::headeronly::IntHeaderOnlyArrayRef({4 * c, c}) &&
                  value_weight.sizes() == torch::headeronly::IntHeaderOnlyArrayRef({c, 4 * c}),
              "CMix weights must have shapes [4C,C] and [C,4C]");
  STD_TORCH_CHECK(mixed.sizes() == x.sizes() && activation.dim() == 2 &&
                  activation.size(0) == b * t && activation.size(1) == 4 * c,
              "invalid saved CMix activation shapes");
  return flashrwkv2::validation::tensor_tuple<5>(statetune_cmix_backward_cuda(
      grad_output, grad_next_shift, x, initial_shift, x_k, key_weight,
      value_weight, mixed, activation));
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("statetune_cmix_backward(Tensor grad_output, Tensor grad_next_shift, Tensor x, Tensor initial_shift, Tensor x_k, Tensor key_weight, Tensor value_weight, Tensor mixed, Tensor activation) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("statetune_cmix_backward", TORCH_BOX(&statetune_cmix_backward));
}
