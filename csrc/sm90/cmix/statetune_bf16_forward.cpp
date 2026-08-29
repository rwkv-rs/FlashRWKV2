// SPDX-License-Identifier: Apache-2.0
// Full ChannelMix operator from RWKV-LM train_temp revision 952102498e9ed367ea0a59ee64106916d474d30f.
// Local adaptation: accept a nonzero chunk-boundary shift and return next shift.

#include "validation.h"

#include <cstdint>
#include <vector>

std::vector<torch::stable::Tensor> statetune_cmix_forward_cuda(
    torch::stable::Tensor x, torch::stable::Tensor initial_shift, torch::stable::Tensor x_k,
    torch::stable::Tensor key_weight, torch::stable::Tensor value_weight);

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

auto statetune_cmix_forward(
    torch::stable::Tensor x, torch::stable::Tensor initial_shift, torch::stable::Tensor x_k,
    torch::stable::Tensor key_weight, torch::stable::Tensor value_weight) {
  check_bf16_cuda(x, "x");
  check_bf16_cuda(initial_shift, "initial_shift");
  check_bf16_cuda(x_k, "x_k");
  check_bf16_cuda(key_weight, "key_weight");
  check_bf16_cuda(value_weight, "value_weight");
  STD_TORCH_CHECK(x.dim() == 3 && x.size(0) > 0 && x.size(1) > 0 &&
                  x.size(2) > 0,
              "x must have non-empty shape [B,T,C]");
  STD_TORCH_CHECK(x.size(2) % 2 == 0,
              "x channel dimension C must be divisible by 2");
  const int64_t b = x.size(0), c = x.size(2);
  STD_TORCH_CHECK(initial_shift.sizes() == torch::headeronly::IntHeaderOnlyArrayRef({b, c}),
              "initial_shift must have shape [B,C]");
  STD_TORCH_CHECK(x_k.sizes() == torch::headeronly::IntHeaderOnlyArrayRef({c}),
              "x_k must have shape [C]");
  STD_TORCH_CHECK(key_weight.sizes() == torch::headeronly::IntHeaderOnlyArrayRef({4 * c, c}),
              "key_weight must have shape [4C,C]");
  STD_TORCH_CHECK(value_weight.sizes() == torch::headeronly::IntHeaderOnlyArrayRef({c, 4 * c}),
              "value_weight must have shape [C,4C]");
  for (const auto* tensor :
       {&initial_shift, &x_k, &key_weight, &value_weight}) {
    STD_TORCH_CHECK(tensor->device() == x.device(),
                "CMix tensors must share x's device");
  }
  return flashrwkv2::validation::tensor_tuple<4>(statetune_cmix_forward_cuda(
      x, initial_shift, x_k, key_weight, value_weight));
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("statetune_cmix_forward(Tensor x, Tensor initial_shift, Tensor x_k, Tensor key_weight, Tensor value_weight) -> (Tensor, Tensor, Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("statetune_cmix_forward", TORCH_BOX(&statetune_cmix_forward));
}
