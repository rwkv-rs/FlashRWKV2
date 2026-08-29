// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Adapted from RWKV-LM train_temp revision
// 952102498e9ed367ea0a59ee64106916d474d30f.

#include "validation.h"

#include <utility>
#include <vector>

std::vector<torch::stable::Tensor> pretrain_tmix_readout_cuda(
    torch::stable::Tensor x,
    torch::stable::Tensor r,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor residual_scale,
    torch::stable::Tensor weight,
    torch::stable::Tensor bias,
    torch::stable::Tensor g,
    int64_t head_size);

namespace {

void check_bf16_cuda(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(
      tensor.is_cuda() && tensor.is_contiguous() &&
          tensor.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      name, " must be contiguous CUDA bfloat16");
}

void check_inputs(
    const torch::stable::Tensor& x,
    const torch::stable::Tensor& r,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    const torch::stable::Tensor& residual_scale,
    const torch::stable::Tensor& weight,
    const torch::stable::Tensor& bias,
    const torch::stable::Tensor& g,
    int64_t head_size) {
  STD_TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256,
              "head_size must be one of 64, 128, or 256");
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{&x, "x"},
           {&r, "r"}, {&k, "k"}, {&v, "v"}, {&residual_scale, "residual_scale"},
           {&weight, "weight"}, {&bias, "bias"}, {&g, "g"},
       }) {
    check_bf16_cuda(*item.first, item.second);
  }
  STD_TORCH_CHECK(
      x.dim() == 3 && x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0 &&
          x.size(2) % head_size == 0,
      "x must have shape [B,T,C] with C divisible by head_size");
  for (const auto& item : {&r, &k, &v, &g}) {
    STD_TORCH_CHECK(item->sizes() == x.sizes(), "token tensors must match x");
  }
  const int64_t heads = x.size(2) / head_size;
  STD_TORCH_CHECK(
      residual_scale.dim() == 2 && residual_scale.size(0) == heads &&
          residual_scale.size(1) == head_size,
      "residual_scale must have shape [C/head_size,head_size]");
  STD_TORCH_CHECK(weight.dim() == 1 && bias.dim() == 1 &&
                  weight.size(0) == x.size(2) && bias.size(0) == x.size(2),
              "weight and bias must have shape [C]");
  STD_TORCH_CHECK(
      x.device() == r.device() && x.device() == k.device() &&
          x.device() == v.device() && x.device() == residual_scale.device() &&
          x.device() == weight.device() && x.device() == bias.device() &&
          x.device() == g.device(),
      "lnx tensors must share a device");
}

}  // namespace

auto pretrain_tmix_readout_forward(
    torch::stable::Tensor x,
    torch::stable::Tensor r,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor residual_scale,
    torch::stable::Tensor weight,
    torch::stable::Tensor bias,
    torch::stable::Tensor g,
    int64_t head_size) {
  check_inputs(x, r, k, v, residual_scale, weight, bias, g, head_size);
  return flashrwkv2::validation::tensor_tuple<3>(pretrain_tmix_readout_cuda(
      x, r, k, v, residual_scale, weight, bias, g, head_size));
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("pretrain_tmix_readout_forward(Tensor x, Tensor r, Tensor k, Tensor v, Tensor residual_scale, Tensor weight, Tensor bias, Tensor g, int head_size) -> (Tensor, Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("pretrain_tmix_readout_forward", TORCH_BOX(&pretrain_tmix_readout_forward));
}
