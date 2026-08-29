// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Adapted from RWKV-LM train_temp revision
// 952102498e9ed367ea0a59ee64106916d474d30f.

#include "validation.h"

#include <utility>
#include <vector>

std::vector<torch::stable::Tensor> pretrain_tmix_readout_backward_cuda(
    torch::stable::Tensor grad_output,
    torch::stable::Tensor x,
    torch::stable::Tensor r,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor residual_scale,
    torch::stable::Tensor weight,
    torch::stable::Tensor bias,
    torch::stable::Tensor g,
    torch::stable::Tensor mean,
    torch::stable::Tensor rstd,
    int64_t head_size);

namespace {

void check_bf16_cuda(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(
      tensor.is_cuda() && tensor.is_contiguous() &&
          tensor.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      name, " must be contiguous CUDA bfloat16");
}

void check_inputs(
    const torch::stable::Tensor& grad_output,
    const torch::stable::Tensor& x,
    const torch::stable::Tensor& r,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    const torch::stable::Tensor& residual_scale,
    const torch::stable::Tensor& weight,
    const torch::stable::Tensor& bias,
    const torch::stable::Tensor& g,
    const torch::stable::Tensor& mean,
    const torch::stable::Tensor& rstd,
    int64_t head_size) {
  STD_TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256,
              "head_size must be one of 64, 128, or 256");
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{&grad_output, "grad_output"},
           {&x, "x"}, {&r, "r"}, {&k, "k"}, {&v, "v"},
           {&residual_scale, "residual_scale"}, {&weight, "weight"},
           {&bias, "bias"}, {&g, "g"},
       }) {
    check_bf16_cuda(*item.first, item.second);
  }
  STD_TORCH_CHECK(x.dim() == 3 && x.size(2) > 0 && x.size(2) % head_size == 0,
              "x must have shape [B,T,C] with C divisible by head_size");
  for (const auto& item : {&grad_output, &r, &k, &v, &g}) {
    STD_TORCH_CHECK(item->sizes() == x.sizes(), "lnx token tensors must match x");
  }
  const int64_t heads = x.size(2) / head_size;
  const auto expected_residual_shape = std::vector<int64_t>{heads, head_size};
  const auto expected_weight_shape = std::vector<int64_t>{x.size(2)};
  const auto expected_stats_shape =
      std::vector<int64_t>{x.size(0), x.size(1), heads};
  STD_TORCH_CHECK(residual_scale.sizes() == expected_residual_shape,
              "residual_scale must have shape [C/head_size,head_size]");
  STD_TORCH_CHECK(weight.sizes() == expected_weight_shape &&
                  bias.sizes() == weight.sizes(),
              "weight and bias must have shape [C]");
  STD_TORCH_CHECK(mean.sizes() == expected_stats_shape &&
                  rstd.sizes() == mean.sizes() &&
                  mean.scalar_type() == torch::headeronly::ScalarType::Float &&
                  rstd.scalar_type() == torch::headeronly::ScalarType::Float,
              "mean and rstd must be float32 [B,T,C/head_size]");
  STD_TORCH_CHECK(x.device() == mean.device() && x.device() == rstd.device(),
              "lnx tensors must share a device");
}

}  // namespace

auto pretrain_tmix_readout_backward(
    torch::stable::Tensor grad_output,
    torch::stable::Tensor x,
    torch::stable::Tensor r,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor residual_scale,
    torch::stable::Tensor weight,
    torch::stable::Tensor bias,
    torch::stable::Tensor g,
    torch::stable::Tensor mean,
    torch::stable::Tensor rstd,
    int64_t head_size) {
  check_inputs(grad_output, x, r, k, v, residual_scale, weight, bias, g, mean, rstd, head_size);
  return flashrwkv2::validation::tensor_tuple<8>(pretrain_tmix_readout_backward_cuda(
      grad_output, x, r, k, v, residual_scale, weight, bias, g, mean, rstd, head_size));
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("pretrain_tmix_readout_backward(Tensor grad_output, Tensor x, Tensor r, Tensor k, Tensor v, Tensor residual_scale, Tensor weight, Tensor bias, Tensor g, Tensor mean, Tensor rstd, int head_size) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("pretrain_tmix_readout_backward", TORCH_BOX(&pretrain_tmix_readout_backward));
}
