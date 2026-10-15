// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Adapted from RWKV-LM train_temp revision
// 952102498e9ed367ea0a59ee64106916d474d30f.

#include <torch/extension.h>
#include "../../internal/linear/backend.cuh"

#include <utility>
#include <vector>

std::vector<torch::Tensor> pretrain_tmix_readout_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor residual_scale,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g,
    torch::Tensor mean,
    torch::Tensor rstd,
    int64_t head_size);

namespace {

void check_bf16_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(
      tensor.is_cuda() && tensor.is_contiguous() &&
          tensor.scalar_type() == torch::kBFloat16,
      name, " must be contiguous CUDA bfloat16");
}

void check_inputs(
    const torch::Tensor& grad_output,
    const torch::Tensor& x,
    const torch::Tensor& r,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& residual_scale,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& g,
    const torch::Tensor& mean,
    const torch::Tensor& rstd,
    int64_t head_size) {
  TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256,
              "head_size must be one of 64, 128, or 256");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&grad_output, "grad_output"},
           {&x, "x"}, {&r, "r"}, {&k, "k"}, {&v, "v"},
           {&residual_scale, "residual_scale"}, {&weight, "weight"},
           {&bias, "bias"}, {&g, "g"},
       }) {
    check_bf16_cuda(*item.first, item.second);
  }
  TORCH_CHECK(x.dim() == 3 && x.size(2) > 0 && x.size(2) % head_size == 0,
              "x must have shape [B,T,C] with C divisible by head_size");
  for (const auto& item : {&grad_output, &r, &k, &v, &g}) {
    TORCH_CHECK(item->sizes() == x.sizes(), "lnx token tensors must match x");
  }
  const int64_t heads = x.size(2) / head_size;
  const auto expected_residual_shape = std::vector<int64_t>{heads, head_size};
  const auto expected_weight_shape = std::vector<int64_t>{x.size(2)};
  const auto expected_stats_shape =
      std::vector<int64_t>{x.size(0), x.size(1), heads};
  TORCH_CHECK(residual_scale.sizes() == expected_residual_shape,
              "residual_scale must have shape [C/head_size,head_size]");
  TORCH_CHECK(weight.sizes() == expected_weight_shape &&
                  bias.sizes() == weight.sizes(),
              "weight and bias must have shape [C]");
  TORCH_CHECK(mean.sizes() == expected_stats_shape &&
                  rstd.sizes() == mean.sizes() &&
                  mean.scalar_type() == torch::kFloat32 &&
                  rstd.scalar_type() == torch::kFloat32,
              "mean and rstd must be float32 [B,T,C/head_size]");
  TORCH_CHECK(x.device() == mean.device() && x.device() == rstd.device(),
              "lnx tensors must share a device");
}

}  // namespace

std::vector<torch::Tensor> pretrain_tmix_readout_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor residual_scale,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g,
    torch::Tensor mean,
    torch::Tensor rstd,
    int64_t head_size,
    std::optional<torch::Tensor> output_weight,
    std::optional<torch::Tensor> output_lora_a,
    std::optional<torch::Tensor> output_lora_b,
    double output_lora_scale, bool output_use_nora, torch::Tensor prelinear,
    torch::Tensor features, torch::Tensor normalized_a, std::vector<bool> needs) {
  TORCH_CHECK(needs.size() == 11, "Readout requires eleven gradient flags");
  std::vector<torch::Tensor> projection_grads(4);
  if (output_weight.has_value()) {
    internal_linear_train_check_backward(grad_output, prelinear, *output_weight);
    projection_grads = internal_linear_train_backward(
        grad_output, prelinear, *output_weight, output_lora_a.value_or(torch::Tensor()),
        output_lora_b.value_or(torch::Tensor()), features, normalized_a,
        output_lora_scale, output_use_nora, true, needs[8], needs[9], needs[10]);
    grad_output = projection_grads[0];
  }
  check_inputs(grad_output, x, r, k, v, residual_scale, weight, bias, g, mean, rstd, head_size);
  auto result = pretrain_tmix_readout_backward_cuda(
      grad_output, x, r, k, v, residual_scale, weight, bias, g, mean, rstd, head_size);
  result.insert(result.end(), {projection_grads[1], projection_grads[2], projection_grads[3]});
  return result;
}

void register_pretrain_tmix_readout_backward_bindings(py::module_& module) {
  module.def(
      "pretrain_tmix_readout_backward",
      &pretrain_tmix_readout_backward,
      "RWKV-7 train_temp fused LN/residual/gate backward",
      py::arg("grad_output"), py::arg("x"), py::arg("r"), py::arg("k"),
      py::arg("v"), py::arg("residual_scale"), py::arg("weight"),
      py::arg("bias"), py::arg("g"), py::arg("mean"), py::arg("rstd"),
      py::arg("head_size"), py::arg("output_weight"), py::arg("output_lora_a"),
      py::arg("output_lora_b"), py::arg("output_lora_scale"), py::arg("output_use_nora"),
      py::arg("prelinear"), py::arg("features"), py::arg("normalized_a"), py::arg("needs"));
}
