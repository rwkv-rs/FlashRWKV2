// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Adapted from RWKV-LM train_temp revision
// 952102498e9ed367ea0a59ee64106916d474d30f.

#include <torch/extension.h>

#include <utility>
#include <vector>

std::vector<torch::Tensor> pretrain_tmix_readout_cuda(
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor residual_scale,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g,
    int64_t head_size);

namespace {

void check_bf16_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(
      tensor.is_cuda() && tensor.is_contiguous() &&
          tensor.scalar_type() == torch::kBFloat16,
      name, " must be contiguous CUDA bfloat16");
}

void check_inputs(
    const torch::Tensor& x,
    const torch::Tensor& r,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& residual_scale,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& g,
    int64_t head_size) {
  TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256,
              "head_size must be one of 64, 128, or 256");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&x, "x"},
           {&r, "r"}, {&k, "k"}, {&v, "v"}, {&residual_scale, "residual_scale"},
           {&weight, "weight"}, {&bias, "bias"}, {&g, "g"},
       }) {
    check_bf16_cuda(*item.first, item.second);
  }
  TORCH_CHECK(
      x.dim() == 3 && x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0 &&
          x.size(2) % head_size == 0,
      "x must have shape [B,T,C] with C divisible by head_size");
  for (const auto& item : {&r, &k, &v, &g}) {
    TORCH_CHECK(item->sizes() == x.sizes(), "token tensors must match x");
  }
  const int64_t heads = x.size(2) / head_size;
  TORCH_CHECK(
      residual_scale.dim() == 2 && residual_scale.size(0) == heads &&
          residual_scale.size(1) == head_size,
      "residual_scale must have shape [C/head_size,head_size]");
  TORCH_CHECK(weight.dim() == 1 && bias.dim() == 1 &&
                  weight.size(0) == x.size(2) && bias.size(0) == x.size(2),
              "weight and bias must have shape [C]");
  TORCH_CHECK(
      x.device() == r.device() && x.device() == k.device() &&
          x.device() == v.device() && x.device() == residual_scale.device() &&
          x.device() == weight.device() && x.device() == bias.device() &&
          x.device() == g.device(),
      "lnx tensors must share a device");
}

}  // namespace

std::vector<torch::Tensor> pretrain_tmix_readout_forward(
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor residual_scale,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g,
    int64_t head_size) {
  check_inputs(x, r, k, v, residual_scale, weight, bias, g, head_size);
  return pretrain_tmix_readout_cuda(
      x, r, k, v, residual_scale, weight, bias, g, head_size);
}

void register_pretrain_tmix_readout_forward_bindings(py::module_& module) {
  module.def(
      "pretrain_tmix_readout_forward",
      &pretrain_tmix_readout_forward,
      "RWKV-7 train_temp fused LN/residual/gate forward",
      py::arg("x"), py::arg("r"), py::arg("k"), py::arg("v"),
      py::arg("residual_scale"), py::arg("weight"), py::arg("bias"),
      py::arg("g"), py::arg("head_size") = 64);
}
