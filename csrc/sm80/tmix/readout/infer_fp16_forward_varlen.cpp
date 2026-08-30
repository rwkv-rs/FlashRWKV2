// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
// Albatross source revision: ee3308f6922e59f2166c7fac3c5a192340a2b48e.

#include "../../../validation.h"
#include "../../internal/linear/backend.cuh"

#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

void tmix_lnx_rkvres_xg_forward_varlen_cuda(
    int batch_size,
    int max_seqlen,
    int total_tokens,
    int channels,
    int heads,
    int head_size,
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g,
    torch::Tensor output);
torch::Tensor tmix_readout_projection_dispatch_f16_cuda(
    torch::Tensor x, torch::Tensor weight);

using flashrwkv2::validation::check_cuda_contiguous;
using flashrwkv2::validation::check_same_device;

namespace {

void check_half(const torch::Tensor& tensor, const torch::Tensor& reference, const char* name) {
  check_cuda_contiguous(tensor, name);
  check_same_device(reference, tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, name, " must be float16");
}

}  // namespace

torch::Tensor tmix_readout_prelinear_internal(
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g,
    int64_t head_size,
    int64_t batch_size,
    int64_t max_seqlen) {
  check_half(x, x, "x");
  TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256,
              "head_size must be one of 64, 128, or 256");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0 &&
                  x.size(1) % head_size == 0,
              "x must have packed shape [total_tokens,H*head_size]");
  const int64_t total_tokens = x.size(0);
  const int64_t channels = x.size(1);
  const int heads = static_cast<int>(channels / head_size);
  TORCH_CHECK(batch_size > 0, "batch_size must be positive");
  TORCH_CHECK(max_seqlen > 0, "max_seqlen must be positive");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&r, "r"},
           {&k, "k"}, {&v, "v"}, {&g, "g"},
       }) {
    check_half(*item.first, x, item.second);
    TORCH_CHECK(item.first->sizes() == x.sizes(), item.second,
                " must match x's packed shape");
  }
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{&r_k, "r_k"},
           {&weight, "weight"}, {&bias, "bias"},
       }) {
    check_half(*item.first, x, item.second);
    TORCH_CHECK(item.first->dim() == 1 && item.first->size(0) == channels,
                item.second, " must have shape [C]");
  }
  auto output = torch::empty_like(x);
  tmix_lnx_rkvres_xg_forward_varlen_cuda(
      static_cast<int>(batch_size), static_cast<int>(max_seqlen),
      static_cast<int>(total_tokens), static_cast<int>(channels), heads,
      static_cast<int>(head_size),
      x, r, k, v, r_k, weight, bias, g, output);
  return output;
}

torch::Tensor tmix_readout_forward_varlen(
    torch::Tensor wkv_output,
    torch::Tensor receptance,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor r_k,
    torch::Tensor ln_weight,
    torch::Tensor ln_bias,
    torch::Tensor gate,
    torch::Tensor output_weight,
    std::optional<torch::Tensor> output_lora_a,
    std::optional<torch::Tensor> output_lora_b,
    double output_lora_scale,
    int64_t head_size,
    int64_t batch_size,
    int64_t max_seqlen) {
  auto prelinear = tmix_readout_prelinear_internal(
      wkv_output, receptance, key, value, r_k, ln_weight, ln_bias, gate,
      head_size, batch_size, max_seqlen);
  check_half(output_weight, wkv_output, "output_weight");
  TORCH_CHECK(output_weight.dim() == 2 &&
                  output_weight.size(0) == prelinear.size(1) &&
                  output_weight.size(1) == prelinear.size(1),
              "output_weight must have shape [C,C]");
  TORCH_CHECK(output_lora_a.has_value() == output_lora_b.has_value(),
              "output_lora_a and output_lora_b must be provided together");
  TORCH_CHECK(std::isfinite(output_lora_scale) &&
                  std::abs(output_lora_scale) <=
                      std::numeric_limits<float>::max(),
              "output_lora_scale must be finite and representable as float32");
  auto output = tmix_readout_projection_dispatch_f16_cuda(
      prelinear, output_weight);
  if (!output_lora_a.has_value()) {
    return output;
  }
  check_half(*output_lora_a, wkv_output, "output_lora_a");
  check_half(*output_lora_b, wkv_output, "output_lora_b");
  TORCH_CHECK(output_lora_a->dim() == 2 &&
                  output_lora_a->size(1) == prelinear.size(1),
              "output_lora_a must have shape [R,C]");
  TORCH_CHECK(output_lora_a->size(0) > 0 && output_lora_a->size(0) <= 512,
              "output LoRA projection requires 0<R<=512");
  TORCH_CHECK(output_lora_b->dim() == 2 &&
                  output_lora_b->size(0) == prelinear.size(1) &&
                  output_lora_b->size(1) == output_lora_a->size(0),
              "output_lora_b must have shape [C,R]");
  if (static_cast<float>(output_lora_scale) == 0.0f) {
    return output;
  }
  return internal_linear_lora_accumulate_f16_cuda(
      prelinear, *output_lora_a, *output_lora_b, output,
      output_lora_scale);
}

void register_tmix_readout_bindings(py::module_& module) {
  module.def(
      "tmix_readout_forward_varlen", &tmix_readout_forward_varlen,
      "Packed TMix readout and output projection",
      py::arg("wkv_output"), py::arg("receptance"), py::arg("key"),
      py::arg("value"), py::arg("r_k"), py::arg("ln_weight"),
      py::arg("ln_bias"), py::arg("gate"), py::arg("output_weight"),
      py::arg("output_lora_a") = py::none(),
      py::arg("output_lora_b") = py::none(),
      py::arg("output_lora_scale") = 1.0, py::arg("head_size") = 64,
      py::arg("batch_size") = 1, py::arg("max_seqlen") = 1);
}
