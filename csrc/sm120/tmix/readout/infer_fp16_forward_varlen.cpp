// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
// Albatross source revision: ee3308f6922e59f2166c7fac3c5a192340a2b48e.

#include "../../internal/linear/backend.cuh"

#include "validation.h"

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
    torch::stable::Tensor x,
    torch::stable::Tensor r,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor r_k,
    torch::stable::Tensor weight,
    torch::stable::Tensor bias,
    torch::stable::Tensor g,
    torch::stable::Tensor output);
torch::stable::Tensor tmix_readout_projection_dispatch_f16_cuda(
    torch::stable::Tensor x, torch::stable::Tensor weight);

using flashrwkv2::validation::check_cuda_contiguous;
using flashrwkv2::validation::check_same_device;

namespace {

void check_half(const torch::stable::Tensor& tensor, const torch::stable::Tensor& reference, const char* name) {
  check_cuda_contiguous(tensor, name);
  check_same_device(reference, tensor, name);
  STD_TORCH_CHECK(tensor.scalar_type() == torch::headeronly::ScalarType::Half, name, " must be float16");
}

}  // namespace

torch::stable::Tensor tmix_readout_prelinear_internal(
    torch::stable::Tensor x,
    torch::stable::Tensor r,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor r_k,
    torch::stable::Tensor weight,
    torch::stable::Tensor bias,
    torch::stable::Tensor g,
    int64_t head_size,
    int64_t batch_size,
    int64_t max_seqlen) {
  check_half(x, x, "x");
  STD_TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256,
              "head_size must be one of 64, 128, or 256");
  STD_TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0 &&
                  x.size(1) % head_size == 0,
              "x must have packed shape [total_tokens,H*head_size]");
  const int64_t total_tokens = x.size(0);
  const int64_t channels = x.size(1);
  const int heads = static_cast<int>(channels / head_size);
  STD_TORCH_CHECK(batch_size > 0, "batch_size must be positive");
  STD_TORCH_CHECK(max_seqlen > 0, "max_seqlen must be positive");
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{&r, "r"},
           {&k, "k"}, {&v, "v"}, {&g, "g"},
       }) {
    check_half(*item.first, x, item.second);
    STD_TORCH_CHECK(item.first->sizes() == x.sizes(), item.second,
                " must match x's packed shape");
  }
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{&r_k, "r_k"},
           {&weight, "weight"}, {&bias, "bias"},
       }) {
    check_half(*item.first, x, item.second);
    STD_TORCH_CHECK(item.first->dim() == 1 && item.first->size(0) == channels,
                item.second, " must have shape [C]");
  }
  auto output = torch::stable::empty_like(x);
  tmix_lnx_rkvres_xg_forward_varlen_cuda(
      static_cast<int>(batch_size), static_cast<int>(max_seqlen),
      static_cast<int>(total_tokens), static_cast<int>(channels), heads,
      static_cast<int>(head_size),
      x, r, k, v, r_k, weight, bias, g, output);
  return output;
}

torch::stable::Tensor tmix_readout_forward_varlen(
    torch::stable::Tensor wkv_output,
    torch::stable::Tensor receptance,
    torch::stable::Tensor key,
    torch::stable::Tensor value,
    torch::stable::Tensor r_k,
    torch::stable::Tensor ln_weight,
    torch::stable::Tensor ln_bias,
    torch::stable::Tensor gate,
    torch::stable::Tensor output_weight,
    std::optional<torch::stable::Tensor> output_lora_a,
    std::optional<torch::stable::Tensor> output_lora_b,
    double output_lora_scale,
    int64_t head_size,
    int64_t batch_size,
    int64_t max_seqlen) {
  auto prelinear = tmix_readout_prelinear_internal(
      wkv_output, receptance, key, value, r_k, ln_weight, ln_bias, gate,
      head_size, batch_size, max_seqlen);
  check_half(output_weight, wkv_output, "output_weight");
  STD_TORCH_CHECK(output_weight.dim() == 2 &&
                  output_weight.size(0) == prelinear.size(1) &&
                  output_weight.size(1) == prelinear.size(1),
              "output_weight must have shape [C,C]");
  STD_TORCH_CHECK(output_lora_a.has_value() == output_lora_b.has_value(),
              "output_lora_a and output_lora_b must be provided together");
  STD_TORCH_CHECK(std::isfinite(output_lora_scale) &&
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
  STD_TORCH_CHECK(output_lora_a->dim() == 2 &&
                  output_lora_a->size(1) == prelinear.size(1),
              "output_lora_a must have shape [R,C]");
  STD_TORCH_CHECK(output_lora_a->size(0) > 0 && output_lora_a->size(0) <= 512,
              "output LoRA projection requires 0<R<=512");
  STD_TORCH_CHECK(output_lora_b->dim() == 2 &&
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

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("tmix_readout_forward_varlen(Tensor wkv_output, Tensor receptance, Tensor key, Tensor value, Tensor r_k, Tensor ln_weight, Tensor ln_bias, Tensor gate, Tensor output_weight, Tensor? output_lora_a, Tensor? output_lora_b, float output_lora_scale, int head_size, int batch_size, int max_seqlen) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("tmix_readout_forward_varlen", TORCH_BOX(&tmix_readout_forward_varlen));
}
