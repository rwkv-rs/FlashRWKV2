// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
//
// The FP16 kernel body is adapted from Albatross faster3a_2607 at revision
// 3e41bc43ed5e8332927ddd7e0ce4816cf200a6ea.  vllm-rwkv is used only for the
// packed metadata/state-slot contract reference.

#include "../../../validation.h"


#include <cstdint>
#include <cmath>
#include <optional>
#include <utility>

void tmix_wkv7_recurrent_fp16_from_decay_logits_cuda(
    torch::stable::Tensor query_start_loc,
    torch::stable::Tensor state_indices,
    torch::stable::Tensor elapsed_state,
    torch::stable::Tensor state,
    torch::stable::Tensor r,
    torch::stable::Tensor decay_logits,
    torch::stable::Tensor decay_bias,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor a,
    torch::stable::Tensor b,
    torch::stable::Tensor output,
    torch::stable::Tensor metadata_status,
    double scale,
    int64_t max_seqlen);
void recurrent_fp16_advance_i32_varlen_cuda(
    torch::stable::Tensor query_start_loc,
    torch::stable::Tensor state_indices,
    torch::stable::Tensor elapsed_state,
    torch::stable::Tensor metadata_status);
using flashrwkv2::validation::check_cuda_contiguous;
using flashrwkv2::validation::check_same_device;
using flashrwkv2::validation::prepare_recurrent_metadata_cuda;

namespace {

void check_fp16_recurrent_layout(
    const torch::stable::Tensor& query_start_loc,
    const torch::stable::Tensor& state_indices,
    const torch::stable::Tensor& elapsed_state,
    const torch::stable::Tensor& state,
    const torch::stable::Tensor& r,
    const torch::stable::Tensor& decay_logits,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    const torch::stable::Tensor& output,
    double scale) {
  check_cuda_contiguous(query_start_loc, "query_start_loc");
  check_cuda_contiguous(state_indices, "state_indices");
  check_cuda_contiguous(elapsed_state, "elapsed_state_pool");
  check_cuda_contiguous(state, "state");
  check_cuda_contiguous(r, "r");
  check_cuda_contiguous(decay_logits, "decay_logits");
  check_cuda_contiguous(k, "k");
  check_cuda_contiguous(v, "v");
  check_cuda_contiguous(a, "a");
  check_cuda_contiguous(b, "b");
  check_cuda_contiguous(output, "output");
  check_same_device(state, query_start_loc, "query_start_loc");
  check_same_device(state, state_indices, "state_indices");
  check_same_device(state, elapsed_state, "elapsed_state_pool");
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{&r, "r"},
           {&decay_logits, "decay_logits"},
           {&k, "k"},
           {&v, "v"},
           {&a, "a"},
           {&b, "b"},
           {&output, "output"},
       }) {
    check_same_device(state, *item.first, item.second);
  }
  STD_TORCH_CHECK(
      query_start_loc.scalar_type() == torch::headeronly::ScalarType::Int &&
          state_indices.scalar_type() == torch::headeronly::ScalarType::Int &&
          elapsed_state.scalar_type() == torch::headeronly::ScalarType::Int,
      "recurrent metadata must be int32");
  STD_TORCH_CHECK(
      state_indices.dim() == 1 && state_indices.numel() > 0 &&
          query_start_loc.dim() == 1 &&
          query_start_loc.numel() == state_indices.numel() + 1,
      "query_start_loc must have shape [B+1] and state_indices shape [B]");
  STD_TORCH_CHECK(
      elapsed_state.dim() == 1 && elapsed_state.size(0) == state.size(0),
      "elapsed_state_pool must have shape [state_pool_slots]");
  STD_TORCH_CHECK(
      state.dim() == 4 && state.scalar_type() == torch::headeronly::ScalarType::Half &&
          state.size(0) > 0 && state.size(1) > 0 &&
          state.size(2) == state.size(3),
      "state must be contiguous float16 [slots,H,D,D]");
  STD_TORCH_CHECK(
      r.dim() == 3 && r.size(0) > 0 && r.size(1) == state.size(1) &&
          r.size(2) == state.size(2),
      "r must have packed shape [total_tokens,H,D] matching state");
  STD_TORCH_CHECK(
      r.sizes() == decay_logits.sizes() && r.sizes() == k.sizes() &&
          r.sizes() == v.sizes() && r.sizes() == a.sizes() &&
          r.sizes() == b.sizes() && r.sizes() == output.sizes(),
      "r,decay_logits,k,v,a,b,output shape mismatch");
  STD_TORCH_CHECK(
      r.scalar_type() == torch::headeronly::ScalarType::Half &&
          decay_logits.scalar_type() == torch::headeronly::ScalarType::Half &&
          k.scalar_type() == torch::headeronly::ScalarType::Half &&
          v.scalar_type() == torch::headeronly::ScalarType::Half &&
          a.scalar_type() == torch::headeronly::ScalarType::Half &&
          b.scalar_type() == torch::headeronly::ScalarType::Half &&
          output.scalar_type() == torch::headeronly::ScalarType::Half,
      "FP16-state recurrent token tensors must be float16");
  STD_TORCH_CHECK(std::isfinite(scale), "scale must be finite");
}

}  // namespace

void tmix_wkv7_recurrent_fp16_from_decay_logits(
    torch::stable::Tensor query_start_loc,
    torch::stable::Tensor state_indices,
    torch::stable::Tensor elapsed_state,
    torch::stable::Tensor state,
    torch::stable::Tensor r,
    torch::stable::Tensor decay_logits,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor a,
    torch::stable::Tensor b,
    torch::stable::Tensor output,
    double scale,
    std::optional<torch::stable::Tensor> decay_bias,
    torch::stable::Tensor metadata_status,
    int64_t max_seqlen) {
  check_fp16_recurrent_layout(
      query_start_loc, state_indices, elapsed_state, state, r, decay_logits,
      k, v, a, b, output, scale);
  if (decay_bias.has_value()) {
    check_cuda_contiguous(*decay_bias, "decay_bias");
    check_same_device(state, *decay_bias, "decay_bias");
    STD_TORCH_CHECK(
        decay_bias->scalar_type() == torch::headeronly::ScalarType::Half &&
            ((decay_bias->dim() == 1 &&
              decay_bias->numel() == state.size(1) * state.size(2)) ||
             (decay_bias->dim() == 2 && decay_bias->size(0) == state.size(1) &&
              decay_bias->size(1) == state.size(2))),
        "decay_bias must be float16 with shape [H*D] or [H,D]");
  }
  tmix_wkv7_recurrent_fp16_from_decay_logits_cuda(
      query_start_loc, state_indices, elapsed_state, state, r,
      decay_logits, decay_bias.value_or(torch::stable::Tensor()), k, v, a, b, output,
      metadata_status, scale, max_seqlen);
  recurrent_fp16_advance_i32_varlen_cuda(
      query_start_loc, state_indices, elapsed_state,
      metadata_status);
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("tmix_wkv7_recurrent_fp16_from_decay_logits(Tensor query_start_loc, Tensor state_indices, Tensor(a!) elapsed_state_pool, Tensor(b!) state, Tensor r, Tensor decay_logits, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(c!) output, float scale, Tensor? decay_bias, Tensor metadata_status, int max_seqlen) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("tmix_wkv7_recurrent_fp16_from_decay_logits", TORCH_BOX(&tmix_wkv7_recurrent_fp16_from_decay_logits));
}
