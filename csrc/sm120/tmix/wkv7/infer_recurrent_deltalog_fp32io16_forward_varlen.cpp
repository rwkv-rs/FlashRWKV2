// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
//
// The DeltaLog algorithm is adapted from Albatross faster3a_2607 at revision
// 3e41bc43ed5e8332927ddd7e0ce4816cf200a6ea.  This binding owns the
// FlashRWKV2 FP32-state packed-varlen and fail-closed contracts.

#include "../../../validation.h"


#include <cmath>
#include <optional>
#include <utility>

void tmix_wkv7_recurrent_deltalog_fp32io16_from_decay_logits_cuda(
    torch::stable::Tensor query_start_loc,
    torch::stable::Tensor state_indices,
    torch::stable::Tensor phase_pool,
    torch::stable::Tensor state_pool,
    torch::stable::Tensor deltalog_pool,
    torch::stable::Tensor r,
    torch::stable::Tensor decay_logits,
    torch::stable::Tensor decay_bias,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor a,
    torch::stable::Tensor b,
    torch::stable::Tensor output,
    torch::stable::Tensor metadata_status,
    torch::stable::Tensor deltalog_status,
    double scale);

void tmix_wkv7_recurrent_deltalog_fp32io16_materialize_slots_cuda(
    torch::stable::Tensor state_indices,
    torch::stable::Tensor phase_pool,
    torch::stable::Tensor state_pool,
    torch::stable::Tensor deltalog_pool,
    torch::stable::Tensor deltalog_status,
    torch::stable::Tensor metadata_status);

using flashrwkv2::validation::check_cuda_contiguous;
using flashrwkv2::validation::check_same_device;
using flashrwkv2::validation::prepare_recurrent_metadata_cuda;

namespace {

void check_deltalog_layout(
    const torch::stable::Tensor& query_start_loc,
    const torch::stable::Tensor& state_indices,
    const torch::stable::Tensor& phase_pool,
    const torch::stable::Tensor& state_pool,
    const torch::stable::Tensor& deltalog_pool,
    const torch::stable::Tensor& r,
    const torch::stable::Tensor& decay_logits,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    const torch::stable::Tensor& output,
    double scale) {
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{
               &query_start_loc, "query_start_loc"},
           {&state_indices, "state_indices"},
           {&phase_pool, "deltalog_phase_pool"},
           {&state_pool, "state_pool"},
           {&deltalog_pool, "deltalog_pool"},
           {&r, "r"},
           {&decay_logits, "decay_logits"},
           {&k, "k"},
           {&v, "v"},
           {&a, "a"},
           {&b, "b"},
           {&output, "output"},
       }) {
    check_cuda_contiguous(*item.first, item.second);
    check_same_device(state_pool, *item.first, item.second);
  }
  STD_TORCH_CHECK(
      query_start_loc.scalar_type() == torch::headeronly::ScalarType::Int &&
          state_indices.scalar_type() == torch::headeronly::ScalarType::Int &&
          phase_pool.scalar_type() == torch::headeronly::ScalarType::Int,
      "DeltaLog metadata and phase tensors must be int32");
  STD_TORCH_CHECK(
      state_indices.dim() == 1 && state_indices.numel() > 0 &&
          query_start_loc.dim() == 1 &&
          query_start_loc.numel() == state_indices.numel() + 1,
      "query_start_loc must have shape [sequence_capacity+1] and "
      "state_indices shape [sequence_capacity]");
  STD_TORCH_CHECK(
      state_pool.dim() == 4 && state_pool.scalar_type() == torch::headeronly::ScalarType::Float &&
          state_pool.size(0) > 0 && state_pool.size(1) > 0 &&
          state_pool.size(2) == 64 && state_pool.size(3) == 64,
      "state_pool must be contiguous float32 [state_pool_slots,H,64,64]");
  STD_TORCH_CHECK(
      phase_pool.dim() == 1 && phase_pool.size(0) == state_pool.size(0),
      "deltalog_phase_pool must have shape [state_pool_slots]");
  STD_TORCH_CHECK(
      deltalog_pool.dim() == 5 &&
          (deltalog_pool.size(0) == 1 || deltalog_pool.size(0) == 2 ||
           deltalog_pool.size(0) == 3 || deltalog_pool.size(0) == 5 ||
           deltalog_pool.size(0) == 7) &&
          deltalog_pool.size(1) == 5 &&
          deltalog_pool.size(2) == state_pool.size(0) &&
          deltalog_pool.size(3) == state_pool.size(1) &&
          deltalog_pool.size(4) == 64 &&
          deltalog_pool.scalar_type() == torch::headeronly::ScalarType::Float,
      "deltalog_pool must be contiguous float32 "
      "[M-1,5,state_pool_slots,H,64] with M in {2,3,4,6,8}");
  STD_TORCH_CHECK(
      r.dim() == 3 && r.size(0) > 0 &&
          r.size(1) == state_pool.size(1) && r.size(2) == 64,
      "r must have packed shape [token_capacity,H,64]");
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
      "FP32IO16 DeltaLog token tensors must be float16");
  STD_TORCH_CHECK(std::isfinite(scale), "scale must be finite");
}

torch::stable::Tensor resolve_status_workspace(
    const torch::stable::Tensor& state_pool,
    const torch::stable::Tensor& state_indices,
    std::optional<torch::stable::Tensor> workspace) {
  auto status = workspace.value_or(torch::stable::Tensor());
  if (status.defined()) {
    check_cuda_contiguous(status, "deltalog_status_workspace");
    check_same_device(state_pool, status, "deltalog_status_workspace");
    STD_TORCH_CHECK(
        status.scalar_type() == torch::headeronly::ScalarType::Int && status.dim() == 1 &&
            status.numel() == 1 + 2 * state_indices.numel(),
        "deltalog_status_workspace must be int32 [1+2*sequence_capacity]");
    return status;
  }
  return torch::stable::new_empty(
      state_pool, {1 + 2 * state_indices.numel()},
      torch::headeronly::ScalarType::Int);
}

void check_materialize_layout(
    const torch::stable::Tensor& state_indices,
    const torch::stable::Tensor& phase_pool,
    const torch::stable::Tensor& state_pool,
    const torch::stable::Tensor& deltalog_pool,
    const torch::stable::Tensor& deltalog_status,
    const std::optional<torch::stable::Tensor>& metadata_status) {
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{
               &state_indices, "state_indices"},
           {&phase_pool, "deltalog_phase_pool"},
           {&state_pool, "state_pool"},
           {&deltalog_pool, "deltalog_pool"},
           {&deltalog_status, "deltalog_status_workspace"},
       }) {
    check_cuda_contiguous(*item.first, item.second);
    check_same_device(state_pool, *item.first, item.second);
  }
  STD_TORCH_CHECK(
      state_indices.scalar_type() == torch::headeronly::ScalarType::Int &&
          state_indices.dim() == 1 && state_indices.numel() > 0,
      "state_indices must be non-empty contiguous CUDA int32 [N]");
  STD_TORCH_CHECK(
      state_pool.dim() == 4 && state_pool.scalar_type() == torch::headeronly::ScalarType::Float &&
          state_pool.size(0) > 0 && state_pool.size(1) > 0 &&
          state_pool.size(2) == 64 && state_pool.size(3) == 64,
      "state_pool must be contiguous float32 [state_pool_slots,H,64,64]");
  STD_TORCH_CHECK(
      phase_pool.scalar_type() == torch::headeronly::ScalarType::Int && phase_pool.dim() == 1 &&
          phase_pool.size(0) == state_pool.size(0),
      "deltalog_phase_pool must be int32 [state_pool_slots]");
  STD_TORCH_CHECK(
      deltalog_pool.dim() == 5 &&
          (deltalog_pool.size(0) == 1 || deltalog_pool.size(0) == 2 ||
           deltalog_pool.size(0) == 3 || deltalog_pool.size(0) == 5 ||
           deltalog_pool.size(0) == 7) &&
          deltalog_pool.size(1) == 5 &&
          deltalog_pool.size(2) == state_pool.size(0) &&
          deltalog_pool.size(3) == state_pool.size(1) &&
          deltalog_pool.size(4) == 64 &&
          deltalog_pool.scalar_type() == torch::headeronly::ScalarType::Float,
      "deltalog_pool must be contiguous float32 "
      "[M-1,5,state_pool_slots,H,64] with M in {2,3,4,6,8}");
  STD_TORCH_CHECK(
      deltalog_status.scalar_type() == torch::headeronly::ScalarType::Int &&
          deltalog_status.dim() == 1 && deltalog_status.numel() >= 1,
      "deltalog_status_workspace must be non-empty int32");
  if (metadata_status.has_value()) {
    check_cuda_contiguous(*metadata_status, "metadata_status");
    check_same_device(state_pool, *metadata_status, "metadata_status");
    STD_TORCH_CHECK(
        metadata_status->scalar_type() == torch::headeronly::ScalarType::Int &&
            metadata_status->dim() == 1 && metadata_status->numel() >= 3,
        "metadata_status must contain validation and active counts");
  }
}

}  // namespace

void tmix_wkv7_recurrent_deltalog_fp32io16_from_decay_logits(
    torch::stable::Tensor query_start_loc,
    torch::stable::Tensor state_indices,
    torch::stable::Tensor phase_pool,
    torch::stable::Tensor state_pool,
    torch::stable::Tensor deltalog_pool,
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
    std::optional<torch::stable::Tensor> deltalog_status_workspace) {
  check_deltalog_layout(
      query_start_loc, state_indices, phase_pool, state_pool, deltalog_pool, r,
      decay_logits, k, v, a, b, output, scale);
  if (decay_bias.has_value()) {
    check_cuda_contiguous(*decay_bias, "decay_bias");
    check_same_device(state_pool, *decay_bias, "decay_bias");
    STD_TORCH_CHECK(
        decay_bias->scalar_type() == torch::headeronly::ScalarType::Half &&
            ((decay_bias->dim() == 1 &&
              decay_bias->numel() == state_pool.size(1) * 64) ||
             (decay_bias->dim() == 2 &&
              decay_bias->size(0) == state_pool.size(1) &&
              decay_bias->size(1) == 64)),
        "decay_bias must be float16 with shape [H*64] or [H,64]");
  }

  auto deltalog_status = resolve_status_workspace(
      state_pool, state_indices, deltalog_status_workspace);
  tmix_wkv7_recurrent_deltalog_fp32io16_from_decay_logits_cuda(
      query_start_loc, state_indices, phase_pool, state_pool,
      deltalog_pool, r, decay_logits, decay_bias.value_or(torch::stable::Tensor()), k,
      v, a, b, output, metadata_status, deltalog_status, scale);
}

void tmix_wkv7_recurrent_deltalog_fp32io16_materialize_slots(
    torch::stable::Tensor state_indices,
    torch::stable::Tensor phase_pool,
    torch::stable::Tensor state_pool,
    torch::stable::Tensor deltalog_pool,
    torch::stable::Tensor deltalog_status,
    std::optional<torch::stable::Tensor> metadata_status) {
  check_materialize_layout(
      state_indices, phase_pool, state_pool, deltalog_pool, deltalog_status,
      metadata_status);
  tmix_wkv7_recurrent_deltalog_fp32io16_materialize_slots_cuda(
      state_indices, phase_pool, state_pool, deltalog_pool, deltalog_status,
      metadata_status.value_or(torch::stable::Tensor()));
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("tmix_wkv7_recurrent_deltalog_fp32io16_from_decay_logits(Tensor query_start_loc, Tensor state_indices, Tensor(a!) deltalog_phase_pool, Tensor(b!) state_pool, Tensor(c!) deltalog_pool, Tensor r, Tensor decay_logits, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(d!) output, float scale, Tensor? decay_bias, Tensor metadata_status, Tensor? deltalog_status_workspace) -> ()");
  module.def("tmix_wkv7_recurrent_deltalog_fp32io16_materialize_slots(Tensor state_indices, Tensor(a!) deltalog_phase_pool, Tensor(b!) state_pool, Tensor(c!) deltalog_pool, Tensor(d!) deltalog_status_workspace, Tensor? metadata_status) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("tmix_wkv7_recurrent_deltalog_fp32io16_from_decay_logits", TORCH_BOX(&tmix_wkv7_recurrent_deltalog_fp32io16_from_decay_logits));
  module.impl("tmix_wkv7_recurrent_deltalog_fp32io16_materialize_slots", TORCH_BOX(&tmix_wkv7_recurrent_deltalog_fp32io16_materialize_slots));
}
