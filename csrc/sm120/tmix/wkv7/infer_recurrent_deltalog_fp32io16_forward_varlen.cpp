// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
//
// The DeltaLog algorithm is adapted from Albatross faster3a_2607 at revision
// 3e41bc43ed5e8332927ddd7e0ce4816cf200a6ea.  This binding owns the
// FlashRWKV2 FP32-state packed-varlen and fail-closed contracts.

#include "../../../validation.h"

#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <optional>
#include <utility>

void tmix_wkv7_recurrent_deltalog_fp32io16_from_decay_logits_cuda(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor phase_pool,
    torch::Tensor state_pool,
    torch::Tensor deltalog_pool,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor decay_bias,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor metadata_status,
    torch::Tensor deltalog_status,
    double scale);

using flashrwkv2::validation::check_cuda_contiguous;
using flashrwkv2::validation::check_same_device;
using flashrwkv2::validation::prepare_recurrent_metadata_cuda;

namespace {

void check_deltalog_layout(
    const torch::Tensor& query_start_loc,
    const torch::Tensor& state_indices,
    const torch::Tensor& phase_pool,
    const torch::Tensor& state_pool,
    const torch::Tensor& deltalog_pool,
    const torch::Tensor& r,
    const torch::Tensor& decay_logits,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& output,
    double scale) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{
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
  TORCH_CHECK(
      query_start_loc.scalar_type() == torch::kInt32 &&
          state_indices.scalar_type() == torch::kInt32 &&
          phase_pool.scalar_type() == torch::kInt32,
      "DeltaLog metadata and phase tensors must be int32");
  TORCH_CHECK(
      state_indices.dim() == 1 && state_indices.numel() > 0 &&
          query_start_loc.dim() == 1 &&
          query_start_loc.numel() == state_indices.numel() + 1,
      "query_start_loc must have shape [sequence_capacity+1] and "
      "state_indices shape [sequence_capacity]");
  TORCH_CHECK(
      state_pool.dim() == 4 && state_pool.scalar_type() == torch::kFloat32 &&
          state_pool.size(0) > 0 && state_pool.size(1) > 0 &&
          state_pool.size(2) == 64 && state_pool.size(3) == 64,
      "state_pool must be contiguous float32 [state_pool_slots,H,64,64]");
  TORCH_CHECK(
      phase_pool.dim() == 1 && phase_pool.size(0) == state_pool.size(0),
      "deltalog_phase_pool must have shape [state_pool_slots]");
  TORCH_CHECK(
      deltalog_pool.dim() == 5 &&
          (deltalog_pool.size(0) == 1 || deltalog_pool.size(0) == 2 ||
           deltalog_pool.size(0) == 3 || deltalog_pool.size(0) == 5 ||
           deltalog_pool.size(0) == 7) &&
          deltalog_pool.size(1) == 5 &&
          deltalog_pool.size(2) == state_pool.size(0) &&
          deltalog_pool.size(3) == state_pool.size(1) &&
          deltalog_pool.size(4) == 64 &&
          deltalog_pool.scalar_type() == torch::kFloat32,
      "deltalog_pool must be contiguous float32 "
      "[M-1,5,state_pool_slots,H,64] with M in {2,3,4,6,8}");
  TORCH_CHECK(
      r.dim() == 3 && r.size(0) > 0 &&
          r.size(1) == state_pool.size(1) && r.size(2) == 64,
      "r must have packed shape [token_capacity,H,64]");
  TORCH_CHECK(
      r.sizes() == decay_logits.sizes() && r.sizes() == k.sizes() &&
          r.sizes() == v.sizes() && r.sizes() == a.sizes() &&
          r.sizes() == b.sizes() && r.sizes() == output.sizes(),
      "r,decay_logits,k,v,a,b,output shape mismatch");
  TORCH_CHECK(
      r.scalar_type() == torch::kFloat16 &&
          decay_logits.scalar_type() == torch::kFloat16 &&
          k.scalar_type() == torch::kFloat16 &&
          v.scalar_type() == torch::kFloat16 &&
          a.scalar_type() == torch::kFloat16 &&
          b.scalar_type() == torch::kFloat16 &&
          output.scalar_type() == torch::kFloat16,
      "FP32IO16 DeltaLog token tensors must be float16");
  TORCH_CHECK(std::isfinite(scale), "scale must be finite");
}

torch::Tensor resolve_status_workspace(
    const torch::Tensor& state_pool,
    const torch::Tensor& state_indices,
    std::optional<torch::Tensor> workspace) {
  auto status = workspace.value_or(torch::Tensor());
  if (status.defined()) {
    check_cuda_contiguous(status, "deltalog_status_workspace");
    check_same_device(state_pool, status, "deltalog_status_workspace");
    TORCH_CHECK(
        status.scalar_type() == torch::kInt32 && status.dim() == 1 &&
            status.numel() == 1 + 2 * state_indices.numel(),
        "deltalog_status_workspace must be int32 [1+2*sequence_capacity]");
    return status;
  }
  return torch::empty(
      {1 + 2 * state_indices.numel()},
      state_pool.options().dtype(torch::kInt32));
}

}  // namespace

void tmix_wkv7_recurrent_deltalog_fp32io16_from_decay_logits(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor phase_pool,
    torch::Tensor state_pool,
    torch::Tensor deltalog_pool,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    double scale,
    std::optional<torch::Tensor> decay_bias,
    py::object validated_metadata,
    std::optional<torch::Tensor> deltalog_status_workspace) {
  check_deltalog_layout(
      query_start_loc, state_indices, phase_pool, state_pool, deltalog_pool, r,
      decay_logits, k, v, a, b, output, scale);
  if (decay_bias.has_value()) {
    check_cuda_contiguous(*decay_bias, "decay_bias");
    check_same_device(state_pool, *decay_bias, "decay_bias");
    TORCH_CHECK(
        decay_bias->scalar_type() == torch::kFloat16 &&
            ((decay_bias->dim() == 1 &&
              decay_bias->numel() == state_pool.size(1) * 64) ||
             (decay_bias->dim() == 2 &&
              decay_bias->size(0) == state_pool.size(1) &&
              decay_bias->size(1) == 64)),
        "decay_bias must be float16 with shape [H*64] or [H,64]");
  }

  torch::Tensor launch_query_start_loc = query_start_loc;
  torch::Tensor launch_state_indices = state_indices;
  torch::Tensor metadata_status;
  if (!validated_metadata.is_none()) {
    validated_metadata.attr("_check_compatible")(
        query_start_loc, state_indices, r.size(0), state_pool.size(0), -1);
    launch_query_start_loc = validated_metadata
        .attr("_query_start_loc_snapshot")()
        .cast<torch::Tensor>();
    launch_state_indices = validated_metadata
        .attr("_state_indices_snapshot")()
        .cast<torch::Tensor>();
    metadata_status =
        validated_metadata.attr("_status")().cast<torch::Tensor>();
  } else {
    auto prepared = prepare_recurrent_metadata_cuda(
        query_start_loc, state_indices, r.size(0), state_pool.size(0));
    launch_query_start_loc = std::move(prepared.query_start_loc);
    launch_state_indices = std::move(prepared.state_indices);
    metadata_status = std::move(prepared.status);
  }
  auto deltalog_status = resolve_status_workspace(
      state_pool, state_indices, deltalog_status_workspace);
  tmix_wkv7_recurrent_deltalog_fp32io16_from_decay_logits_cuda(
      launch_query_start_loc, launch_state_indices, phase_pool, state_pool,
      deltalog_pool, r, decay_logits, decay_bias.value_or(torch::Tensor()), k,
      v, a, b, output, metadata_status, deltalog_status, scale);
}

void register_infer_tmix_wkv7_recurrent_deltalog_fp32io16_bindings(
    py::module_& module) {
  module.def(
      "tmix_wkv7_recurrent_deltalog_fp32io16_from_decay_logits",
      &tmix_wkv7_recurrent_deltalog_fp32io16_from_decay_logits,
      "Slot-native packed DeltaLog recurrent forward with FP32 state",
      py::arg("query_start_loc"), py::arg("state_indices"),
      py::arg("deltalog_phase_pool"), py::arg("state_pool"),
      py::arg("deltalog_pool"), py::arg("r"), py::arg("decay_logits"),
      py::arg("k"), py::arg("v"), py::arg("a"), py::arg("b"),
      py::arg("output"), py::arg("scale"),
      py::arg("decay_bias") = py::none(),
      py::arg("validated_metadata") = py::none(),
      py::arg("deltalog_status_workspace") = py::none());
}
