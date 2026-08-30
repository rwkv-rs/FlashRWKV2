// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
// FlashKDA source revision: 1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b.
// This binding keeps the RWKV-7 K1/K2 chunk algebra and exposes only the raw
// decay-logit boundary.  It is not a KDA-attention compatibility wrapper.

#include "../../../validation.h"

#include <torch/extension.h>

#include <cmath>
#include <optional>
#include <utility>
#include <vector>

void infer_tmix_wkv7_chunk_bf16_forward_varlen_cuda(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor state_pool,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor decay_bias,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor chunk_transform,
    torch::Tensor chunk_bias,
    torch::Tensor token_transform,
    torch::Tensor token_bias,
    int64_t chunk_size,
    int64_t max_seqlen,
    double scale,
    torch::Tensor metadata_status);

using flashrwkv2::validation::check_cuda_contiguous;
using flashrwkv2::validation::check_same_device;
using flashrwkv2::validation::prepare_recurrent_metadata_cuda;

namespace {

void check_bf16(
    const torch::Tensor& tensor,
    const torch::Tensor& reference,
    const char* name) {
  check_cuda_contiguous(tensor, name);
  check_same_device(reference, tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16, name, " must be bf16");
}

void check_metadata(
    const torch::Tensor& query_start_loc,
    const torch::Tensor& state_indices,
    const torch::Tensor& reference,
    int64_t total_tokens,
    int64_t state_pool_size) {
  check_cuda_contiguous(query_start_loc, "cu_seqlens");
  check_cuda_contiguous(state_indices, "state_indices");
  check_same_device(reference, query_start_loc, "cu_seqlens");
  check_same_device(reference, state_indices, "state_indices");
  TORCH_CHECK(
      query_start_loc.scalar_type() == torch::kInt32 &&
          state_indices.scalar_type() == torch::kInt32,
      "chunk metadata must be int32");
  TORCH_CHECK(
      query_start_loc.dim() == 1 && state_indices.dim() == 1 &&
          state_indices.numel() > 0 &&
          query_start_loc.numel() == state_indices.numel() + 1,
      "cu_seqlens must have shape [B+1] and state_indices [B]");
  (void)total_tokens;
  (void)state_pool_size;
}

}  // namespace

py::tuple infer_tmix_wkv7_chunk_bf16_forward_varlen(
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor state_pool,
    torch::Tensor cu_seqlens,
    torch::Tensor state_indices,
    int64_t chunk_size,
    int64_t max_seqlen,
    double scale,
    std::optional<torch::Tensor> decay_bias,
    py::object validated_metadata) {
  check_bf16(r, r, "r");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{
               &decay_logits, "decay_logits"},
           {&k, "k"},
           {&v, "v"},
           {&a, "a"},
           {&b, "b"},
           {&state_pool, "state_pool"},
       }) {
    check_bf16(*item.first, r, item.second);
  }
  TORCH_CHECK(std::isfinite(scale), "scale must be finite");
  TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
  TORCH_CHECK(max_seqlen > 0, "max_seqlen must be positive");
  TORCH_CHECK(
      r.dim() == 3 && r.size(0) > 0 && r.size(1) > 0 && r.size(2) == 64,
      "chunk token tensors must have shape [total_tokens,H,64]");
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{
               &decay_logits, "decay_logits"},
           {&k, "k"},
           {&v, "v"},
           {&a, "a"},
           {&b, "b"},
       }) {
    TORCH_CHECK(item.first->sizes() == r.sizes(), item.second,
                " must have the same shape as r");
  }
  TORCH_CHECK(
      state_pool.dim() == 4 && state_pool.size(0) > 0 &&
          state_pool.size(1) == r.size(1) && state_pool.size(2) == 64 &&
          state_pool.size(3) == 64,
      "state_pool must have shape [slots,H,64,64]");
  check_metadata(
      cu_seqlens, state_indices, r, r.size(0), state_pool.size(0));
  if (decay_bias.has_value()) {
    check_bf16(*decay_bias, r, "decay_bias");
    TORCH_CHECK(
        (decay_bias->dim() == 1 && decay_bias->numel() == r.size(1) * 64) ||
            (decay_bias->dim() == 2 && decay_bias->size(0) == r.size(1) &&
             decay_bias->size(1) == 64),
        "decay_bias must have shape [H*64] or [H,64]");
  }

  const int64_t batch_size = state_indices.numel();
  const int64_t max_chunks = (max_seqlen + chunk_size - 1) / chunk_size;
  auto output = torch::empty_like(v);
  const std::vector<int64_t> chunk_workspace_shape{
      batch_size, max_chunks, r.size(1), 64, 64};
  auto chunk_transform = torch::empty(
      chunk_workspace_shape,
      r.options().dtype(torch::kFloat32));
  auto chunk_bias = torch::empty_like(chunk_transform);
  auto token_transform = torch::empty(
      r.sizes(), r.options().dtype(torch::kFloat32));
  auto token_bias = torch::empty_like(token_transform);
  torch::Tensor launch_query_start_loc = cu_seqlens;
  torch::Tensor launch_state_indices = state_indices;
  torch::Tensor metadata_status;
  if (!validated_metadata.is_none()) {
    validated_metadata.attr("_check_compatible")(
        cu_seqlens, state_indices, r.size(0), state_pool.size(0),
        max_seqlen);
    launch_query_start_loc = validated_metadata
        .attr("_query_start_loc_snapshot")()
        .cast<torch::Tensor>();
    launch_state_indices = validated_metadata
        .attr("_state_indices_snapshot")()
        .cast<torch::Tensor>();
    metadata_status = validated_metadata.attr("_status")().cast<torch::Tensor>();
    max_seqlen = validated_metadata.attr("_max_seqlen")().cast<int64_t>();
  } else {
    auto prepared = prepare_recurrent_metadata_cuda(
        cu_seqlens, state_indices, r.size(0), state_pool.size(0));
    launch_query_start_loc = std::move(prepared.query_start_loc);
    launch_state_indices = std::move(prepared.state_indices);
    metadata_status = std::move(prepared.status);
  }
  infer_tmix_wkv7_chunk_bf16_forward_varlen_cuda(
      launch_query_start_loc,
      launch_state_indices,
      state_pool,
      r,
      decay_logits,
      decay_bias.value_or(torch::Tensor()),
      k,
      v,
      a,
      b,
      output,
      chunk_transform,
      chunk_bias,
      token_transform,
      token_bias,
      chunk_size,
      max_seqlen,
      scale,
      metadata_status);
  return py::make_tuple(output, state_pool);
}

void register_infer_tmix_wkv7_chunk_bindings(py::module_& module) {
  module.def(
      "infer_tmix_wkv7_chunk_bf16_forward_varlen",
      &infer_tmix_wkv7_chunk_bf16_forward_varlen,
      "Packed RWKV-7 BF16 K1/K2 chunk inference",
      py::arg("r"), py::arg("decay_logits"), py::arg("k"), py::arg("v"),
      py::arg("a"), py::arg("b"), py::arg("state_pool"),
      py::arg("cu_seqlens"), py::arg("state_indices"),
      py::arg("chunk_size") = 16, py::arg("max_seqlen") = -1,
      py::arg("scale") = 1.0, py::arg("decay_bias") = py::none(),
      py::arg("validated_metadata") = py::none());
}
