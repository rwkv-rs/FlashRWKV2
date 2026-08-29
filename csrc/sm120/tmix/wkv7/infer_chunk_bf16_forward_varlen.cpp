// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
// FlashKDA source revision: 1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b.
// This binding keeps the RWKV-7 K1/K2 chunk algebra and exposes only the raw
// decay-logit boundary.  It is not a KDA-attention compatibility wrapper.

#include "../../../validation.h"

#include "validation.h"

#include <cmath>
#include <optional>
#include <utility>
#include <vector>

void infer_tmix_wkv7_chunk_bf16_forward_varlen_cuda(
    torch::stable::Tensor query_start_loc,
    torch::stable::Tensor state_indices,
    torch::stable::Tensor state_pool,
    torch::stable::Tensor r,
    torch::stable::Tensor decay_logits,
    torch::stable::Tensor decay_bias,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor a,
    torch::stable::Tensor b,
    torch::stable::Tensor output,
    torch::stable::Tensor chunk_transform,
    torch::stable::Tensor chunk_bias,
    torch::stable::Tensor token_transform,
    torch::stable::Tensor token_bias,
    int64_t chunk_size,
    int64_t max_seqlen,
    double scale,
    torch::stable::Tensor metadata_status);

using flashrwkv2::validation::check_cuda_contiguous;
using flashrwkv2::validation::check_same_device;
using flashrwkv2::validation::prepare_recurrent_metadata_cuda;

namespace {

void check_bf16(
    const torch::stable::Tensor& tensor,
    const torch::stable::Tensor& reference,
    const char* name) {
  check_cuda_contiguous(tensor, name);
  check_same_device(reference, tensor, name);
  STD_TORCH_CHECK(tensor.scalar_type() == torch::headeronly::ScalarType::BFloat16, name, " must be bf16");
}

void check_metadata(
    const torch::stable::Tensor& query_start_loc,
    const torch::stable::Tensor& state_indices,
    const torch::stable::Tensor& reference,
    int64_t total_tokens,
    int64_t state_pool_size) {
  check_cuda_contiguous(query_start_loc, "cu_seqlens");
  check_cuda_contiguous(state_indices, "state_indices");
  check_same_device(reference, query_start_loc, "cu_seqlens");
  check_same_device(reference, state_indices, "state_indices");
  STD_TORCH_CHECK(
      query_start_loc.scalar_type() == torch::headeronly::ScalarType::Int &&
          state_indices.scalar_type() == torch::headeronly::ScalarType::Int,
      "chunk metadata must be int32");
  STD_TORCH_CHECK(
      query_start_loc.dim() == 1 && state_indices.dim() == 1 &&
          state_indices.numel() > 0 &&
          query_start_loc.numel() == state_indices.numel() + 1,
      "cu_seqlens must have shape [B+1] and state_indices [B]");
  (void)total_tokens;
  (void)state_pool_size;
}

}  // namespace

std::tuple<torch::stable::Tensor, torch::stable::Tensor> infer_tmix_wkv7_chunk_bf16_forward_varlen(
    torch::stable::Tensor r,
    torch::stable::Tensor decay_logits,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor a,
    torch::stable::Tensor b,
    torch::stable::Tensor state_pool,
    torch::stable::Tensor cu_seqlens,
    torch::stable::Tensor state_indices,
    torch::stable::Tensor metadata_status,
    int64_t chunk_size,
    int64_t max_seqlen,
    double scale,
    std::optional<torch::stable::Tensor> decay_bias) {
  check_bf16(r, r, "r");
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{
               &decay_logits, "decay_logits"},
           {&k, "k"},
           {&v, "v"},
           {&a, "a"},
           {&b, "b"},
           {&state_pool, "state_pool"},
       }) {
    check_bf16(*item.first, r, item.second);
  }
  STD_TORCH_CHECK(std::isfinite(scale), "scale must be finite");
  STD_TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
  STD_TORCH_CHECK(max_seqlen > 0, "max_seqlen must be positive");
  STD_TORCH_CHECK(
      r.dim() == 3 && r.size(0) > 0 && r.size(1) > 0 && r.size(2) == 64,
      "chunk token tensors must have shape [total_tokens,H,64]");
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{
               &decay_logits, "decay_logits"},
           {&k, "k"},
           {&v, "v"},
           {&a, "a"},
           {&b, "b"},
       }) {
    STD_TORCH_CHECK(item.first->sizes() == r.sizes(), item.second,
                " must have the same shape as r");
  }
  STD_TORCH_CHECK(
      state_pool.dim() == 4 && state_pool.size(0) > 0 &&
          state_pool.size(1) == r.size(1) && state_pool.size(2) == 64 &&
          state_pool.size(3) == 64,
      "state_pool must have shape [slots,H,64,64]");
  check_metadata(
      cu_seqlens, state_indices, r, r.size(0), state_pool.size(0));
  if (decay_bias.has_value()) {
    check_bf16(*decay_bias, r, "decay_bias");
    STD_TORCH_CHECK(
        (decay_bias->dim() == 1 && decay_bias->numel() == r.size(1) * 64) ||
            (decay_bias->dim() == 2 && decay_bias->size(0) == r.size(1) &&
             decay_bias->size(1) == 64),
        "decay_bias must have shape [H*64] or [H,64]");
  }

  const int64_t batch_size = state_indices.numel();
  const int64_t max_chunks = (max_seqlen + chunk_size - 1) / chunk_size;
  auto output = torch::stable::empty_like(v);
  const std::vector<int64_t> chunk_workspace_shape{
      batch_size, max_chunks, r.size(1), 64, 64};
  auto chunk_transform = torch::stable::new_empty(
      r, chunk_workspace_shape, torch::headeronly::ScalarType::Float);
  auto chunk_bias = torch::stable::empty_like(chunk_transform);
  auto token_transform = torch::stable::new_empty(
      r, r.sizes(), torch::headeronly::ScalarType::Float);
  auto token_bias = torch::stable::empty_like(token_transform);
  infer_tmix_wkv7_chunk_bf16_forward_varlen_cuda(
      cu_seqlens,
      state_indices,
      state_pool,
      r,
      decay_logits,
      decay_bias.value_or(torch::stable::Tensor()),
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
  return std::make_tuple(std::move(output), std::move(state_pool));
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("infer_tmix_wkv7_chunk_bf16_forward_varlen(Tensor r, Tensor decay_logits, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) state_pool, Tensor cu_seqlens, Tensor state_indices, Tensor metadata_status, int chunk_size, int max_seqlen, float scale, Tensor? decay_bias) -> (Tensor, Tensor(a!))");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("infer_tmix_wkv7_chunk_bf16_forward_varlen", TORCH_BOX(&infer_tmix_wkv7_chunk_bf16_forward_varlen));
}
