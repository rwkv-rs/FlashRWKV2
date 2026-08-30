// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from vllm-rwkv rwkv7_wkv_fp32_v2 at commit
// 6d683f9e49a2997e405c47edc147872c8609513b.

#include "../../../validation.h"


#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>

void tmix_wkv7_recurrent_fp32_from_decay_logits_cuda(
    torch::stable::Tensor query_start_loc,
    torch::stable::Tensor state_indices,
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

using flashrwkv2::validation::check_cuda_contiguous;
using flashrwkv2::validation::check_same_device;
using flashrwkv2::validation::prepare_recurrent_graph_metadata_cuda;
using flashrwkv2::validation::prepare_recurrent_metadata_cuda;

namespace {

using PreparedMetadataTuple = std::tuple<
    torch::stable::Tensor,
    torch::stable::Tensor,
    torch::stable::Tensor,
    torch::stable::Tensor,
    torch::stable::Tensor>;

PreparedMetadataTuple prepare_tmix_wkv7_recurrent_metadata(
    torch::stable::Tensor query_start_loc,
    torch::stable::Tensor state_indices,
    int64_t total_tokens,
    int64_t state_pool_size) {
  check_cuda_contiguous(query_start_loc, "query_start_loc");
  check_cuda_contiguous(state_indices, "state_indices");
  check_same_device(query_start_loc, state_indices, "state_indices");
  STD_TORCH_CHECK(
      query_start_loc.scalar_type() == torch::headeronly::ScalarType::Int &&
          state_indices.scalar_type() == torch::headeronly::ScalarType::Int,
      "recurrent metadata must be int32");
  STD_TORCH_CHECK(
      query_start_loc.dim() == 1 && state_indices.dim() == 1 &&
          state_indices.numel() > 0 &&
          query_start_loc.numel() == state_indices.numel() + 1,
      "query_start_loc must have shape [B+1] and state_indices shape [B]");
  STD_TORCH_CHECK(
      state_indices.numel() <= 65535,
      "state_indices must contain at most 65535 sequences");
  STD_TORCH_CHECK(
      total_tokens > 0 && total_tokens <= std::numeric_limits<int>::max(),
      "total_tokens must be positive and fit in int32");
  STD_TORCH_CHECK(
      state_pool_size > 0 &&
          state_pool_size <= std::numeric_limits<int>::max(),
      "state_pool_size must be positive and fit in int32");
  auto prepared = prepare_recurrent_metadata_cuda(
      query_start_loc, state_indices, total_tokens, state_pool_size);
  return {
      std::move(prepared.query_start_loc),
      std::move(prepared.state_indices),
      std::move(prepared.status),
      std::move(prepared.token_predecessor),
      std::move(prepared.workspace)};
}

PreparedMetadataTuple prepare_tmix_wkv7_recurrent_graph_metadata(
    torch::stable::Tensor query_start_loc,
    torch::stable::Tensor state_indices,
    torch::stable::Tensor num_active_tokens,
    torch::stable::Tensor num_active_sequences,
    int64_t token_capacity,
    int64_t sequence_capacity,
    int64_t state_pool_size,
    int64_t max_seqlen_capacity) {
  check_cuda_contiguous(query_start_loc, "query_start_loc");
  check_cuda_contiguous(state_indices, "state_indices");
  check_same_device(query_start_loc, state_indices, "state_indices");
  STD_TORCH_CHECK(
      query_start_loc.scalar_type() == torch::headeronly::ScalarType::Int &&
          state_indices.scalar_type() == torch::headeronly::ScalarType::Int,
      "recurrent metadata must be int32");
  STD_TORCH_CHECK(
      sequence_capacity > 0 && sequence_capacity <= 65535 &&
          state_indices.dim() == 1 &&
          state_indices.numel() == sequence_capacity &&
          query_start_loc.dim() == 1 &&
          query_start_loc.numel() == sequence_capacity + 1,
      "graph metadata shapes must match sequence_capacity");
  STD_TORCH_CHECK(
      token_capacity > 0 && token_capacity <= std::numeric_limits<int>::max(),
      "token_capacity must be positive and fit in int32");
  STD_TORCH_CHECK(
      state_pool_size > 0 &&
          state_pool_size <= std::numeric_limits<int>::max(),
      "state_pool_size must be positive and fit in int32");
  STD_TORCH_CHECK(
      max_seqlen_capacity > 0 &&
          max_seqlen_capacity <= token_capacity,
      "max_seqlen_capacity must be positive and not exceed token_capacity");

  auto prepared = prepare_recurrent_graph_metadata_cuda(
      query_start_loc, state_indices, num_active_tokens,
      num_active_sequences, token_capacity, sequence_capacity,
      state_pool_size, max_seqlen_capacity);
  return {
      std::move(prepared.query_start_loc),
      std::move(prepared.state_indices),
      std::move(prepared.status),
      std::move(prepared.token_predecessor),
      std::move(prepared.workspace)};
}

}  // namespace

void tmix_wkv7_recurrent_fp32_from_decay_logits(
    torch::stable::Tensor query_start_loc,
    torch::stable::Tensor state_indices,
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
  tmix_wkv7_recurrent_fp32_from_decay_logits_cuda(
      query_start_loc, state_indices, state, r, decay_logits,
      decay_bias.value_or(torch::stable::Tensor()), k, v, a, b, output,
      metadata_status, scale, max_seqlen);
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def(
      "prepare_tmix_wkv7_recurrent_metadata(Tensor query_start_loc, Tensor state_indices, int total_tokens, int state_pool_size) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  module.def(
      "prepare_tmix_wkv7_recurrent_graph_metadata(Tensor query_start_loc, Tensor state_indices, Tensor num_active_tokens, Tensor num_active_sequences, int token_capacity, int sequence_capacity, int state_pool_size, int max_seqlen_capacity) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  module.def(
      "tmix_wkv7_recurrent_fp32_from_decay_logits(Tensor query_start_loc, Tensor state_indices, Tensor(a!) state, Tensor r, Tensor decay_logits, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(b!) output, float scale, Tensor? decay_bias, Tensor metadata_status, int max_seqlen) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl(
      "prepare_tmix_wkv7_recurrent_metadata",
      TORCH_BOX(&prepare_tmix_wkv7_recurrent_metadata));
  module.impl(
      "prepare_tmix_wkv7_recurrent_graph_metadata",
      TORCH_BOX(&prepare_tmix_wkv7_recurrent_graph_metadata));
  module.impl(
      "tmix_wkv7_recurrent_fp32_from_decay_logits",
      TORCH_BOX(&tmix_wkv7_recurrent_fp32_from_decay_logits));
}
