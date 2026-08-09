// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project

#pragma once

#include <torch/extension.h>

#include <optional>

namespace flashrwkv2::validation {

struct RecurrentDimensions {
  int64_t num_sequences;
  int64_t num_heads;
  int64_t head_size;
};

struct PreparedRecurrentMetadata {
  torch::Tensor query_start_loc;
  torch::Tensor state_indices;
  torch::Tensor status;
  torch::Tensor workspace;
};

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name);
void check_same_device(
    const torch::Tensor& reference,
    const torch::Tensor& tensor,
    const char* name);
RecurrentDimensions check_recurrent_layout(
    const torch::Tensor& query_start_loc,
    const torch::Tensor& state_indices,
    const torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& decay_logits,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& output,
    double scale);
PreparedRecurrentMetadata prepare_recurrent_metadata_cuda(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    int64_t total_tokens,
    int64_t state_pool_size);
PreparedRecurrentMetadata prepare_recurrent_graph_metadata_cuda(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor num_active_tokens,
    torch::Tensor num_active_sequences,
    int64_t token_capacity,
    int64_t sequence_capacity,
    int64_t state_pool_size,
    int64_t max_seqlen_capacity);

}  // namespace flashrwkv2::validation
