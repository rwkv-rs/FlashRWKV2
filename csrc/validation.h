// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project

#pragma once

#include <torch/extension.h>

namespace flashrwkv2::validation {

enum RecurrentMetadataError : int {
  kInvalidEndpoint = 1 << 0,
  kInvalidSequenceRange = 1 << 1,
  kInvalidStateSlot = 1 << 2,
  kDuplicateStateSlot = 1 << 3,
  kInvalidActiveCount = 1 << 4,
  kSequenceTooLong = 1 << 5,
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
PreparedRecurrentMetadata prepare_recurrent_metadata_cuda(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    int64_t total_tokens,
    int64_t state_pool_size,
    torch::Tensor validation_summary = torch::Tensor());
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
