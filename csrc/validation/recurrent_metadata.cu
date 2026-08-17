// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project

#include "../validation.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <utility>

namespace flashrwkv2::validation {
namespace {

enum MetadataError : int {
  kInvalidEndpoint = 1 << 0,
  kInvalidSequenceRange = 1 << 1,
  kInvalidStateSlot = 1 << 2,
  kDuplicateStateSlot = 1 << 3,
  kInvalidActiveCount = 1 << 4,
  kSequenceTooLong = 1 << 5,
};

__global__ void validate_recurrent_metadata_kernel(
    const int* __restrict__ query_start_loc,
    const int* __restrict__ state_indices,
    int num_sequences,
    int total_tokens,
    int state_pool_size,
    int max_seqlen,
    const int* __restrict__ num_active_tokens,
    const int* __restrict__ num_active_sequences,
    int* __restrict__ status,
    int* __restrict__ seen_slots,
    int* __restrict__ query_start_loc_snapshot,
    int* __restrict__ state_indices_snapshot) {
  __shared__ int shared_status;
  __shared__ int active_tokens;
  __shared__ int active_sequences;
  if (threadIdx.x == 0) {
    shared_status = 0;
    active_tokens = num_active_tokens == nullptr ? total_tokens
                                                  : num_active_tokens[0];
    active_sequences = num_active_sequences == nullptr
        ? num_sequences
        : num_active_sequences[0];
    if (active_tokens < 0 || active_tokens > total_tokens ||
        active_sequences < 0 || active_sequences > num_sequences ||
        ((active_tokens == 0) != (active_sequences == 0))) {
      shared_status |= kInvalidActiveCount;
      active_tokens = 0;
      active_sequences = 0;
    }
  }
  __syncthreads();

  for (int slot = static_cast<int>(threadIdx.x); slot < state_pool_size;
       slot += static_cast<int>(blockDim.x)) {
    seen_slots[slot] = -1;
  }

  if (query_start_loc_snapshot != nullptr) {
    for (int sequence_index = static_cast<int>(threadIdx.x);
         sequence_index < num_sequences;
         sequence_index += static_cast<int>(blockDim.x)) {
      query_start_loc_snapshot[sequence_index] =
          query_start_loc[sequence_index];
      state_indices_snapshot[sequence_index] = state_indices[sequence_index];
    }
    if (threadIdx.x == 0) {
      query_start_loc_snapshot[num_sequences] =
          query_start_loc[num_sequences];
    }
  }
  __syncthreads();

  const int* validated_query_start_loc = query_start_loc_snapshot != nullptr
      ? query_start_loc_snapshot
      : query_start_loc;
  const int* validated_state_indices = state_indices_snapshot != nullptr
      ? state_indices_snapshot
      : state_indices;

  if (threadIdx.x == 0) {
    const int current_status = atomicAdd(&shared_status, 0);
    if (current_status == 0 &&
        (validated_query_start_loc[0] != 0 ||
         validated_query_start_loc[active_sequences] != active_tokens)) {
      atomicOr(&shared_status, kInvalidEndpoint);
    }
  }
  __syncthreads();

  for (int sequence_index = static_cast<int>(threadIdx.x);
       sequence_index < active_sequences;
       sequence_index += static_cast<int>(blockDim.x)) {
    int error = 0;
    const int token_start = validated_query_start_loc[sequence_index];
    const int token_end = validated_query_start_loc[sequence_index + 1];
    if (token_start < 0 || token_end <= token_start ||
        token_end > active_tokens) {
      error |= kInvalidSequenceRange;
    } else if (token_end - token_start > max_seqlen) {
      error |= kSequenceTooLong;
    }

    const int state_slot = validated_state_indices[sequence_index];
    if (state_slot < 0 || state_slot >= state_pool_size) {
      error |= kInvalidStateSlot;
    } else if (atomicCAS(seen_slots + state_slot, -1, sequence_index) != -1) {
      error |= kDuplicateStateSlot;
    }

    if (error != 0) {
      atomicOr(&shared_status, error);
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    status[0] = shared_status;
    if (shared_status == 0) {
      status[1] = active_tokens;
      status[2] = active_sequences;
    } else {
      // Every consumer checks status[0] before the active counts.  Zeroing the
      // counts is a second fail-closed guard for future consumers.
      status[1] = 0;
      status[2] = 0;
    }
  }
}

void check_graph_scalar(
    const torch::Tensor& tensor,
    const torch::Tensor& reference,
    const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.device() == reference.device(),
              name, " must share the metadata CUDA device");
  TORCH_CHECK(tensor.scalar_type() == torch::kInt32,
              name, " must have dtype int32");
  TORCH_CHECK(tensor.numel() == 1, name, " must contain one element");
}

PreparedRecurrentMetadata launch_recurrent_metadata_validation(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor num_active_tokens,
    torch::Tensor num_active_sequences,
    int64_t total_tokens,
    int64_t num_sequences,
    int64_t state_pool_size,
    int64_t max_seqlen,
    bool snapshot) {
  const c10::cuda::CUDAGuard device_guard(query_start_loc.device());
  auto status = torch::empty({3}, query_start_loc.options());
  auto seen_slots = torch::empty({state_pool_size}, query_start_loc.options());
  auto query_start_loc_snapshot = snapshot
      ? torch::empty_like(query_start_loc)
      : query_start_loc;
  auto state_indices_snapshot = snapshot
      ? torch::empty_like(state_indices)
      : state_indices;
  constexpr int threads = 256;
  validate_recurrent_metadata_kernel<<<
      1,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      query_start_loc.data_ptr<int>(),
      state_indices.data_ptr<int>(),
      static_cast<int>(num_sequences),
      static_cast<int>(total_tokens),
      static_cast<int>(state_pool_size),
      static_cast<int>(max_seqlen),
      num_active_tokens.defined() ? num_active_tokens.data_ptr<int>() : nullptr,
      num_active_sequences.defined()
          ? num_active_sequences.data_ptr<int>()
          : nullptr,
      status.data_ptr<int>(),
      seen_slots.data_ptr<int>(),
      snapshot ? query_start_loc_snapshot.data_ptr<int>() : nullptr,
      snapshot ? state_indices_snapshot.data_ptr<int>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return PreparedRecurrentMetadata{
      std::move(query_start_loc_snapshot),
      std::move(state_indices_snapshot),
      std::move(status),
      std::move(seen_slots)};
}

}  // namespace

PreparedRecurrentMetadata prepare_recurrent_metadata_cuda(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    int64_t total_tokens,
    int64_t state_pool_size) {
  const int64_t num_sequences = state_indices.numel();
  return launch_recurrent_metadata_validation(
      std::move(query_start_loc), std::move(state_indices), torch::Tensor(),
      torch::Tensor(), total_tokens, num_sequences, state_pool_size,
      total_tokens, true);
}

PreparedRecurrentMetadata prepare_recurrent_graph_metadata_cuda(
    torch::Tensor query_start_loc,
    torch::Tensor state_indices,
    torch::Tensor num_active_tokens,
    torch::Tensor num_active_sequences,
    int64_t token_capacity,
    int64_t sequence_capacity,
    int64_t state_pool_size,
    int64_t max_seqlen_capacity) {
  check_graph_scalar(num_active_tokens, query_start_loc, "num_active_tokens");
  check_graph_scalar(
      num_active_sequences, query_start_loc, "num_active_sequences");
  return launch_recurrent_metadata_validation(
      std::move(query_start_loc), std::move(state_indices),
      std::move(num_active_tokens), std::move(num_active_sequences),
      token_capacity, sequence_capacity, state_pool_size,
      max_seqlen_capacity, false);
}

}  // namespace flashrwkv2::validation
