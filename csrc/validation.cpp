// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project

#include "validation.h"

#include <cmath>
#include <limits>
#include <utility>

namespace flashrwkv2::validation {

void check_cuda_contiguous(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(
    const torch::stable::Tensor& reference,
    const torch::stable::Tensor& tensor,
    const char* name) {
  STD_TORCH_CHECK(
      tensor.device() == reference.device(),
      name,
      " must be on the same device as state");
}

RecurrentDimensions check_recurrent_layout(
    const torch::stable::Tensor& query_start_loc,
    const torch::stable::Tensor& state_indices,
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
  check_cuda_contiguous(state, "state");
  check_cuda_contiguous(r, "r");
  check_cuda_contiguous(decay_logits, "decay_logits");
  check_cuda_contiguous(k, "k");
  check_cuda_contiguous(v, "v");
  check_cuda_contiguous(a, "a");
  check_cuda_contiguous(b, "b");
  check_cuda_contiguous(output, "output");

  STD_TORCH_CHECK(
      query_start_loc.scalar_type() == torch::headeronly::ScalarType::Int,
      "query_start_loc must be int32");
  STD_TORCH_CHECK(
      state_indices.scalar_type() == torch::headeronly::ScalarType::Int,
      "state_indices must be int32");
  STD_TORCH_CHECK(std::isfinite(scale), "scale must be finite");

  const int64_t num_sequences = state_indices.numel();
  STD_TORCH_CHECK(
      num_sequences > 0 && num_sequences <= 65535,
      "state_indices must contain 1..65535 sequences");
  STD_TORCH_CHECK(state_indices.dim() == 1, "state_indices must have shape [N]");
  STD_TORCH_CHECK(
      query_start_loc.dim() == 1 &&
          query_start_loc.size(0) == num_sequences + 1,
      "query_start_loc must have shape [N+1]");
  STD_TORCH_CHECK(
      state.dim() == 4 && state.size(0) > 0 && state.size(1) > 0 &&
          state.size(2) == state.size(3),
      "state must have square shape [slots,H,D,D]");
  STD_TORCH_CHECK(
      state.scalar_type() == torch::headeronly::ScalarType::Float,
      "state must be float32");

  const int64_t head_size = state.size(2);
  STD_TORCH_CHECK(
      head_size == 64 || head_size == 128 || head_size == 256,
      "recurrent head size must be 64, 128, or 256, got ",
      head_size);
  STD_TORCH_CHECK(
      state.size(0) <= std::numeric_limits<int>::max(),
      "state slot count must fit in int32");

  const int64_t num_heads = state.size(1);
  STD_TORCH_CHECK(
      num_heads <= std::numeric_limits<int>::max(),
      "head count must fit in int32");
  STD_TORCH_CHECK(
      r.dim() == 3 && r.size(0) > 0 && r.size(1) == num_heads &&
          r.size(2) == head_size,
      "r must have shape [total_tokens,H,D] matching state");
  STD_TORCH_CHECK(
      r.size(0) <= std::numeric_limits<int>::max(),
      "token count must fit in int32");
  STD_TORCH_CHECK(
      r.sizes() == decay_logits.sizes() && r.sizes() == k.sizes() &&
          r.sizes() == v.sizes() && r.sizes() == a.sizes() &&
          r.sizes() == b.sizes() && r.sizes() == output.sizes(),
      "r,decay_logits,k,v,a,b,output shape mismatch");
  STD_TORCH_CHECK(
      r.scalar_type() == torch::headeronly::ScalarType::Half ||
          r.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "token tensors must be float16 or bfloat16");
  STD_TORCH_CHECK(
      r.scalar_type() == decay_logits.scalar_type() &&
          r.scalar_type() == k.scalar_type() &&
          r.scalar_type() == v.scalar_type() &&
          r.scalar_type() == a.scalar_type() &&
          r.scalar_type() == b.scalar_type() &&
          r.scalar_type() == output.scalar_type(),
      "r,decay_logits,k,v,a,b,output dtype mismatch");

  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{
               &query_start_loc, "query_start_loc"},
           {&state_indices, "state_indices"},
           {&r, "r"},
           {&decay_logits, "decay_logits"},
           {&k, "k"},
           {&v, "v"},
           {&a, "a"},
           {&b, "b"},
           {&output, "output"},
       }) {
    check_same_device(state, *item.first, item.second);
  }

  return RecurrentDimensions{num_sequences, num_heads, head_size};
}

}  // namespace flashrwkv2::validation
