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

  const auto query_start_loc_sizes = query_start_loc.sizes();
  const auto state_indices_sizes = state_indices.sizes();
  const auto state_sizes = state.sizes();
  const auto r_sizes = r.sizes();
  const auto decay_logits_sizes = decay_logits.sizes();
  const auto k_sizes = k.sizes();
  const auto v_sizes = v.sizes();
  const auto a_sizes = a.sizes();
  const auto b_sizes = b.sizes();
  const auto output_sizes = output.sizes();
  const int64_t num_sequences = state_indices.numel();
  STD_TORCH_CHECK(
      num_sequences > 0 && num_sequences <= 65535,
      "state_indices must contain 1..65535 sequences");
  STD_TORCH_CHECK(state_indices_sizes.size() == 1, "state_indices must have shape [N]");
  STD_TORCH_CHECK(
      query_start_loc_sizes.size() == 1 &&
          query_start_loc_sizes[0] == num_sequences + 1,
      "query_start_loc must have shape [N+1]");
  STD_TORCH_CHECK(
      state_sizes.size() == 4 && state_sizes[0] > 0 && state_sizes[1] > 0 &&
          state_sizes[2] == state_sizes[3],
      "state must have square shape [slots,H,D,D]");
  STD_TORCH_CHECK(
      state.scalar_type() == torch::headeronly::ScalarType::Float,
      "state must be float32");

  const int64_t head_size = state_sizes[2];
  STD_TORCH_CHECK(
      head_size == 64 || head_size == 128 || head_size == 256,
      "recurrent head size must be 64, 128, or 256, got ",
      head_size);
  STD_TORCH_CHECK(
      state_sizes[0] <= std::numeric_limits<int>::max(),
      "state slot count must fit in int32");

  const int64_t num_heads = state_sizes[1];
  STD_TORCH_CHECK(
      num_heads <= std::numeric_limits<int>::max(),
      "head count must fit in int32");
  STD_TORCH_CHECK(
      r_sizes.size() == 3 && r_sizes[0] > 0 && r_sizes[1] == num_heads &&
          r_sizes[2] == head_size,
      "r must have shape [total_tokens,H,D] matching state");
  STD_TORCH_CHECK(
      r_sizes[0] <= std::numeric_limits<int>::max(),
      "token count must fit in int32");
  STD_TORCH_CHECK(
      r_sizes == decay_logits_sizes && r_sizes == k_sizes &&
          r_sizes == v_sizes && r_sizes == a_sizes &&
          r_sizes == b_sizes && r_sizes == output_sizes,
      "r,decay_logits,k,v,a,b,output shape mismatch");
  const auto token_dtype = r.scalar_type();
  STD_TORCH_CHECK(
      token_dtype == torch::headeronly::ScalarType::Half ||
          token_dtype == torch::headeronly::ScalarType::BFloat16,
      "token tensors must be float16 or bfloat16");
  STD_TORCH_CHECK(
      token_dtype == decay_logits.scalar_type() &&
          token_dtype == k.scalar_type() &&
          token_dtype == v.scalar_type() &&
          token_dtype == a.scalar_type() &&
          token_dtype == b.scalar_type() &&
          token_dtype == output.scalar_type(),
      "r,decay_logits,k,v,a,b,output dtype mismatch");

  const auto state_device = state.get_device_index();
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
    STD_TORCH_CHECK(
        item.first->get_device_index() == state_device,
        item.second,
        " must be on the same device as state");
  }

  return RecurrentDimensions{num_sequences, num_heads, head_size};
}

}  // namespace flashrwkv2::validation
