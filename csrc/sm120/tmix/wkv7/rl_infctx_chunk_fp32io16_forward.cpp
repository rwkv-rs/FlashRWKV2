// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
// Canonical module owner: tmix/wkv7; RL/Infctx is the workload.
// RL/Infctx binding restores the retained D64 materialized and factor-recompute
// families and dispatches D128/256 to the local 64-wide tiled recurrence. The
// public decay boundary is raw logits.

#include "validation.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

void materialized_chunk_fp32_from_decay_logits_cuda(
    torch::stable::Tensor sequence_chunk_offsets,
    torch::stable::Tensor chunk_token_starts,
    torch::stable::Tensor chunk_token_ends,
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
    torch::stable::Tensor transform,
    torch::stable::Tensor bias,
    torch::stable::Tensor boundary,
    torch::stable::Tensor state_dot_a,
    int64_t build_warps,
    int64_t stages,
    int64_t state_tile,
    double scale);

void recompute_chunk_fp32_from_decay_logits_cuda(
    torch::stable::Tensor sequence_chunk_offsets,
    torch::stable::Tensor chunk_token_starts,
    torch::stable::Tensor chunk_token_ends,
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
    torch::stable::Tensor boundary,
    double scale);

void tiled_chunk_fp32_from_decay_logits_cuda(
    torch::stable::Tensor sequence_chunk_offsets,torch::stable::Tensor chunk_token_starts,
    torch::stable::Tensor chunk_token_ends,torch::stable::Tensor state_indices,torch::stable::Tensor state,
    torch::stable::Tensor r,torch::stable::Tensor decay_logits,torch::stable::Tensor decay_bias,
    torch::stable::Tensor k,torch::stable::Tensor v,torch::stable::Tensor a,torch::stable::Tensor b,
    torch::stable::Tensor output,torch::stable::Tensor boundary,torch::stable::Tensor state_dot_a,double scale);

using flashrwkv2::validation::check_cuda_contiguous;
using flashrwkv2::validation::check_recurrent_layout;
using flashrwkv2::validation::check_same_device;

namespace {

void check_chunk_metadata(
    const torch::stable::Tensor& sequence_chunk_offsets,
    const torch::stable::Tensor& chunk_token_starts,
    const torch::stable::Tensor& chunk_token_ends,
    int64_t num_sequences,
    int64_t total_tokens,
    const torch::stable::Tensor& state) {
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{
               &sequence_chunk_offsets, "sequence_chunk_offsets"},
           {&chunk_token_starts, "chunk_token_starts"},
           {&chunk_token_ends, "chunk_token_ends"},
       }) {
    check_cuda_contiguous(*item.first, item.second);
    check_same_device(state, *item.first, item.second);
    STD_TORCH_CHECK(item.first->scalar_type() == torch::headeronly::ScalarType::Int,
                item.second, " must be int32");
  }
  STD_TORCH_CHECK(
      sequence_chunk_offsets.dim() == 1 &&
          sequence_chunk_offsets.numel() == num_sequences + 1,
      "sequence_chunk_offsets must have shape [B+1]");
  STD_TORCH_CHECK(
      chunk_token_starts.dim() == 1 && chunk_token_starts.numel() > 0 &&
          chunk_token_starts.sizes() == chunk_token_ends.sizes(),
      "chunk token metadata must have matching shape [C]");
  STD_TORCH_CHECK(
      chunk_token_starts.numel() * state.size(1) <=
          std::numeric_limits<int>::max(),
      "chunk/head grid must fit in int32");

}

void check_decay_bias(
    const std::optional<torch::stable::Tensor>& decay_bias,
    const torch::stable::Tensor& state,
    const torch::stable::Tensor& reference) {
  if (!decay_bias.has_value()) {
    return;
  }
  check_cuda_contiguous(*decay_bias, "decay_bias");
  check_same_device(state, *decay_bias, "decay_bias");
  STD_TORCH_CHECK(decay_bias->scalar_type() == reference.scalar_type(),
              "decay_bias must match token dtype");
  STD_TORCH_CHECK(
      (decay_bias->dim() == 1 &&
       decay_bias->numel() == state.size(1) * state.size(2)) ||
          (decay_bias->dim() == 2 && decay_bias->size(0) == state.size(1) &&
           decay_bias->size(1) == state.size(2)),
      "decay_bias must have shape [H*D] or [H,D]");
}

}  // namespace

std::tuple<torch::stable::Tensor, torch::stable::Tensor> rl_infctx_tmix_wkv7_chunk_fp32io16_forward(
    torch::stable::Tensor sequence_chunk_offsets,
    torch::stable::Tensor chunk_token_starts,
    torch::stable::Tensor chunk_token_ends,
    torch::stable::Tensor state_indices,
    torch::stable::Tensor state,
    torch::stable::Tensor r,
    torch::stable::Tensor decay_logits,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor a,
    torch::stable::Tensor b,
    int64_t strategy,
    double scale,
    std::optional<torch::stable::Tensor> decay_bias) {
  STD_TORCH_CHECK(strategy == 0 || strategy == 1,
              "strategy must be 0 (materialized) or 1 (recompute)");
  STD_TORCH_CHECK(state.is_cuda() && state.is_contiguous() &&
                  state.scalar_type() == torch::headeronly::ScalarType::Float,
              "RL/Infctx state must be contiguous CUDA float32");
  STD_TORCH_CHECK(state.dim() == 4 && state.size(0) > 0 && state.size(1) > 0 &&
                  state.size(2) == state.size(3) &&
                  (state.size(2) == 64 || state.size(2) == 128 || state.size(2) == 256),
              "RL/Infctx state must have shape [B,H,D,D], D in {64,128,256}");

  auto working_state = torch::stable::clone(state);
  auto output = torch::stable::empty_like(v);
  check_recurrent_layout(
      sequence_chunk_offsets, state_indices, working_state, r,
      decay_logits, k, v, a, b, output, scale);
  check_chunk_metadata(
      sequence_chunk_offsets, chunk_token_starts, chunk_token_ends,
      state.size(0), r.size(0), working_state);
  check_decay_bias(decay_bias, working_state, r);

  const int64_t num_chunks = chunk_token_starts.numel();
  auto boundary = torch::stable::new_empty(
      state,
      {num_chunks, state.size(1), state.size(2), state.size(2)},
      torch::headeronly::ScalarType::Float);

  if (state.size(2) != 64) {
    auto state_dot_a = strategy == 0
        ? torch::stable::new_empty(r, r.sizes(), torch::headeronly::ScalarType::Float)
        : torch::stable::Tensor();
    tiled_chunk_fp32_from_decay_logits_cuda(
        sequence_chunk_offsets, chunk_token_starts, chunk_token_ends,
        state_indices, working_state, r, decay_logits,
        decay_bias.value_or(torch::stable::Tensor()), k, v, a, b, output, boundary,
        state_dot_a, scale);
  } else if (strategy == 0) {
    auto transform = torch::stable::empty_like(boundary);
    auto bias = torch::stable::empty_like(boundary);
    auto state_dot_a = torch::stable::new_empty(
        r, r.sizes(), torch::headeronly::ScalarType::Float);
    // This is the retained old explicit-chunk selection: (2 warps, one
    // stage, 64-row state tile).  It is a canonical source configuration,
    // not a generic replacement for the materialized family.
    materialized_chunk_fp32_from_decay_logits_cuda(
        sequence_chunk_offsets, chunk_token_starts, chunk_token_ends,
        state_indices, working_state, r, decay_logits,
        decay_bias.value_or(torch::stable::Tensor()), k, v, a, b, output, transform,
        bias, boundary, state_dot_a, 2, 1, 64, scale);
  } else {
    recompute_chunk_fp32_from_decay_logits_cuda(
        sequence_chunk_offsets, chunk_token_starts, chunk_token_ends,
        state_indices, working_state, r, decay_logits,
        decay_bias.value_or(torch::stable::Tensor()), k, v, a, b, output, boundary,
        scale);
  }
  return std::make_tuple(std::move(output), std::move(working_state));
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("rl_infctx_tmix_wkv7_chunk_fp32io16_forward(Tensor sequence_chunk_offsets, Tensor chunk_token_starts, Tensor chunk_token_ends, Tensor state_indices, Tensor state, Tensor r, Tensor decay_logits, Tensor k, Tensor v, Tensor a, Tensor b, int strategy, float scale, Tensor? decay_bias) -> (Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("rl_infctx_tmix_wkv7_chunk_fp32io16_forward", TORCH_BOX(&rl_infctx_tmix_wkv7_chunk_fp32io16_forward));
}
