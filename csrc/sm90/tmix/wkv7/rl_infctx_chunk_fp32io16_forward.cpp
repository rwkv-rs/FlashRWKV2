// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
// Canonical module owner: tmix/wkv7; RL/Infctx is the workload.
// RL/Infctx binding restores the retained D64 materialized and factor-recompute
// families and dispatches D128/256 to the local 64-wide tiled recurrence. The
// public decay boundary is raw logits.

#include "../../../validation.h"

#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

void materialized_chunk_fp32_from_decay_logits_cuda(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor state_indices,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor decay_bias,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor transform,
    torch::Tensor bias,
    torch::Tensor boundary,
    torch::Tensor state_dot_a,
    int64_t build_warps,
    int64_t stages,
    int64_t state_tile,
    double scale);

void recompute_chunk_fp32_from_decay_logits_cuda(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor state_indices,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor decay_bias,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor output,
    torch::Tensor boundary,
    double scale);

void tiled_chunk_fp32_from_decay_logits_cuda(
    torch::Tensor sequence_chunk_offsets,torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,torch::Tensor state_indices,torch::Tensor state,
    torch::Tensor r,torch::Tensor decay_logits,torch::Tensor decay_bias,
    torch::Tensor k,torch::Tensor v,torch::Tensor a,torch::Tensor b,
    torch::Tensor output,torch::Tensor boundary,torch::Tensor state_dot_a,double scale);

using flashrwkv2::validation::check_cuda_contiguous;
using flashrwkv2::validation::check_recurrent_layout;
using flashrwkv2::validation::check_same_device;

namespace {

void check_chunk_metadata(
    const torch::Tensor& sequence_chunk_offsets,
    const torch::Tensor& chunk_token_starts,
    const torch::Tensor& chunk_token_ends,
    int64_t num_sequences,
    int64_t total_tokens,
    const torch::Tensor& state) {
  for (const auto& item : {
           std::pair<const torch::Tensor*, const char*>{
               &sequence_chunk_offsets, "sequence_chunk_offsets"},
           {&chunk_token_starts, "chunk_token_starts"},
           {&chunk_token_ends, "chunk_token_ends"},
       }) {
    check_cuda_contiguous(*item.first, item.second);
    check_same_device(state, *item.first, item.second);
    TORCH_CHECK(item.first->scalar_type() == torch::kInt32,
                item.second, " must be int32");
  }
  TORCH_CHECK(
      sequence_chunk_offsets.dim() == 1 &&
          sequence_chunk_offsets.numel() == num_sequences + 1,
      "sequence_chunk_offsets must have shape [B+1]");
  TORCH_CHECK(
      chunk_token_starts.dim() == 1 && chunk_token_starts.numel() > 0 &&
          chunk_token_starts.sizes() == chunk_token_ends.sizes(),
      "chunk token metadata must have matching shape [C]");
  TORCH_CHECK(
      chunk_token_starts.numel() * state.size(1) <=
          std::numeric_limits<int>::max(),
      "chunk/head grid must fit in int32");

  // Chunk metadata is prepared once per RL request.  This explicit host read
  // is retained at the preparation boundary; it is not part of a recurrent
  // launch and does not copy token or state tensors.
  auto sequence_cpu = sequence_chunk_offsets.to(torch::kCPU).contiguous();
  auto starts_cpu = chunk_token_starts.to(torch::kCPU).contiguous();
  auto ends_cpu = chunk_token_ends.to(torch::kCPU).contiguous();
  const auto* sequence = sequence_cpu.data_ptr<int32_t>();
  const auto* starts = starts_cpu.data_ptr<int32_t>();
  const auto* ends = ends_cpu.data_ptr<int32_t>();
  const int64_t chunks = chunk_token_starts.numel();
  TORCH_CHECK(sequence[0] == 0 && sequence[num_sequences] == chunks,
              "sequence_chunk_offsets must cover all chunks");
  int32_t previous_token_end = 0;
  for (int64_t sequence_index = 0; sequence_index < num_sequences;
       ++sequence_index) {
    const int32_t chunk_start = sequence[sequence_index];
    const int32_t chunk_end = sequence[sequence_index + 1];
    TORCH_CHECK(chunk_start >= 0 && chunk_end > chunk_start &&
                    chunk_end <= chunks,
                "each sequence must own at least one ordered chunk");
    TORCH_CHECK(starts[chunk_start] == previous_token_end,
                "chunks must cover packed tokens without gaps");
    for (int32_t chunk = chunk_start; chunk < chunk_end; ++chunk) {
      TORCH_CHECK(starts[chunk] >= 0 && ends[chunk] > starts[chunk] &&
                      ends[chunk] <= total_tokens &&
                      (chunk == chunk_start ||
                       starts[chunk] == ends[chunk - 1]),
                  "chunk token ranges must be contiguous and non-empty");
    }
    previous_token_end = ends[chunk_end - 1];
  }
  TORCH_CHECK(previous_token_end == total_tokens,
              "chunks must cover exactly all packed tokens");
}

void check_decay_bias(
    const std::optional<torch::Tensor>& decay_bias,
    const torch::Tensor& state,
    const torch::Tensor& reference) {
  if (!decay_bias.has_value()) {
    return;
  }
  check_cuda_contiguous(*decay_bias, "decay_bias");
  check_same_device(state, *decay_bias, "decay_bias");
  TORCH_CHECK(decay_bias->scalar_type() == reference.scalar_type(),
              "decay_bias must match token dtype");
  TORCH_CHECK(
      (decay_bias->dim() == 1 &&
       decay_bias->numel() == state.size(1) * state.size(2)) ||
          (decay_bias->dim() == 2 && decay_bias->size(0) == state.size(1) &&
           decay_bias->size(1) == state.size(2)),
      "decay_bias must have shape [H*D] or [H,D]");
}

}  // namespace

py::tuple rl_infctx_chunk_fp32io16_forward(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    int64_t strategy,
    double scale,
    std::optional<torch::Tensor> decay_bias) {
  TORCH_CHECK(strategy == 0 || strategy == 1,
              "strategy must be 0 (materialized) or 1 (recompute)");
  TORCH_CHECK(state.is_cuda() && state.is_contiguous() &&
                  state.scalar_type() == torch::kFloat32,
              "RL/Infctx state must be contiguous CUDA float32");
  TORCH_CHECK(state.dim() == 4 && state.size(0) > 0 && state.size(1) > 0 &&
                  state.size(2) == state.size(3) &&
                  (state.size(2) == 64 || state.size(2) == 128 || state.size(2) == 256),
              "RL/Infctx state must have shape [B,H,D,D], D in {64,128,256}");

  auto working_state = state.clone();
  auto output = torch::empty_like(v);
  auto internal_state_indices = torch::arange(
      state.size(0),
      torch::TensorOptions().device(state.device()).dtype(torch::kInt32));
  check_recurrent_layout(
      sequence_chunk_offsets, internal_state_indices, working_state, r,
      decay_logits, k, v, a, b, output, scale);
  check_chunk_metadata(
      sequence_chunk_offsets, chunk_token_starts, chunk_token_ends,
      state.size(0), r.size(0), working_state);
  check_decay_bias(decay_bias, working_state, r);

  const int64_t num_chunks = chunk_token_starts.numel();
  auto boundary = torch::empty(
      {num_chunks, state.size(1), state.size(2), state.size(2)},
      torch::TensorOptions().device(state.device()).dtype(torch::kFloat32));

  if (state.size(2) != 64) {
    auto state_dot_a = strategy == 0
        ? torch::empty(r.sizes(), r.options().dtype(torch::kFloat32))
        : torch::Tensor();
    tiled_chunk_fp32_from_decay_logits_cuda(
        sequence_chunk_offsets, chunk_token_starts, chunk_token_ends,
        internal_state_indices, working_state, r, decay_logits,
        decay_bias.value_or(torch::Tensor()), k, v, a, b, output, boundary,
        state_dot_a, scale);
  } else if (strategy == 0) {
    auto transform = torch::empty_like(boundary);
    auto bias = torch::empty_like(boundary);
    auto state_dot_a = torch::empty(
        r.sizes(), r.options().dtype(torch::kFloat32));
    // This is the retained old explicit-chunk selection: (2 warps, one
    // stage, 64-row state tile).  It is a canonical source configuration,
    // not a generic replacement for the materialized family.
    materialized_chunk_fp32_from_decay_logits_cuda(
        sequence_chunk_offsets, chunk_token_starts, chunk_token_ends,
        internal_state_indices, working_state, r, decay_logits,
        decay_bias.value_or(torch::Tensor()), k, v, a, b, output, transform,
        bias, boundary, state_dot_a, 2, 1, 64, scale);
  } else {
    recompute_chunk_fp32_from_decay_logits_cuda(
        sequence_chunk_offsets, chunk_token_starts, chunk_token_ends,
        internal_state_indices, working_state, r, decay_logits,
        decay_bias.value_or(torch::Tensor()), k, v, a, b, output, boundary,
        scale);
  }
  return py::make_tuple(output, working_state);
}

void register_rl_infctx_forward_bindings(py::module_& module) {
  module.def(
      "rl_infctx_chunk_fp32io16_forward",
      &rl_infctx_chunk_fp32io16_forward,
      "RL/Infctx raw-decay chunk forward",
      py::arg("sequence_chunk_offsets"), py::arg("chunk_token_starts"),
      py::arg("chunk_token_ends"), py::arg("state"), py::arg("r"),
      py::arg("decay_logits"), py::arg("k"), py::arg("v"), py::arg("a"),
      py::arg("b"), py::arg("strategy") = 1, py::arg("scale") = 1.0,
      py::arg("decay_bias") = py::none());
}
