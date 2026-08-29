// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
// Canonical module owner: tmix/wkv7; RL/Infctx is the workload.
// This binding exposes the mechanically migrated RL/Infctx output replay
// stage.  It is not a train_temp backward alias.

#include "rl_infctx_chunk_fp32io16_replay.cuh"

#include "../../../validation.h"

#include "validation.h"

#include <cmath>
#include <optional>
#include <utility>

using flashrwkv2::validation::check_cuda_contiguous;
using flashrwkv2::validation::check_same_device;

namespace {

void check_replay_inputs(
    const torch::stable::Tensor& chunk_token_starts,
    const torch::stable::Tensor& chunk_token_ends,
    const torch::stable::Tensor& boundary,
    const torch::stable::Tensor& r,
    const torch::stable::Tensor& decay_logits,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    const torch::stable::Tensor& output,
    const torch::stable::Tensor& state_dot_a,
    double scale,
    const std::optional<torch::stable::Tensor>& decay_bias) {
  check_cuda_contiguous(boundary, "boundary");
  check_cuda_contiguous(r, "r");
  check_cuda_contiguous(decay_logits, "decay_logits");
  check_cuda_contiguous(k, "k");
  check_cuda_contiguous(v, "v");
  check_cuda_contiguous(a, "a");
  check_cuda_contiguous(b, "b");
  check_cuda_contiguous(output, "output");
  check_cuda_contiguous(state_dot_a, "state_dot_a");
  check_cuda_contiguous(chunk_token_starts, "chunk_token_starts");
  check_cuda_contiguous(chunk_token_ends, "chunk_token_ends");
  STD_TORCH_CHECK(std::isfinite(scale), "scale must be finite");
  STD_TORCH_CHECK(chunk_token_starts.scalar_type() == torch::headeronly::ScalarType::Int &&
                  chunk_token_ends.scalar_type() == torch::headeronly::ScalarType::Int,
              "chunk token metadata must be int32");
  STD_TORCH_CHECK(chunk_token_starts.dim() == 1 &&
                  chunk_token_starts.numel() > 0 &&
                  chunk_token_starts.sizes() == chunk_token_ends.sizes(),
              "chunk token metadata must have shape [C]");
  STD_TORCH_CHECK(boundary.dim() == 4 && boundary.size(0) == chunk_token_starts.numel() &&
                  boundary.size(1) == r.size(1) && boundary.size(2) == r.size(2) &&
                  boundary.size(3) == r.size(2) &&
                  boundary.scalar_type() == torch::headeronly::ScalarType::Float,
              "boundary must have shape [C,H,D,D] and be float32");
  STD_TORCH_CHECK(r.dim() == 3 && r.size(0) > 0 &&
                  (r.size(2) == 64 || r.size(2) == 128 || r.size(2) == 256),
              "replay token tensors must have shape [N,H,D], D in {64,128,256}");
  for (const auto& item : {
           std::pair<const torch::stable::Tensor*, const char*>{&decay_logits,
                                                        "decay_logits"},
           {&k, "k"},
           {&v, "v"},
           {&a, "a"},
           {&b, "b"},
           {&output, "output"},
       }) {
    STD_TORCH_CHECK(item.first->sizes() == r.sizes(), item.second,
                " must match r shape");
    STD_TORCH_CHECK(item.first->scalar_type() == r.scalar_type(), item.second,
                " must match r dtype");
    check_same_device(r, *item.first, item.second);
  }
  STD_TORCH_CHECK(r.scalar_type() == torch::headeronly::ScalarType::Half ||
                  r.scalar_type() == torch::headeronly::ScalarType::BFloat16,
              "replay token tensors must be float16 or bfloat16");
  STD_TORCH_CHECK(state_dot_a.sizes() == r.sizes() &&
                  state_dot_a.scalar_type() == torch::headeronly::ScalarType::Float,
              "state_dot_a must match r shape and be float32");
  for (const auto& item : {&chunk_token_starts, &chunk_token_ends, &boundary,
                           &state_dot_a}) {
    check_same_device(r, *item, "replay metadata/workspace");
  }
  if (decay_bias.has_value()) {
    check_cuda_contiguous(*decay_bias, "decay_bias");
    check_same_device(r, *decay_bias, "decay_bias");
    STD_TORCH_CHECK(decay_bias->scalar_type() == r.scalar_type(),
                "decay_bias must match r dtype");
    STD_TORCH_CHECK(
        (decay_bias->dim() == 1 && decay_bias->numel() == r.size(1) * r.size(2)) ||
            (decay_bias->dim() == 2 && decay_bias->size(0) == r.size(1) &&
             decay_bias->size(1) == r.size(2)),
        "decay_bias must have shape [H*D] or [H,D]");
  }
}

}  // namespace

void rl_infctx_tmix_wkv7_chunk_fp32io16_backward_replay(
    torch::stable::Tensor chunk_token_starts,
    torch::stable::Tensor chunk_token_ends,
    torch::stable::Tensor boundary,
    torch::stable::Tensor r,
    torch::stable::Tensor decay_logits,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor a,
    torch::stable::Tensor b,
    torch::stable::Tensor output,
    torch::stable::Tensor state_dot_a,
    double scale,
    std::optional<torch::stable::Tensor> decay_bias) {
  check_replay_inputs(
      chunk_token_starts, chunk_token_ends, boundary, r, decay_logits, k, v,
      a, b, output, state_dot_a, scale, decay_bias);
  rl_infctx_tmix_wkv7_chunk_fp32io16_backward_replay_cuda(
      chunk_token_starts, chunk_token_ends, boundary, r, decay_logits,
      decay_bias.value_or(torch::stable::Tensor()), k, v, a, b, output, state_dot_a,
      scale);
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("rl_infctx_tmix_wkv7_chunk_fp32io16_backward_replay(Tensor chunk_token_starts, Tensor chunk_token_ends, Tensor boundary, Tensor r, Tensor decay_logits, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) output, Tensor state_dot_a, float scale=1.0, Tensor? decay_bias=None) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("rl_infctx_tmix_wkv7_chunk_fp32io16_backward_replay", TORCH_BOX(&rl_infctx_tmix_wkv7_chunk_fp32io16_backward_replay));
}
