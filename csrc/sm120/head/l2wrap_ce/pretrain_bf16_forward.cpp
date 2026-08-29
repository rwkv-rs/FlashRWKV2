// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Adapted from RWKV-LM train_temp revision
// 952102498e9ed367ea0a59ee64106916d474d30f.
// This binding keeps the memory-bounded head-loss contract and the canonical
// 65536-way L2Wrap gradient; chunking is an internal implementation detail.

#include "validation.h"

#include <cstdint>
#include <vector>

std::vector<torch::stable::Tensor> pretrain_head_l2wrap_ce_cuda(
    torch::stable::Tensor hidden,
    torch::stable::Tensor weight,
    torch::stable::Tensor targets,
    int64_t chunk_rows);

namespace {

void check_inputs(
    const torch::stable::Tensor& hidden,
    const torch::stable::Tensor& weight,
    const torch::stable::Tensor& targets,
    int64_t chunk_rows) {
  STD_TORCH_CHECK(
      hidden.is_cuda() && hidden.is_contiguous() &&
          hidden.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "hidden must be contiguous CUDA bfloat16");
  STD_TORCH_CHECK(
      weight.is_cuda() && weight.is_contiguous() &&
          weight.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "weight must be contiguous CUDA bfloat16");
  STD_TORCH_CHECK(
      targets.is_cuda() && targets.is_contiguous() &&
          targets.scalar_type() == torch::headeronly::ScalarType::Long,
      "targets must be contiguous CUDA int64");
  STD_TORCH_CHECK(
      hidden.dim() == 3 && hidden.size(0) > 0 && hidden.size(1) > 0 &&
          hidden.size(2) > 0,
      "hidden must have non-empty shape [B,T,C]");
  STD_TORCH_CHECK(weight.dim() == 2 && weight.size(0) == 65536 &&
                  weight.size(1) == hidden.size(2),
              "weight must have shape [65536,C]");
  STD_TORCH_CHECK(targets.numel() == hidden.size(0) * hidden.size(1),
              "targets must contain one token per hidden row");
  STD_TORCH_CHECK(hidden.device() == weight.device() &&
                  hidden.device() == targets.device(),
              "head loss tensors must share a device");
  STD_TORCH_CHECK(chunk_rows > 0, "chunk_rows must be positive");
}

}  // namespace

auto pretrain_head_l2wrap_ce(
    torch::stable::Tensor hidden,
    torch::stable::Tensor weight,
    torch::stable::Tensor targets,
    int64_t chunk_rows) {
  check_inputs(hidden, weight, targets, chunk_rows);
  return flashrwkv2::validation::tensor_tuple<3>(pretrain_head_l2wrap_ce_cuda(hidden, weight, targets, chunk_rows));
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("pretrain_head_l2wrap_ce_forward(Tensor hidden, Tensor weight, Tensor targets, int chunk_rows) -> (Tensor, Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("pretrain_head_l2wrap_ce_forward", TORCH_BOX(&pretrain_head_l2wrap_ce));
}
