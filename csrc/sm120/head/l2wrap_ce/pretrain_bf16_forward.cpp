// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Adapted from RWKV-LM train_temp revision
// 952102498e9ed367ea0a59ee64106916d474d30f.
// This binding keeps the memory-bounded head-loss contract and the canonical
// 65536-way L2Wrap gradient; chunking is an internal implementation detail.

#include <torch/extension.h>

#include <cstdint>
#include <vector>

std::vector<torch::Tensor> pretrain_head_l2wrap_ce_cuda(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor targets,
    int64_t chunk_rows);

namespace {

void check_inputs(
    const torch::Tensor& hidden,
    const torch::Tensor& weight,
    const torch::Tensor& targets,
    int64_t chunk_rows) {
  TORCH_CHECK(
      hidden.is_cuda() && hidden.is_contiguous() &&
          hidden.scalar_type() == torch::kBFloat16,
      "hidden must be contiguous CUDA bfloat16");
  TORCH_CHECK(
      weight.is_cuda() && weight.is_contiguous() &&
          weight.scalar_type() == torch::kBFloat16,
      "weight must be contiguous CUDA bfloat16");
  TORCH_CHECK(
      targets.is_cuda() && targets.is_contiguous() &&
          targets.scalar_type() == torch::kLong,
      "targets must be contiguous CUDA int64");
  TORCH_CHECK(
      hidden.dim() == 3 && hidden.size(0) > 0 && hidden.size(1) > 0 &&
          hidden.size(2) > 0,
      "hidden must have non-empty shape [B,T,C]");
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == 65536 &&
                  weight.size(1) == hidden.size(2),
              "weight must have shape [65536,C]");
  TORCH_CHECK(targets.numel() == hidden.size(0) * hidden.size(1),
              "targets must contain one token per hidden row");
  TORCH_CHECK(hidden.device() == weight.device() &&
                  hidden.device() == targets.device(),
              "head loss tensors must share a device");
  TORCH_CHECK(chunk_rows > 0, "chunk_rows must be positive");
}

}  // namespace

std::vector<torch::Tensor> pretrain_head_l2wrap_ce(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor targets,
    int64_t chunk_rows) {
  check_inputs(hidden, weight, targets, chunk_rows);
  return pretrain_head_l2wrap_ce_cuda(hidden, weight, targets, chunk_rows);
}

void register_pretrain_head_l2wrap_ce_bindings(py::module_& module) {
  module.def(
      "pretrain_head_l2wrap_ce_forward", &pretrain_head_l2wrap_ce,
      "RWKV-7 train_temp memory-bounded head L2Wrap CE",
      py::arg("hidden"), py::arg("weight"), py::arg("targets"),
      py::arg("chunk_rows") = 4096);
}
