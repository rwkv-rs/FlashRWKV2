// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Adapted from RWKV-LM train_temp commit 952102498e9ed367ea0a59ee64106916d474d30f.

#include "validation.h"

#include <vector>

std::vector<torch::stable::Tensor> l2wrap_ce_forward_cuda(
    torch::stable::Tensor logits, torch::stable::Tensor targets, int64_t vocab);

namespace {

void check_inputs(const torch::stable::Tensor& logits, const torch::stable::Tensor& targets) {
  STD_TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(),
              "logits must be contiguous CUDA");
  STD_TORCH_CHECK(targets.is_cuda() && targets.is_contiguous(),
              "targets must be contiguous CUDA");
  STD_TORCH_CHECK(logits.scalar_type() == torch::headeronly::ScalarType::BFloat16 ||
                  logits.scalar_type() == torch::headeronly::ScalarType::Float,
              "logits must be bfloat16 or float32");
  STD_TORCH_CHECK(targets.scalar_type() == torch::headeronly::ScalarType::Long,
              "targets must be int64");
  STD_TORCH_CHECK(logits.dim() >= 2 && logits.size(-1) > 0,
              "logits must have shape [...,vocab]");
  STD_TORCH_CHECK(targets.numel() == logits.numel() / logits.size(-1),
              "targets shape mismatch");
  STD_TORCH_CHECK(targets.device() == logits.device(),
              "targets must share logits' device");
}

}  // namespace

auto pretrain_l2wrap_ce_forward(
    torch::stable::Tensor logits, torch::stable::Tensor targets) {
  check_inputs(logits, targets);
  return l2wrap_ce_forward_cuda(logits, targets, logits.size(-1));
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("pretrain_l2wrap_ce_forward(Tensor logits, Tensor targets) -> (Tensor, Tensor, Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("pretrain_l2wrap_ce_forward", TORCH_BOX(&pretrain_l2wrap_ce_forward));
}
