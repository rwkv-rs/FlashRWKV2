// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Adapted from RWKV-LM train_temp commit 952102498e9ed367ea0a59ee64106916d474d30f.

#include "validation.h"

torch::stable::Tensor l2wrap_ce_backward_cuda(
    torch::stable::Tensor grad_loss,
    torch::stable::Tensor logits,
    torch::stable::Tensor targets,
    torch::stable::Tensor lse,
    torch::stable::Tensor max_vals,
    torch::stable::Tensor argmax,
    int64_t vocab);

torch::stable::Tensor pretrain_l2wrap_ce_backward(
    torch::stable::Tensor grad_loss,
    torch::stable::Tensor logits,
    torch::stable::Tensor targets,
    torch::stable::Tensor lse,
    torch::stable::Tensor max_vals,
    torch::stable::Tensor argmax) {
  STD_TORCH_CHECK(grad_loss.is_cuda() && grad_loss.is_contiguous() &&
                  grad_loss.scalar_type() == torch::headeronly::ScalarType::Float && grad_loss.numel() == 1,
              "grad_loss must be one contiguous CUDA float32 value");
  STD_TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(),
              "logits must be contiguous CUDA");
  STD_TORCH_CHECK(targets.is_cuda() && targets.is_contiguous() &&
                  targets.scalar_type() == torch::headeronly::ScalarType::Long,
              "targets must be contiguous CUDA int64");
  STD_TORCH_CHECK(lse.is_cuda() && lse.is_contiguous() && lse.scalar_type() == torch::headeronly::ScalarType::Float,
              "lse must be contiguous CUDA float32");
  STD_TORCH_CHECK(max_vals.is_cuda() && max_vals.is_contiguous() &&
                  max_vals.scalar_type() == torch::headeronly::ScalarType::Float,
              "max_vals must be contiguous CUDA float32");
  STD_TORCH_CHECK(argmax.is_cuda() && argmax.is_contiguous() &&
                  argmax.scalar_type() == torch::headeronly::ScalarType::Int,
              "argmax must be contiguous CUDA int32");
  return l2wrap_ce_backward_cuda(
      grad_loss, logits, targets, lse, max_vals, argmax, logits.size(-1));
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("pretrain_l2wrap_ce_backward(Tensor grad_loss, Tensor logits, Tensor targets, Tensor lse, Tensor max_vals, Tensor argmax) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("pretrain_l2wrap_ce_backward", TORCH_BOX(&pretrain_l2wrap_ce_backward));
}
