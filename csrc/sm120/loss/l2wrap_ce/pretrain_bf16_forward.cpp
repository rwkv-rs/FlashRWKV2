// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Adapted from RWKV-LM train_temp commit 952102498e9ed367ea0a59ee64106916d474d30f.

#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> l2wrap_ce_forward_cuda(
    torch::Tensor logits, torch::Tensor targets, int64_t vocab);

namespace {

void check_inputs(const torch::Tensor& logits, const torch::Tensor& targets) {
  TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(),
              "logits must be contiguous CUDA");
  TORCH_CHECK(targets.is_cuda() && targets.is_contiguous(),
              "targets must be contiguous CUDA");
  TORCH_CHECK(logits.scalar_type() == torch::kBFloat16 ||
                  logits.scalar_type() == torch::kFloat32,
              "logits must be bfloat16 or float32");
  TORCH_CHECK(targets.scalar_type() == torch::kInt64,
              "targets must be int64");
  TORCH_CHECK(logits.dim() >= 2 && logits.size(-1) > 0,
              "logits must have shape [...,vocab]");
  TORCH_CHECK(targets.numel() == logits.numel() / logits.size(-1),
              "targets shape mismatch");
  TORCH_CHECK(targets.device() == logits.device(),
              "targets must share logits' device");
}

}  // namespace

std::vector<torch::Tensor> pretrain_l2wrap_ce_forward(
    torch::Tensor logits, torch::Tensor targets) {
  check_inputs(logits, targets);
  return l2wrap_ce_forward_cuda(logits, targets, logits.size(-1));
}

void register_pretrain_l2wrap_ce_forward_bindings(py::module_& module) {
  module.def("pretrain_l2wrap_ce_forward", &pretrain_l2wrap_ce_forward,
             py::arg("logits"), py::arg("targets"));
}
