// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to BlinkDL/RWKV-LM
// Adapted from RWKV-LM train_temp commit 952102498e9ed367ea0a59ee64106916d474d30f.

#include <torch/extension.h>

torch::Tensor l2wrap_ce_backward_cuda(
    torch::Tensor grad_loss,
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor lse,
    torch::Tensor max_vals,
    torch::Tensor argmax,
    int64_t vocab);

torch::Tensor pretrain_l2wrap_ce_backward(
    torch::Tensor grad_loss,
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor lse,
    torch::Tensor max_vals,
    torch::Tensor argmax) {
  TORCH_CHECK(grad_loss.is_cuda() && grad_loss.is_contiguous() &&
                  grad_loss.scalar_type() == torch::kFloat32 && grad_loss.numel() == 1,
              "grad_loss must be one contiguous CUDA float32 value");
  TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(),
              "logits must be contiguous CUDA");
  TORCH_CHECK(targets.is_cuda() && targets.is_contiguous() &&
                  targets.scalar_type() == torch::kInt64,
              "targets must be contiguous CUDA int64");
  TORCH_CHECK(lse.is_cuda() && lse.is_contiguous() && lse.scalar_type() == torch::kFloat32,
              "lse must be contiguous CUDA float32");
  TORCH_CHECK(max_vals.is_cuda() && max_vals.is_contiguous() &&
                  max_vals.scalar_type() == torch::kFloat32,
              "max_vals must be contiguous CUDA float32");
  TORCH_CHECK(argmax.is_cuda() && argmax.is_contiguous() &&
                  argmax.scalar_type() == torch::kInt32,
              "argmax must be contiguous CUDA int32");
  return l2wrap_ce_backward_cuda(
      grad_loss, logits, targets, lse, max_vals, argmax, logits.size(-1));
}

void register_pretrain_l2wrap_ce_backward_bindings(py::module_& module) {
  module.def("pretrain_l2wrap_ce_backward", &pretrain_l2wrap_ce_backward,
             py::arg("grad_loss"), py::arg("logits"), py::arg("targets"),
             py::arg("lse"), py::arg("max_vals"), py::arg("argmax"));
}
