// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the RWKV-LM project
// StateTune contract from RWKV-LM train_temp revision
// 952102498e9ed367ea0a59ee64106916d474d30f.
// StateTune owns a mechanically migrated train_temp recurrent body; the
// public binding and CUDA symbols remain independent from pretrain.

#include <torch/extension.h>

void statetune_tmix_wkv7_recurrent_fp32io16_forward_cuda(
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
    torch::Tensor output,
    torch::Tensor boundary,
    torch::Tensor state_dot_a,
    double scale);

void statetune_tmix_wkv7_recurrent_fp32io16_forward(
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
    torch::Tensor output,
    torch::Tensor boundary,
    torch::Tensor state_dot_a,
    double scale) {
  statetune_tmix_wkv7_recurrent_fp32io16_forward_cuda(
      sequence_chunk_offsets, chunk_token_starts, chunk_token_ends, state, r,
      decay_logits, k, v, a, b, output, boundary, state_dot_a, scale);
}

void register_statetune_tmix_wkv7_recurrent_forward_bindings(py::module_& module) {
  module.def(
      "statetune_tmix_wkv7_recurrent_fp32io16_forward",
      &statetune_tmix_wkv7_recurrent_fp32io16_forward,
      "StateTune recurrent forward with nonzero initial state",
      py::arg("sequence_chunk_offsets"), py::arg("chunk_token_starts"),
      py::arg("chunk_token_ends"), py::arg("state"), py::arg("r"),
      py::arg("decay_logits"), py::arg("k"), py::arg("v"), py::arg("a"),
      py::arg("b"), py::arg("output"), py::arg("boundary"),
      py::arg("state_dot_a"), py::arg("scale") = 1.0);
}
