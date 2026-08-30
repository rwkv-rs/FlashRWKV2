// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the RWKV-LM project
// StateTune contract from RWKV-LM train_temp revision
// 952102498e9ed367ea0a59ee64106916d474d30f.

#include <torch/extension.h>

#include <optional>

void statetune_tmix_wkv7_recurrent_fp32io16_backward_cuda(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor final_state,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor state_dot_a,
    torch::Tensor grad_output,
    torch::Tensor grad_final_state,
    torch::Tensor boundary,
    torch::Tensor grad_r,
    torch::Tensor grad_decay_logits,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor grad_a,
    torch::Tensor grad_b,
    torch::Tensor grad_initial_state,
    double scale);

void statetune_tmix_wkv7_recurrent_fp32io16_backward(
    torch::Tensor sequence_chunk_offsets,
    torch::Tensor chunk_token_starts,
    torch::Tensor chunk_token_ends,
    torch::Tensor final_state,
    torch::Tensor r,
    torch::Tensor decay_logits,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor state_dot_a,
    std::optional<torch::Tensor> grad_output,
    std::optional<torch::Tensor> grad_final_state,
    torch::Tensor boundary,
    std::optional<torch::Tensor> grad_r,
    std::optional<torch::Tensor> grad_decay_logits,
    std::optional<torch::Tensor> grad_k,
    std::optional<torch::Tensor> grad_v,
    std::optional<torch::Tensor> grad_a,
    std::optional<torch::Tensor> grad_b,
    std::optional<torch::Tensor> grad_initial_state,
    double scale) {
  statetune_tmix_wkv7_recurrent_fp32io16_backward_cuda(
      sequence_chunk_offsets, chunk_token_starts, chunk_token_ends, final_state,
      r, decay_logits, k, v, a, b, state_dot_a,
      grad_output.value_or(torch::Tensor()),
      grad_final_state.value_or(torch::Tensor()), boundary,
      grad_r.value_or(torch::Tensor()),
      grad_decay_logits.value_or(torch::Tensor()),
      grad_k.value_or(torch::Tensor()), grad_v.value_or(torch::Tensor()),
      grad_a.value_or(torch::Tensor()), grad_b.value_or(torch::Tensor()),
      grad_initial_state.value_or(torch::Tensor()), scale);
}

void register_statetune_tmix_wkv7_recurrent_backward_bindings(py::module_& module) {
  module.def(
      "statetune_tmix_wkv7_recurrent_fp32io16_backward",
      &statetune_tmix_wkv7_recurrent_fp32io16_backward,
      "StateTune recurrent backward with initial-state gradient",
      py::arg("sequence_chunk_offsets"), py::arg("chunk_token_starts"),
      py::arg("chunk_token_ends"), py::arg("final_state"), py::arg("r"),
      py::arg("decay_logits"), py::arg("k"), py::arg("v"), py::arg("a"),
      py::arg("b"), py::arg("state_dot_a"), py::arg("grad_output"),
      py::arg("grad_final_state"), py::arg("boundary"), py::arg("grad_r"),
      py::arg("grad_decay_logits"), py::arg("grad_k"), py::arg("grad_v"),
      py::arg("grad_a"), py::arg("grad_b"), py::arg("grad_initial_state"),
      py::arg("scale") = 1.0);
}
