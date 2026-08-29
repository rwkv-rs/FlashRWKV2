// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the RWKV-LM project
// StateTune contract from RWKV-LM train_temp revision
// 952102498e9ed367ea0a59ee64106916d474d30f.

#include "validation.h"

#include <optional>

void statetune_tmix_wkv7_recurrent_fp32io16_backward_cuda(
    torch::stable::Tensor sequence_chunk_offsets,
    torch::stable::Tensor chunk_token_starts,
    torch::stable::Tensor chunk_token_ends,
    torch::stable::Tensor final_state,
    torch::stable::Tensor r,
    torch::stable::Tensor decay_logits,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor a,
    torch::stable::Tensor b,
    torch::stable::Tensor state_dot_a,
    torch::stable::Tensor grad_output,
    torch::stable::Tensor grad_final_state,
    torch::stable::Tensor boundary,
    torch::stable::Tensor grad_r,
    torch::stable::Tensor grad_decay_logits,
    torch::stable::Tensor grad_k,
    torch::stable::Tensor grad_v,
    torch::stable::Tensor grad_a,
    torch::stable::Tensor grad_b,
    torch::stable::Tensor grad_initial_state,
    double scale);

void statetune_tmix_wkv7_recurrent_fp32io16_backward(
    torch::stable::Tensor sequence_chunk_offsets,
    torch::stable::Tensor chunk_token_starts,
    torch::stable::Tensor chunk_token_ends,
    torch::stable::Tensor final_state,
    torch::stable::Tensor r,
    torch::stable::Tensor decay_logits,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor a,
    torch::stable::Tensor b,
    torch::stable::Tensor state_dot_a,
    std::optional<torch::stable::Tensor> grad_output,
    std::optional<torch::stable::Tensor> grad_final_state,
    torch::stable::Tensor boundary,
    std::optional<torch::stable::Tensor> grad_r,
    std::optional<torch::stable::Tensor> grad_decay_logits,
    std::optional<torch::stable::Tensor> grad_k,
    std::optional<torch::stable::Tensor> grad_v,
    std::optional<torch::stable::Tensor> grad_a,
    std::optional<torch::stable::Tensor> grad_b,
    std::optional<torch::stable::Tensor> grad_initial_state,
    double scale) {
  statetune_tmix_wkv7_recurrent_fp32io16_backward_cuda(
      sequence_chunk_offsets, chunk_token_starts, chunk_token_ends, final_state,
      r, decay_logits, k, v, a, b, state_dot_a,
      grad_output.value_or(torch::stable::Tensor()),
      grad_final_state.value_or(torch::stable::Tensor()), boundary,
      grad_r.value_or(torch::stable::Tensor()),
      grad_decay_logits.value_or(torch::stable::Tensor()),
      grad_k.value_or(torch::stable::Tensor()), grad_v.value_or(torch::stable::Tensor()),
      grad_a.value_or(torch::stable::Tensor()), grad_b.value_or(torch::stable::Tensor()),
      grad_initial_state.value_or(torch::stable::Tensor()), scale);
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("statetune_tmix_wkv7_recurrent_fp32io16_backward(Tensor sequence_chunk_offsets, Tensor chunk_token_starts, Tensor chunk_token_ends, Tensor final_state, Tensor r, Tensor decay_logits, Tensor k, Tensor v, Tensor a, Tensor b, Tensor state_dot_a, Tensor? grad_output, Tensor? grad_final_state, Tensor boundary, Tensor(a!)? grad_r, Tensor(b!)? grad_decay_logits, Tensor(c!)? grad_k, Tensor(d!)? grad_v, Tensor(e!)? grad_a, Tensor(f!)? grad_b, Tensor(g!)? grad_initial_state, float scale) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("statetune_tmix_wkv7_recurrent_fp32io16_backward", TORCH_BOX(&statetune_tmix_wkv7_recurrent_fp32io16_backward));
}
