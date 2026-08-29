// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the RWKV-LM project
// StateTune contract from RWKV-LM train_temp revision
// 952102498e9ed367ea0a59ee64106916d474d30f.
// StateTune owns a mechanically migrated train_temp recurrent body; the
// public binding and CUDA symbols remain independent from pretrain.

#include "validation.h"

void statetune_tmix_wkv7_recurrent_fp32io16_forward_cuda(
    torch::stable::Tensor sequence_chunk_offsets,
    torch::stable::Tensor chunk_token_starts,
    torch::stable::Tensor chunk_token_ends,
    torch::stable::Tensor state,
    torch::stable::Tensor r,
    torch::stable::Tensor decay_logits,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor a,
    torch::stable::Tensor b,
    torch::stable::Tensor output,
    torch::stable::Tensor boundary,
    torch::stable::Tensor state_dot_a,
    double scale);

void statetune_tmix_wkv7_recurrent_fp32io16_forward(
    torch::stable::Tensor sequence_chunk_offsets,
    torch::stable::Tensor chunk_token_starts,
    torch::stable::Tensor chunk_token_ends,
    torch::stable::Tensor state,
    torch::stable::Tensor r,
    torch::stable::Tensor decay_logits,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor a,
    torch::stable::Tensor b,
    torch::stable::Tensor output,
    torch::stable::Tensor boundary,
    torch::stable::Tensor state_dot_a,
    double scale) {
  statetune_tmix_wkv7_recurrent_fp32io16_forward_cuda(
      sequence_chunk_offsets, chunk_token_starts, chunk_token_ends, state, r,
      decay_logits, k, v, a, b, output, boundary, state_dot_a, scale);
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("statetune_tmix_wkv7_recurrent_fp32io16_forward(Tensor sequence_chunk_offsets, Tensor chunk_token_starts, Tensor chunk_token_ends, Tensor(a!) state, Tensor r, Tensor decay_logits, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(b!) output, Tensor(c!) boundary, Tensor(d!) state_dot_a, float scale) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("statetune_tmix_wkv7_recurrent_fp32io16_forward", TORCH_BOX(&statetune_tmix_wkv7_recurrent_fp32io16_forward));
}
