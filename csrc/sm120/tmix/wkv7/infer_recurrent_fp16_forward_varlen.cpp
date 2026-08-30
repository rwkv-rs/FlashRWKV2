// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
//
// The FP16 kernel body is adapted from Albatross faster3a_2607 at revision
// 3e41bc43ed5e8332927ddd7e0ce4816cf200a6ea.  vllm-rwkv is used only for the
// packed metadata/state-slot contract reference.

#include "../../../validation.h"


#include <cstdint>
#include <optional>

void tmix_wkv7_recurrent_fp16_from_decay_logits_cuda(
    torch::stable::Tensor query_start_loc,
    torch::stable::Tensor state_indices,
    torch::stable::Tensor elapsed_state,
    torch::stable::Tensor state,
    torch::stable::Tensor r,
    torch::stable::Tensor decay_logits,
    torch::stable::Tensor decay_bias,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor a,
    torch::stable::Tensor b,
    torch::stable::Tensor output,
    torch::stable::Tensor metadata_status,
    double scale,
    int64_t max_seqlen);
void recurrent_fp16_advance_i32_varlen_cuda(
    torch::stable::Tensor query_start_loc,
    torch::stable::Tensor state_indices,
    torch::stable::Tensor elapsed_state,
    torch::stable::Tensor metadata_status);
void tmix_wkv7_recurrent_fp16_from_decay_logits(
    torch::stable::Tensor query_start_loc,
    torch::stable::Tensor state_indices,
    torch::stable::Tensor elapsed_state,
    torch::stable::Tensor state,
    torch::stable::Tensor r,
    torch::stable::Tensor decay_logits,
    torch::stable::Tensor k,
    torch::stable::Tensor v,
    torch::stable::Tensor a,
    torch::stable::Tensor b,
    torch::stable::Tensor output,
    double scale,
    std::optional<torch::stable::Tensor> decay_bias,
    torch::stable::Tensor metadata_status,
    int64_t max_seqlen) {
  tmix_wkv7_recurrent_fp16_from_decay_logits_cuda(
      query_start_loc, state_indices, elapsed_state, state, r,
      decay_logits, decay_bias.value_or(torch::stable::Tensor()), k, v, a, b, output,
      metadata_status, scale, max_seqlen);
  recurrent_fp16_advance_i32_varlen_cuda(
      query_start_loc, state_indices, elapsed_state,
      metadata_status);
}

STABLE_TORCH_LIBRARY_FRAGMENT(flashrwkv2, module) {
  module.def("tmix_wkv7_recurrent_fp16_from_decay_logits(Tensor query_start_loc, Tensor state_indices, Tensor(a!) elapsed_state_pool, Tensor(b!) state, Tensor r, Tensor decay_logits, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(c!) output, float scale, Tensor? decay_bias, Tensor metadata_status, int max_seqlen) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(flashrwkv2, CUDA, module) {
  module.impl("tmix_wkv7_recurrent_fp16_from_decay_logits", TORCH_BOX(&tmix_wkv7_recurrent_fp16_from_decay_logits));
}
