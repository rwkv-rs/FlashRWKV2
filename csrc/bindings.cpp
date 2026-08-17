// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project

#include "bindings.h"

void register_flashrwkv2_bindings(py::module_& module) {
#if defined(FLASHRWKV_BACKEND_SM120)
  register_infer_tmix_wkv7_recurrent_fp32io16_bindings(module);
  register_infer_tmix_wkv7_recurrent_fp16_bindings(module);
  register_tmix_tokenshift_bindings(module);
  register_tmix_wkv_prepare_bindings(module);
  register_tmix_readout_bindings(module);
  register_cmix_bindings(module);
  register_post_norm_bindings(module);
  register_embedding_bindings(module);
  register_head_linear_bindings(module);
  register_sampling_bindings(module);
  register_infer_tmix_wkv7_chunk_bindings(module);
#elif defined(FLASHRWKV_BACKEND_SM90)
  register_pretrain_l2wrap_ce_forward_bindings(module);
  register_pretrain_l2wrap_ce_backward_bindings(module);
  register_pretrain_tmix_a_gate_forward_bindings(module);
  register_pretrain_tmix_a_gate_backward_bindings(module);
  register_pretrain_tmix_vres_gate_forward_bindings(module);
  register_pretrain_tmix_vres_gate_backward_bindings(module);
  register_pretrain_tmix_tokenshift_forward_bindings(module);
  register_pretrain_tmix_tokenshift_backward_bindings(module);
  register_statetune_tmix_tokenshift_forward_bindings(module);
  register_statetune_tmix_tokenshift_backward_bindings(module);
  register_pretrain_cmix_forward_bindings(module);
  register_pretrain_cmix_backward_bindings(module);
  register_statetune_cmix_forward_bindings(module);
  register_statetune_cmix_backward_bindings(module);
  register_pretrain_tmix_kk_pre_bindings(module);
  register_pretrain_tmix_kk_pre_backward_bindings(module);
  register_pretrain_tmix_readout_forward_bindings(module);
  register_pretrain_tmix_readout_backward_bindings(module);
  register_pretrain_head_l2wrap_ce_bindings(module);
  register_statetune_tmix_wkv7_recurrent_forward_bindings(module);
  register_statetune_tmix_wkv7_recurrent_backward_bindings(module);
  register_rl_infctx_tmix_wkv7_chunk_forward_bindings(module);
  register_rl_infctx_tmix_wkv7_chunk_backward_bindings(module);
#else
#error "FlashRWKV2 private extension requires an architecture backend macro"
#endif
}
