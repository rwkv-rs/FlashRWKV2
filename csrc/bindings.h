// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project

#pragma once

#include <torch/extension.h>

void register_flashrwkv2_bindings(py::module_&);
void register_infer_tmix_wkv7_recurrent_fp32io16_bindings(py::module_&);
void register_infer_tmix_wkv7_recurrent_fp16_bindings(py::module_&);
void register_tmix_tokenshift_bindings(py::module_&);
void register_tmix_wkv_prepare_bindings(py::module_&);
void register_tmix_readout_bindings(py::module_&);
void register_cmix_bindings(py::module_&);
void register_post_norm_bindings(py::module_&);
void register_embedding_bindings(py::module_&);
void register_head_linear_bindings(py::module_&);
void register_sampling_bindings(py::module_&);
void register_pretrain_l2wrap_ce_forward_bindings(py::module_&);
void register_pretrain_l2wrap_ce_backward_bindings(py::module_&);
void register_infer_tmix_wkv7_chunk_bindings(py::module_&);
void register_pretrain_tmix_a_gate_forward_bindings(py::module_&);
void register_pretrain_tmix_a_gate_backward_bindings(py::module_&);
void register_pretrain_tmix_vres_gate_forward_bindings(py::module_&);
void register_pretrain_tmix_vres_gate_backward_bindings(py::module_&);
void register_pretrain_tmix_tokenshift_forward_bindings(py::module_&);
void register_pretrain_tmix_tokenshift_backward_bindings(py::module_&);
void register_statetune_tmix_tokenshift_forward_bindings(py::module_&);
void register_statetune_tmix_tokenshift_backward_bindings(py::module_&);
void register_pretrain_cmix_forward_bindings(py::module_&);
void register_pretrain_cmix_backward_bindings(py::module_&);
void register_statetune_cmix_forward_bindings(py::module_&);
void register_statetune_cmix_backward_bindings(py::module_&);
void register_pretrain_tmix_kk_pre_bindings(py::module_&);
void register_pretrain_tmix_kk_pre_backward_bindings(py::module_&);
void register_pretrain_tmix_readout_forward_bindings(py::module_&);
void register_pretrain_tmix_readout_backward_bindings(py::module_&);
void register_pretrain_head_l2wrap_ce_bindings(py::module_&);
void register_statetune_tmix_wkv7_recurrent_forward_bindings(py::module_&);
void register_statetune_tmix_wkv7_recurrent_backward_bindings(py::module_&);
void register_rl_infctx_tmix_wkv7_chunk_forward_bindings(py::module_&);
void register_rl_infctx_tmix_wkv7_chunk_backward_bindings(py::module_&);
