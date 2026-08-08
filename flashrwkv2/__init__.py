# SPDX-License-Identifier: MIT

# Load PyTorch before the native module.  The extension links against
# ``libc10`` and the other libraries loaded by PyTorch; importing ``_C``
# first makes the optional import fail even when the editable build exists.
import torch as _torch  # noqa: F401

try:
    from . import _C
except ImportError:
    _C = None

from .cmix.mix import (
    infer_cmix_add_layer_norm_mix_forward_varlen,
    infer_cmix_linear_ffn_down_forward_varlen,
    infer_cmix_mix_forward_varlen,
    infer_cmix_relu_square_forward_varlen,
    pretrain_cmix_bf16,
    statetune_cmix_bf16,
)
from .cmix.sparse import (
    infer_cmix_sparse_down_relu_forward_varlen,
    infer_cmix_sparse_forward_varlen,
    infer_cmix_sparse_up_forward_varlen,
)
from .embedding import infer_embedding_ln0_forward_varlen
from .head.l2wrap_ce import pretrain_head_l2wrap_ce_bf16
from .head.linear import (
    infer_head_last_norm_forward_varlen,
    infer_head_linear_all_forward_varlen,
    infer_head_linear_forward_varlen,
    infer_head_linear_last_forward_varlen,
)
from .loss.l2wrap_ce import pretrain_l2wrap_ce_bf16
from .rl_infctx.wkv7 import (
    rl_infctx_chunk_fp32io16,
    rl_infctx_chunk_fp32io16_factor_recompute,
)
from .sampling import (
    infer_sampling_six_parameter_forward_varlen,
    infer_sampling_temperature_topk_topp_forward_varlen,
    setup_sampling_states,
)
from .tmix.a_gate import pretrain_tmix_a_gate_bf16
from .tmix.kk_a_gate import infer_tmix_kk_a_gate_forward_varlen
from .tmix.kk_pre import pretrain_tmix_kk_pre_bf16
from .tmix.linear import (
    infer_tmix_linear_act_sigmoid_forward_varlen,
    infer_tmix_linear_act_tanh_forward_varlen,
    infer_tmix_linear_attention_c2c_forward_varlen,
    infer_tmix_linear_ffn_key_forward_varlen,
    infer_tmix_linear_forward_varlen,
    infer_tmix_linear_rank_in_forward_varlen,
    infer_tmix_linear_rank_out_forward_varlen,
    infer_tmix_linear_rank_out_sigmoid_forward_varlen,
    infer_tmix_linear_rank_out_tanh_forward_varlen,
    infer_tmix_linear_t_forward_varlen,
    infer_tmix_linear_t_sigmoid_forward_varlen,
    infer_tmix_linear_t_tanh_forward_varlen,
    infer_tmix_linear_t_vres_forward_varlen,
    infer_tmix_lowrank_in_forward_varlen,
    infer_tmix_lowrank_out_forward_varlen,
    infer_tmix_lowrank_vres_forward_varlen,
    infer_tmix_lowrank_wagv_in_forward_varlen,
)
from .tmix.lnx_rkvres_xg import (
    infer_tmix_lnx_rkvres_xg_forward_varlen,
    pretrain_tmix_lnx_rkvres_xg_bf16,
)
from .tmix.mix6 import (
    infer_tmix_mix6_add_layer_norm_forward_varlen,
    infer_tmix_mix6_forward_varlen,
    pretrain_tmix_mix6_bf16,
    statetune_tmix_mix6_bf16,
)
from .tmix.normalization import (
    infer_tmix_add_forward_varlen,
    infer_tmix_add_last_layer_norm_forward_varlen,
    infer_tmix_add_layer_norm_forward_varlen,
    infer_tmix_layer_norm_forward_varlen,
)
from .tmix.vres_gate import (
    infer_tmix_vres_gate_forward_varlen,
    pretrain_tmix_vres_gate_bf16,
)
from .tmix.wkv7 import (
    infer_chunk_bf16_forward_varlen,
    infer_recurrent_add_vec_forward_varlen,
    infer_recurrent_fp16_advance_i32,
    infer_recurrent_fp16_advance_i32_varlen,
    infer_recurrent_fp16_forward_varlen,
    infer_recurrent_fp32io16_forward_varlen,
    prepare_recurrent_metadata,
    pretrain_recurrent_bf16,
)
from .tmix.wkv7.statetune import statetune_recurrent_fp32io16

__all__ = [
    "infer_chunk_bf16_forward_varlen",
    "infer_cmix_add_layer_norm_mix_forward_varlen",
    "infer_cmix_linear_ffn_down_forward_varlen",
    "infer_cmix_mix_forward_varlen",
    "infer_cmix_relu_square_forward_varlen",
    "infer_cmix_sparse_down_relu_forward_varlen",
    "infer_cmix_sparse_forward_varlen",
    "infer_cmix_sparse_up_forward_varlen",
    "infer_embedding_ln0_forward_varlen",
    "infer_head_last_norm_forward_varlen",
    "infer_head_linear_all_forward_varlen",
    "infer_head_linear_forward_varlen",
    "infer_head_linear_last_forward_varlen",
    "infer_recurrent_add_vec_forward_varlen",
    "infer_recurrent_fp16_advance_i32",
    "infer_recurrent_fp16_advance_i32_varlen",
    "infer_recurrent_fp16_forward_varlen",
    "infer_recurrent_fp32io16_forward_varlen",
    "infer_sampling_six_parameter_forward_varlen",
    "infer_sampling_temperature_topk_topp_forward_varlen",
    "infer_tmix_add_forward_varlen",
    "infer_tmix_add_last_layer_norm_forward_varlen",
    "infer_tmix_add_layer_norm_forward_varlen",
    "infer_tmix_kk_a_gate_forward_varlen",
    "infer_tmix_layer_norm_forward_varlen",
    "infer_tmix_linear_act_sigmoid_forward_varlen",
    "infer_tmix_linear_act_tanh_forward_varlen",
    "infer_tmix_linear_attention_c2c_forward_varlen",
    "infer_tmix_linear_ffn_key_forward_varlen",
    "infer_tmix_linear_forward_varlen",
    "infer_tmix_linear_rank_in_forward_varlen",
    "infer_tmix_linear_rank_out_forward_varlen",
    "infer_tmix_linear_rank_out_sigmoid_forward_varlen",
    "infer_tmix_linear_rank_out_tanh_forward_varlen",
    "infer_tmix_linear_t_forward_varlen",
    "infer_tmix_linear_t_sigmoid_forward_varlen",
    "infer_tmix_linear_t_tanh_forward_varlen",
    "infer_tmix_linear_t_vres_forward_varlen",
    "infer_tmix_lnx_rkvres_xg_forward_varlen",
    "infer_tmix_lowrank_in_forward_varlen",
    "infer_tmix_lowrank_out_forward_varlen",
    "infer_tmix_lowrank_vres_forward_varlen",
    "infer_tmix_lowrank_wagv_in_forward_varlen",
    "infer_tmix_mix6_add_layer_norm_forward_varlen",
    "infer_tmix_mix6_forward_varlen",
    "infer_tmix_vres_gate_forward_varlen",
    "prepare_recurrent_metadata",
    "pretrain_cmix_bf16",
    "pretrain_head_l2wrap_ce_bf16",
    "pretrain_l2wrap_ce_bf16",
    "pretrain_recurrent_bf16",
    "pretrain_tmix_a_gate_bf16",
    "pretrain_tmix_kk_pre_bf16",
    "pretrain_tmix_lnx_rkvres_xg_bf16",
    "pretrain_tmix_mix6_bf16",
    "pretrain_tmix_vres_gate_bf16",
    "rl_infctx_chunk_fp32io16",
    "rl_infctx_chunk_fp32io16_factor_recompute",
    "setup_sampling_states",
    "statetune_cmix_bf16",
    "statetune_recurrent_fp32io16",
    "statetune_tmix_mix6_bf16",
]

__version__ = "0.1.0a5"
