# SPDX-License-Identifier: MIT

import sys as _sys

# PyTorch must be loaded before the private extensions, which link against its
# native libraries.


_C = None


def _extension():
    global _C
    if _C is None:
        from .compile import load_extension

        _C = load_extension().module
        _sys.modules[f"{__name__}._C"] = _C
    return _C

from .cmix import infer_cmix_forward_varlen, pretrain_cmix_bf16, statetune_cmix_bf16
from .embedding import infer_embedding_ln0_forward_varlen
from .head.l2wrap_ce import pretrain_head_l2wrap_ce_bf16
from .head.linear import (
    infer_head_linear_all_forward_varlen,
    infer_head_linear_last_forward_varlen,
)
from .loss.l2wrap_ce import pretrain_l2wrap_ce_bf16
from .post_norm import infer_post_norm_output_forward_varlen
from .sampling import (
    infer_sampling_six_parameter_forward_varlen,
    infer_sampling_temperature_topk_topp_forward_varlen,
    setup_sampling_states,
)
from .tmix.a_gate import pretrain_tmix_a_gate_bf16
from .tmix.kk_pre import pretrain_tmix_kk_pre_bf16
from .tmix.readout import (
    infer_tmix_readout_forward_varlen,
    pretrain_tmix_readout_bf16,
)
from .tmix.tokenshift import (
    infer_tmix_postnorm_tokenshift_forward_varlen,
    pretrain_tmix_tokenshift_bf16,
    statetune_tmix_tokenshift_bf16,
)
from .tmix.vres_gate import pretrain_tmix_vres_gate_bf16
from .tmix.wkv7 import (
    infer_tmix_wkv7_chunk_bf16_forward_varlen,
    infer_tmix_wkv7_recurrent_fp16_forward_varlen,
    infer_tmix_wkv7_recurrent_fp32io16_forward_varlen,
    prepare_tmix_wkv7_recurrent_fp16_state,
    prepare_tmix_wkv7_recurrent_fp32io16_state,
    prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor,
    prepare_tmix_wkv7_recurrent_metadata,
    pretrain_tmix_wkv7_recurrent_bf16,
    rl_infctx_tmix_wkv7_chunk_fp32io16,
    rl_infctx_tmix_wkv7_chunk_fp32io16_factor_recompute,
)
from .tmix.wkv7.statetune import statetune_tmix_wkv7_recurrent_fp32io16
from .tmix.wkv_prepare import infer_tmix_wkv_prepare_forward_varlen

__all__ = [
    "infer_cmix_forward_varlen",
    "infer_embedding_ln0_forward_varlen",
    "infer_head_linear_all_forward_varlen",
    "infer_head_linear_last_forward_varlen",
    "infer_post_norm_output_forward_varlen",
    "infer_sampling_six_parameter_forward_varlen",
    "infer_sampling_temperature_topk_topp_forward_varlen",
    "infer_tmix_postnorm_tokenshift_forward_varlen",
    "infer_tmix_readout_forward_varlen",
    "infer_tmix_wkv7_chunk_bf16_forward_varlen",
    "infer_tmix_wkv7_recurrent_fp16_forward_varlen",
    "infer_tmix_wkv7_recurrent_fp32io16_forward_varlen",
    "infer_tmix_wkv_prepare_forward_varlen",
    "prepare_tmix_wkv7_recurrent_fp16_state",
    "prepare_tmix_wkv7_recurrent_fp32io16_state",
    "prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor",
    "prepare_tmix_wkv7_recurrent_metadata",
    "pretrain_cmix_bf16",
    "pretrain_head_l2wrap_ce_bf16",
    "pretrain_l2wrap_ce_bf16",
    "pretrain_tmix_a_gate_bf16",
    "pretrain_tmix_kk_pre_bf16",
    "pretrain_tmix_readout_bf16",
    "pretrain_tmix_tokenshift_bf16",
    "pretrain_tmix_vres_gate_bf16",
    "pretrain_tmix_wkv7_recurrent_bf16",
    "rl_infctx_tmix_wkv7_chunk_fp32io16",
    "rl_infctx_tmix_wkv7_chunk_fp32io16_factor_recompute",
    "setup_sampling_states",
    "statetune_cmix_bf16",
    "statetune_tmix_tokenshift_bf16",
    "statetune_tmix_wkv7_recurrent_fp32io16",
]

__version__ = "0.1.0a13"
