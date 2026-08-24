# SPDX-License-Identifier: MIT

import importlib as _importlib
import sys as _sys

import torch as _torch

# PyTorch must be loaded before the private extensions, which link against its
# native libraries.


_NATIVE_BACKENDS = (
    ((9, 0), "_C_sm90"),
    ((12, 0), "_C_sm120"),
)


def _backend_module_name(capability: tuple[int, int]) -> str:
    major, minor = capability
    compatible = (
        (target, module_name)
        for target, module_name in _NATIVE_BACKENDS
        if target[0] == major and target[1] <= minor
    )
    try:
        return max(compatible, key=lambda backend: backend[0])[1]
    except ValueError as error:
        rendered = f"sm{major}{minor}"
        available = ", ".join(
            f"sm{target[0]}{target[1]}" for target, _ in _NATIVE_BACKENDS
        )
        raise RuntimeError(
            "FlashRWKV2 has no binary-compatible native backend for "
            f"{rendered}; available cubin backends are {available}"
        ) from error


def _load_native_backend():
    if not _torch.cuda.is_available():
        return None
    capability = tuple(_torch.cuda.get_device_capability())
    module_name = _backend_module_name(capability)
    try:
        module = _importlib.import_module(f"{__name__}.{module_name}")
    except ImportError as error:
        rendered = f"sm{capability[0]}{capability[1]}"
        raise RuntimeError(
            f"FlashRWKV2 selected backend {module_name} for {rendered}, "
            "but that private extension is not installed or cannot be loaded"
        ) from error
    _sys.modules[f"{__name__}._C"] = module
    return module


_C = _load_native_backend()

from .cmix import infer_cmix_forward_varlen, pretrain_cmix_bf16, statetune_cmix_bf16
from .embedding import infer_embedding_ln0_forward_varlen
from .head.l2wrap_ce import pretrain_head_l2wrap_ce_bf16
from .head.linear import infer_head_linear_all_forward_varlen, infer_head_linear_last_forward_varlen
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
from .tmix.wkv_prepare import infer_tmix_wkv_prepare_forward_varlen
from .tmix.wkv7 import (
    infer_tmix_wkv7_chunk_bf16_forward_varlen,
    infer_tmix_wkv7_recurrent_fp16_forward_varlen,
    infer_tmix_wkv7_recurrent_fp32io16_forward_varlen,
    prepare_tmix_wkv7_recurrent_metadata,
    pretrain_tmix_wkv7_recurrent_bf16,
    rl_infctx_tmix_wkv7_chunk_fp32io16,
    rl_infctx_tmix_wkv7_chunk_fp32io16_factor_recompute,
)
from .tmix.wkv7.statetune import statetune_tmix_wkv7_recurrent_fp32io16

__all__ = [
    "infer_tmix_wkv7_chunk_bf16_forward_varlen",
    "infer_cmix_forward_varlen",
    "infer_embedding_ln0_forward_varlen",
    "infer_head_linear_all_forward_varlen",
    "infer_head_linear_last_forward_varlen",
    "infer_post_norm_output_forward_varlen",
    "infer_tmix_wkv7_recurrent_fp16_forward_varlen",
    "infer_tmix_wkv7_recurrent_fp32io16_forward_varlen",
    "infer_sampling_six_parameter_forward_varlen",
    "infer_sampling_temperature_topk_topp_forward_varlen",
    "infer_tmix_readout_forward_varlen",
    "infer_tmix_wkv_prepare_forward_varlen",
    "infer_tmix_postnorm_tokenshift_forward_varlen",
    "prepare_tmix_wkv7_recurrent_metadata",
    "pretrain_cmix_bf16",
    "pretrain_head_l2wrap_ce_bf16",
    "pretrain_l2wrap_ce_bf16",
    "pretrain_tmix_wkv7_recurrent_bf16",
    "pretrain_tmix_a_gate_bf16",
    "pretrain_tmix_kk_pre_bf16",
    "pretrain_tmix_readout_bf16",
    "pretrain_tmix_tokenshift_bf16",
    "pretrain_tmix_vres_gate_bf16",
    "rl_infctx_tmix_wkv7_chunk_fp32io16",
    "rl_infctx_tmix_wkv7_chunk_fp32io16_factor_recompute",
    "setup_sampling_states",
    "statetune_cmix_bf16",
    "statetune_tmix_wkv7_recurrent_fp32io16",
    "statetune_tmix_tokenshift_bf16",
]

__version__ = "0.1.0a8"
