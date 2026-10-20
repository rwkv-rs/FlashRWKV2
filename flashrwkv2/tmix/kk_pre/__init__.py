# SPDX-License-Identifier: MIT
"""RWKV-7 normalized keys and recurrent directions."""


def pretrain_tmix_kk_pre_bf16(
    key, key_scale, learning_rate, learning_rate_scale, *, head_size=64
):
    from .tilelang import prepare

    return prepare(key, key_scale, learning_rate, learning_rate_scale, head_size)


__all__ = ["pretrain_tmix_kk_pre_bf16"]
