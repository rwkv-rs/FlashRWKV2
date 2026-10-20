# SPDX-License-Identifier: MIT
"""RWKV-7 value residual gate."""


def pretrain_tmix_vres_gate_bf16(value, first_value, v0, v12):
    """Mix first-layer values with a shared BF16 forward."""
    from .tilelang import vres

    return vres(value, first_value, v0, v12)


__all__ = ["pretrain_tmix_vres_gate_bf16"]
