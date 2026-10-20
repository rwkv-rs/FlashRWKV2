# SPDX-License-Identifier: MIT
"""RWKV-7 learning-rate gate."""


def pretrain_tmix_a_gate_bf16(a0, a12):
    """Apply the shared FP32 sigmoid and round the result to BF16."""
    from ..._internal.elementwise import activation

    return activation(a12, "sigmoid", a0)


__all__ = ["pretrain_tmix_a_gate_bf16"]
