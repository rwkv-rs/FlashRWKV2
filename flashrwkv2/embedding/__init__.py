# SPDX-License-Identifier: MIT
"""Embedding LN0, shared by training and inference."""


def infer_embedding_ln0_forward_varlen(embedding, weight, bias, *, eps=1.0e-5):
    """BF16 embedding/affine inputs, FP32 statistics, BF16 output."""
    from ..post_norm.tilelang import norm

    return norm(embedding, weight, bias, eps=eps)


__all__ = ["infer_embedding_ln0_forward_varlen"]
