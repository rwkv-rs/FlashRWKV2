# SPDX-License-Identifier: MIT
"""Shared BF16 residual addition and layer normalization."""


def infer_post_norm_output_forward_varlen(x, res, weight, bias, *, eps=1.0e-5):
    """Normalize the rounded BF16 residual sum with FP32 statistics."""
    from .tilelang import norm

    return norm(x, weight, bias, eps=eps, res=res)


__all__ = ["infer_post_norm_output_forward_varlen"]
