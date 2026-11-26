# SPDX-License-Identifier: MIT
"""Embedding LN0, shared by training and inference."""

from __future__ import annotations

import torch


def infer_embedding_ln0_forward_varlen(
    embedding: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, *, eps: float = 1.0e-5,
) -> torch.Tensor:
    """Apply initial LayerNorm (ln0) to BF16 embeddings using FP32 statistics."""
    from ..post_norm.tilelang import norm

    return norm(embedding, weight, bias, eps=eps)


__all__ = ["infer_embedding_ln0_forward_varlen"]
