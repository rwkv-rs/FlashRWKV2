# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from ..tmix.wkv7 import _extension


def _check_rows(tensor: torch.Tensor, name: str, reference: torch.Tensor | None = None) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != torch.float16:
        raise TypeError(f"{name} must have dtype torch.float16")
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"{name} must be CUDA and contiguous")
    if tensor.ndim != 2 or tensor.shape[0] <= 0:
        raise ValueError(f"{name} must have packed shape [total_tokens,C]")
    if reference is not None and (tensor.shape != reference.shape or tensor.device != reference.device):
        raise ValueError(f"{name} must match the packed input")


def _check_affine(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> None:
    for name, tensor in (("weight", weight), ("bias", bias)):
        if tensor.dtype != torch.float16 or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA float16")
        if tensor.device != x.device or tensor.shape != (x.shape[1],):
            raise ValueError(f"{name} must have shape [C]")


def infer_post_norm_output_forward_varlen(
    x: torch.Tensor,
    res: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    _check_rows(x, "x")
    _check_rows(res, "res", x)
    _check_affine(x, weight, bias)
    return _extension().post_norm_output_forward_varlen(
        x, res, weight, bias, float(eps)
    )
__all__ = ["infer_post_norm_output_forward_varlen"]
