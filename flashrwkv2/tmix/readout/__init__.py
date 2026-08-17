# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch

from ..wkv7 import _extension


def infer_tmix_readout_forward_varlen(
    wkv_output: torch.Tensor,
    receptance: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    r_k: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    gate: torch.Tensor,
    output_weight: torch.Tensor,
    *,
    output_lora_a: torch.Tensor | None = None,
    output_lora_b: torch.Tensor | None = None,
    output_lora_scale: float = 1.0,
    head_size: int = 64,
    batch_size: int = 1,
    max_seqlen: int | None = None,
) -> torch.Tensor:
    """Run head normalization, RKV residual, gate and output projection."""

    if head_size not in {64, 128, 256}:
        raise ValueError("head_size must be one of 64, 128, or 256")
    for name, tensor in (
        ("wkv_output", wkv_output),
        ("receptance", receptance),
        ("key", key),
        ("value", value),
        ("gate", gate),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.float16 or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA float16")
        if tensor.device != wkv_output.device or tensor.shape != wkv_output.shape:
            raise ValueError(f"{name} must match wkv_output's packed shape and device")
    if wkv_output.ndim != 2 or wkv_output.shape[0] <= 0 or wkv_output.shape[1] % head_size:
        raise ValueError("wkv_output must have packed shape [total_tokens,H*head_size]")
    channels = wkv_output.shape[1]
    for name, tensor in (("r_k", r_k), ("ln_weight", ln_weight), ("ln_bias", ln_bias)):
        if tensor.dtype != torch.float16 or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA float16")
        if tensor.device != wkv_output.device or tensor.shape != (channels,):
            raise ValueError(f"{name} must have shape [C] on wkv_output's device")
    if output_weight.dtype != torch.float16 or not output_weight.is_cuda or not output_weight.is_contiguous():
        raise ValueError("output_weight must be contiguous CUDA float16")
    if output_weight.device != wkv_output.device or output_weight.shape != (channels, channels):
        raise ValueError("output_weight must have shape [C,C]")
    if (output_lora_a is None) != (output_lora_b is None):
        raise ValueError("output_lora_a and output_lora_b must be provided together")
    if max_seqlen is None:
        max_seqlen = wkv_output.shape[0]
    return _extension().tmix_readout_forward_varlen(
        wkv_output,
        receptance,
        key,
        value,
        r_k,
        ln_weight,
        ln_bias,
        gate,
        output_weight,
        output_lora_a,
        output_lora_b,
        float(output_lora_scale),
        int(head_size),
        int(batch_size),
        int(max_seqlen),
    )


__all__ = ["infer_tmix_readout_forward_varlen"]

from .pretrain import pretrain_tmix_readout_bf16

__all__.append("pretrain_tmix_readout_bf16")
