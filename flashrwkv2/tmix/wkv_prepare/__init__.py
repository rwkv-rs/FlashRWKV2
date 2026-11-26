# SPDX-License-Identifier: MIT
"""Shared BF16 WKV preparation for training and packed inference."""

from __future__ import annotations

import torch


def _original_weight(original: torch.Tensor | None, runtime: torch.Tensor | None, name: str) -> torch.Tensor:
    if original is not None:
        return original
    if runtime is None:
        raise ValueError(f"{name} requires an original or runtime weight")
    return runtime.t()


def infer_tmix_wkv_prepare_forward_varlen(
    x_r: torch.Tensor, x_w: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor, x_a: torch.Tensor, x_g: torch.Tensor,
    receptance_weight: torch.Tensor, key_weight: torch.Tensor, value_weight: torch.Tensor,
    w1: torch.Tensor | None, a1: torch.Tensor | None, g1: torch.Tensor | None, v1: torch.Tensor | None,
    w2: torch.Tensor | None, a2: torch.Tensor | None, g2: torch.Tensor | None, v2: torch.Tensor | None,
    v0: torch.Tensor | None, k_k: torch.Tensor, a0: torch.Tensor, k_a: torch.Tensor,
    *,
    v_first: torch.Tensor | None = None,
    w1_runtime: torch.Tensor | None = None, a1_runtime: torch.Tensor | None = None,
    g1_runtime: torch.Tensor | None = None, v1_runtime: torch.Tensor | None = None,
    w2_runtime: torch.Tensor | None = None, a2_runtime: torch.Tensor | None = None,
    g2_runtime: torch.Tensor | None = None, v2_runtime: torch.Tensor | None = None,
    receptance_lora_a: torch.Tensor | None = None, receptance_lora_b: torch.Tensor | None = None,
    receptance_lora_scale: float = 1.0,
    key_lora_a: torch.Tensor | None = None, key_lora_b: torch.Tensor | None = None, key_lora_scale: float = 1.0,
    value_lora_a: torch.Tensor | None = None, value_lora_b: torch.Tensor | None = None, value_lora_scale: float = 1.0,
    receptance_use_nora: bool = False, key_use_nora: bool = False, value_use_nora: bool = False,
    head_size: int = 64, batch_size: int = 1, max_seqlen: int | None = None,
) -> tuple[torch.Tensor, ...]:
    """Prepare packed WKV7 inputs -> ``(r, decay_logits, key, value, recurrent_a, recurrent_b, gate, v_first)``.

    Accepts original ``[out,in]`` or transposed ``*_runtime`` ``[in,out]`` low-rank weights without copying.
    Pass ``w0`` as ``decay_bias`` to the WKV7 inference kernel.
    """
    from .tilelang import prepare

    del batch_size, max_seqlen
    lowrank = tuple(
        _original_weight(original, runtime, name)
        for original, runtime, name in (
            (w1, w1_runtime, "w1"), (w2, w2_runtime, "w2"),
            (a1, a1_runtime, "a1"), (a2, a2_runtime, "a2"),
            (g1, g1_runtime, "g1"), (g2, g2_runtime, "g2"),
        )
    )
    return prepare(
        x_r, x_w, x_k, x_v, x_a, x_g, receptance_weight, key_weight, value_weight, *lowrank,
        k_k=k_k, a0=a0, k_a=k_a, v_first=v_first, v0=v0,
        v1=None if v_first is None else _original_weight(v1, v1_runtime, "v1"),
        v2=None if v_first is None else _original_weight(v2, v2_runtime, "v2"),
        head_size=head_size,
        adapters=(
            (receptance_lora_a, receptance_lora_b, receptance_lora_scale, receptance_use_nora),
            (key_lora_a, key_lora_b, key_lora_scale, key_use_nora),
            (value_lora_a, value_lora_b, value_lora_scale, value_use_nora),
        ),
    )


def pretrain_tmix_wkv_prepare_bf16(
    x_r: torch.Tensor, x_w: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor, x_a: torch.Tensor, x_g: torch.Tensor,
    receptance_weight: torch.Tensor, key_weight: torch.Tensor, value_weight: torch.Tensor,
    w0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
    a0: torch.Tensor, a1: torch.Tensor, a2: torch.Tensor,
    g1: torch.Tensor, g2: torch.Tensor, k_k: torch.Tensor, k_a: torch.Tensor,
    *,
    v_first: torch.Tensor | None = None, v0: torch.Tensor | None = None,
    v1: torch.Tensor | None = None, v2: torch.Tensor | None = None,
    receptance_lora_a: torch.Tensor | None = None, receptance_lora_b: torch.Tensor | None = None,
    key_lora_a: torch.Tensor | None = None, key_lora_b: torch.Tensor | None = None,
    value_lora_a: torch.Tensor | None = None, value_lora_b: torch.Tensor | None = None,
    receptance_lora_scale: float = 1.0, key_lora_scale: float = 1.0, value_lora_scale: float = 1.0,
    receptance_use_nora: bool = False, key_use_nora: bool = False, value_use_nora: bool = False,
    head_size: int = 64,
) -> tuple[torch.Tensor, ...]:
    """Differentiable WKV7 preparation using training ``[in,out]`` low-rank weights and ``w0`` bias."""
    from .tilelang import prepare

    return prepare(
        x_r, x_w, x_k, x_v, x_a, x_g, receptance_weight, key_weight, value_weight,
        w1.t(), w2.t(), a1.t(), a2.t(), g1.t(), g2.t(),
        w0=w0, k_k=k_k, a0=a0, k_a=k_a, v_first=v_first, v0=v0,
        v1=None if v1 is None else v1.t(), v2=None if v2 is None else v2.t(),
        head_size=head_size,
        adapters=(
            (receptance_lora_a, receptance_lora_b, receptance_lora_scale, receptance_use_nora),
            (key_lora_a, key_lora_b, key_lora_scale, key_use_nora),
            (value_lora_a, value_lora_b, value_lora_scale, value_use_nora),
        ),
    )


__all__ = ["infer_tmix_wkv_prepare_forward_varlen", "pretrain_tmix_wkv_prepare_bf16"]
