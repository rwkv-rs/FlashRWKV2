# SPDX-License-Identifier: MIT

from __future__ import annotations

import math

import torch

from . import _extension


def _check_cuda_contiguous(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous CUDA")


def _check_metadata(
    sequence_chunk_offsets: torch.Tensor,
    chunk_token_starts: torch.Tensor,
    chunk_token_ends: torch.Tensor,
    state: torch.Tensor,
) -> None:
    for name, tensor in (
        ("sequence_chunk_offsets", sequence_chunk_offsets),
        ("chunk_token_starts", chunk_token_starts),
        ("chunk_token_ends", chunk_token_ends),
    ):
        _check_cuda_contiguous(tensor, name)
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be int32")
        if tensor.device != state.device:
            raise ValueError(f"{name} must share the state device")
    if sequence_chunk_offsets.ndim != 1:
        raise ValueError("sequence_chunk_offsets must be one-dimensional")
    if chunk_token_starts.ndim != 1 or chunk_token_ends.shape != chunk_token_starts.shape:
        raise ValueError("chunk token metadata must have matching shape [C]")
    if chunk_token_starts.numel() == 0:
        raise ValueError("at least one training chunk is required")


def _check_token_inputs(
    state: torch.Tensor,
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> None:
    _check_cuda_contiguous(state, "state")
    if state.dtype != torch.float32:
        raise TypeError("state must be float32")
    if state.ndim != 4 or state.shape[0] <= 0 or state.shape[1] <= 0:
        raise ValueError("state must have shape [B,H,D,D]")
    if state.shape[2] != state.shape[3] or state.shape[2] not in (64, 128, 256):
        raise ValueError("state head size must be 64, 128, or 256")
    tensors = {
        "r": r,
        "decay_logits": decay_logits,
        "k": k,
        "v": v,
        "a": a,
        "b": b,
    }
    for name, tensor in tensors.items():
        _check_cuda_contiguous(tensor, name)
        if tensor.device != state.device:
            raise ValueError(f"{name} must share the state device")
        if tensor.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError(f"{name} must be float16 or bfloat16")
        if tensor.shape != r.shape:
            raise ValueError(f"{name} must match r shape")
    if r.ndim != 3 or r.shape[0] <= 0 or r.shape[1:] != state.shape[1:3]:
        raise ValueError("token tensors must have shape [total_tokens,H,D]")


class _StateTune(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        initial_state,
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        scale,
    ):
        state = initial_state.detach().contiguous().clone()
        output = torch.empty_like(v)
        boundary = torch.empty(
            (chunk_token_starts.numel(), state.shape[1], state.shape[2], state.shape[3]),
            device=state.device,
            dtype=torch.float32,
        )
        state_dot_a = torch.empty_like(r, dtype=torch.float32)
        _extension().statetune_tmix_wkv7_recurrent_fp32io16_forward(
            sequence_chunk_offsets,
            chunk_token_starts,
            chunk_token_ends,
            state,
            r,
            decay_logits,
            k,
            v,
            a,
            b,
            output,
            boundary,
            state_dot_a,
            float(scale),
        )
        ctx.save_for_backward(
            sequence_chunk_offsets,
            chunk_token_starts,
            chunk_token_ends,
            state,
            r,
            decay_logits,
            k,
            v,
            a,
            b,
            state_dot_a,
            boundary,
        )
        ctx.scale = float(scale)
        ctx.mark_non_differentiable(boundary, state_dot_a)
        return output, state, boundary, state_dot_a

    @staticmethod
    def backward(ctx, grad_output, grad_final_state, _grad_boundary, _grad_state_dot_a):
        saved = ctx.saved_tensors
        (
            sequence_chunk_offsets,
            chunk_token_starts,
            chunk_token_ends,
            final_state,
            r,
            decay_logits,
            k,
            v,
            a,
            b,
            state_dot_a,
            boundary,
        ) = saved
        gradients = [
            torch.empty_like(r),
            torch.empty_like(decay_logits),
            torch.empty_like(k),
            torch.empty_like(v),
            torch.empty_like(a),
            torch.empty_like(b),
            torch.empty_like(final_state),
        ]
        _extension().statetune_tmix_wkv7_recurrent_fp32io16_backward(
            sequence_chunk_offsets,
            chunk_token_starts,
            chunk_token_ends,
            final_state,
            r,
            decay_logits,
            k,
            v,
            a,
            b,
            state_dot_a,
            grad_output,
            grad_final_state,
            boundary,
            gradients[0],
            gradients[1],
            gradients[2],
            gradients[3],
            gradients[4],
            gradients[5],
            gradients[6],
            ctx.scale,
        )
        return (
            gradients[6],
            None,
            None,
            None,
            gradients[0],
            gradients[1],
            gradients[2],
            gradients[3],
            gradients[4],
            gradients[5],
            None,
        )


def statetune_tmix_wkv7_recurrent_fp32io16(
    initial_state,
    sequence_chunk_offsets,
    chunk_token_starts,
    chunk_token_ends,
    r,
    decay_logits,
    k,
    v,
    a,
    b,
    *,
    scale: float = 1.0,
):
    """Run StateTune with a nonzero state and direct initial-state gradient."""

    _check_token_inputs(initial_state, r, decay_logits, k, v, a, b)
    _check_metadata(sequence_chunk_offsets, chunk_token_starts, chunk_token_ends, initial_state)
    if sequence_chunk_offsets.numel() != initial_state.shape[0] + 1:
        raise ValueError("sequence_chunk_offsets must have shape [B+1]")
    if not isinstance(scale, (int, float)) or not math.isfinite(float(scale)):
        raise ValueError("scale must be finite")
    return _StateTune.apply(
        initial_state,
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        float(scale),
    )


__all__ = ["statetune_tmix_wkv7_recurrent_fp32io16"]
