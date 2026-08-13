# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

import flashrwkv2
from flashrwkv2.tmix.wkv7 import rl_infctx_chunk_fp32io16


def _reference(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float = 1.0,
    decay_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent FP32-state oracle for the mechanically migrated family."""

    decay_rate = 0.6065306597126334
    expected_state = state_pool.float().clone()
    expected_output = torch.empty_like(v)
    bias = None if decay_bias is None else decay_bias.reshape(r.shape[1], r.shape[2])
    offsets = tuple(int(value) for value in cu_seqlens.cpu().tolist())
    slots = tuple(int(value) for value in state_indices.cpu().tolist())
    for sequence_index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        state = expected_state[slots[sequence_index]].clone()
        for token_index in range(start, end):
            logits = decay_logits[token_index].float()
            if bias is not None:
                logits = logits + bias.float()
            retention = torch.exp(-decay_rate * torch.sigmoid(logits))
            state_dot_a = torch.einsum("hk,hkv->hv", a[token_index].float(), state)
            state = (
                retention.unsqueeze(-1) * state
                + b[token_index].float().unsqueeze(-1) * state_dot_a.unsqueeze(-2)
                + k[token_index].float().unsqueeze(-1)
                * v[token_index].float().unsqueeze(-2)
            )
            expected_output[token_index] = (
                float(scale)
                * torch.einsum("hk,hkv->hv", r[token_index].float(), state)
            ).to(expected_output.dtype)
        expected_state[slots[sequence_index]] = state
    return expected_output, expected_state


def _chunk_metadata(
    lengths: tuple[int, ...], chunk_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence_offsets = [0]
    starts: list[int] = []
    ends: list[int] = []
    token_start = 0
    for length in lengths:
        token_end = token_start + length
        for start in range(token_start, token_end, chunk_size):
            starts.append(start)
            ends.append(min(start + chunk_size, token_end))
        sequence_offsets.append(len(starts))
        token_start = token_end
    return (
        torch.tensor(sequence_offsets, device=device, dtype=torch.int32),
        torch.tensor(starts, device=device, dtype=torch.int32),
        torch.tensor(ends, device=device, dtype=torch.int32),
        torch.tensor(
            [0, *[sum(lengths[:index + 1]) for index in range(len(lengths))]],
            device=device,
            dtype=torch.int32,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("d", (64, 128, 256))
def test_rl_infctx_ragged_materialized_and_recompute(d: int) -> None:
    torch.manual_seed(37)
    device = torch.device("cuda")
    lengths = (2, 5)
    total = sum(lengths)
    h = 1
    inputs = [
        (torch.randn(total, h, d, device=device) * 0.01).to(torch.float16)
        for _ in range(6)
    ]
    pool = (torch.randn(4, h, d, d, device=device) * 0.01).to(torch.float32)
    original_pool = pool.clone()
    cu = torch.tensor([0, 2, 7], device=device, dtype=torch.int32)
    slots = torch.tensor([3, 1], device=device, dtype=torch.int32)
    decay_bias = (torch.randn(h, d, device=device) * 0.01).to(torch.float16)
    expected_output, expected_pool = _reference(
        *inputs,
        state_pool=pool,
        cu_seqlens=cu,
        state_indices=slots,
        decay_bias=decay_bias,
    )
    materialized_output, materialized_pool = rl_infctx_chunk_fp32io16(
        *inputs,
        state_pool=pool,
        cu_seqlens=cu,
        state_indices=slots,
        chunk_size=16,
        strategy="materialized",
        decay_bias=decay_bias,
    )
    recompute_output, recompute_pool = rl_infctx_chunk_fp32io16(
        *inputs,
        state_pool=pool,
        cu_seqlens=cu,
        state_indices=slots,
        chunk_size=16,
        strategy="recompute",
        decay_bias=decay_bias,
    )
    torch.cuda.synchronize()
    assert materialized_output.shape == inputs[3].shape
    assert materialized_output.dtype == torch.float16
    assert torch.allclose(materialized_output.float(), expected_output.float(), rtol=0.0, atol=2e-3)
    assert torch.allclose(recompute_output.float(), expected_output.float(), rtol=0.0, atol=2e-3)
    assert torch.allclose(materialized_pool, expected_pool, rtol=0.0, atol=2e-3)
    assert torch.allclose(recompute_pool, expected_pool, rtol=0.0, atol=2e-3)
    assert torch.allclose(materialized_output, recompute_output, rtol=0.0, atol=2e-3)
    inactive = torch.tensor([0, 2], device=device, dtype=torch.long)
    assert torch.equal(materialized_pool.index_select(0, inactive), pool.index_select(0, inactive))
    assert torch.equal(pool, original_pool)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("d", (64, 128, 256))
def test_rl_infctx_bf16_chunk_sizes_and_tail_match_reference(d: int) -> None:
    torch.manual_seed(43)
    device = torch.device("cuda")
    total = 65
    h = 1
    inputs = [
        (torch.randn(total, h, d, device=device) * 0.01).to(torch.bfloat16)
        for _ in range(6)
    ]
    pool = (torch.randn(3, h, d, d, device=device) * 0.01).to(torch.float32)
    original_pool = pool.clone()
    cu = torch.tensor([0, total], device=device, dtype=torch.int32)
    slots = torch.tensor([2], device=device, dtype=torch.int32)
    decay_bias = (torch.randn(h, d, device=device) * 0.01).to(torch.bfloat16)
    expected_output, expected_pool = _reference(
        *inputs,
        state_pool=pool,
        cu_seqlens=cu,
        state_indices=slots,
        scale=0.75,
        decay_bias=decay_bias,
    )

    results = []
    for chunk_size in (16, 32, 64):
        for strategy in ("materialized", "recompute"):
            output, final_pool = rl_infctx_chunk_fp32io16(
                *inputs,
                state_pool=pool,
                cu_seqlens=cu,
                state_indices=slots,
                chunk_size=chunk_size,
                strategy=strategy,
                scale=0.75,
                decay_bias=decay_bias,
            )
            assert output.dtype == torch.bfloat16
            assert torch.allclose(
                output.float(), expected_output.float(), rtol=0.0, atol=6e-3
            )
            assert torch.allclose(
                final_pool, expected_pool, rtol=0.0, atol=6e-3
            )
            results.append((output, final_pool))

    torch.cuda.synchronize()
    assert torch.equal(pool, original_pool)
    assert torch.allclose(
        results[0][0].float(), results[-1][0].float(), atol=6e-3, rtol=0.0
    )
    assert torch.allclose(results[0][1], results[-1][1], atol=6e-3, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("d", (64, 128, 256))
def test_rl_infctx_replay_stage_matches_reference(d: int) -> None:
    torch.manual_seed(41)
    device = torch.device("cuda")
    lengths = (3, 2)
    total = sum(lengths)
    h = 1
    inputs = [
        (torch.randn(total, h, d, device=device) * 0.01).to(torch.float16)
        for _ in range(6)
    ]
    r, decay_logits, k, v, a, b = inputs
    cu = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)
    slots = torch.tensor([0, 1], device=device, dtype=torch.int32)
    sequence_offsets, starts, ends, _ = _chunk_metadata(lengths, 2, device)
    initial_pool = torch.randn(2, h, d, d, device=device, dtype=torch.float32) * 0.01
    boundary = torch.empty(starts.numel(), h, d, d, device=device, dtype=torch.float32)
    expected_output, _ = _reference(
        *inputs,
        state_pool=initial_pool,
        cu_seqlens=cu,
        state_indices=slots,
    )
    boundary_states = initial_pool.clone()
    for sequence_index, slot in enumerate(slots.cpu().tolist()):
        state = boundary_states[slot].clone()
        chunk_start = int(sequence_offsets[sequence_index].item())
        chunk_end = int(sequence_offsets[sequence_index + 1].item())
        for chunk_index in range(chunk_start, chunk_end):
            boundary[chunk_index].copy_(state)
            token_start = int(starts[chunk_index].item())
            token_end = int(ends[chunk_index].item())
            for token_index in range(token_start, token_end):
                logits = decay_logits[token_index].float()
                retention = torch.exp(-0.6065306597126334 * torch.sigmoid(logits))
                state_dot_a = torch.einsum("hk,hkv->hv", a[token_index].float(), state)
                state = (
                    retention.unsqueeze(-1) * state
                    + b[token_index].float().unsqueeze(-1) * state_dot_a.unsqueeze(-2)
                    + k[token_index].float().unsqueeze(-1)
                    * v[token_index].float().unsqueeze(-2)
                )
        boundary_states[slot].copy_(state)
    output = torch.empty_like(v)
    state_dot_a = torch.empty_like(r, dtype=torch.float32)
    flashrwkv2._C.rl_infctx_chunk_fp32io16_backward_replay(
        starts,
        ends,
        boundary,
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        output,
        state_dot_a,
    )
    torch.cuda.synchronize()
    assert torch.allclose(output.float(), expected_output.float(), rtol=0.0, atol=2e-3)
    assert torch.isfinite(state_dot_a).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rl_infctx_rejects_duplicate_slots() -> None:
    device = torch.device("cuda")
    shape = (2, 1, 64)
    values = [torch.zeros(shape, device=device, dtype=torch.float16) for _ in range(6)]
    with pytest.raises(ValueError, match="duplicates"):
        rl_infctx_chunk_fp32io16(
            *values,
            state_pool=torch.zeros(2, 1, 64, 64, device=device),
            cu_seqlens=torch.tensor([0, 1, 2], device=device, dtype=torch.int32),
            state_indices=torch.tensor([0, 0], device=device, dtype=torch.int32),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rl_infctx_rejects_invalid_packed_boundaries() -> None:
    device = torch.device("cuda")
    values = [
        torch.zeros(2, 1, 64, device=device, dtype=torch.float16)
        for _ in range(6)
    ]
    state = torch.zeros(1, 1, 64, 64, device=device, dtype=torch.float32)
    slots = torch.zeros(1, device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match="start at zero"):
        rl_infctx_chunk_fp32io16(
            *values,
            state_pool=state,
            cu_seqlens=torch.tensor([1, 2], device=device, dtype=torch.int32),
            state_indices=slots,
        )
    with pytest.raises(ValueError, match="end at total_tokens"):
        rl_infctx_chunk_fp32io16(
            *values,
            state_pool=state,
            cu_seqlens=torch.tensor([0, 1], device=device, dtype=torch.int32),
            state_indices=slots,
        )
