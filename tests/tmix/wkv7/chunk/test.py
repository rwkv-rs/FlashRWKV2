# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flashrwkv2.tmix.wkv7 import (
    infer_chunk_bf16_forward_varlen,
    prepare_recurrent_metadata,
)


def _retention(logits: torch.Tensor) -> torch.Tensor:
    return torch.exp2(
        -0.8750387749145276
        / (1.0 + torch.exp2(-1.4426950408889634 * logits.float()))
    )


def _reference(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    state_pool: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(v)
    expected_state = state_pool.clone()
    for sequence, slot in enumerate(state_indices.tolist()):
        start = cu_seqlens[sequence].item()
        end = cu_seqlens[sequence + 1].item()
        state = expected_state[slot].float()
        for token in range(start, end):
            previous = state
            a_state = torch.einsum("hk,hkv->hv", a[token].float(), previous)
            state = (
                _retention(decay_logits[token]).unsqueeze(-1) * previous
                + b[token].float().unsqueeze(-1) * a_state.unsqueeze(-2)
                + k[token].float().unsqueeze(-1) * v[token].float().unsqueeze(-2)
            )
            output[token] = (
                float(scale)
                * torch.einsum("hk,hkv->hv", r[token].float(), state)
            ).to(torch.bfloat16)
        expected_state[slot] = state.to(torch.bfloat16)
    return output, expected_state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_ragged_tail_and_slot_update() -> None:
    torch.manual_seed(11)
    lengths = [1, 9, 17]
    total = sum(lengths)
    h = 1
    d = 64
    device = torch.device("cuda")
    inputs = [
        (torch.randn(total, h, d, device=device) * 0.02).to(torch.bfloat16)
        for _ in range(6)
    ]
    r, decay_logits, k, v, a, b = inputs
    cu_seqlens = torch.tensor(
        [0, 1, 10, 27], device=device, dtype=torch.int32
    )
    state_indices = torch.tensor([2, 0, 3], device=device, dtype=torch.int32)
    state_pool = (torch.randn(5, h, d, d, device=device) * 0.002).to(torch.bfloat16)
    before = state_pool.clone()
    expected_output, expected_state = _reference(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        state_pool,
        cu_seqlens,
        state_indices,
        0.5,
    )
    output = infer_chunk_bf16_forward_varlen(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        state_pool=state_pool,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        chunk_size=8,
        max_seqlen=17,
        scale=0.5,
    )
    torch.cuda.synchronize()
    assert output.shape == v.shape
    assert torch.allclose(output.float(), expected_output.float(), atol=0.2, rtol=0.08)
    assert torch.allclose(state_pool.float(), expected_state.float(), atol=0.2, rtol=0.08)
    inactive = torch.tensor([1, 4], device=device)
    assert torch.equal(state_pool.index_select(0, inactive), before.index_select(0, inactive))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_rejects_non_bf16_and_duplicate_slots() -> None:
    device = torch.device("cuda")
    shape = (1, 1, 64)
    tensors = [torch.zeros(shape, device=device, dtype=torch.bfloat16) for _ in range(6)]
    state = torch.zeros((2, 1, 64, 64), device=device, dtype=torch.bfloat16)
    cu = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
    duplicate = torch.tensor([0, 0], device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match="bfloat16"):
        infer_chunk_bf16_forward_varlen(
            tensors[0].float(), *tensors[1:], state_pool=state,
            cu_seqlens=cu, state_indices=torch.tensor([0, 1], device=device, dtype=torch.int32),
        )
    with pytest.raises(RuntimeError):
        infer_chunk_bf16_forward_varlen(
            *tensors, state_pool=state, cu_seqlens=cu, state_indices=duplicate,
            max_seqlen=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_cuda_graph_consumes_live_metadata_ticket() -> None:
    torch.manual_seed(23)
    device = torch.device("cuda")
    token_capacity, sequence_capacity, num_slots = 3, 2, 4
    shape = (token_capacity, 1, 64)
    inputs = tuple(
        (torch.randn(shape, device=device) * 0.02).to(torch.bfloat16)
        for _ in range(6)
    )
    initial_state = (
        torch.randn(num_slots, 1, 64, 64, device=device) * 0.002
    ).to(torch.bfloat16)
    state = initial_state.clone()
    cu_seqlens = torch.tensor([0, 1, 3], device=device, dtype=torch.int32)
    state_indices = torch.tensor([3, 1], device=device, dtype=torch.int32)
    num_active_tokens = torch.tensor([3], device=device, dtype=torch.int32)
    num_active_sequences = torch.tensor([2], device=device, dtype=torch.int32)

    infer_chunk_bf16_forward_varlen(
        *inputs,
        state_pool=initial_state.clone(),
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        chunk_size=2,
        max_seqlen=2,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ticket = prepare_recurrent_metadata(
            cu_seqlens,
            state_indices,
            state_pool_size=num_slots,
            token_capacity=token_capacity,
            sequence_capacity=sequence_capacity,
            max_seqlen_capacity=2,
            num_active_tokens=num_active_tokens,
            num_active_sequences=num_active_sequences,
        )
        output = infer_chunk_bf16_forward_varlen(
            *inputs,
            state_pool=state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            chunk_size=2,
            validated_metadata=ticket,
        )

    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()
    assert not torch.equal(state[3], initial_state[3])
    assert not torch.equal(state[1], initial_state[1])
    assert torch.equal(state[0], initial_state[0])
    assert torch.equal(state[2], initial_state[2])

    state.copy_(initial_state)
    cu_seqlens.copy_(torch.tensor([0, 2, -1], device=device, dtype=torch.int32))
    state_indices.copy_(torch.tensor([2, 99], device=device, dtype=torch.int32))
    num_active_tokens.fill_(2)
    num_active_sequences.fill_(1)
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(state[2], initial_state[2])
    assert torch.equal(state[[0, 1, 3]], initial_state[[0, 1, 3]])

    state.copy_(initial_state)
    cu_seqlens.copy_(torch.tensor([0, 1, 2], device=device, dtype=torch.int32))
    state_indices.copy_(torch.tensor([1, 1], device=device, dtype=torch.int32))
    num_active_tokens.fill_(2)
    num_active_sequences.fill_(2)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(state, initial_state)
