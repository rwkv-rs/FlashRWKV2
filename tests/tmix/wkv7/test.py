# SPDX-License-Identifier: MIT
"""FP32 state-pool lifecycle, metadata reuse, and CUDA Graph behavior."""

import pytest
import torch
from utils import require_cuda_backend

from flashrwkv2.tmix.wkv7 import (
    infer_tmix_wkv7_recurrent_fp32io16_forward_varlen,
    prepare_tmix_wkv7_recurrent_fp32io16_state,
    prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor,
    prepare_tmix_wkv7_recurrent_metadata,
)

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")]


def _require_cuda_extension():
    require_cuda_backend("_C", 8, "prepare_tmix_wkv7_recurrent_metadata")


def _case():
    torch.manual_seed(93)
    inputs = tuple((torch.randn(7, 2, 64, device="cuda") * 0.03).bfloat16() for _ in range(6))
    state = torch.randn(4, 2, 64, 64, device="cuda", dtype=torch.float32) * 0.02
    cu = torch.tensor([0, 2, 7], device="cuda", dtype=torch.int32)
    slots = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    return inputs, state, cu, slots


@pytest.mark.memcheck
@pytest.mark.racecheck
def test_public_recurrence_matches_functional_and_preserves_inactive_slots():
    from flashrwkv2 import rl_infctx_tmix_wkv7_recurrent_bf16

    inputs, initial, cu, slots = _case()
    expected, expected_pool = rl_infctx_tmix_wkv7_recurrent_bf16(
        *inputs, state_pool=initial, cu_seqlens=cu, state_indices=slots,
    )
    pool = initial.clone()
    actual = infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
        *inputs, state=prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(pool),
        cu_seqlens=cu, state_indices=slots,
    )
    assert torch.equal(actual, expected)
    assert torch.equal(pool, expected_pool)
    assert torch.equal(pool[[0, 2]], initial[[0, 2]])
    assert actual.dtype == torch.bfloat16


def test_state_checkpoint_cow_reset_and_memory():
    state = prepare_tmix_wkv7_recurrent_fp32io16_state(4, 128, sequence_capacity=3)
    pool = state._components()[0]
    pool.copy_(torch.randn_like(pool))
    checkpoint = state.clone()
    src = torch.tensor([3, 0], device="cuda", dtype=torch.int32)
    dst = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
    selected = state.clone_slots(src)
    state.copy_slots_(state, src, dst)
    assert torch.equal(pool[dst.long()], selected._components()[0])
    state.reset_slots_(dst)
    assert torch.count_nonzero(pool[dst.long()]) == 0
    state.copy_(checkpoint)
    assert torch.equal(pool, checkpoint._components()[0])
    state.materialize_slots_(src)
    assert state.memory_layout["total_nbytes"] == pool.numel() * 4
    assert state.memory_layout["private_bytes_per_slot"] == 0
    with pytest.raises(ValueError, match="unique"):
        state.copy_slots_(checkpoint, src, torch.tensor([1, 1], device="cuda", dtype=torch.int32))
    with pytest.raises(ValueError, match="within"):
        state.reset_slots_(torch.tensor([4], device="cuda", dtype=torch.int32))
    assert torch.equal(pool, checkpoint._components()[0])
    state.zero_()
    assert torch.count_nonzero(pool) == 0


@pytest.mark.parametrize("change", ["identity", "version", "total_tokens", "state_pool_size"])
def test_ticket_rejects_incompatible_launch_before_state_write(change):
    inputs, pool, cu, slots = _case()
    ticket = prepare_tmix_wkv7_recurrent_metadata(cu, slots, total_tokens=7, state_pool_size=4)
    if change == "identity":
        cu = cu.clone()
    elif change == "version":
        cu.add_(0)
    elif change == "total_tokens":
        inputs = tuple(x[:-1] for x in inputs)
    else:
        pool = torch.cat((pool, pool[:1]), dim=0)
    before = pool.clone()
    with pytest.raises(RuntimeError, match=change):
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            *inputs, state=prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(pool),
            cu_seqlens=cu, state_indices=slots, validated_metadata=ticket,
        )
    assert torch.equal(pool, before)


@pytest.mark.parametrize("cu,slots", [([0, 0, 7], [3, 1]), ([0, 2, 6], [3, 1]), ([0, 2, 7], [1, 1]), ([0, 2, 7], [4, 1])])
def test_invalid_metadata_preserves_state(cu, slots):
    inputs, pool, _, _ = _case()
    before = pool.clone()
    with pytest.raises(RuntimeError):
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            *inputs, state=prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(pool),
            cu_seqlens=torch.tensor(cu, device="cuda", dtype=torch.int32),
            state_indices=torch.tensor(slots, device="cuda", dtype=torch.int32),
        )
    assert torch.equal(pool, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda_graph
def test_fp32_wkv_zero_active_precapture_warmup_then_graph_replay(
) -> None:
    _require_cuda_extension()
    torch.manual_seed(20260809)
    device = torch.device("cuda")
    token_capacity, sequence_capacity, slots, heads, head_size = 4, 4, 2, 1, 64
    shape = (token_capacity, heads, head_size)
    packed = tuple(
        (torch.randn(shape, device=device) * scale).bfloat16()
        for scale in (0.1, 2.0, 0.05, 0.05, 0.05, 0.05)
    )
    initial_state = torch.randn(
        slots, heads, head_size, head_size, device=device, dtype=torch.float32
    ).mul_(0.02)
    state = initial_state.clone()
    cu_seqlens = torch.tensor([0, -1, -1, -1, -1], device=device, dtype=torch.int32)
    state_indices = torch.full(
        (sequence_capacity,), 99, device=device, dtype=torch.int32
    )
    num_active_tokens = torch.zeros(1, device=device, dtype=torch.int32)
    num_active_sequences = torch.zeros(1, device=device, dtype=torch.int32)
    metadata_addresses = (cu_seqlens.data_ptr(), state_indices.data_ptr())
    stream = torch.cuda.Stream(device=device)
    state_handle = prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(state)

    # vLLM V2 warms the exact capture shape before torch.cuda.graph().  The
    # padded sequence capacity can exceed the physical state-pool capacity;
    # zero active counts must therefore ignore every metadata tail entry.
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        warmup_ticket = prepare_tmix_wkv7_recurrent_metadata(
            cu_seqlens,
            state_indices,
            state_pool_size=slots,
            token_capacity=token_capacity,
            sequence_capacity=sequence_capacity,
            max_seqlen_capacity=3,
            num_active_tokens=num_active_tokens,
            num_active_sequences=num_active_sequences,
        )
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            *packed,
            state=state_handle,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            validated_metadata=warmup_ticket,
        )
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize()
    assert warmup_ticket._is_graph()
    assert sequence_capacity > slots
    assert torch.equal(state, initial_state)
    assert metadata_addresses == (cu_seqlens.data_ptr(), state_indices.data_ptr())

    cu_seqlens.copy_(
        torch.tensor([0, 1, 4, -1, -1], device=device, dtype=torch.int32)
    )
    state_indices.copy_(
        torch.tensor([1, 0, 99, 99], device=device, dtype=torch.int32)
    )
    num_active_tokens.fill_(4)
    num_active_sequences.fill_(2)

    expected_state = initial_state.clone()
    expected_state_handle = (
        prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(expected_state)
    )
    expected_ticket = prepare_tmix_wkv7_recurrent_metadata(
        cu_seqlens,
        state_indices,
        state_pool_size=slots,
        token_capacity=token_capacity,
        sequence_capacity=sequence_capacity,
        max_seqlen_capacity=3,
        num_active_tokens=num_active_tokens,
        num_active_sequences=num_active_sequences,
    )
    expected_output = infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
        *packed,
        state=expected_state_handle,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        validated_metadata=expected_ticket,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        ticket = prepare_tmix_wkv7_recurrent_metadata(
            cu_seqlens,
            state_indices,
            state_pool_size=slots,
            token_capacity=token_capacity,
            sequence_capacity=sequence_capacity,
            max_seqlen_capacity=3,
            num_active_tokens=num_active_tokens,
            num_active_sequences=num_active_sequences,
        )
        graph_output = infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            *packed,
            state=state_handle,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            validated_metadata=ticket,
        )

    graph.replay()
    torch.cuda.synchronize()
    assert metadata_addresses == (cu_seqlens.data_ptr(), state_indices.data_ptr())
    assert torch.equal(graph_output, expected_output)
    assert torch.equal(state, expected_state)

    state.copy_(initial_state)
    cu_seqlens.copy_(
        torch.tensor([0, 2, -1, -1, -1], device=device, dtype=torch.int32)
    )
    state_indices.copy_(
        torch.tensor([0, 99, 99, 99], device=device, dtype=torch.int32)
    )
    num_active_tokens.fill_(2)
    num_active_sequences.fill_(1)
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(state[0], initial_state[0])
    assert torch.equal(state[1], initial_state[1])

    state.copy_(initial_state)
    cu_seqlens.copy_(
        torch.tensor([0, 1, 2, -1, -1], device=device, dtype=torch.int32)
    )
    state_indices.copy_(
        torch.tensor([1, 1, 99, 99], device=device, dtype=torch.int32)
    )
    num_active_tokens.fill_(2)
    num_active_sequences.fill_(2)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(state, initial_state)


def test_static_metadata_validates_once_and_reads_only_a_summary() -> None:
    _require_cuda_extension()
    cu = torch.tensor([0, 3, 4, 8, 10, 11], dtype=torch.int32, device="cuda")
    slots = torch.tensor([0, 5, 13, 21, 34], dtype=torch.int32, device="cuda")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as profile:
        ticket = prepare_tmix_wkv7_recurrent_metadata(
            cu, slots, total_tokens=11, state_pool_size=8192
        )
        torch.cuda.synchronize()
    assert ticket._max_seqlen() == 4
    assert ticket._active_status().cpu().tolist() == [0, 11, 5]
    assert torch.equal(ticket._query_start_loc_snapshot(), cu)
    assert torch.equal(ticket._state_indices_snapshot(), slots)
    copies = [
        event.input_shapes[0]
        for event in profile.events()
        if event.name == "aten::_to_copy"
    ]
    assert copies == [[4]], copies
    kernels = [
        event.name
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert sum("validate_recurrent_metadata_kernel" in name for name in kernels) == 1, (
        kernels
    )
