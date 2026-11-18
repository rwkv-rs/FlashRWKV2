# SPDX-License-Identifier: MIT
"""FP32 state-pool lifecycle, metadata reuse, and CUDA Graph behavior."""

import pytest
import torch

from flashrwkv2.tmix.wkv7 import (
    infer_tmix_wkv7_recurrent_fp16_forward_varlen,
    infer_tmix_wkv7_recurrent_fp32io16_forward_varlen,
    prepare_tmix_wkv7_recurrent_fp16_state_from_tensor,
    prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor,
    prepare_tmix_wkv7_recurrent_metadata,
)

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")]


def _require_cuda_extension():
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("CUDA source requires CC >= 8.0")
    import flashrwkv2

    assert hasattr(flashrwkv2._extension(), "prepare_tmix_wkv7_recurrent_metadata")


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
    from flashrwkv2 import pretrain_tmix_wkv7_recurrent_bf16

    inputs, initial, cu, slots = _case()
    expected, final = pretrain_tmix_wkv7_recurrent_bf16(
        *inputs, initial_state=initial[slots.long()], cu_seqlens=cu, return_final_state=True,
    )
    pool = initial.clone()
    actual = infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
        *inputs, state=prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(pool),
        cu_seqlens=cu, state_indices=slots,
    )
    assert torch.equal(actual, expected)
    assert torch.equal(pool[slots.long()], final)
    assert torch.equal(pool[[0, 2]], initial[[0, 2]])
    assert actual.dtype == torch.bfloat16


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.float16])
def test_state_checkpoint_cow_reset_and_memory(state_dtype):
    prepare = prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor if state_dtype == torch.float32 else prepare_tmix_wkv7_recurrent_fp16_state_from_tensor
    state = prepare(torch.empty(4, 2, 64, 64, device="cuda", dtype=state_dtype))
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
    assert state.memory_layout["total_nbytes"] == pool.numel() * pool.element_size()
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
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.float16])
def test_wkv_zero_active_precapture_warmup_then_graph_replay(
    state_dtype,
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
        slots, heads, head_size, head_size, device=device, dtype=state_dtype
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
    prepare_state = prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor if state_dtype == torch.float32 else prepare_tmix_wkv7_recurrent_fp16_state_from_tensor
    infer = infer_tmix_wkv7_recurrent_fp32io16_forward_varlen if state_dtype == torch.float32 else infer_tmix_wkv7_recurrent_fp16_forward_varlen
    state_handle = prepare_state(state)

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
        infer(
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
        prepare_state(expected_state)
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
    expected_output = infer(
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
        graph_output = infer(
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


@pytest.mark.memcheck
@pytest.mark.racecheck
@pytest.mark.parametrize("heads", [40, 64])
def test_fp16_deltalog_materializes_reads_and_preserves_overlapping_copies(heads):
    from flashrwkv2.tmix.wkv7.tilelang import _infer_recurrent

    torch.manual_seed(67)
    batch, dim = 16, 64
    slots = torch.arange(batch, device="cuda", dtype=torch.int32)
    bias = torch.randn(heads, dim, device="cuda", dtype=torch.bfloat16)
    raw = torch.randn(batch + 2, heads, dim, dim, device="cuda").mul_(0.03).half()
    initial, reference = raw.clone(), raw.clone()
    state = prepare_tmix_wkv7_recurrent_fp16_state_from_tensor(raw)
    base = heads * dim * dim * raw.element_size()
    props = torch.cuda.get_device_properties(raw.device)
    compact = heads == 64 and (props.major, props.minor, props.multi_processor_count) == (8, 9, 128)
    private = heads * (1540 if compact else 1284)
    assert state.memory_layout == {
        "base_bytes_per_slot": base, "private_bytes_per_slot": private,
        "bytes_per_slot": base + private, "fixed_workspace_nbytes": 0,
        "total_nbytes": (batch + 2) * (base + private),
    }

    def advance(offsets, indices):
        n = offsets[-1]
        values = tuple(torch.randn(n, heads, dim, device="cuda").mul_(.06).bfloat16() for _ in range(6))
        cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
        ticket = prepare_tmix_wkv7_recurrent_metadata(cu, indices, total_tokens=n, state_pool_size=raw.shape[0])
        expected = _infer_recurrent(
            *values, reference, cu, indices, ticket._status(), scale=.75, decay_bias=bias,
            block_values=64, prefetch_tokens=1,
        )
        actual = infer_tmix_wkv7_recurrent_fp16_forward_varlen(
            *values, state=state, cu_seqlens=cu, state_indices=indices,
            validated_metadata=ticket, scale=.75, decay_bias=bias,
        )
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))

    decode_cu = list(range(batch + 1))
    advance(decode_cu, slots)
    assert torch.equal(raw, initial)
    selected = slots[::2].contiguous()
    state.materialize_slots_(selected.repeat(2))
    assert torch.equal(raw[selected.long()], reference[selected.long()])
    assert torch.equal(raw[1:batch:2], initial[1:batch:2])
    copied = state.clone_slots(slots[1:4])
    assert torch.equal(copied._components()[0], reference[1:4])
    src, dst = slots[4:8], slots[4:8].flip((0,))
    state.copy_slots_(state, src, dst)
    reference.index_copy_(0, dst.long(), reference.index_select(0, src))
    state.reset_slots_(slots[11:13])
    reference[11:13].zero_()
    advance(decode_cu, slots.flip((0,)))
    # Run a mixed prefill directly against a partially logged batch, with unequal sequence lengths.
    mixed_cu = [0]
    for i in range(batch):
        mixed_cu.append(mixed_cu[-1] + (2 if i % 2 else 1))
    advance(mixed_cu, slots)
    assert torch.equal(state.clone()._components()[0].view(torch.int16), reference.view(torch.int16))
    advance(decode_cu, slots)
    advance(list(range(0, 4 * (batch + 1), 4)), slots.flip((0,)))
    assert torch.equal(raw.view(torch.int16), reference.view(torch.int16))
    advance(decode_cu, slots)
    target = prepare_tmix_wkv7_recurrent_fp16_state_from_tensor(torch.empty_like(raw))
    target.copy_(state)
    assert torch.equal(target._components()[0].view(torch.int16), reference.view(torch.int16))
    advance(decode_cu, slots)
    state.zero_()
    state.materialize_slots_(slots)
    assert torch.count_nonzero(raw) == 0


@pytest.mark.cuda_graph
@pytest.mark.parametrize("batch", (16, 128))
def test_fp16_deltalog_graph_replays_live_slot_counts_and_rejects_invalid_tails(batch):
    from flashrwkv2.tmix.wkv7.tilelang import _infer_recurrent

    torch.manual_seed(68)
    heads = 64
    inputs = tuple(torch.randn(batch, heads, 64, device="cuda").mul_(.04).bfloat16() for _ in range(6))
    raw = torch.randn(batch, heads, 64, 64, device="cuda").mul_(.02).half()
    reference = raw.clone()
    state = prepare_tmix_wkv7_recurrent_fp16_state_from_tensor(raw)
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    slots = torch.arange(batch, device="cuda", dtype=torch.int32)
    count = torch.zeros(1, device="cuda", dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def launch():
        ticket = prepare_tmix_wkv7_recurrent_metadata(
            cu, slots, state_pool_size=batch, token_capacity=batch, sequence_capacity=batch,
            max_seqlen_capacity=1, num_active_tokens=count, num_active_sequences=count,
        )
        return infer_tmix_wkv7_recurrent_fp16_forward_varlen(
            *inputs, state=state, cu_seqlens=cu, state_indices=slots, validated_metadata=ticket,
        )

    with torch.cuda.stream(stream):
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = launch()

    all_slots = torch.arange(batch, device="cuda", dtype=torch.int32)
    for step, active in enumerate((batch, batch, batch, 7, 0, 12, batch)):
        slots.copy_(all_slots if step % 2 else all_slots.flip((0,)))
        slots[active:] = -99
        count.fill_(active)
        graph.replay()
        if active:
            status = torch.tensor([0, active, active], dtype=torch.int32, device="cuda")
            expected = _infer_recurrent(*inputs, reference, cu, slots, status, block_values=64, prefetch_tokens=1)
            assert torch.equal(output[:active].view(torch.int16), expected[:active].view(torch.int16))
        assert torch.isnan(output[active:]).all()
    slots[1] = slots[0]
    graph.replay()
    assert torch.isnan(output).all()
    state.materialize_slots_(all_slots)
    assert torch.equal(raw.view(torch.int16), reference.view(torch.int16))


@pytest.mark.cuda_graph
@pytest.mark.parametrize("batch", [16, 257])
def test_live_mixed_prefill_matches_token_decode_after_row_reordering(batch):
    torch.manual_seed(137)
    heads, dim, capacity = 64, 64, batch + 17
    slots = torch.randperm(batch, device="cuda", dtype=torch.int32)
    cu = torch.zeros(batch + 1, device="cuda", dtype=torch.int32)
    tokens = torch.zeros(1, device="cuda", dtype=torch.int32)
    sequences = torch.zeros_like(tokens)
    raw = torch.randn(batch, heads, dim, dim, device="cuda").mul_(.03).half()
    reference = raw.clone()
    state = prepare_tmix_wkv7_recurrent_fp16_state_from_tensor(raw)
    control = prepare_tmix_wkv7_recurrent_fp16_state_from_tensor(reference)
    xs = [torch.empty(capacity, heads, dim, device="cuda", dtype=torch.bfloat16) for _ in range(6)]

    def launch():
        ticket = prepare_tmix_wkv7_recurrent_metadata(
            cu, slots, token_capacity=capacity, sequence_capacity=batch, max_seqlen_capacity=17,
            state_pool_size=batch, num_active_tokens=tokens, num_active_sequences=sequences,
        )
        return infer_tmix_wkv7_recurrent_fp16_forward_varlen(
            *xs, state=state, cu_seqlens=cu, state_indices=slots, validated_metadata=ticket,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = launch()
    for lengths in ([1] * batch, [1] * batch, [1] * (batch - 1) + [17],
                    [1] * batch, [5, 2, 7, 4, 3] + [1] * (batch - 9)):
        slots.copy_(slots.roll(3))
        lengths_gpu = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        offsets = torch.cat((lengths_gpu.new_zeros(1), lengths_gpu.cumsum(0, dtype=torch.int32)))
        active, rows = sum(lengths), len(lengths)
        cu[:rows + 1].copy_(offsets)
        tokens.fill_(active)
        sequences.fill_(rows)
        for x in xs:
            x.fill_(float("nan"))
            x[:active].normal_(std=.05)
        graph.replay()
        for t in range(max(lengths)):
            selected = torch.where(lengths_gpu > t)[0]
            positions = offsets[selected].long() + t
            step_slots = slots[selected].contiguous()
            single_offsets = torch.arange(len(selected) + 1, device="cuda", dtype=torch.int32)
            y = infer_tmix_wkv7_recurrent_fp16_forward_varlen(
                *(x[positions] for x in xs), state=control, cu_seqlens=single_offsets, state_indices=step_slots,
            )
            assert torch.equal(output[positions].view(torch.int16), y.view(torch.int16))
        assert output[active:].isnan().all()
    state.materialize_slots_(slots)
    control.materialize_slots_(slots)
    assert torch.equal(raw.view(torch.int16), reference.view(torch.int16))
