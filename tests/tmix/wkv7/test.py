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


def _wkv_ops(dtype):
    if dtype == torch.float16:
        return infer_tmix_wkv7_recurrent_fp16_forward_varlen, prepare_tmix_wkv7_recurrent_fp16_state_from_tensor
    return infer_tmix_wkv7_recurrent_fp32io16_forward_varlen, prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor


def _capture_graph(launch):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = launch()
    return graph, result


def _case():
    torch.manual_seed(93)
    inputs = tuple((torch.randn(7, 2, 64, device="cuda") * 0.03).bfloat16() for _ in range(6))
    state = torch.randn(4, 2, 64, 64, device="cuda", dtype=torch.float32) * 0.02
    cu = torch.tensor([0, 2, 7], device="cuda", dtype=torch.int32)
    slots = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    return inputs, state, cu, slots


def _decode_reference(infer, prepare_state, inputs, initial, lengths, slots):
    pool = initial.clone()
    handle = prepare_state(pool)
    cu = slots.new_tensor([0, 1])
    output, checkpoints = [], []
    start = 0
    for seq, length in enumerate(lengths):
        indices = slots[seq:seq + 1]
        slot = int(indices[0])
        ticket = prepare_tmix_wkv7_recurrent_metadata(
            cu, indices, total_tokens=1, state_pool_size=pool.shape[0], max_seqlen=1,
        )
        for token in range(start, start + length):
            checkpoints.append(pool[slot].clone())
            output.append(infer(
                *(tensor[token:token + 1] for tensor in inputs),
                state=handle, cu_seqlens=cu, state_indices=indices, validated_metadata=ticket,
            ))
        start += length
    return torch.cat(output), pool, checkpoints


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


@pytest.mark.parametrize("dtype,interval", [
    pytest.param(torch.float32, 1, marks=(pytest.mark.memcheck, pytest.mark.racecheck)),
    (torch.float32, 3),
    (torch.float32, 16),
    (torch.float16, 1),
])
def test_infer_recurrent_checkpoints_match_token_decode(dtype, interval):
    _require_cuda_extension()
    torch.manual_seed(634)
    inputs = tuple(
        (torch.randn(24, 2, 64, device="cuda") * 0.1).bfloat16() for _ in range(6)
    )
    initial = (torch.randn(7, 2, 64, 64, device="cuda") * 0.02).to(dtype)
    slots = torch.tensor([5, 1, 3], dtype=torch.int32, device="cuda")
    cu = slots.new_tensor([0, 2, 7, 24])
    starts, lengths = (0, 2, 7), (2, 5, 17)
    infer, prepare = _wkv_ops(dtype)
    expected, expected_pool, checkpoints = _decode_reference(
        infer, prepare, inputs, initial, lengths, slots,
    )
    pool = initial.clone()
    state = prepare(pool)
    boundaries = initial.new_full((24 // interval + 3, 2, 64, 64), 123)
    wanted = boundaries.clone()
    written = []
    for seq, (start, length) in enumerate(zip(starts, lengths, strict=True)):
        for token in range(0, length, interval):
            index = start // interval + seq + token // interval
            wanted[index] = checkpoints[start + token]
            written.append(index)

    result, saved = infer(
        *inputs, state=state, cu_seqlens=cu, state_indices=slots,
        save=True, boundaries=boundaries, checkpoint_interval=interval,
    )
    assert saved is boundaries
    assert torch.equal(result, expected)
    assert torch.equal(pool, expected_pool)
    assert torch.equal(saved, wanted)

    if dtype == torch.float32 and interval == 16:
        pool.copy_(initial)
        result, allocated = infer(
            *inputs, state=state, cu_seqlens=cu, state_indices=slots, save=True,
        )
        assert torch.equal(result, expected)
        assert torch.equal(allocated[written], wanted[written])
        assert torch.equal(pool, expected_pool)


def test_invalid_checkpoint_requests_preserve_pool():
    _require_cuda_extension()
    inputs, pool, cu, slots = _case()
    initial = pool.clone()
    infer = infer_tmix_wkv7_recurrent_fp32io16_forward_varlen
    state = prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(pool)
    args = {"state": state, "cu_seqlens": cu, "state_indices": slots, "save": True}
    for interval in (0, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            infer(*inputs, **args, checkpoint_interval=interval)
    for invalid in (pool[:1], pool[:2].half(), pool[:2].transpose(-1, -2)):
        with pytest.raises(ValueError, match="contiguous .* with shape"):
            infer(*inputs, **args, boundaries=invalid)
    with pytest.raises(ValueError, match="overlap"):
        infer(*inputs, **args, boundaries=pool[:2])
    boundaries = pool.new_empty((2, 2, 64, 64))
    bias = boundaries.flatten()[: 2 * 64].view(2, 64)
    with pytest.raises(ValueError, match="overlap"):
        infer(*inputs, **args, boundaries=boundaries, decay_bias=bias)
    assert torch.equal(pool, initial)


@pytest.mark.parametrize("heads", [40, 64])
def test_fp16_checkpoints_include_pending_decode_updates(heads):
    _require_cuda_extension()
    torch.manual_seed(643)
    prepare = prepare_tmix_wkv7_recurrent_fp16_state_from_tensor
    infer = infer_tmix_wkv7_recurrent_fp16_forward_varlen
    slots = torch.arange(16, dtype=torch.int32, device="cuda").flip(0) + 1
    pool = torch.randn(18, heads, 64, 64, dtype=torch.float16, device="cuda") * 0.02
    state = prepare(pool)
    reference_pool = pool.index_select(0, slots.long())
    reference = prepare(reference_pool)
    reference_slots = slots.new_tensor(range(16))
    decode_lengths = slots.new_tensor(range(17))
    warmup = tuple((torch.randn(16, heads, 64, device="cuda") * 0.1).bfloat16() for _ in range(6))
    infer(*warmup, state=state, cu_seqlens=decode_lengths, state_indices=slots)
    infer(*warmup, state=reference, cu_seqlens=decode_lengths, state_indices=reference_slots)
    reference.materialize_slots_(reference_slots)
    assert not torch.equal(pool.index_select(0, slots.long()), reference_pool)

    inputs = tuple((torch.randn(20, heads, 64, device="cuda") * 0.1).bfloat16() for _ in range(6))
    expected, final, checkpoints = _decode_reference(
        infer, prepare, inputs, reference_pool, [5] + [1] * 15, reference_slots,
    )
    result, saved = infer(
        *inputs, state=state, cu_seqlens=slots.new_tensor([0, *range(5, 21)]),
        state_indices=slots, save=True, checkpoint_interval=1,
    )
    assert torch.equal(result, expected)
    assert torch.equal(pool.index_select(0, slots.long()), final)
    saved_indices = slots.new_tensor([*range(5), *range(6, 35, 2)]).long()
    assert torch.equal(saved.index_select(0, saved_indices), torch.stack(checkpoints))


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


@pytest.mark.cuda_graph
def test_verify_checkpoint_graph_writes_only_active_prefixes():
    _require_cuda_extension()
    torch.manual_seed(635)
    inputs = tuple(
        (torch.randn(8, 1, 64, device="cuda") * 0.1).bfloat16() for _ in range(6)
    )
    initial = torch.randn(4, 1, 64, 64, device="cuda").mul_(0.02)
    pool = initial.clone()
    prepare = prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor
    infer = infer_tmix_wkv7_recurrent_fp32io16_forward_varlen
    handle = prepare(pool)
    boundaries = initial.new_full((10, 1, 64, 64), 123)
    offsets = torch.tensor([0, -1, -1], device="cuda", dtype=torch.int32)
    slots = offsets.new_full((2,), 99)
    active_tokens = offsets.new_zeros(1)
    active_sequences = offsets.new_zeros(1)

    def launch():
        ticket = prepare_tmix_wkv7_recurrent_metadata(
            offsets, slots, state_pool_size=4, token_capacity=8, sequence_capacity=2,
            max_seqlen_capacity=8, num_active_tokens=active_tokens,
            num_active_sequences=active_sequences,
        )
        return infer(
            *inputs, state=handle, cu_seqlens=offsets, state_indices=slots,
            validated_metadata=ticket, save=True, boundaries=boundaries, checkpoint_interval=1,
        )[0]

    current = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    side.wait_stream(current)
    with torch.cuda.stream(side):
        launch()
    current.wait_stream(side)
    assert torch.equal(pool, initial)
    assert (boundaries == 123).all()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        output = launch()

    for lengths, slot_values in [((5,), [3, 99]), ((2, 4), [1, 0])]:
        counts = [0, lengths[0], sum(lengths) if len(lengths) == 2 else -1]
        offsets.copy_(offsets.new_tensor(counts))
        slots.copy_(slots.new_tensor(slot_values))
        active_tokens.fill_(sum(lengths))
        active_sequences.fill_(len(lengths))
        pool.copy_(initial)
        boundaries.fill_(123)
        expected, final, rows = _decode_reference(infer, prepare, inputs, initial, lengths, slots)
        wanted = torch.full_like(boundaries, 123)
        start = 0
        for seq, length in enumerate(lengths):
            wanted[start + seq:start + seq + length] = torch.stack(rows[start:start + length])
            start += length

        graph.replay()
        assert torch.equal(output[:start], expected)
        assert torch.isnan(output[start:]).all()
        assert torch.equal(boundaries, wanted)
        assert torch.equal(pool, final)

    pool.copy_(initial)
    boundaries.fill_(123)
    slots.fill_(1)
    graph.replay()
    assert torch.isnan(output).all()
    assert torch.equal(pool, initial)
    assert (boundaries == 123).all()


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

    def launch():
        ticket = prepare_tmix_wkv7_recurrent_metadata(
            cu, slots, state_pool_size=batch, token_capacity=batch, sequence_capacity=batch,
            max_seqlen_capacity=1, num_active_tokens=count, num_active_sequences=count,
        )
        return infer_tmix_wkv7_recurrent_fp16_forward_varlen(
            *inputs, state=state, cu_seqlens=cu, state_indices=slots, validated_metadata=ticket,
        )

    graph, output = _capture_graph(launch)

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


@pytest.mark.parametrize(
    "dtype,batch,heads",
    [(torch.float32, batch, 2) for batch in (1, 4, 32, 128)]
    + [(torch.float16, 1, 40), (torch.float16, 4, 64)]
    + [(torch.float16, batch, heads) for batch in (32, 128) for heads in (40, 64)],
)
def test_scratch_verify_and_state_only_commit_match_token_decode(dtype, batch, heads):
    torch.manual_seed(692)
    infer, prepare = _wkv_ops(dtype)
    span, dim = 4, 64
    raw = torch.randn(batch + 2, heads, dim, dim, device="cuda").mul_(0.02).to(dtype)
    reference_raw = raw.clone()
    main, reference, scratch = prepare(raw), prepare(reference_raw), prepare(torch.empty_like(raw[:batch]))
    rows = torch.arange(batch, device="cuda", dtype=torch.int32)
    slots = (batch - rows).contiguous()
    unit_cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    bias = torch.randn(heads, dim, device="cuda", dtype=torch.bfloat16)
    history = tuple(torch.randn(batch, heads, dim, device="cuda").mul_(0.04).bfloat16() for _ in range(6))
    infer(*history, state=reference, cu_seqlens=unit_cu, state_indices=slots, decay_bias=bias)
    assert infer(None, *history[1:], state=main, cu_seqlens=unit_cu, state_indices=slots, decay_bias=bias) is None

    inputs = tuple(torch.randn(batch, span, heads, dim, device="cuda").mul_(0.06).bfloat16() for _ in range(6))
    count = unit_cu[-1:].clone()
    ticket = prepare_tmix_wkv7_recurrent_metadata(
        unit_cu, slots, state_pool_size=raw.shape[0], token_capacity=batch, sequence_capacity=batch,
        max_seqlen_capacity=1, num_active_tokens=count, num_active_sequences=count,
    )
    bits = torch.int16 if dtype == torch.float16 else torch.int32
    for lengths in (([1], [span]) if batch == 1 else ([1 + i % span for i in range(batch)],)):
        scratch.copy_slots_(main, slots, rows, validated_metadata=ticket)
        before = raw.clone()
        verify = infer(
            *(value.flatten(0, 1) for value in inputs), state=scratch, cu_seqlens=unit_cu * span,
            state_indices=rows, decay_bias=bias, save=False,
        )
        assert verify.shape == (batch * span, heads, dim) and torch.equal(raw, before)

        lens = unit_cu.new_tensor(lengths)
        keep = torch.arange(span, device="cuda") < lens[:, None]
        commit_cu = torch.cat((unit_cu[:1], lens.cumsum(0, dtype=torch.int32)))
        assert infer(
            None, *(value[keep] for value in inputs[1:]), state=main, cu_seqlens=commit_cu,
            state_indices=slots, decay_bias=bias,
        ) is None
        for step in range(max(lengths)):
            active = [i for i, length in enumerate(lengths) if step < length]
            infer(
                *(value[active, step] for value in inputs), state=reference, cu_seqlens=unit_cu[:len(active) + 1],
                state_indices=slots[active], decay_bias=bias,
            )
        main.materialize_slots_(slots)
        reference.materialize_slots_(slots)
        assert torch.equal(raw.view(bits), reference_raw.view(bits))


@pytest.mark.cuda_graph
def test_fp16_state_only_commit_replays_live_prefixes():
    torch.manual_seed(694)
    batch, span, heads, dim = 128, 4, 64, 64
    infer, prepare = _wkv_ops(torch.float16)
    pool = torch.randn(batch + 1, heads, dim, dim, device="cuda").mul_(0.02).half()
    expected = pool.clone()
    state, reference = prepare(pool), prepare(expected)
    inputs = tuple(torch.randn(batch, span, heads, dim, device="cuda").mul_(0.04).bfloat16() for _ in range(6))
    packed = tuple(torch.empty_like(value.flatten(0, 1)) for value in inputs[1:])
    cu = torch.zeros(batch + 1, device="cuda", dtype=torch.int32)
    slots = torch.zeros(batch, device="cuda", dtype=torch.int32)
    num_sequences, num_tokens = cu[:1].clone(), cu[:1].clone()

    def launch():
        ticket = prepare_tmix_wkv7_recurrent_metadata(
            cu, slots, state_pool_size=pool.shape[0], token_capacity=batch * span, sequence_capacity=batch,
            max_seqlen_capacity=span, num_active_tokens=num_tokens, num_active_sequences=num_sequences,
        )
        assert infer(None, *packed, state=state, cu_seqlens=cu, state_indices=slots, validated_metadata=ticket) is None

    graph, _ = _capture_graph(launch)
    for active in (batch, 0, batch // 2):
        lens = cu.new_tensor([1 + i % span for i in range(active)])
        tokens = int(lens.sum())
        cu[1:active + 1].copy_(lens.cumsum(0, dtype=torch.int32))
        slots[:active].copy_(torch.arange(1, active + 1, device="cuda", dtype=torch.int32))
        slots[active:].zero_()
        keep = torch.arange(span, device="cuda") < lens[:, None]
        compact = tuple(value[:active][keep] for value in inputs)
        for destination, value in zip(packed, compact[1:]):
            destination[:tokens].copy_(value)
        num_sequences.fill_(active)
        num_tokens.fill_(tokens)
        graph.replay()
        if active:
            infer(*compact, state=reference, cu_seqlens=cu[:active + 1], state_indices=slots[:active])
    indices = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    state.materialize_slots_(indices)
    reference.materialize_slots_(indices)
    assert torch.equal(pool.view(torch.int16), expected.view(torch.int16))


@pytest.mark.cuda_graph
@pytest.mark.parametrize("overlap", (False, True))
def test_live_copy_slots_graph_ignores_duplicate_padding(overlap):
    torch.manual_seed(693)
    batch, heads, dim = 16, 64, 64
    infer, prepare = _wkv_ops(torch.float16)
    src_pool = torch.randn(batch, heads, dim, dim, device="cuda").mul_(0.02).half()
    dst_pool = src_pool if overlap else -src_pool
    src_state = prepare(src_pool)
    dst_state = src_state if overlap else prepare(dst_pool)
    expected_src, expected_dst = src_pool.clone(), dst_pool.clone()
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    slots = cu[:-1].clone()
    inputs = tuple(torch.randn(batch, heads, dim, device="cuda").mul_(0.04).bfloat16() for _ in range(6))
    for state, expected in (((src_state, expected_src),) if overlap else ((src_state, expected_src), (dst_state, expected_dst))):
        ref = prepare(expected)
        infer(*inputs, state=state, cu_seqlens=cu, state_indices=slots)
        infer(*inputs, state=ref, cu_seqlens=cu, state_indices=slots)
        ref.materialize_slots_(slots)
    if overlap:
        expected_dst = expected_src
    pending_zero = src_pool[0].clone()
    assert not torch.equal(pending_zero, expected_src[0])

    static = prepare_tmix_wkv7_recurrent_metadata(cu, slots, total_tokens=batch, state_pool_size=batch)
    with pytest.raises(ValueError, match="live"):
        dst_state.copy_slots_(src_state, slots, slots, validated_metadata=static)

    src, dst, count = torch.zeros_like(slots), torch.zeros_like(slots), cu[:1].clone()

    def launch():
        live = prepare_tmix_wkv7_recurrent_metadata(
            cu, dst, state_pool_size=batch, token_capacity=batch, sequence_capacity=batch,
            max_seqlen_capacity=1, num_active_tokens=count, num_active_sequences=count,
        )
        dst_state.copy_slots_(src_state, src, dst, validated_metadata=live)

    graph, _ = _capture_graph(launch)
    for source_rows, destination_rows in (([1, 1, 2], [2, 3, 1]), ([], []), ([0, 2], [4, 0])):
        active = len(source_rows)
        src.zero_()
        dst.zero_()
        src[:active] = src.new_tensor(source_rows)
        dst[:active] = dst.new_tensor(destination_rows)
        count.fill_(active)
        graph.replay()
        if 0 not in source_rows:
            assert torch.equal(src_pool[0], pending_zero)
        if active:
            expected_dst[destination_rows] = expected_src[source_rows]
            assert torch.equal(dst_pool[destination_rows], expected_dst[destination_rows])
    dst_state.materialize_slots_(slots)
    assert torch.equal(dst_pool, expected_dst)
    if not overlap:
        src_state.materialize_slots_(slots)
        assert torch.equal(src_pool, expected_src)


@pytest.mark.cuda_graph
@pytest.mark.parametrize("batch", [16, 257])
def test_live_mixed_prefill_matches_token_decode_after_row_reordering(batch):
    torch.manual_seed(137)
    heads, dim, capacity = 64, 64, batch + max(17, batch)
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

    graph, output = _capture_graph(launch)
    # Eight-token prompts change the decode traversal on the large Ada graph.
    prompts = (batch + 7) // 8
    mixed = [1] * (batch - prompts) + [8] * prompts
    for lengths in ([1] * batch, [1] * batch, [1] * (batch - 1) + [17],
                    [1] * batch, mixed, [5, 2, 7, 4, 3] + [1] * (batch - 9)):
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
