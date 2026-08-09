# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flashrwkv2.cmix.mix import (
    infer_cmix_add_layer_norm_mix_forward_varlen,
    infer_cmix_linear_ffn_down_forward_varlen,
    infer_cmix_mix_forward_varlen,
    infer_cmix_relu_square_forward_varlen,
    pretrain_cmix_bf16,
    statetune_cmix_bf16,
)
from flashrwkv2.tmix.wkv7 import prepare_recurrent_metadata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("batch_size", "max_seqlen", "total_tokens"),
    ((255, 257, 65535), (256, 256, 65536), (320, 256, 81920)),
)
def test_infer_cmix_cuda_grid_y_boundaries(
    batch_size: int, max_seqlen: int, total_tokens: int
) -> None:
    device = torch.device("cuda")
    channels = 4096
    lengths = [max_seqlen] * (batch_size - 1)
    lengths.append(total_tokens - sum(lengths))
    assert min(lengths) > 0 and max(lengths) == max_seqlen
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)

    x = torch.zeros(total_tokens, channels, device=device, dtype=torch.float16)
    x_k = torch.full((channels,), 0.25, device=device, dtype=torch.float16)
    shift = torch.ones(batch_size, channels, device=device, dtype=torch.float16)
    cu = torch.tensor(starts, device=device, dtype=torch.int32)
    slots = torch.arange(batch_size, device=device, dtype=torch.int32)

    output = infer_cmix_mix_forward_varlen(
        x,
        x_k,
        shift_state_pool=shift,
        cu_seqlens=cu,
        state_indices=slots,
        max_seqlen=max_seqlen,
    )
    first_rows = torch.tensor(starts[:-1], device=device)
    second_rows = first_rows + 1
    assert torch.isfinite(output).all()
    assert torch.allclose(
        output[first_rows], x_k.expand(batch_size, -1), atol=0.001, rtol=0.001
    )
    assert torch.count_nonzero(output[second_rows]) == 0
    assert torch.count_nonzero(shift) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_pretrain_cmix_forward_backward() -> None:
    torch.manual_seed(5)
    device = torch.device("cuda")
    b, t, c = 1, 3, 4
    x = (torch.randn(b, t, c, device=device) * 0.03).to(torch.bfloat16).requires_grad_()
    x_k = (torch.randn(c, device=device) * 0.1).to(torch.bfloat16).requires_grad_()
    key = (
        (torch.randn(4 * c, c, device=device) * 0.05)
        .to(torch.bfloat16)
        .requires_grad_()
    )
    value = (
        (torch.randn(c, 4 * c, device=device) * 0.05)
        .to(torch.bfloat16)
        .requires_grad_()
    )
    output = pretrain_cmix_bf16(x, x_k, key, value)
    previous = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)
    mixed = x + (previous - x) * x_k
    preact = mixed.float().reshape(-1, c) @ key.float().t()
    activation = torch.relu(preact) ** 2
    expected = (activation @ value.float().t()).reshape_as(x)
    assert torch.allclose(output.float(), expected.float(), atol=0.03, rtol=0.03)
    relu_square_input = torch.randn(3, 8, device=device, dtype=torch.float16)
    relu_square_output = infer_cmix_relu_square_forward_varlen(relu_square_input)
    assert torch.allclose(
        relu_square_output.float(),
        torch.relu(relu_square_input.float()).square(),
        atol=0.002,
        rtol=0.002,
    )
    output.float().sum().backward()
    assert torch.isfinite(x.grad.float()).all()
    assert torch.isfinite(x_k.grad.float()).all()
    assert torch.isfinite(key.grad.float()).all()
    assert torch.isfinite(value.grad.float()).all()

    # CMix's dense FFN-down caller owns the Albatross C=4096 tuned table.
    # The table is selected internally for the canonical 48-row shape; this
    # test only observes the exact GEMM result, not a forced algorithm API.
    rows, hidden, channels = 48, 16384, 4096
    down_x = torch.randn(rows, hidden, device=device, dtype=torch.float16) * 0.01
    down_weight = (
        torch.randn(hidden, channels, device=device, dtype=torch.float16) * 0.01
    )
    down_output = infer_cmix_linear_ffn_down_forward_varlen(down_x, down_weight)
    expected_down = down_x.float() @ down_weight.float()
    assert torch.allclose(down_output.float(), expected_down, atol=0.04, rtol=0.04)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch_size", (1, 2))
@pytest.mark.parametrize("seqlen", (1, 2, 7, 16, 31))
@pytest.mark.parametrize("zero_shift", (False, True))
def test_statetune_cmix_matches_torch_forward_backward(
    batch_size: int, seqlen: int, zero_shift: bool
) -> None:
    torch.manual_seed(2000 + batch_size * 100 + seqlen)
    device = torch.device("cuda")
    channels = 16
    x = (
        (torch.randn(batch_size, seqlen, channels, device=device) * 0.05)
        .to(torch.bfloat16)
        .requires_grad_()
    )
    initial = (torch.randn(batch_size, channels, device=device) * 0.05).to(
        torch.bfloat16
    )
    if zero_shift:
        initial.zero_()
    initial.requires_grad_()
    x_k = (
        (torch.randn(channels, device=device) * 0.1).to(torch.bfloat16).requires_grad_()
    )
    key = (
        (torch.randn(4 * channels, channels, device=device) * 0.05)
        .to(torch.bfloat16)
        .requires_grad_()
    )
    value = (
        (torch.randn(channels, 4 * channels, device=device) * 0.05)
        .to(torch.bfloat16)
        .requires_grad_()
    )
    actual = statetune_cmix_bf16(x, initial, x_k, key, value)

    x_ref = x.detach().clone().requires_grad_()
    initial_ref = initial.detach().clone().requires_grad_()
    x_k_ref = x_k.detach().clone().requires_grad_()
    key_ref = key.detach().clone().requires_grad_()
    value_ref = value.detach().clone().requires_grad_()
    previous = torch.cat((initial_ref[:, None, :], x_ref[:, :-1, :]), dim=1)
    mixed = x_ref + (previous - x_ref) * x_k_ref
    activation = torch.relu(mixed @ key_ref.t()).square()
    expected = (activation @ value_ref.t(), x_ref[:, -1, :].contiguous())
    upstream = [torch.randn_like(output) for output in actual]
    actual_grads = torch.autograd.grad(
        actual, (x, initial, x_k, key, value), grad_outputs=upstream
    )
    expected_grads = torch.autograd.grad(
        expected,
        (x_ref, initial_ref, x_k_ref, key_ref, value_ref),
        grad_outputs=upstream,
    )
    for output, reference in zip(actual, expected, strict=True):
        assert torch.allclose(output, reference, atol=0.03, rtol=0.03)
    for gradient, reference in zip(actual_grads, expected_grads, strict=True):
        assert torch.allclose(gradient, reference, atol=0.06, rtol=0.06)

    if zero_shift:
        pretrain = pretrain_cmix_bf16(
            x.detach(), x_k.detach(), key.detach(), value.detach()
        )
        assert torch.equal(actual[0], pretrain)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_statetune_cmix_chunk_composition() -> None:
    torch.manual_seed(29)
    device = torch.device("cuda")
    b, t, c, split = 2, 7, 16, 3
    # Keep the chunk test in the operator's normal activation/weight range.
    # Unit-scale BF16 FFN matrices make whole-vs-split GEMM weight gradients
    # differ at cancellation points solely because each chunk is rounded before
    # autograd adds it, which does not exercise the recurrent shift contract.
    x = (torch.randn(b, t, c, device=device) * 0.05).to(torch.bfloat16).requires_grad_()
    initial = (torch.randn(b, c, device=device) * 0.05).to(torch.bfloat16).requires_grad_()
    x_k = (torch.randn(c, device=device) * 0.1).to(torch.bfloat16).requires_grad_()
    key = (torch.randn(4 * c, c, device=device) * 0.05).to(torch.bfloat16).requires_grad_()
    value = (torch.randn(c, 4 * c, device=device) * 0.05).to(torch.bfloat16).requires_grad_()
    whole = statetune_cmix_bf16(x, initial, x_k, key, value)
    upstream = [torch.randn_like(output) for output in whole]
    whole_grads = torch.autograd.grad(
        whole, (x, initial, x_k, key, value), grad_outputs=upstream
    )

    x_chunked = x.detach().clone().requires_grad_()
    initial_chunked = initial.detach().clone().requires_grad_()
    x_k_chunked = x_k.detach().clone().requires_grad_()
    key_chunked = key.detach().clone().requires_grad_()
    value_chunked = value.detach().clone().requires_grad_()
    first = statetune_cmix_bf16(
        x_chunked[:, :split].contiguous(),
        initial_chunked,
        x_k_chunked,
        key_chunked,
        value_chunked,
    )
    second = statetune_cmix_bf16(
        x_chunked[:, split:].contiguous(),
        first[1],
        x_k_chunked,
        key_chunked,
        value_chunked,
    )
    chunked = (torch.cat((first[0], second[0]), dim=1), second[1])
    chunked_grads = torch.autograd.grad(
        chunked,
        (x_chunked, initial_chunked, x_k_chunked, key_chunked, value_chunked),
        grad_outputs=upstream,
    )
    for expected, actual in zip(whole, chunked, strict=True):
        assert torch.equal(expected, actual)
    for expected, actual in zip(whole_grads, chunked_grads, strict=True):
        assert torch.allclose(expected, actual, atol=0.06, rtol=0.06)


def test_statetune_cmix_rejects_cpu_and_invalid_shape() -> None:
    x = torch.zeros(1, 1, 8, dtype=torch.bfloat16)
    initial = torch.zeros(1, 8, dtype=torch.bfloat16)
    x_k = torch.zeros(8, dtype=torch.bfloat16)
    key = torch.zeros(32, 8, dtype=torch.bfloat16)
    value = torch.zeros(8, 32, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA"):
        statetune_cmix_bf16(x, initial, x_k, key, value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_statetune_cmix_rejects_misaligned_vec2_input() -> None:
    device = torch.device("cuda")
    x = torch.zeros(1, 2, 8, device=device, dtype=torch.bfloat16)
    initial = torch.empty(1 * 8 + 1, device=device, dtype=torch.bfloat16)[1:].view(
        1, 8
    )
    x_k = torch.zeros(8, device=device, dtype=torch.bfloat16)
    key = torch.zeros(32, 8, device=device, dtype=torch.bfloat16)
    value = torch.zeros(8, 32, device=device, dtype=torch.bfloat16)
    assert initial.is_contiguous() and initial.data_ptr() % 4 == 2
    with pytest.raises(ValueError, match="initial_shift must be 4-byte aligned"):
        statetune_cmix_bf16(x, initial, x_k, key, value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_statetune_cmix_rejects_invalid_cuda_contracts() -> None:
    device = torch.device("cuda")
    x = torch.zeros(1, 2, 8, device=device, dtype=torch.bfloat16)
    initial = torch.zeros(1, 8, device=device, dtype=torch.bfloat16)
    x_k = torch.zeros(8, device=device, dtype=torch.bfloat16)
    key = torch.zeros(32, 8, device=device, dtype=torch.bfloat16)
    value = torch.zeros(8, 32, device=device, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="x must have dtype"):
        statetune_cmix_bf16(x.float(), initial, x_k, key, value)
    with pytest.raises(TypeError, match="initial_shift must have dtype"):
        statetune_cmix_bf16(x, initial.half(), x_k, key, value)
    with pytest.raises(ValueError, match="x must be contiguous"):
        statetune_cmix_bf16(x.transpose(1, 2), initial, x_k, key, value)
    with pytest.raises(ValueError, match="non-empty"):
        statetune_cmix_bf16(x[:, :0], initial, x_k, key, value)
    with pytest.raises(ValueError, match="initial_shift must have shape"):
        statetune_cmix_bf16(x, initial[:, :-2].contiguous(), x_k, key, value)
    with pytest.raises(ValueError, match="key_weight must have shape"):
        statetune_cmix_bf16(x, initial, x_k, key[:-1].contiguous(), value)

    odd_x = torch.zeros(1, 2, 7, device=device, dtype=torch.bfloat16)
    odd_initial = torch.zeros(1, 7, device=device, dtype=torch.bfloat16)
    odd_x_k = torch.zeros(7, device=device, dtype=torch.bfloat16)
    odd_key = torch.zeros(28, 7, device=device, dtype=torch.bfloat16)
    odd_value = torch.zeros(7, 28, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="divisible by 2"):
        statetune_cmix_bf16(odd_x, odd_initial, odd_x_k, odd_key, odd_value)

    if torch.cuda.device_count() > 1:
        foreign = x_k.to("cuda:1")
        with pytest.raises(ValueError, match="x_k must share x's device"):
            statetune_cmix_bf16(x, initial, foreign, key, value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_infer_cmix_ragged_ticket_updates_only_selected_shift_slots() -> None:
    device = torch.device("cuda")
    c = 8
    lengths = (1, 3)
    total = sum(lengths)
    x = torch.arange(total * c, device=device, dtype=torch.float16).reshape(total, c)
    x_k = torch.full((c,), 0.25, device=device, dtype=torch.float16)
    initial = torch.randn(5, c, device=device, dtype=torch.float16)
    shift_state = initial.clone()
    cu_seqlens = torch.tensor([0, lengths[0], total], device=device, dtype=torch.int32)
    state_indices = torch.tensor([0, 4], device=device, dtype=torch.int32)
    ticket = prepare_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=total,
        state_pool_size=shift_state.shape[0],
    )

    output = infer_cmix_mix_forward_varlen(
        x,
        x_k,
        shift_state_pool=shift_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        validated_metadata=ticket,
    )

    expected = []
    expected_shift = initial.clone()
    start = 0
    for sequence, length in enumerate(lengths):
        slot = int(state_indices[sequence].item())
        previous = initial[slot]
        for token in range(start, start + length):
            current = x[token]
            expected.append(current + (previous - current) * x_k)
            previous = current
        expected_shift[slot] = previous
        start += length

    assert torch.allclose(output, torch.stack(expected), atol=0.01, rtol=0.01)
    assert torch.allclose(shift_state, expected_shift, atol=0.01, rtol=0.01)
    untouched = [slot for slot in range(initial.shape[0]) if slot not in (0, 4)]
    assert torch.equal(shift_state[untouched], initial[untouched])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_infer_cmix_ticket_rejects_metadata_mutation_without_state_write() -> None:
    device = torch.device("cuda")
    c = 8
    total = 5
    x = torch.randn(total, c, device=device, dtype=torch.float16)
    x_k = torch.randn(c, device=device, dtype=torch.float16)
    shift_state = torch.randn(4, c, device=device, dtype=torch.float16)
    cu_seqlens = torch.tensor([0, 2, total], device=device, dtype=torch.int32)
    state_indices = torch.tensor([1, 3], device=device, dtype=torch.int32)
    ticket = prepare_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=total,
        state_pool_size=shift_state.shape[0],
    )

    before = shift_state.clone()
    state_indices[1] = 2
    with pytest.raises(RuntimeError, match="version"):
        infer_cmix_mix_forward_varlen(
            x,
            x_k,
            shift_state_pool=shift_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            validated_metadata=ticket,
        )
    assert torch.equal(shift_state, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_infer_cmix_cuda_graph_ignores_inactive_tail_state_slots() -> None:
    device = torch.device("cuda")
    token_capacity, sequence_capacity, channels, num_slots = 3, 2, 8, 4
    x = torch.randn(token_capacity, channels, device=device, dtype=torch.float16)
    x_k = torch.randn(channels, device=device, dtype=torch.float16)
    initial = torch.randn(num_slots, channels, device=device, dtype=torch.float16)
    shift_state = initial.clone()
    cu_seqlens = torch.tensor([0, 1, 3], device=device, dtype=torch.int32)
    state_indices = torch.tensor([3, 1], device=device, dtype=torch.int32)
    num_active_tokens = torch.tensor([3], device=device, dtype=torch.int32)
    num_active_sequences = torch.tensor([2], device=device, dtype=torch.int32)

    infer_cmix_mix_forward_varlen(
        x,
        x_k,
        shift_state_pool=initial.clone(),
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
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
        infer_cmix_mix_forward_varlen(
            x,
            x_k,
            shift_state_pool=shift_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            validated_metadata=ticket,
        )

    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(shift_state[3], x[0])
    assert torch.equal(shift_state[1], x[2])
    assert torch.equal(shift_state[0], initial[0])
    assert torch.equal(shift_state[2], initial[2])

    shift_state.copy_(initial)
    cu_seqlens.copy_(torch.tensor([0, 2, -1], device=device, dtype=torch.int32))
    state_indices.copy_(torch.tensor([2, 99], device=device, dtype=torch.int32))
    num_active_tokens.fill_(2)
    num_active_sequences.fill_(1)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(shift_state[2], x[1])
    assert torch.equal(shift_state[[0, 1, 3]], initial[[0, 1, 3]])

    shift_state.copy_(initial)
    cu_seqlens.copy_(torch.tensor([0, 1, 2], device=device, dtype=torch.int32))
    state_indices.copy_(torch.tensor([1, 1], device=device, dtype=torch.int32))
    num_active_tokens.fill_(2)
    num_active_sequences.fill_(2)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(shift_state, initial)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_infer_cmix_fused_add_layer_norm_matches_albatross_t1_path() -> None:
    torch.manual_seed(17)
    device = torch.device("cuda")
    b, c = 2, 4096
    eps = 1.0e-5
    x = (torch.randn(b, c, device=device) * 0.03).to(torch.float16)
    residual = (torch.randn(b, c, device=device) * 0.02).to(torch.float16)
    weight = (torch.randn(c, device=device) * 0.1 + 1.0).to(torch.float16)
    bias = (torch.randn(c, device=device) * 0.01).to(torch.float16)
    x_k = torch.full((c,), 0.25, device=device, dtype=torch.float16)
    initial = torch.randn(6, c, device=device, dtype=torch.float16)
    shift_state = initial.clone()
    cu_seqlens = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
    state_indices = torch.tensor([4, 1], device=device, dtype=torch.int32)
    ticket = prepare_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=b,
        state_pool_size=shift_state.shape[0],
        max_seqlen=1,
    )

    x_out, mixed = infer_cmix_add_layer_norm_mix_forward_varlen(
        x,
        residual,
        weight,
        bias,
        x_k,
        shift_state_pool=shift_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        validated_metadata=ticket,
    )

    summed = x.float() + residual.float()
    mean = summed.mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt((summed - mean).square().mean(dim=-1, keepdim=True) + eps)
    normalized = ((summed - mean) * rstd * weight.float() + bias.float()).to(
        torch.float16
    )
    expected_mixed = (
        normalized.float()
        + (initial[state_indices.long()].float() - normalized.float()) * x_k.float()
    ).to(torch.float16)

    assert torch.allclose(x_out, summed.to(torch.float16), atol=0.01, rtol=0.01)
    assert torch.allclose(mixed, expected_mixed, atol=0.01, rtol=0.01)
    expected_state = initial.clone()
    expected_state[state_indices.long()] = normalized
    assert torch.allclose(shift_state, expected_state, atol=0.01, rtol=0.01)
    assert torch.equal(shift_state[[0, 2, 3, 5]], initial[[0, 2, 3, 5]])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_infer_cmix_fused_cuda_graph_ignores_inactive_tail_state_slots() -> None:
    torch.manual_seed(20260809)
    device = torch.device("cuda")
    capacity, channels, num_slots = 2, 4096, 4
    x = torch.randn(capacity, channels, device=device, dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn(channels, device=device, dtype=torch.float16)
    bias = torch.randn(channels, device=device, dtype=torch.float16)
    x_k = torch.randn(channels, device=device, dtype=torch.float16)
    initial = torch.randn(num_slots, channels, device=device, dtype=torch.float16)
    shift_state = initial.clone()
    cu_seqlens = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
    state_indices = torch.tensor([3, 1], device=device, dtype=torch.int32)
    num_active_tokens = torch.tensor([2], device=device, dtype=torch.int32)
    num_active_sequences = torch.tensor([2], device=device, dtype=torch.int32)

    infer_cmix_add_layer_norm_mix_forward_varlen(
        x,
        residual,
        weight,
        bias,
        x_k,
        shift_state_pool=initial.clone(),
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        max_seqlen=1,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ticket = prepare_recurrent_metadata(
            cu_seqlens,
            state_indices,
            state_pool_size=num_slots,
            token_capacity=capacity,
            sequence_capacity=capacity,
            max_seqlen_capacity=1,
            num_active_tokens=num_active_tokens,
            num_active_sequences=num_active_sequences,
        )
        infer_cmix_add_layer_norm_mix_forward_varlen(
            x,
            residual,
            weight,
            bias,
            x_k,
            shift_state_pool=shift_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            validated_metadata=ticket,
        )

    shift_state.copy_(initial)
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(shift_state[3], initial[3])
    assert not torch.equal(shift_state[1], initial[1])
    assert torch.equal(shift_state[[0, 2]], initial[[0, 2]])

    shift_state.copy_(initial)
    cu_seqlens.copy_(torch.tensor([0, 1, -1], device=device, dtype=torch.int32))
    state_indices.copy_(torch.tensor([2, 99], device=device, dtype=torch.int32))
    num_active_tokens.fill_(1)
    num_active_sequences.fill_(1)
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(shift_state[2], initial[2])
    assert torch.equal(shift_state[[0, 1, 3]], initial[[0, 1, 3]])

    shift_state.copy_(initial)
    num_active_tokens.zero_()
    num_active_sequences.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(shift_state, initial)

    shift_state.copy_(initial)
    cu_seqlens.copy_(torch.tensor([0, 1, 2], device=device, dtype=torch.int32))
    state_indices.copy_(torch.tensor([1, 1], device=device, dtype=torch.int32))
    num_active_tokens.fill_(2)
    num_active_sequences.fill_(2)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(shift_state, initial)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_infer_cmix_fused_rejects_non_t1_dispatch() -> None:
    device = torch.device("cuda")
    x = torch.zeros(2, 4096, device=device, dtype=torch.float16)
    residual = torch.zeros_like(x)
    parameter = torch.ones(4096, device=device, dtype=torch.float16)
    shift_state = torch.zeros(4, 4096, device=device, dtype=torch.float16)
    cu_seqlens = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
    state_indices = torch.tensor([0, 1], device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match="max_seqlen=1"):
        infer_cmix_add_layer_norm_mix_forward_varlen(
            x,
            residual,
            parameter,
            torch.zeros_like(parameter),
            parameter,
            shift_state_pool=shift_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            max_seqlen=2,
        )
