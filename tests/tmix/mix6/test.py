# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flashrwkv2.tmix.mix6 import (
    infer_tmix_mix6_add_layer_norm_forward_varlen,
    infer_tmix_mix6_forward_varlen,
    pretrain_tmix_mix6_bf16,
    statetune_tmix_mix6_bf16,
)
from flashrwkv2.tmix.wkv7 import prepare_recurrent_metadata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("batch_size", "max_seqlen", "total_tokens"),
    ((255, 257, 65535), (256, 256, 65536), (320, 256, 81920)),
)
def test_infer_mix6_cuda_grid_y_boundaries(
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
    params = [
        torch.full((channels,), value, device=device, dtype=torch.float16)
        for value in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    ]
    shift = torch.ones(batch_size, channels, device=device, dtype=torch.float16)
    cu = torch.tensor(starts, device=device, dtype=torch.int32)
    slots = torch.arange(batch_size, device=device, dtype=torch.int32)

    outputs = infer_tmix_mix6_forward_varlen(
        x,
        *params,
        shift_state_pool=shift,
        cu_seqlens=cu,
        state_indices=slots,
        max_seqlen=max_seqlen,
    )
    first_rows = torch.tensor(starts[:-1], device=device)
    second_rows = first_rows + 1
    for output, parameter in zip(outputs, params, strict=True):
        assert torch.isfinite(output).all()
        assert torch.allclose(
            output[first_rows], parameter.expand(batch_size, -1), atol=0.001, rtol=0.001
        )
        assert torch.count_nonzero(output[second_rows]) == 0
    assert torch.count_nonzero(shift) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_pretrain_mix6_forward_backward() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    x = (torch.randn(2, 3, 8, device=device) * 0.1).to(torch.bfloat16).requires_grad_()
    params = [
        (torch.randn(8, device=device) * 0.1).to(torch.bfloat16).requires_grad_()
        for _ in range(6)
    ]
    outputs = pretrain_tmix_mix6_bf16(x, *params)
    reference = []
    previous = torch.zeros_like(x[:, :1])
    for parameter in params:
        mixed = x + (torch.cat((previous, x[:, :-1]), dim=1) - x) * parameter
        reference.append(mixed)
    for output, expected in zip(outputs, reference, strict=True):
        assert torch.allclose(output.float(), expected.float(), atol=0.01, rtol=0.01)
    sum(output.float().sum() for output in outputs).backward()
    assert torch.isfinite(x.grad.float()).all()
    for parameter in params:
        assert torch.isfinite(parameter.grad.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch_size", (1, 2))
@pytest.mark.parametrize("seqlen", (1, 2, 7, 16, 31))
@pytest.mark.parametrize("zero_shift", (False, True))
def test_statetune_mix6_matches_torch_forward_backward(
    batch_size: int, seqlen: int, zero_shift: bool
) -> None:
    torch.manual_seed(1000 + batch_size * 100 + seqlen)
    device = torch.device("cuda")
    channels = 128
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
    params = [
        (torch.randn(channels, device=device) * 0.1).to(torch.bfloat16).requires_grad_()
        for _ in range(6)
    ]
    actual = statetune_tmix_mix6_bf16(x, initial, *params)

    x_ref = x.detach().clone().requires_grad_()
    initial_ref = initial.detach().clone().requires_grad_()
    params_ref = [parameter.detach().clone().requires_grad_() for parameter in params]
    previous = torch.cat((initial_ref[:, None, :], x_ref[:, :-1, :]), dim=1)
    expected = tuple(
        x_ref + (previous - x_ref) * parameter for parameter in params_ref
    ) + (x_ref[:, -1, :].contiguous(),)
    upstream = [torch.randn_like(output) for output in actual]
    actual_grads = torch.autograd.grad(
        actual, (x, initial, *params), grad_outputs=upstream
    )
    expected_grads = torch.autograd.grad(
        expected, (x_ref, initial_ref, *params_ref), grad_outputs=upstream
    )
    for output, reference in zip(actual, expected, strict=True):
        assert torch.allclose(output, reference, atol=0.01, rtol=0.01)
    for gradient, reference in zip(actual_grads, expected_grads, strict=True):
        assert torch.allclose(gradient, reference, atol=0.03, rtol=0.03)

    if zero_shift:
        pretrain = pretrain_tmix_mix6_bf16(x.detach(), *(p.detach() for p in params))
        for output, reference in zip(actual[:6], pretrain, strict=True):
            assert torch.equal(output, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_statetune_mix6_chunk_composition() -> None:
    torch.manual_seed(23)
    device = torch.device("cuda")
    b, t, c, split = 2, 7, 128, 3
    x = torch.randn(b, t, c, device=device, dtype=torch.bfloat16).requires_grad_()
    initial = torch.randn(b, c, device=device, dtype=torch.bfloat16).requires_grad_()
    params = [
        torch.randn(c, device=device, dtype=torch.bfloat16).requires_grad_()
        for _ in range(6)
    ]
    whole = statetune_tmix_mix6_bf16(x, initial, *params)
    upstream = [torch.randn_like(output) for output in whole]
    whole_grads = torch.autograd.grad(
        whole, (x, initial, *params), grad_outputs=upstream
    )

    x_chunked = x.detach().clone().requires_grad_()
    initial_chunked = initial.detach().clone().requires_grad_()
    params_chunked = [parameter.detach().clone().requires_grad_() for parameter in params]
    first = statetune_tmix_mix6_bf16(
        x_chunked[:, :split].contiguous(), initial_chunked, *params_chunked
    )
    second = statetune_tmix_mix6_bf16(
        x_chunked[:, split:].contiguous(), first[-1], *params_chunked
    )
    chunked = tuple(
        torch.cat((left, right), dim=1)
        for left, right in zip(first[:6], second[:6], strict=True)
    ) + (second[-1],)
    chunked_grads = torch.autograd.grad(
        chunked,
        (x_chunked, initial_chunked, *params_chunked),
        grad_outputs=upstream,
    )
    for expected, actual in zip(whole, chunked, strict=True):
        assert torch.equal(expected, actual)
    for expected, actual in zip(whole_grads, chunked_grads, strict=True):
        assert torch.allclose(expected, actual, atol=0.03, rtol=0.03)


def test_statetune_mix6_rejects_cpu_and_invalid_shape() -> None:
    x = torch.zeros(1, 1, 8, dtype=torch.bfloat16)
    initial = torch.zeros(1, 8, dtype=torch.bfloat16)
    params = [torch.zeros(8, dtype=torch.bfloat16) for _ in range(6)]
    with pytest.raises(ValueError, match="CUDA"):
        statetune_tmix_mix6_bf16(x, initial, *params)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_statetune_mix6_rejects_misaligned_vec2_input() -> None:
    device = torch.device("cuda")
    x = torch.empty(1 * 2 * 8 + 1, device=device, dtype=torch.bfloat16)[1:].view(
        1, 2, 8
    )
    initial = torch.zeros(1, 8, device=device, dtype=torch.bfloat16)
    params = [torch.zeros(8, device=device, dtype=torch.bfloat16) for _ in range(6)]
    assert x.is_contiguous() and x.data_ptr() % 4 == 2
    with pytest.raises(ValueError, match="x must be 4-byte aligned"):
        statetune_tmix_mix6_bf16(x, initial, *params)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_statetune_mix6_rejects_invalid_cuda_contracts() -> None:
    device = torch.device("cuda")
    x = torch.zeros(1, 2, 8, device=device, dtype=torch.bfloat16)
    initial = torch.zeros(1, 8, device=device, dtype=torch.bfloat16)
    params = [torch.zeros(8, device=device, dtype=torch.bfloat16) for _ in range(6)]
    with pytest.raises(TypeError, match="x must have dtype"):
        statetune_tmix_mix6_bf16(x.float(), initial, *params)
    with pytest.raises(TypeError, match="initial_shift must have dtype"):
        statetune_tmix_mix6_bf16(x, initial.half(), *params)
    with pytest.raises(ValueError, match="x must be contiguous"):
        statetune_tmix_mix6_bf16(x.transpose(1, 2), initial, *params)
    with pytest.raises(ValueError, match="non-empty"):
        statetune_tmix_mix6_bf16(x[:, :0], initial, *params)
    with pytest.raises(ValueError, match="initial_shift must have shape"):
        statetune_tmix_mix6_bf16(x, initial[:, :-2].contiguous(), *params)
    with pytest.raises(ValueError, match="x_r must have shape"):
        statetune_tmix_mix6_bf16(x, initial, params[0][:-2].contiguous(), *params[1:])

    odd_x = torch.zeros(1, 2, 7, device=device, dtype=torch.bfloat16)
    odd_initial = torch.zeros(1, 7, device=device, dtype=torch.bfloat16)
    odd_params = [torch.zeros(7, device=device, dtype=torch.bfloat16) for _ in range(6)]
    with pytest.raises(ValueError, match="divisible by 2"):
        statetune_tmix_mix6_bf16(odd_x, odd_initial, *odd_params)

    if torch.cuda.device_count() > 1:
        foreign = params[0].to("cuda:1")
        with pytest.raises(ValueError, match="x_r must share x's device"):
            statetune_tmix_mix6_bf16(x, initial, foreign, *params[1:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_infer_mix6_consumes_packed_metadata_ticket_and_updates_last_shift() -> None:
    device = torch.device("cuda")
    c = 8
    lengths = (2, 3)
    total = sum(lengths)
    x = torch.arange(total * c, device=device, dtype=torch.float16).reshape(total, c)
    params = [
        torch.full((c,), value, device=device, dtype=torch.float16)
        for value in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    ]
    initial = torch.randn(4, c, device=device, dtype=torch.float16)
    shift_state = initial.clone()
    cu_seqlens = torch.tensor([0, lengths[0], total], device=device, dtype=torch.int32)
    state_indices = torch.tensor([1, 3], device=device, dtype=torch.int32)
    ticket = prepare_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=total,
        state_pool_size=shift_state.shape[0],
    )

    outputs = infer_tmix_mix6_forward_varlen(
        x,
        *params,
        shift_state_pool=shift_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        validated_metadata=ticket,
    )

    expected = [[] for _ in range(6)]
    expected_shift = initial.clone()
    start = 0
    for sequence, length in enumerate(lengths):
        slot = int(state_indices[sequence].item())
        previous = initial[slot]
        for token in range(start, start + length):
            current = x[token]
            for index, parameter in enumerate(params):
                expected[index].append(current + (previous - current) * parameter)
            previous = current
        expected_shift[slot] = previous
        start += length

    for output, rows in zip(outputs, expected, strict=True):
        assert torch.allclose(output, torch.stack(rows), atol=0.01, rtol=0.01)
    assert torch.allclose(shift_state, expected_shift, atol=0.01, rtol=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_infer_mix6_ticket_rejects_metadata_mutation_without_state_write() -> None:
    device = torch.device("cuda")
    c = 8
    total = 5
    x = torch.randn(total, c, device=device, dtype=torch.float16)
    params = [torch.randn(c, device=device, dtype=torch.float16) for _ in range(6)]
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
    cu_seqlens[1] += 1
    with pytest.raises(RuntimeError, match="version"):
        infer_tmix_mix6_forward_varlen(
            x,
            *params,
            shift_state_pool=shift_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            validated_metadata=ticket,
        )
    assert torch.equal(shift_state, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_infer_mix6_cuda_graph_ignores_inactive_tail_state_slots() -> None:
    device = torch.device("cuda")
    token_capacity, sequence_capacity, channels, num_slots = 3, 2, 8, 4
    x = torch.randn(token_capacity, channels, device=device, dtype=torch.float16)
    params = [
        torch.randn(channels, device=device, dtype=torch.float16)
        for _ in range(6)
    ]
    initial = torch.randn(num_slots, channels, device=device, dtype=torch.float16)
    shift_state = initial.clone()
    cu_seqlens = torch.tensor([0, 1, 3], device=device, dtype=torch.int32)
    state_indices = torch.tensor([3, 1], device=device, dtype=torch.int32)
    num_active_tokens = torch.tensor([3], device=device, dtype=torch.int32)
    num_active_sequences = torch.tensor([2], device=device, dtype=torch.int32)

    infer_tmix_mix6_forward_varlen(
        x,
        *params,
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
        infer_tmix_mix6_forward_varlen(
            x,
            *params,
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
def test_infer_mix6_fused_add_layer_norm_matches_albatross_t1_path() -> None:
    torch.manual_seed(19)
    device = torch.device("cuda")
    c = 4096
    eps = 1.0e-5
    x = (torch.randn(1, c, device=device) * 0.03).to(torch.float16)
    residual = (torch.randn(1, c, device=device) * 0.02).to(torch.float16)
    weight = (torch.randn(c, device=device) * 0.1 + 1.0).to(torch.float16)
    bias = (torch.randn(c, device=device) * 0.01).to(torch.float16)
    params = [(torch.randn(c, device=device) * 0.1).to(torch.float16) for _ in range(6)]
    initial = torch.randn(5, c, device=device, dtype=torch.float16)
    shift_state = initial.clone()
    cu_seqlens = torch.tensor([0, 1], device=device, dtype=torch.int32)
    state_indices = torch.tensor([3], device=device, dtype=torch.int32)
    ticket = prepare_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=1,
        state_pool_size=shift_state.shape[0],
        max_seqlen=1,
    )

    outputs = infer_tmix_mix6_add_layer_norm_forward_varlen(
        x,
        residual,
        weight,
        bias,
        *params,
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
    previous = initial[3]
    expected = [
        (
            normalized.float()
            + (previous.float() - normalized.float()) * parameter.float()
        ).to(torch.float16)
        for parameter in params
    ]

    assert torch.allclose(outputs[0], summed.to(torch.float16), atol=0.01, rtol=0.01)
    for output, expected_output in zip(outputs[1:], expected, strict=True):
        assert torch.allclose(output, expected_output, atol=0.01, rtol=0.01)
    expected_state = initial.clone()
    expected_state[3] = normalized[0]
    assert torch.allclose(shift_state, expected_state, atol=0.01, rtol=0.01)
    assert torch.equal(shift_state[[0, 1, 2, 4]], initial[[0, 1, 2, 4]])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_infer_mix6_fused_cuda_graph_zero_active_has_no_state_side_effect() -> None:
    torch.manual_seed(20260809)
    device = torch.device("cuda")
    channels, num_slots = 4096, 4
    x = torch.randn(1, channels, device=device, dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn(channels, device=device, dtype=torch.float16)
    bias = torch.randn(channels, device=device, dtype=torch.float16)
    params = [
        torch.randn(channels, device=device, dtype=torch.float16)
        for _ in range(6)
    ]
    initial = torch.randn(num_slots, channels, device=device, dtype=torch.float16)
    shift_state = initial.clone()
    cu_seqlens = torch.tensor([0, 1], device=device, dtype=torch.int32)
    state_indices = torch.tensor([3], device=device, dtype=torch.int32)
    num_active_tokens = torch.tensor([1], device=device, dtype=torch.int32)
    num_active_sequences = torch.tensor([1], device=device, dtype=torch.int32)

    infer_tmix_mix6_add_layer_norm_forward_varlen(
        x,
        residual,
        weight,
        bias,
        *params,
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
            token_capacity=1,
            sequence_capacity=1,
            max_seqlen_capacity=1,
            num_active_tokens=num_active_tokens,
            num_active_sequences=num_active_sequences,
        )
        infer_tmix_mix6_add_layer_norm_forward_varlen(
            x,
            residual,
            weight,
            bias,
            *params,
            shift_state_pool=shift_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            validated_metadata=ticket,
        )

    shift_state.copy_(initial)
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(shift_state[3], initial[3])
    assert torch.equal(shift_state[[0, 1, 2]], initial[[0, 1, 2]])

    shift_state.copy_(initial)
    cu_seqlens.zero_()
    state_indices.fill_(99)
    num_active_tokens.zero_()
    num_active_sequences.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(shift_state, initial)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_infer_mix6_fused_rejects_non_b1_dispatch() -> None:
    device = torch.device("cuda")
    x = torch.zeros(1, 4096, device=device, dtype=torch.float16)
    residual = torch.zeros_like(x)
    parameters = [
        torch.ones(4096, device=device, dtype=torch.float16) for _ in range(8)
    ]
    cu_seqlens = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
    state_indices = torch.tensor([0, 1], device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match="B=1"):
        infer_tmix_mix6_add_layer_norm_forward_varlen(
            x,
            residual,
            *parameters[:2],
            *parameters[2:],
            shift_state_pool=torch.zeros(4, 4096, device=device, dtype=torch.float16),
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
        )
