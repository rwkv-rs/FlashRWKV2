# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect
import json
from itertools import pairwise
from pathlib import Path

import pytest
import torch
from utils import require_cuda_backend

import flashrwkv2
import flashrwkv2.tmix.wkv7 as wkv7_module
from flashrwkv2.tmix.wkv7 import (
    infer_tmix_wkv7_recurrent_fp16_forward_varlen,
    infer_tmix_wkv7_recurrent_fp32io16_forward_varlen,
    prepare_tmix_wkv7_recurrent_fp16_state,
    prepare_tmix_wkv7_recurrent_fp32io16_state,
    prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor,
    prepare_tmix_wkv7_recurrent_metadata,
)

pytestmark = pytest.mark.cuda

ROOT = Path(__file__).resolve().parents[3]
TOLERANCES = json.loads(
    (ROOT / "tests/fixtures/tolerances-v1.json").read_text(encoding="utf-8")
)["fp32io16_recurrent"]


def rwkv7_decay_logits_reference(
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
    """Independent FP32 oracle for the kernel precision tests."""

    decay_rate = 0.6065306597126334
    if r.ndim != 3:
        raise ValueError("reference expects packed inputs with shape [total_tokens,H,D]")
    total_tokens, num_heads, head_size = r.shape
    offsets = tuple(int(value) for value in cu_seqlens.cpu().tolist())
    slots = tuple(int(value) for value in state_indices.cpu().tolist())
    state_pool = state_pool.float().clone()

    bias = None if decay_bias is None else decay_bias.reshape(num_heads, head_size)
    output = torch.empty(
        (total_tokens, num_heads, head_size),
        device=r.device,
        dtype=torch.float32,
    )
    r_f32 = r.float()
    logits_f32 = decay_logits.float()
    k_f32 = k.float()
    v_f32 = v.float()
    a_f32 = a.float()
    b_f32 = b.float()

    for sequence_index, (start, end) in enumerate(pairwise(offsets)):
        state = state_pool[slots[sequence_index]].clone()
        for token_index in range(start, end):
            token_logits = logits_f32[token_index]
            if bias is not None:
                token_logits = token_logits + bias.float()
            retention = torch.exp(-decay_rate * torch.sigmoid(token_logits))
            a_state = torch.einsum("hk,hkv->hv", a_f32[token_index], state)
            state = (
                retention.unsqueeze(-1) * state
                + b_f32[token_index].unsqueeze(-1) * a_state.unsqueeze(-2)
                + k_f32[token_index].unsqueeze(-1)
                * v_f32[token_index].unsqueeze(-2)
            )
            output[token_index] = float(scale) * torch.einsum(
                "hk,hkv->hv", r_f32[token_index], state
            )
        state_pool[slots[sequence_index]] = state

    return output, state_pool


def rwkv7_fp16_reference(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    elapsed_state_pool: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float = 1.0,
    decay_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference the Albatross FP16 delta and elapsed-phase contract."""

    decay_rate = 0.6065306597126334
    _, num_heads, head_size = r.shape
    offsets = tuple(int(value) for value in cu_seqlens.cpu().tolist())
    slots = tuple(int(value) for value in state_indices.cpu().tolist())
    expected_state = state_pool.clone()
    expected = torch.empty_like(r)
    elapsed = elapsed_state_pool.clone()
    bias = None if decay_bias is None else decay_bias.reshape(num_heads, head_size)
    for sequence_index, (start, end) in enumerate(pairwise(offsets)):
        slot = slots[sequence_index]
        state = expected_state[slot].clone()
        elapsed_base = int(elapsed[slot].item())
        for token_index in range(start, end):
            logits = decay_logits[token_index].float()
            if bias is not None:
                logits = logits + bias.float()
            retention = torch.exp(-decay_rate * torch.sigmoid(logits))
            phase = (
                torch.arange(num_heads * head_size, device=r.device, dtype=torch.int64)
                + elapsed_base
                + (token_index - start)
            )
            bits = (phase * 2654435769) & 0xFFFFFFFF
            signed_bits = torch.where(bits >= 0x80000000, bits - 0x100000000, bits)
            dither = (signed_bits.float() * 4.547473508864641e-13).reshape(
                num_heads, head_size
            )
            delta = (retention - 1.0 + dither).half().float()
            state_float = state.float()
            a_state = torch.einsum(
                "hk,hkv->hv", a[token_index].float(), state_float
            )
            state = (
                state_float
                + state_float * delta.unsqueeze(-1)
                + b[token_index].float().unsqueeze(-1) * a_state.unsqueeze(-2)
                + k[token_index].float().unsqueeze(-1)
                * v[token_index].float().unsqueeze(-2)
            ).half()
            expected[token_index] = (
                float(scale)
                * torch.einsum("hk,hkv->hv", r[token_index].float(), state.float())
            ).half()
        expected_state[slot] = state
    return expected, expected_state


def _require_cuda_extension() -> None:
    require_cuda_backend(
        "_C", 8, "tmix_wkv7_recurrent_fp32_from_decay_logits"
    )


def _require_fp16_extension() -> None:
    require_cuda_backend(
        "_C", 8, "tmix_wkv7_recurrent_fp16_from_decay_logits"
    )


def _require_deltalog_extension() -> None:
    require_cuda_backend(
        "_C",
        8,
        "tmix_wkv7_recurrent_deltalog_fp16_from_decay_logits",
    )


def _require_fp32io16_deltalog_extension() -> None:
    require_cuda_backend(
        "_C",
        8,
        "tmix_wkv7_recurrent_deltalog_fp32io16_from_decay_logits",
    )


def _prepare_fp32io16_state(
    state_pool: torch.Tensor,
    state_indices: torch.Tensor | list[int],
) -> object:
    sequence_capacity = (
        state_indices.numel()
        if isinstance(state_indices, torch.Tensor)
        else len(state_indices)
    )
    return _bind_fp32io16_state(
        state_pool,
        sequence_capacity=sequence_capacity,
    )


def _bind_fp16_state(
    state_pool: torch.Tensor,
    elapsed_state_pool: torch.Tensor,
    *,
    sequence_capacity: int,
) -> object:
    prepare = wkv7_module.prepare_tmix_wkv7_recurrent_fp16_state
    state = prepare(
        state_pool.shape[0],
        state_pool.shape[1] * state_pool.shape[2],
        sequence_capacity=sequence_capacity,
        head_size=state_pool.shape[2],
        device=state_pool.device,
    )
    state._state_pool = state_pool
    state._elapsed_state_pool = elapsed_state_pool
    return state


def _bind_fp32io16_state(
    state_pool: torch.Tensor,
    *,
    sequence_capacity: int,
) -> object:
    prepare = wkv7_module.prepare_tmix_wkv7_recurrent_fp32io16_state
    state = prepare(
        state_pool.shape[0],
        state_pool.shape[1] * state_pool.shape[2],
        sequence_capacity=sequence_capacity,
        head_size=state_pool.shape[2],
        device=state_pool.device,
    )
    state._state_pool = state_pool
    return state


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.float() - expected.float()
    baseline = expected.float().square().mean().sqrt().clamp_min(1.0e-6)
    return float((difference.square().mean().sqrt() / baseline).item())


def _make_case(
    *,
    dtype: torch.dtype,
    head_size: int,
    lengths: tuple[int, ...] = (1, 15, 16, 17, 65),
    with_decay_bias: bool = False,
    num_heads: int = 2,
    state_pool_slots: int | None = None,
) -> dict[str, torch.Tensor | None]:
    torch.manual_seed(20260804 + head_size + int(with_decay_bias))
    device = torch.device("cuda")
    total_tokens = sum(lengths)
    if state_pool_slots is None:
        state_pool_slots = max(9, len(lengths))
    if state_pool_slots < len(lengths):
        raise ValueError("state_pool_slots must cover every sequence")
    shape = (total_tokens, num_heads, head_size)

    def random(scale: float) -> torch.Tensor:
        return torch.randn(shape, device=device, dtype=torch.float32).mul(scale).to(
            dtype
        )

    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, device=device, dtype=torch.int32)
    if len(lengths) <= 5:
        state_indices = torch.tensor(
            [5, 0, 7, 2, 4], device=device, dtype=torch.int32
        )[: len(lengths)]
    else:
        state_indices = torch.arange(
            len(lengths), device=device, dtype=torch.int32
        )
    state_pool = torch.randn(
        (state_pool_slots, num_heads, head_size, head_size),
        device=device,
        dtype=torch.float32,
    ).mul_(0.03)
    elapsed_state_pool = torch.arange(
        state_pool_slots, device=device, dtype=torch.int32
    ).mul_(11)
    decay_bias = None
    if with_decay_bias:
        decay_bias = torch.randn(
            (num_heads, head_size), device=device, dtype=dtype
        ).mul_(0.2)
    return {
        "r": random(0.12),
        "decay_logits": random(3.5),
        "k": random(0.06),
        "v": random(0.06),
        "a": random(0.06),
        "b": random(0.06),
        "cu_seqlens": cu_seqlens,
        "state_indices": state_indices,
        "state_pool": state_pool,
        "elapsed_state_pool": elapsed_state_pool,
        "decay_bias": decay_bias,
    }


def _args(case: dict[str, torch.Tensor | None]) -> tuple[torch.Tensor, ...]:
    values = tuple(case[name] for name in ("r", "decay_logits", "k", "v", "a", "b"))
    assert all(isinstance(value, torch.Tensor) for value in values)
    return values  # type: ignore[return-value]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda_graph
def test_fp32_wkv_zero_active_precapture_warmup_then_graph_replay() -> None:
    _require_cuda_extension()
    torch.manual_seed(20260809)
    device = torch.device("cuda")
    token_capacity, sequence_capacity, slots, heads, head_size = 4, 4, 2, 1, 64
    shape = (token_capacity, heads, head_size)
    packed = tuple(
        (torch.randn(shape, device=device) * scale).half()
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda_graph
def test_metadata_ticket_predecessor_matches_cpu_reference_and_live_updates() -> None:
    _require_cuda_extension()
    device = torch.device("cuda")
    inactive = torch.iinfo(torch.int32).min

    static_cu = torch.tensor([0, 1, 5, 7], device=device, dtype=torch.int32)
    static_slots = torch.tensor([8, 2, 6], device=device, dtype=torch.int32)
    static_ticket = prepare_tmix_wkv7_recurrent_metadata(
        static_cu,
        static_slots,
        total_tokens=7,
        state_pool_size=10,
        max_seqlen=4,
    )
    torch.cuda.synchronize()
    assert static_ticket._active_status().data_ptr() == static_ticket._status().data_ptr()
    assert static_ticket._token_predecessor().cpu().tolist() == [
        -9,
        -3,
        1,
        2,
        3,
        -7,
        5,
    ]

    token_capacity, sequence_capacity, state_pool_size = 8, 4, 6
    cu = torch.tensor([0, -1, -1, -1, -1], device=device, dtype=torch.int32)
    slots = torch.full((sequence_capacity,), 99, device=device, dtype=torch.int32)
    active_tokens = torch.zeros(1, device=device, dtype=torch.int32)
    active_sequences = torch.zeros(1, device=device, dtype=torch.int32)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        live_ticket = prepare_tmix_wkv7_recurrent_metadata(
            cu,
            slots,
            state_pool_size=state_pool_size,
            token_capacity=token_capacity,
            sequence_capacity=sequence_capacity,
            max_seqlen_capacity=4,
            num_active_tokens=active_tokens,
            num_active_sequences=active_sequences,
        )

    graph.replay()
    torch.cuda.synchronize()
    assert live_ticket._active_status().cpu().tolist() == [0, 0, 0]
    assert live_ticket._token_predecessor().cpu().tolist() == [inactive] * 8

    cu.copy_(torch.tensor([0, 1, 4, 6, -1], device=device, dtype=torch.int32))
    slots.copy_(torch.tensor([5, 2, 0, 99], device=device, dtype=torch.int32))
    active_tokens.fill_(6)
    active_sequences.fill_(3)
    graph.replay()
    torch.cuda.synchronize()
    assert live_ticket._active_status().cpu().tolist() == [0, 6, 3]
    assert live_ticket._token_predecessor().cpu().tolist() == [
        -6,
        -3,
        1,
        2,
        -1,
        4,
        inactive,
        inactive,
    ]

    slots.copy_(torch.tensor([2, 2, 0, 99], device=device, dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()
    assert live_ticket._active_status()[0].item() != 0
    assert live_ticket._token_predecessor().cpu().tolist() == [inactive] * 8


def test_public_signature_is_raw_only_and_old_symbols_are_absent() -> None:
    signature = inspect.signature(infer_tmix_wkv7_recurrent_fp32io16_forward_varlen)
    assert "decay_logits" in signature.parameters
    assert "state" in signature.parameters
    assert "state_pool" not in signature.parameters
    assert "deltalog_phase_pool" not in signature.parameters
    assert "deltalog_pool" not in signature.parameters
    assert "initial_state" not in signature.parameters
    assert "output_final_state" not in signature.parameters
    assert "log_decay" not in signature.parameters
    assert "elapsed_t" not in signature.parameters
    assert not hasattr(flashrwkv2, "rwkv7_recurrent_stateful")
    with pytest.raises(TypeError, match="log_decay"):
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            None,
            None,
            None,
            None,
            None,
            None,
            state=None,
            cu_seqlens=None,
            log_decay=None,
        )
    caller_backed_signature = inspect.signature(
        prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor
    )
    assert tuple(caller_backed_signature.parameters) == ("state",)
    if flashrwkv2._C is not None:
        assert hasattr(
            flashrwkv2._C, "tmix_wkv7_recurrent_fp32_from_decay_logits"
        )


@pytest.mark.parametrize(
    ("dtype", "error_type"),
    [
        (torch.float32, TypeError),
        (torch.int32, TypeError),
    ],
)
def test_public_rejects_unsupported_token_dtype(
    dtype: torch.dtype,
    error_type: type[Exception],
) -> None:
    _require_cuda_extension()
    case = _make_case(dtype=torch.float16, head_size=64)
    bad = case["r"].float() if dtype == torch.float32 else case["r"].to(dtype)
    with pytest.raises((error_type, RuntimeError)):
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            bad,
            case["decay_logits"],
            case["k"],
            case["v"],
            case["a"],
            case["b"],
            state=_prepare_fp32io16_state(
                case["state_pool"], case["state_indices"]
            ),
            cu_seqlens=case["cu_seqlens"],
            state_indices=case["state_indices"],
        )


def test_public_rejects_structural_arguments_without_launching() -> None:
    _require_cuda_extension()
    case = _make_case(dtype=torch.float16, head_size=64)
    args = _args(case)
    state_pool = case["state_pool"]
    assert isinstance(state_pool, torch.Tensor)
    bad_shape = torch.zeros(
        (args[0].shape[0], args[0].shape[1], args[0].shape[2] - 1),
        device=args[0].device,
        dtype=args[0].dtype,
    )
    bad_calls = [
        (args[0].transpose(0, 1), ValueError, "contiguous"),
        (bad_shape, RuntimeError, "shape"),
        (args[0].float(), RuntimeError, "float16 or bfloat16"),
    ]
    for bad_r, error_type, message in bad_calls:
        with pytest.raises(error_type, match=message):
            infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
                bad_r,
                *args[1:],
                state=_prepare_fp32io16_state(
                    state_pool, case["state_indices"]
                ),
                cu_seqlens=case["cu_seqlens"],
                state_indices=case["state_indices"],
            )

    with pytest.raises(TypeError, match="r must be a torch.Tensor"):
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            object(),
            *args[1:],
            state=_prepare_fp32io16_state(state_pool, case["state_indices"]),
            cu_seqlens=case["cu_seqlens"],
            state_indices=case["state_indices"],
        )
    with pytest.raises(ValueError, match=r"packed shape \[total_tokens,H,D\]"):
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            args[0].reshape(-1, args[0].shape[-1]),
            *args[1:],
            state=_prepare_fp32io16_state(state_pool, case["state_indices"]),
            cu_seqlens=case["cu_seqlens"],
            state_indices=case["state_indices"],
        )
    with pytest.raises(RuntimeError, match="dtype mismatch"):
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            args[0],
            args[1].to(torch.bfloat16),
            *args[2:],
            state=_prepare_fp32io16_state(state_pool, case["state_indices"]),
            cu_seqlens=case["cu_seqlens"],
            state_indices=case["state_indices"],
        )
    with pytest.raises(ValueError, match=r"packed shape \[total_tokens,H,D\]"):
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            args[0].unsqueeze(0),
            args[1].unsqueeze(0),
            args[2].unsqueeze(0),
            args[3].unsqueeze(0),
            args[4].unsqueeze(0),
            args[5].unsqueeze(0),
            state=_prepare_fp32io16_state(state_pool, case["state_indices"]),
            cu_seqlens=case["cu_seqlens"],
            state_indices=case["state_indices"],
        )

    with pytest.raises(ValueError, match="state_pool_size"):
        prepare_tmix_wkv7_recurrent_fp32io16_state(
            0,
            128,
            sequence_capacity=case["state_indices"].numel(),
            device=state_pool.device,
        )
    with pytest.raises(ValueError, match="divisible"):
        prepare_tmix_wkv7_recurrent_fp32io16_state(
            state_pool.shape[0],
            129,
            sequence_capacity=case["state_indices"].numel(),
            device=state_pool.device,
        )
    with pytest.raises(ValueError, match="CUDA"):
        prepare_tmix_wkv7_recurrent_fp32io16_state(
            state_pool.shape[0],
            128,
            sequence_capacity=case["state_indices"].numel(),
            device="cpu",
        )
    with pytest.raises(TypeError, match="dtype float32"):
        prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(
            state_pool.half()
        )
    with pytest.raises(ValueError, match="contiguous CUDA"):
        prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(
            state_pool.transpose(-1, -2)
        )
    unsupported = _make_case(dtype=torch.float16, head_size=32, lengths=(4,))
    with pytest.raises(RuntimeError, match="64, 128, or 256"):
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            *_args(unsupported),
            state=_prepare_fp32io16_state(
                unsupported["state_pool"], unsupported["state_indices"]
            ),
            cu_seqlens=unsupported["cu_seqlens"],
            state_indices=unsupported["state_indices"],
        )


@pytest.mark.parametrize(
    "metadata_kind",
    [
        "wrong_rank",
        "wrong_length",
        "wrong_dtype",
        "wrong_start",
        "wrong_end",
        "non_increasing",
        "empty_sequence",
        "wrong_slot_length",
        "duplicate_slot",
        "negative_slot",
        "out_of_range_slot",
    ],
)
def test_packed_public_rejects_invalid_metadata_and_preserves_state(
    metadata_kind: str,
) -> None:
    _require_cuda_extension()
    case = _make_case(dtype=torch.bfloat16, head_size=64)
    args = _args(case)
    cu_seqlens = case["cu_seqlens"]
    state_indices = case["state_indices"]
    state_pool = case["state_pool"]
    assert isinstance(cu_seqlens, torch.Tensor)
    assert isinstance(state_indices, torch.Tensor)
    assert isinstance(state_pool, torch.Tensor)
    if metadata_kind == "wrong_rank":
        bad_cu = cu_seqlens.reshape(1, -1)
        bad_slots = state_indices
    elif metadata_kind == "wrong_length":
        bad_cu = cu_seqlens[:-1]
        bad_slots = state_indices
    elif metadata_kind == "wrong_dtype":
        bad_cu = cu_seqlens.to(torch.int64)
        bad_slots = state_indices
    elif metadata_kind == "wrong_start":
        bad_cu = cu_seqlens + 1
        bad_slots = state_indices
    elif metadata_kind == "wrong_end":
        bad_cu = cu_seqlens.clone()
        bad_cu[-1] -= 1
        bad_slots = state_indices
    elif metadata_kind == "non_increasing":
        bad_cu = cu_seqlens.clone()
        bad_cu[2] = bad_cu[1] - 1
        bad_slots = state_indices
    elif metadata_kind == "empty_sequence":
        bad_cu = cu_seqlens.clone()
        bad_cu[2] = bad_cu[1]
        bad_slots = state_indices
    elif metadata_kind == "wrong_slot_length":
        bad_cu = cu_seqlens
        bad_slots = state_indices[:-1]
    elif metadata_kind == "duplicate_slot":
        bad_cu = cu_seqlens
        bad_slots = state_indices.clone()
        bad_slots[1] = bad_slots[0]
    elif metadata_kind == "negative_slot":
        bad_cu = cu_seqlens
        bad_slots = state_indices.clone()
        bad_slots[0] = -1
    else:
        bad_cu = cu_seqlens
        bad_slots = state_indices.clone()
        bad_slots[0] = state_pool.shape[0]

    before = state_pool.clone()
    state_handle = prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(
        state_pool
    )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            *args,
            state=state_handle,
            cu_seqlens=bad_cu,
            state_indices=bad_slots,
        )
    assert torch.equal(state_pool, before)


def test_public_rejects_devices_bias_scale_and_state_indices_contracts() -> None:
    _require_cuda_extension()
    case = _make_case(dtype=torch.float16, head_size=64)
    args = _args(case)
    state_pool = case["state_pool"]
    cu_seqlens = case["cu_seqlens"]
    state_indices = case["state_indices"]
    assert isinstance(state_pool, torch.Tensor)
    assert isinstance(cu_seqlens, torch.Tensor)
    assert isinstance(state_indices, torch.Tensor)
    state_handle = prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(
        state_pool
    )

    with pytest.raises((ValueError, RuntimeError), match="CUDA"):
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            *(tensor.cpu() for tensor in args),
            state=state_handle,
            cu_seqlens=cu_seqlens.cpu(),
            state_indices=state_indices.cpu(),
        )
    if torch.cuda.device_count() >= 2:
        with pytest.raises(RuntimeError, match="same device"):
            infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
                *args,
                state=state_handle,
                cu_seqlens=cu_seqlens,
                state_indices=state_indices,
                decay_bias=torch.zeros(
                    2, 64, device="cuda:1", dtype=args[0].dtype
                ),
            )

    with pytest.raises((ValueError, RuntimeError), match="decay_bias"):
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            *args,
            state=state_handle,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            decay_bias=torch.zeros(2, 63, device=state_pool.device, dtype=args[0].dtype),
        )
    with pytest.raises((TypeError, RuntimeError), match="decay_bias"):
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            *args,
            state=state_handle,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            decay_bias=torch.zeros(2, 64, device=state_pool.device, dtype=torch.float32),
        )
    with pytest.raises((ValueError, RuntimeError), match="contiguous"):
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            *args,
            state=state_handle,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            decay_bias=torch.zeros(
                64, 2, device=state_pool.device, dtype=args[0].dtype
            ).transpose(0, 1),
        )
    with pytest.raises(TypeError, match="cu_seqlens must be a torch.Tensor"):
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            *args,
            state=state_handle,
            cu_seqlens=[0, args[0].shape[0]],
            state_indices=state_indices,
        )
    with pytest.raises(TypeError, match="state_indices must be a torch.Tensor"):
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            *args,
            state=state_handle,
            cu_seqlens=cu_seqlens,
            state_indices=[0] * state_indices.numel(),
        )
    for bad_scale in (float("nan"), float("inf")):
        with pytest.raises(RuntimeError, match="finite"):
            infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
                *args,
                state=state_handle,
                cu_seqlens=cu_seqlens,
                state_indices=state_indices,
                scale=bad_scale,
            )
    with pytest.raises(TypeError, match="state"):
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            *args,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
        )


def test_ticket_rejects_identity_version_and_stream_mismatches() -> None:
    _require_cuda_extension()
    case = _make_case(dtype=torch.bfloat16, head_size=64, lengths=(4, 5))
    args = _args(case)
    cu_seqlens = case["cu_seqlens"]
    state_indices = case["state_indices"]
    state_pool = case["state_pool"]
    assert isinstance(cu_seqlens, torch.Tensor)
    assert isinstance(state_indices, torch.Tensor)
    assert isinstance(state_pool, torch.Tensor)
    ticket = prepare_tmix_wkv7_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=args[0].shape[0],
        state_pool_size=state_pool.shape[0],
    )
    with pytest.raises(RuntimeError, match="identity"):
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            *args,
            state=_prepare_fp32io16_state(state_pool.clone(), state_indices),
            cu_seqlens=cu_seqlens.clone(),
            state_indices=state_indices,
            validated_metadata=ticket,
        )

    short_args = tuple(tensor[:-1].contiguous() for tensor in args)
    with pytest.raises(RuntimeError, match="total_tokens"):
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            *short_args,
            state=_prepare_fp32io16_state(state_pool.clone(), state_indices),
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            validated_metadata=ticket,
        )
    with pytest.raises(RuntimeError, match="state_pool_size"):
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            *args,
            state=_prepare_fp32io16_state(
                torch.cat(
                    (state_pool, torch.zeros_like(state_pool[:1])), dim=0
                ),
                state_indices,
            ),
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            validated_metadata=ticket,
        )

    before = cu_seqlens.clone()
    cu_seqlens.add_(0)
    assert torch.equal(cu_seqlens, before)
    cu_seqlens[1] += 1
    with pytest.raises(RuntimeError, match="version"):
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            *args,
            state=_prepare_fp32io16_state(state_pool.clone(), state_indices),
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            validated_metadata=ticket,
        )

    cu_seqlens[1] -= 1
    ticket_stream = prepare_tmix_wkv7_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=args[0].shape[0],
        state_pool_size=state_pool.shape[0],
    )
    other_stream = torch.cuda.Stream(device=cu_seqlens.device)
    with torch.cuda.stream(other_stream), pytest.raises(RuntimeError, match="stream"):
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            *args,
            state=_prepare_fp32io16_state(state_pool.clone(), state_indices),
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            validated_metadata=ticket_stream,
        )
    other_stream.synchronize()


def test_packed_low_level_invalid_metadata_fails_closed_without_state_write() -> None:
    _require_cuda_extension()
    case = _make_case(dtype=torch.float16, head_size=64, lengths=(4,))
    args = _args(case)
    state_pool = case["state_pool"]
    assert isinstance(state_pool, torch.Tensor)
    flat = tuple(tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1]) for tensor in args)
    bad_cu = torch.tensor([1, 4], device=state_pool.device, dtype=torch.int32)
    slots = torch.tensor([0], device=state_pool.device, dtype=torch.int32)
    output = torch.zeros_like(flat[3])
    before = state_pool.clone()
    flashrwkv2._C.tmix_wkv7_recurrent_fp32_from_decay_logits(
        bad_cu,
        slots,
        state_pool,
        *flat,
        output,
        1.0,
        None,
        None,
    )
    torch.cuda.synchronize()
    assert torch.isnan(output).all()
    assert torch.equal(state_pool, before)


@pytest.mark.parametrize(
    "dtype,head_size,with_decay_bias",
    [
        pytest.param(
            torch.float16,
            64,
            False,
            marks=(pytest.mark.memcheck, pytest.mark.racecheck),
        ),
        *((dtype, head_size, with_bias)
          for dtype in (torch.float16, torch.bfloat16)
          for head_size in (64, 128, 256)
          for with_bias in (False, True)
          if (dtype, head_size, with_bias) != (torch.float16, 64, False)),
    ],
)
def test_packed_fp32_state_precision_and_state_safety(
    dtype: torch.dtype,
    head_size: int,
    with_decay_bias: bool,
) -> None:
    _require_cuda_extension()
    case = _make_case(
        dtype=dtype,
        head_size=head_size,
        with_decay_bias=with_decay_bias,
    )
    args = _args(case)
    cu_seqlens = case["cu_seqlens"]
    state_indices = case["state_indices"]
    state_pool = case["state_pool"]
    decay_bias = case["decay_bias"]
    assert isinstance(cu_seqlens, torch.Tensor)
    assert isinstance(state_indices, torch.Tensor)
    assert isinstance(state_pool, torch.Tensor)
    expected_output, expected_state = rwkv7_decay_logits_reference(
        *args,
        state_pool=state_pool,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        decay_bias=decay_bias,
    )

    before = state_pool.clone()
    state_handle = prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(
        state_pool
    )
    assert state_handle._state_pool is state_pool
    assert state_handle._sequence_capacity is None
    assert state_handle._merge_interval == 0
    assert state_handle._deltalog_phase_pool is None
    assert state_handle._deltalog_pool is None
    observed_output = infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
        *args,
        state=state_handle,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        decay_bias=decay_bias,
    )
    torch.cuda.synchronize()
    assert observed_output.dtype == dtype
    assert state_pool.dtype == torch.float32
    assert torch.isfinite(observed_output).all()
    assert torch.isfinite(state_pool.index_select(0, state_indices.long())).all()

    active = state_indices.long()
    observed_active = state_pool.index_select(0, active)
    expected_active = expected_state.index_select(0, active)
    output_rmse = _relative_rmse(observed_output, expected_output)
    state_rmse = _relative_rmse(observed_active, expected_active)
    assert output_rmse <= TOLERANCES["output_relative_rmse"]
    assert state_rmse <= TOLERANCES["state_relative_rmse"]
    untouched = [index for index in range(state_pool.shape[0]) if index not in set(state_indices.cpu().tolist())]
    assert torch.equal(
        state_pool.index_select(
            0, torch.tensor(untouched, device=state_pool.device, dtype=torch.long)
        ),
        before.index_select(
            0, torch.tensor(untouched, device=state_pool.device, dtype=torch.long)
        ),
    )
    assert not torch.equal(observed_active, before.index_select(0, active))

    second_state = before.clone()
    second_handle = prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(
        second_state
    )
    second_output = infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
        *args,
        state=second_handle,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        decay_bias=decay_bias,
    )
    torch.cuda.synchronize()
    assert torch.equal(observed_output, second_output)
    assert torch.equal(state_pool, second_state)


def test_public_signatures_use_only_prepared_state_handles() -> None:
    for infer in (
        infer_tmix_wkv7_recurrent_fp16_forward_varlen,
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen,
    ):
        signature = inspect.signature(infer)
        assert "decay_logits" in signature.parameters
        assert "state" in signature.parameters
        assert "max_seqlen" in signature.parameters
        assert "state_pool" not in signature.parameters
        assert "elapsed_state_pool" not in signature.parameters
        assert "deltalog_phase_pool" not in signature.parameters
        assert "deltalog_pool" not in signature.parameters
    prepare_signature = inspect.signature(prepare_tmix_wkv7_recurrent_fp16_state)
    assert tuple(prepare_signature.parameters) == (
        "state_pool_size",
        "channels",
        "sequence_capacity",
        "head_size",
        "device",
    )
    fp32_prepare_signature = inspect.signature(
        prepare_tmix_wkv7_recurrent_fp32io16_state
    )
    assert tuple(fp32_prepare_signature.parameters) == (
        "state_pool_size",
        "channels",
        "sequence_capacity",
        "head_size",
        "device",
    )
    assert not hasattr(
        wkv7_module, "get_tmix_wkv7_recurrent_state_memory_layout"
    )
    assert not hasattr(
        flashrwkv2, "get_tmix_wkv7_recurrent_state_memory_layout"
    )
    assert not hasattr(
        wkv7_module,
        "infer_tmix_wkv7_recurrent_deltalog_fp16_forward_varlen",
    )
    assert not hasattr(
        wkv7_module,
        "infer_tmix_wkv7_recurrent_deltalog_fp32io16_forward_varlen",
    )
    if flashrwkv2._C is not None:
        assert hasattr(
            flashrwkv2._C, "tmix_wkv7_recurrent_fp16_from_decay_logits"
        )


@pytest.mark.parametrize(
    ("prepare", "state_dtype", "has_elapsed"),
    (
        (prepare_tmix_wkv7_recurrent_fp16_state, torch.float16, True),
        (prepare_tmix_wkv7_recurrent_fp32io16_state, torch.float32, False),
    ),
)
def test_public_state_preparation_owns_complete_zeroed_allocation(
    prepare,
    state_dtype: torch.dtype,
    has_elapsed: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_cuda_extension()
    monkeypatch.setattr(
        wkv7_module,
        "_select_deltalog_merge_interval",
        lambda *_: 2,
    )
    state = prepare(
        9,
        4096,
        sequence_capacity=8,
        head_size=64,
        device="cuda",
    )
    assert state._state_pool.shape == (9, 64, 64, 64)
    assert state._state_pool.dtype == state_dtype
    assert torch.count_nonzero(state._state_pool).item() == 0
    assert (state._elapsed_state_pool is not None) is has_elapsed
    if state._elapsed_state_pool is not None:
        assert state._elapsed_state_pool.shape == (9,)
        assert state._elapsed_state_pool.dtype == torch.int32
        assert torch.count_nonzero(state._elapsed_state_pool).item() == 0
    layout = state.memory_layout
    element_size = state._state_pool.element_size()
    assert layout["base_bytes_per_slot"] == (
        64 * 64 * 64 * element_size + (4 if has_elapsed else 0)
    )
    assert layout["private_bytes_per_slot"] == (
        4 + 5 * 64 * 64 * element_size
    )
    assert layout["fixed_workspace_nbytes"] == (1 + 2 * 8) * 4
    assert layout["total_nbytes"] == (
        9 * layout["bytes_per_slot"] + layout["fixed_workspace_nbytes"]
    )


@pytest.mark.parametrize("head_size", (64, 128, 256))
@pytest.mark.parametrize("lengths", ((1,), (1, 2, 4), (8, 8)))
def test_packed_fp16_state_family_correctness(
    head_size: int,
    lengths: tuple[int, ...],
) -> None:
    _require_fp16_extension()
    case = _make_case(
        dtype=torch.float16,
        head_size=head_size,
        lengths=lengths,
        with_decay_bias=True,
    )
    args = _args(case)
    cu_seqlens = case["cu_seqlens"]
    state_indices = case["state_indices"]
    decay_bias = case["decay_bias"]
    assert isinstance(cu_seqlens, torch.Tensor)
    assert isinstance(state_indices, torch.Tensor)
    assert isinstance(decay_bias, torch.Tensor)
    initial_state = case["state_pool"].half()
    expected_output, expected_state = rwkv7_fp16_reference(
        *args,
        state_pool=initial_state,
        elapsed_state_pool=case["elapsed_state_pool"],
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        decay_bias=decay_bias,
    )
    observed_state = initial_state.clone()
    observed_elapsed = case["elapsed_state_pool"].clone()
    state = _bind_fp16_state(
        observed_state,
        observed_elapsed,
        sequence_capacity=state_indices.numel(),
    )
    observed_output = infer_tmix_wkv7_recurrent_fp16_forward_varlen(
        *args,
        state=state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        decay_bias=decay_bias,
    )
    state.materialize_slots_(state_indices.contiguous())
    torch.cuda.synchronize()
    assert observed_output.dtype == torch.float16
    assert observed_state.dtype == torch.float16
    assert torch.isfinite(observed_output).all()
    assert torch.isfinite(observed_state).all()
    active = state_indices.long()
    output_rmse = _relative_rmse(observed_output, expected_output)
    state_rmse = _relative_rmse(
        observed_state.index_select(0, active), expected_state.index_select(0, active)
    )
    assert output_rmse <= 4.0e-3
    assert state_rmse <= 4.0e-3
    untouched = [
        index
        for index in range(observed_state.shape[0])
        if index not in set(state_indices.cpu().tolist())
    ]
    assert torch.equal(
        observed_state.index_select(
            0, torch.tensor(untouched, device=observed_state.device, dtype=torch.long)
        ),
        initial_state.index_select(
            0, torch.tensor(untouched, device=observed_state.device, dtype=torch.long)
        ),
    )


@pytest.mark.parametrize(
    "lengths",
    (
        (1,),
        (1,) * 17,
        (1,) * 128,
        (2,) * 32,
        (8,),
        (4,) * 16,
    ),
)
def test_fp16_dispatch_covers_albatross_grid_and_family_boundaries(
    lengths: tuple[int, ...],
) -> None:
    """Exercise clone/one/exact/seq-v2 and both packed grid specializations."""

    _require_fp16_extension()
    case = _make_case(
        dtype=torch.float16,
        head_size=64,
        lengths=lengths,
        with_decay_bias=True,
        num_heads=64,
        state_pool_slots=max(9, len(lengths)),
    )
    args = _args(case)
    cu_seqlens = case["cu_seqlens"]
    state_indices = case["state_indices"]
    decay_bias = case["decay_bias"]
    elapsed_state = case["elapsed_state_pool"]
    assert isinstance(cu_seqlens, torch.Tensor)
    assert isinstance(state_indices, torch.Tensor)
    assert isinstance(decay_bias, torch.Tensor)
    assert isinstance(elapsed_state, torch.Tensor)
    initial_state = case["state_pool"].half()
    expected_output, expected_state = rwkv7_fp16_reference(
        *args,
        state_pool=initial_state,
        elapsed_state_pool=elapsed_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        decay_bias=decay_bias,
    )
    observed_state = initial_state.clone()
    observed_elapsed = elapsed_state.clone()
    state = _bind_fp16_state(
        observed_state,
        observed_elapsed,
        sequence_capacity=state_indices.numel(),
    )
    observed_output = infer_tmix_wkv7_recurrent_fp16_forward_varlen(
        *args,
        state=state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        decay_bias=decay_bias,
    )
    state.materialize_slots_(state_indices.contiguous())
    torch.cuda.synchronize()
    assert torch.isfinite(observed_output).all()
    assert torch.isfinite(observed_state).all()
    assert _relative_rmse(observed_output, expected_output) <= 4.0e-3
    active = state_indices.long()
    assert (
        _relative_rmse(
            observed_state.index_select(0, active),
            expected_state.index_select(0, active),
        )
        <= 4.0e-3
    )


def test_fp16_consumes_the_same_metadata_ticket_contract() -> None:
    _require_fp16_extension()
    case = _make_case(dtype=torch.float16, head_size=64, lengths=(2, 3))
    args = _args(case)
    cu_seqlens = case["cu_seqlens"]
    state_indices = case["state_indices"]
    state_pool = case["state_pool"].half()
    assert isinstance(cu_seqlens, torch.Tensor)
    assert isinstance(state_indices, torch.Tensor)
    ticket = prepare_tmix_wkv7_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=args[0].shape[0],
        state_pool_size=state_pool.shape[0],
    )
    expected_state = state_pool.clone()
    observed_state = state_pool.clone()
    expected_elapsed = case["elapsed_state_pool"].clone()
    observed_elapsed = case["elapsed_state_pool"].clone()
    expected_handle = _bind_fp16_state(
        expected_state,
        expected_elapsed,
        sequence_capacity=state_indices.numel(),
    )
    observed_handle = _bind_fp16_state(
        observed_state,
        observed_elapsed,
        sequence_capacity=state_indices.numel(),
    )
    expected_output = infer_tmix_wkv7_recurrent_fp16_forward_varlen(
        *args,
        state=expected_handle,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
    )
    observed_output = infer_tmix_wkv7_recurrent_fp16_forward_varlen(
        *args,
        state=observed_handle,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        validated_metadata=ticket,
    )
    torch.cuda.synchronize()
    assert torch.equal(observed_output, expected_output)
    assert torch.equal(observed_state, expected_state)
    assert torch.equal(observed_elapsed, expected_elapsed)

    cu_seqlens[1] += 1
    invalid_handle = _bind_fp16_state(
        state_pool.clone(),
        case["elapsed_state_pool"].clone(),
        sequence_capacity=state_indices.numel(),
    )
    with pytest.raises(RuntimeError, match="version"):
        infer_tmix_wkv7_recurrent_fp16_forward_varlen(
            *args,
            state=invalid_handle,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            validated_metadata=ticket,
        )
    cu_seqlens[1] -= 1


def test_fp16_public_rejects_non_fp16_state_and_tokens() -> None:
    _require_fp16_extension()
    case = _make_case(dtype=torch.float16, head_size=64, lengths=(2,))
    args = _args(case)
    wrong_state = _bind_fp32io16_state(
        case["state_pool"],
        sequence_capacity=case["state_indices"].numel(),
    )
    with pytest.raises(TypeError, match="recurrent_fp16_state"):
        infer_tmix_wkv7_recurrent_fp16_forward_varlen(
            *args,
            state=wrong_state,
            cu_seqlens=case["cu_seqlens"],
            state_indices=case["state_indices"],
        )
    state = _bind_fp16_state(
        case["state_pool"].half(),
        case["elapsed_state_pool"],
        sequence_capacity=case["state_indices"].numel(),
    )
    with pytest.raises((TypeError, RuntimeError), match="float16"):
        infer_tmix_wkv7_recurrent_fp16_forward_varlen(
            args[0].bfloat16(),
            *args[1:],
            state=state,
            cu_seqlens=case["cu_seqlens"],
            state_indices=case["state_indices"],
        )


@pytest.mark.parametrize(
    ("lengths", "num_heads", "dtype"),
    [
        ((5,) * 10, 4, torch.float16),
        ((6,) * 10, 4, torch.float16),
        ((6,) * 11, 4, torch.float16),
        ((10,) * 12, 4, torch.float16),
        ((10,) * 13, 4, torch.float16),
        ((2, 5, 3, 1, 4, 5, 2, 5), 4, torch.float16),
        ((6,) * 10, 4, torch.bfloat16),
    ],
)
def test_fp32_group4_selector_boundaries_and_fallbacks(
    lengths: tuple[int, ...],
    num_heads: int,
    dtype: torch.dtype,
) -> None:
    """Cover every 3e41bc4 group4 admission edge with packed slot reorder."""

    _require_cuda_extension()
    case = _make_case(
        dtype=dtype,
        head_size=64,
        lengths=lengths,
        num_heads=num_heads,
        state_pool_slots=len(lengths) + 3,
        with_decay_bias=True,
    )
    args = _args(case)
    state_indices = torch.arange(
        len(lengths) + 1,
        0,
        -1,
        device=args[0].device,
        dtype=torch.int32,
    )[: len(lengths)]
    state = case["state_pool"]
    cu_seqlens = case["cu_seqlens"]
    decay_bias = case["decay_bias"]
    assert isinstance(state, torch.Tensor)
    assert isinstance(cu_seqlens, torch.Tensor)
    expected_output, expected_state = rwkv7_decay_logits_reference(
        *args,
        state_pool=state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        decay_bias=decay_bias,
    )
    observed_state = state.clone()
    observed_handle = _prepare_fp32io16_state(observed_state, state_indices)
    observed_output = infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
        *args,
        state=observed_handle,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        decay_bias=decay_bias,
    )
    torch.cuda.synchronize()
    assert _relative_rmse(observed_output, expected_output) <= TOLERANCES[
        "output_relative_rmse"
    ]
    active = state_indices.long()
    assert _relative_rmse(
        observed_state.index_select(0, active),
        expected_state.index_select(0, active),
    ) <= TOLERANCES["state_relative_rmse"]


@pytest.mark.parametrize(
    ("seqlen", "output_index", "output_bits", "state_index", "state_bits"),
    (
        (1, 877, 5899, 153784, -1132557914),
        (5, 7510, -26213, 178, 1011067375),
        (10, 14699, 7268, 77652, 998222305),
        (16, 725, 6368, 35618, 990122828),
    ),
)
def test_fp32io16_matches_ee3308_recurrent_arithmetic(
    seqlen: int,
    output_index: int,
    output_bits: int,
    state_index: int,
    state_bits: int,
) -> None:
    """Lock the ee3308 small/large selector and fast-division arithmetic."""

    _require_cuda_extension()
    batch_size, num_heads, head_size = 1, 64, 64
    channels = num_heads * head_size
    values = torch.arange(
        batch_size * seqlen * channels,
        device="cuda",
        dtype=torch.int32,
    )
    specs = (
        (31, 15, 256.0),
        (67, 33, 8.0),
        (37, 18, 512.0),
        (41, 20, 512.0),
        (43, 21, 512.0),
        (47, 23, 512.0),
    )
    packed = tuple(
        (((values % period) - offset).float() / divisor)
        .half()
        .view(-1, num_heads, head_size)
        .contiguous()
        for period, offset, divisor in specs
    )
    state_values = torch.arange(
        batch_size * num_heads * head_size * head_size,
        device="cuda",
        dtype=torch.int32,
    )
    state = (
        (((state_values % 61) - 30).float() / 1024.0)
        .view(batch_size, num_heads, head_size, head_size)
        .contiguous()
    )
    cu_seqlens = torch.tensor([0, seqlen], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([0], device="cuda", dtype=torch.int32)
    state_handle = _prepare_fp32io16_state(state, state_indices)
    output = infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
        *packed,
        state=state_handle,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        max_seqlen=seqlen,
    )
    torch.cuda.synchronize()

    assert int(output.view(torch.int16).view(-1)[output_index]) == output_bits
    assert int(state.view(torch.int32).view(-1)[state_index]) == state_bits


@pytest.mark.parametrize(
    ("batch_size", "max_seqlen", "num_heads"),
    ((250, 1, 60), (400, 1, 50), (20, 2, 64)),
)
def test_fp16_kv_native_selector_workloads(
    batch_size: int,
    max_seqlen: int,
    num_heads: int,
) -> None:
    """Exercise the exact 15000, 20000, and non-T1 1280 boundaries."""

    _require_fp16_extension()
    lengths = (max_seqlen,) * batch_size
    case = _make_case(
        dtype=torch.float16,
        head_size=64,
        lengths=lengths,
        num_heads=num_heads,
        state_pool_slots=batch_size + 1,
        with_decay_bias=True,
    )
    args = _args(case)
    state = case["state_pool"].half()
    elapsed = case["elapsed_state_pool"]
    cu_seqlens = case["cu_seqlens"]
    state_indices = torch.arange(
        batch_size - 1,
        -1,
        -1,
        device=args[0].device,
        dtype=torch.int32,
    )
    decay_bias = case["decay_bias"]
    assert isinstance(elapsed, torch.Tensor)
    assert isinstance(cu_seqlens, torch.Tensor)
    assert isinstance(state_indices, torch.Tensor)
    expected_output, expected_state = rwkv7_fp16_reference(
        *args,
        state_pool=state,
        elapsed_state_pool=elapsed,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        decay_bias=decay_bias,
    )
    observed_state = state.clone()
    untouched_before = observed_state[-1].clone()
    observed_elapsed = elapsed.clone()
    expected_elapsed = elapsed.clone()
    expected_elapsed[state_indices.long()] += max_seqlen
    state_handle = _bind_fp16_state(
        observed_state,
        observed_elapsed,
        sequence_capacity=state_indices.numel(),
    )
    observed_output = infer_tmix_wkv7_recurrent_fp16_forward_varlen(
        *args,
        state=state_handle,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        decay_bias=decay_bias,
    )
    torch.cuda.synchronize()
    assert _relative_rmse(observed_output, expected_output) <= 4.0e-3
    active = state_indices.long()
    assert _relative_rmse(
        observed_state.index_select(0, active),
        expected_state.index_select(0, active),
    ) <= 4.0e-3
    assert torch.equal(observed_state[-1], untouched_before)
    assert torch.equal(observed_elapsed, expected_elapsed)


def _make_deltalog_cycle_case(
    merge_interval: int,
    *,
    batch_size: int = 3,
    num_heads: int = 2,
    with_decay_bias: bool = True,
) -> tuple[
    tuple[tuple[torch.Tensor, ...], ...],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    case = _make_case(
        dtype=torch.float16,
        head_size=64,
        lengths=(merge_interval,) * batch_size,
        num_heads=num_heads,
        state_pool_slots=batch_size + 3,
        with_decay_bias=with_decay_bias,
    )
    packed = _args(case)
    steps = tuple(
        tuple(
            tensor.reshape(batch_size, merge_interval, num_heads, 64)[
                :, step
            ].contiguous()
            for tensor in packed
        )
        for step in range(merge_interval)
    )
    cu_seqlens = torch.arange(
        batch_size + 1, device=packed[0].device, dtype=torch.int32
    )
    initial_state = case["state_pool"].half()
    elapsed = case["elapsed_state_pool"]
    assert isinstance(elapsed, torch.Tensor)
    phase = torch.zeros_like(elapsed)
    logs = torch.zeros(
        (
            merge_interval - 1,
            5,
            initial_state.shape[0],
            num_heads,
            64,
        ),
        device=initial_state.device,
        dtype=torch.float16,
    )
    return (
        steps,
        cu_seqlens,
        initial_state,
        elapsed,
        phase,
        logs,
        case["decay_bias"],
    )


@pytest.mark.parametrize("merge_interval", (2, 3, 4, 6, 8))
@pytest.mark.parametrize("with_decay_bias", (False, True))
@pytest.mark.parametrize("operator", ("fp16", "fp32io16"))
def test_deltalog_complete_cycles_slot_reorder_and_merge_state(
    merge_interval: int,
    with_decay_bias: bool,
    operator: str,
) -> None:
    if operator == "fp16":
        _require_deltalog_extension()
    else:
        _require_fp32io16_deltalog_extension()
    (
        steps,
        cu_seqlens,
        initial_state,
        initial_elapsed,
        phase,
        logs,
        decay_bias,
    ) = _make_deltalog_cycle_case(
        merge_interval, with_decay_bias=with_decay_bias
    )
    if operator == "fp32io16":
        initial_state = initial_state.float()
        logs = logs.float()
    normal_state = initial_state.clone()
    normal_elapsed = initial_elapsed.clone()
    if operator == "fp16":
        normal_handle = _bind_fp16_state(
            normal_state,
            normal_elapsed,
            sequence_capacity=cu_seqlens.numel() - 1,
        )
    else:
        normal_handle = _bind_fp32io16_state(
            normal_state,
            sequence_capacity=cu_seqlens.numel() - 1,
        )
    deltalog_state = initial_state.clone()
    deltalog_elapsed = initial_elapsed.clone()
    initial_logs = logs.clone()
    slot_orders = (
        torch.tensor([4, 1, 3], device=initial_state.device, dtype=torch.int32),
        torch.tensor([1, 3, 4], device=initial_state.device, dtype=torch.int32),
    )
    for step_index, packed in enumerate(steps):
        slots = slot_orders[step_index & 1]
        normal_infer = (
            infer_tmix_wkv7_recurrent_fp16_forward_varlen
            if operator == "fp16"
            else infer_tmix_wkv7_recurrent_fp32io16_forward_varlen
        )
        normal_output = normal_infer(
            *packed,
            state=normal_handle,
            cu_seqlens=cu_seqlens,
            state_indices=slots,
            decay_bias=decay_bias,
            max_seqlen=1,
        )
        if operator == "fp16":
            deltalog_output = wkv7_module._run_deltalog_fp16(
                *packed,
                state_pool=deltalog_state,
                elapsed_state_pool=deltalog_elapsed,
                deltalog_phase_pool=phase,
                deltalog_pool=logs,
                cu_seqlens=cu_seqlens,
                state_indices=slots,
                decay_bias=decay_bias,
            )
        else:
            deltalog_output = wkv7_module._run_deltalog_fp32io16(
                *packed,
                state_pool=deltalog_state,
                deltalog_phase_pool=phase,
                deltalog_pool=logs,
                cu_seqlens=cu_seqlens,
                state_indices=slots,
                decay_bias=decay_bias,
            )
        torch.cuda.synchronize()
        assert _relative_rmse(deltalog_output, normal_output) <= 4.0e-3
        if operator == "fp16":
            assert torch.equal(deltalog_elapsed, normal_elapsed)
        expected_phase = (step_index + 1) % merge_interval
        assert torch.equal(
            phase.index_select(0, slots.long()),
            torch.full(
                (slots.numel(),),
                expected_phase,
                device=phase.device,
                dtype=torch.int32,
            ),
        )
        if expected_phase != 0:
            assert torch.equal(
                deltalog_state.index_select(0, slots.long()),
                initial_state.index_select(0, slots.long()),
            )

    active = slot_orders[0].long()
    assert _relative_rmse(
        deltalog_state.index_select(0, active),
        normal_state.index_select(0, active),
    ) <= 4.0e-3
    untouched = torch.tensor([0, 2, 5], device=phase.device, dtype=torch.long)
    assert torch.equal(
        deltalog_state.index_select(0, untouched),
        initial_state.index_select(0, untouched),
    )
    assert torch.equal(logs[:, :, untouched], initial_logs[:, :, untouched])


@pytest.mark.memcheck
@pytest.mark.racecheck
@pytest.mark.parametrize("operator", ("fp16", "fp32io16"))
def test_deltalog_append_merge_multiple_staggered_slots(operator: str) -> None:
    """Minimal sanitizer case: one launch has append and merge CTAs together."""

    if operator == "fp16":
        _require_deltalog_extension()
    else:
        _require_fp32io16_deltalog_extension()
    (
        steps,
        _,
        initial_state,
        initial_elapsed,
        phase,
        logs,
        decay_bias,
    ) = _make_deltalog_cycle_case(3)
    if operator == "fp32io16":
        initial_state = initial_state.float()
        logs = logs.float()
    normal_state = initial_state.clone()
    normal_elapsed = initial_elapsed.clone()
    if operator == "fp16":
        normal_handle = _bind_fp16_state(
            normal_state,
            normal_elapsed,
            sequence_capacity=3,
        )
    else:
        normal_handle = _bind_fp32io16_state(
            normal_state,
            sequence_capacity=3,
        )
    deltalog_state = initial_state.clone()
    deltalog_elapsed = initial_elapsed.clone()

    def run_normal(
        packed: tuple[torch.Tensor, ...],
        cu: torch.Tensor,
        selected: torch.Tensor,
    ) -> torch.Tensor:
        infer = (
            infer_tmix_wkv7_recurrent_fp16_forward_varlen
            if operator == "fp16"
            else infer_tmix_wkv7_recurrent_fp32io16_forward_varlen
        )
        return infer(
            *packed,
            state=normal_handle,
            cu_seqlens=cu,
            state_indices=selected,
            decay_bias=decay_bias,
            max_seqlen=1,
        )

    def run_deltalog(
        packed: tuple[torch.Tensor, ...],
        cu: torch.Tensor,
        selected: torch.Tensor,
    ) -> torch.Tensor:
        if operator == "fp16":
            return wkv7_module._run_deltalog_fp16(
                *packed,
                state_pool=deltalog_state,
                elapsed_state_pool=deltalog_elapsed,
                deltalog_phase_pool=phase,
                deltalog_pool=logs,
                cu_seqlens=cu,
                state_indices=selected,
                decay_bias=decay_bias,
            )
        return wkv7_module._run_deltalog_fp32io16(
            *packed,
            state_pool=deltalog_state,
            deltalog_phase_pool=phase,
            deltalog_pool=logs,
            cu_seqlens=cu,
            state_indices=selected,
            decay_bias=decay_bias,
        )

    def run_subset(step: int, slots: tuple[int, ...]) -> None:
        selected = torch.tensor(
            slots, device=initial_state.device, dtype=torch.int32
        )
        cu = torch.arange(
            len(slots) + 1, device=initial_state.device, dtype=torch.int32
        )
        packed = tuple(tensor[: len(slots)].contiguous() for tensor in steps[step])
        run_normal(packed, cu, selected)
        run_deltalog(packed, cu, selected)

    run_subset(0, (1, 4))
    run_subset(1, (1,))
    slots = torch.tensor([4, 3, 1], device=initial_state.device, dtype=torch.int32)
    cu = torch.arange(4, device=initial_state.device, dtype=torch.int32)
    normal_output = run_normal(steps[2], cu, slots)
    deltalog_output = run_deltalog(steps[2], cu, slots)
    torch.cuda.synchronize()
    assert _relative_rmse(deltalog_output, normal_output) <= 4.0e-3
    assert phase[1].item() == 0
    assert phase[4].item() == 2
    assert phase[3].item() == 1
    assert _relative_rmse(deltalog_state[1], normal_state[1]) <= 4.0e-3


@pytest.mark.parametrize(
    "failure",
    ("non_t1", "illegal_phase", "duplicate_slot"),
)
@pytest.mark.parametrize("operator", ("fp16", "fp32io16"))
def test_deltalog_runtime_failures_preserve_entire_state_bundle(
    failure: str,
    operator: str,
) -> None:
    if operator == "fp16":
        _require_deltalog_extension()
    else:
        _require_fp32io16_deltalog_extension()
    (
        steps,
        cu_seqlens,
        initial_state,
        initial_elapsed,
        phase,
        logs,
        decay_bias,
    ) = _make_deltalog_cycle_case(3)
    if operator == "fp32io16":
        initial_state = initial_state.float()
        logs = logs.float()
    packed = steps[0]
    slots = torch.tensor([4, 1, 3], device=initial_state.device, dtype=torch.int32)
    if failure == "non_t1":
        packed = tuple(torch.cat((tensor, tensor[:1]), dim=0) for tensor in packed)
        cu_seqlens = torch.tensor(
            [0, 2, 3, 4], device=initial_state.device, dtype=torch.int32
        )
    elif failure == "illegal_phase":
        phase[1] = 3
    else:
        slots[1] = slots[0]

    state = initial_state.clone()
    elapsed = initial_elapsed.clone()
    bundle = (
        (state, elapsed, phase, logs)
        if operator == "fp16"
        else (state, phase, logs)
    )
    before = tuple(tensor.clone() for tensor in bundle)

    def run() -> torch.Tensor:
        if operator == "fp16":
            return wkv7_module._run_deltalog_fp16(
                *packed,
                state_pool=state,
                elapsed_state_pool=elapsed,
                deltalog_phase_pool=phase,
                deltalog_pool=logs,
                cu_seqlens=cu_seqlens,
                state_indices=slots,
                decay_bias=decay_bias,
            )
        return wkv7_module._run_deltalog_fp32io16(
            *packed,
            state_pool=state,
            deltalog_phase_pool=phase,
            deltalog_pool=logs,
            cu_seqlens=cu_seqlens,
            state_indices=slots,
            decay_bias=decay_bias,
        )

    if failure == "duplicate_slot":
        with pytest.raises(RuntimeError, match="unique"):
            run()
    else:
        output = run()
        torch.cuda.synchronize()
        assert torch.isnan(output).all()
    for observed, expected in zip(bundle, before, strict=True):
        assert torch.equal(observed, expected)


@pytest.mark.parametrize(
    "bad_bundle",
    ("log_shape", "log_dtype", "phase_dtype", "log_device"),
)
@pytest.mark.parametrize("operator", ("fp16", "fp32io16"))
def test_deltalog_rejects_invalid_bundle_without_state_write(
    bad_bundle: str,
    operator: str,
) -> None:
    if operator == "fp16":
        _require_deltalog_extension()
    else:
        _require_fp32io16_deltalog_extension()
    (
        steps,
        cu_seqlens,
        state,
        elapsed,
        phase,
        logs,
        decay_bias,
    ) = _make_deltalog_cycle_case(3)
    if operator == "fp32io16":
        state = state.float()
        logs = logs.float()
    slots = torch.tensor([4, 1, 3], device=state.device, dtype=torch.int32)
    if bad_bundle == "log_shape":
        logs = logs[..., :-1].contiguous()
    elif bad_bundle == "log_dtype":
        logs = logs.float() if operator == "fp16" else logs.half()
    elif bad_bundle == "phase_dtype":
        phase = phase.long()
    else:
        logs = logs.cpu()
    before_state = state.clone()
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        if operator == "fp16":
            wkv7_module._run_deltalog_fp16(
                *steps[0],
                state_pool=state,
                elapsed_state_pool=elapsed,
                deltalog_phase_pool=phase,
                deltalog_pool=logs,
                cu_seqlens=cu_seqlens,
                state_indices=slots,
                decay_bias=decay_bias,
            )
        else:
            wkv7_module._run_deltalog_fp32io16(
                *steps[0],
                state_pool=state,
                deltalog_phase_pool=phase,
                deltalog_pool=logs,
                cu_seqlens=cu_seqlens,
                state_indices=slots,
                decay_bias=decay_bias,
            )
    assert torch.equal(state, before_state)


def test_deltalog_native_launcher_is_private_provider_detail() -> None:
    assert "_run_deltalog_fp16" not in wkv7_module.__all__
    assert "_run_deltalog_fp32io16" not in wkv7_module.__all__
    if flashrwkv2._C is not None:
        assert hasattr(
            flashrwkv2._C,
            "tmix_wkv7_recurrent_deltalog_fp16_from_decay_logits",
        )
        assert hasattr(
            flashrwkv2._C,
            "tmix_wkv7_recurrent_deltalog_fp32io16_from_decay_logits",
        )
        assert hasattr(
            flashrwkv2._C,
            "tmix_wkv7_recurrent_deltalog_fp16_materialize_slots",
        )
        assert hasattr(
            flashrwkv2._C,
            "tmix_wkv7_recurrent_deltalog_fp32io16_materialize_slots",
        )


def test_unified_fp16_exact_policy_and_complete_state_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_deltalog_extension()
    monkeypatch.setattr(
        wkv7_module,
        "_select_deltalog_merge_interval",
        lambda *_: 2,
    )
    (
        steps,
        cu_seqlens,
        initial_state,
        initial_elapsed,
        _,
        _,
        decay_bias,
    ) = _make_deltalog_cycle_case(2, batch_size=8, num_heads=64)
    slots = torch.arange(8, device=initial_state.device, dtype=torch.int32)
    ordinary_state = initial_state.clone()
    ordinary_elapsed = initial_elapsed.clone()
    selected_state = initial_state.clone()
    selected_elapsed = initial_elapsed.clone()
    state = _bind_fp16_state(
        selected_state,
        selected_elapsed,
        sequence_capacity=8,
    )
    assert state._merge_interval == 2
    assert state._deltalog_phase_pool is not None
    assert state._deltalog_pool is not None

    for step, packed in enumerate(steps):
        ordinary_output = wkv7_module._run_fp16(
            *packed,
            state_pool=ordinary_state,
            elapsed_state_pool=ordinary_elapsed,
            cu_seqlens=cu_seqlens,
            state_indices=slots,
            decay_bias=decay_bias,
            max_seqlen=1,
        )
        selected_output = infer_tmix_wkv7_recurrent_fp16_forward_varlen(
            *packed,
            state=state,
            cu_seqlens=cu_seqlens,
            state_indices=slots,
            decay_bias=decay_bias,
            max_seqlen=1,
        )
        torch.cuda.synchronize()
        assert _relative_rmse(selected_output, ordinary_output) <= 4.0e-3
        assert torch.equal(selected_elapsed, ordinary_elapsed)
        if step == 0:
            snapshot = state.clone()
            state.zero_()
            assert all(
                torch.count_nonzero(tensor).item() == 0
                for tensor in state._components()
            )
            state.copy_(snapshot)
            assert all(
                torch.equal(observed, expected)
                for observed, expected in zip(
                    state._components(), snapshot._components(), strict=True
                )
            )

    assert torch.count_nonzero(state._deltalog_phase_pool).item() == 0
    assert _relative_rmse(selected_state, ordinary_state) <= 4.0e-3


def test_unified_fp32io16_exact_policy_and_complete_state_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_fp32io16_deltalog_extension()
    monkeypatch.setattr(
        wkv7_module,
        "_select_deltalog_merge_interval",
        lambda *_: 2,
    )
    (
        steps,
        cu_seqlens,
        initial_state,
        _,
        _,
        _,
        decay_bias,
    ) = _make_deltalog_cycle_case(2, batch_size=8, num_heads=64)
    slots = torch.arange(8, device=initial_state.device, dtype=torch.int32)
    ordinary_state = initial_state.float()
    selected_state = ordinary_state.clone()
    state = _bind_fp32io16_state(
        selected_state,
        sequence_capacity=8,
    )
    assert state._merge_interval == 2
    assert state._deltalog_phase_pool is not None
    assert state._deltalog_pool is not None
    assert state._deltalog_pool.dtype == torch.float32

    for step, packed in enumerate(steps):
        ordinary_output = wkv7_module._run_fp32io16(
            *packed,
            state=ordinary_state,
            cu_seqlens=cu_seqlens,
            state_indices=slots,
            decay_bias=decay_bias,
            max_seqlen=1,
            validated_metadata=None,
            scale=1.0,
        )
        selected_output = infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            *packed,
            state=state,
            cu_seqlens=cu_seqlens,
            state_indices=slots,
            decay_bias=decay_bias,
            max_seqlen=1,
        )
        torch.cuda.synchronize()
        assert _relative_rmse(selected_output, ordinary_output) <= 4.0e-3
        if step == 0:
            snapshot = state.clone()
            state.zero_()
            assert all(
                torch.count_nonzero(tensor).item() == 0
                for tensor in state._components()
            )
            state.copy_(snapshot)
            assert all(
                torch.equal(observed, expected)
                for observed, expected in zip(
                    state._components(), snapshot._components(), strict=True
                )
            )

    assert torch.count_nonzero(state._deltalog_phase_pool).item() == 0
    assert _relative_rmse(selected_state, ordinary_state) <= 4.0e-3


@pytest.mark.parametrize("operator", ("fp16", "fp32io16"))
def test_unified_state_slot_checkpoint_cow_reset_and_memory(
    operator: str,
) -> None:
    if operator == "fp16":
        _require_deltalog_extension()
    else:
        _require_fp32io16_deltalog_extension()
    (
        steps,
        cu_seqlens,
        initial_state,
        initial_elapsed,
        _,
        _,
        decay_bias,
    ) = _make_deltalog_cycle_case(2, batch_size=8, num_heads=64)
    slots = torch.arange(8, device=initial_state.device, dtype=torch.int32)
    state_pool = (
        initial_state.clone()
        if operator == "fp16"
        else initial_state.float()
    )
    if operator == "fp16":
        elapsed = initial_elapsed.clone()
        state = _bind_fp16_state(
            state_pool,
            elapsed,
            sequence_capacity=8,
        )
        infer_tmix_wkv7_recurrent_fp16_forward_varlen(
            *steps[0],
            state=state,
            cu_seqlens=cu_seqlens,
            state_indices=slots,
            decay_bias=decay_bias,
            max_seqlen=1,
        )
    else:
        state = _bind_fp32io16_state(
            state_pool,
            sequence_capacity=8,
        )
        infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            *steps[0],
            state=state,
            cu_seqlens=cu_seqlens,
            state_indices=slots,
            decay_bias=decay_bias,
            max_seqlen=1,
        )
    torch.cuda.synchronize()

    checkpoint_slots = torch.tensor(
        [5, 2, 7], device=state_pool.device, dtype=torch.int32
    )
    checkpoint = state.clone_slots(checkpoint_slots)
    for original, compact in zip(
        state._components(), checkpoint._components(), strict=True
    ):
        dimension = 2 if original.ndim == 5 else 0
        assert torch.equal(
            compact,
            original.index_select(dimension, checkpoint_slots.long()),
        )

    destination_slots = torch.tensor(
        [6, 1, 4], device=state_pool.device, dtype=torch.int32
    )
    compact_slots = torch.arange(
        3, device=state_pool.device, dtype=torch.int32
    )
    state.copy_slots_(checkpoint, compact_slots, destination_slots)
    for observed, expected in zip(
        state._components(), checkpoint._components(), strict=True
    ):
        dimension = 2 if observed.ndim == 5 else 0
        assert torch.equal(
            observed.index_select(dimension, destination_slots.long()),
            expected,
        )

    before_cow = tuple(tensor.clone() for tensor in state._components())
    cow_source = torch.tensor(
        [1, 4], device=state_pool.device, dtype=torch.int32
    )
    cow_destination = torch.tensor(
        [4, 0], device=state_pool.device, dtype=torch.int32
    )
    state.copy_slots_(state, cow_source, cow_destination)
    for before, observed in zip(
        before_cow, state._components(), strict=True
    ):
        dimension = 2 if observed.ndim == 5 else 0
        assert torch.equal(
            observed.index_select(dimension, cow_destination.long()),
            before.index_select(dimension, cow_source.long()),
        )

    reset_slots = torch.tensor(
        [2, 7], device=state_pool.device, dtype=torch.int32
    )
    untouched_before = tuple(tensor.clone() for tensor in state._components())
    state.reset_slots_(reset_slots)
    untouched = torch.tensor(
        [0, 1, 3, 4, 5, 6], device=state_pool.device, dtype=torch.long
    )
    for before, observed in zip(
        untouched_before, state._components(), strict=True
    ):
        dimension = 2 if observed.ndim == 5 else 0
        assert torch.count_nonzero(
            observed.index_select(dimension, reset_slots.long())
        ).item() == 0
        assert torch.equal(
            observed.index_select(dimension, untouched),
            before.index_select(dimension, untouched),
        )

    layout = state.memory_layout
    assert layout["total_nbytes"] == (
        layout["bytes_per_slot"] * state_pool.shape[0]
        + layout["fixed_workspace_nbytes"]
    )


@pytest.mark.memcheck
@pytest.mark.racecheck
@pytest.mark.parametrize("operator", ("fp16", "fp32io16"))
def test_unified_state_materialize_and_pending_log_fallback(
    operator: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if operator == "fp16":
        _require_deltalog_extension()
    else:
        _require_fp32io16_deltalog_extension()
    (
        steps,
        cu_seqlens,
        initial_state,
        initial_elapsed,
        _,
        _,
        decay_bias,
    ) = _make_deltalog_cycle_case(2, batch_size=8, num_heads=64)
    slots = torch.arange(8, device=initial_state.device, dtype=torch.int32)
    ordinary_state = (
        initial_state.clone()
        if operator == "fp16"
        else initial_state.float()
    )
    selected_state = ordinary_state.clone()
    ordinary_elapsed = initial_elapsed.clone()
    selected_elapsed = initial_elapsed.clone()
    monkeypatch.setattr(
        wkv7_module,
        "_select_deltalog_merge_interval",
        lambda *_: 0,
    )
    if operator == "fp16":
        ordinary_handle = _bind_fp16_state(
            ordinary_state,
            ordinary_elapsed,
            sequence_capacity=8,
        )
        monkeypatch.setattr(
            wkv7_module,
            "_select_deltalog_merge_interval",
            lambda *_: 2,
        )
        selected_handle = _bind_fp16_state(
            selected_state,
            selected_elapsed,
            sequence_capacity=8,
        )
        infer = infer_tmix_wkv7_recurrent_fp16_forward_varlen
    else:
        ordinary_handle = _bind_fp32io16_state(
            ordinary_state,
            sequence_capacity=8,
        )
        monkeypatch.setattr(
            wkv7_module,
            "_select_deltalog_merge_interval",
            lambda *_: 2,
        )
        selected_handle = _bind_fp32io16_state(
            selected_state,
            sequence_capacity=8,
        )
        infer = infer_tmix_wkv7_recurrent_fp32io16_forward_varlen

    infer(
        *steps[0],
        state=ordinary_handle,
        cu_seqlens=cu_seqlens,
        state_indices=slots,
        decay_bias=decay_bias,
        max_seqlen=1,
    )
    infer(
        *steps[0],
        state=selected_handle,
        cu_seqlens=cu_seqlens,
        state_indices=slots,
        decay_bias=decay_bias,
        max_seqlen=1,
    )
    assert ordinary_handle._deltalog_phase_pool is None
    assert selected_handle._deltalog_phase_pool is not None
    assert selected_handle._deltalog_pool is not None
    state_before_materialize = selected_state.clone()
    phase_before_materialize = selected_handle._deltalog_phase_pool.clone()
    logs_before_materialize = selected_handle._deltalog_pool.clone()
    ordinary_handle.materialize_slots_(slots[:4].contiguous())
    selected_handle.materialize_slots_(slots[:4].contiguous())
    torch.cuda.synchronize()
    assert _relative_rmse(selected_state[:4], ordinary_state[:4]) <= 4.0e-3
    assert torch.count_nonzero(
        selected_handle._deltalog_phase_pool[:4]
    ).item() == 0
    assert torch.equal(
        selected_handle._deltalog_phase_pool[4:8],
        torch.ones_like(selected_handle._deltalog_phase_pool[4:8]),
    )
    assert torch.equal(selected_state[4:], state_before_materialize[4:])
    assert torch.equal(
        selected_handle._deltalog_phase_pool[4:],
        phase_before_materialize[4:],
    )
    assert torch.count_nonzero(
        selected_handle._deltalog_pool[:, :, :4]
    ).item() == 0
    assert torch.equal(
        selected_handle._deltalog_pool[:, :, 4:],
        logs_before_materialize[:, :, 4:],
    )
    assert torch.count_nonzero(
        selected_handle._deltalog_phase_pool[8:]
    ).item() == 0

    fallback_slots = slots[4:].contiguous()
    fallback_cu = torch.arange(
        fallback_slots.numel() + 1,
        device=slots.device,
        dtype=torch.int32,
    )
    fallback_inputs = tuple(tensor[4:].contiguous() for tensor in steps[1])
    expected_output = infer(
        *fallback_inputs,
        state=ordinary_handle,
        cu_seqlens=fallback_cu,
        state_indices=fallback_slots,
        decay_bias=decay_bias,
        max_seqlen=1,
    )
    observed_output = infer(
        *fallback_inputs,
        state=selected_handle,
        cu_seqlens=fallback_cu,
        state_indices=fallback_slots,
        decay_bias=decay_bias,
        max_seqlen=1,
    )
    torch.cuda.synchronize()
    assert _relative_rmse(observed_output, expected_output) <= 4.0e-3
    assert _relative_rmse(selected_state, ordinary_state) <= 4.0e-3
    assert torch.count_nonzero(
        selected_handle._deltalog_phase_pool
    ).item() == 0
    assert torch.count_nonzero(selected_handle._deltalog_pool).item() == 0
    if operator == "fp16":
        assert torch.equal(selected_elapsed, ordinary_elapsed)


def test_state_prepare_uses_mode_specific_profitable_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_deltalog_extension()
    _require_fp32io16_deltalog_extension()
    sequence_capacity, num_heads, head_size = 64, 12, 64
    assert (
        wkv7_module._select_deltalog_merge_interval(
            768, sequence_capacity, head_size, (12, 0), "fp16"
        )
        == 0
    )
    assert (
        wkv7_module._select_deltalog_merge_interval(
            768, sequence_capacity, head_size, (12, 0), "fp32io16"
        )
        == 3
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (8, 9))
    fp16_state_pool = torch.zeros(
        (sequence_capacity, num_heads, head_size, head_size),
        device="cuda",
        dtype=torch.float16,
    )
    elapsed_state_pool = torch.zeros(
        sequence_capacity,
        device="cuda",
        dtype=torch.int32,
    )
    fp16_state = _bind_fp16_state(
        fp16_state_pool,
        elapsed_state_pool,
        sequence_capacity=sequence_capacity,
    )
    fp32io16_state = _bind_fp32io16_state(
        fp16_state_pool.float(),
        sequence_capacity=sequence_capacity,
    )
    assert fp16_state._merge_interval == 0
    assert fp16_state._deltalog_phase_pool is None
    assert fp16_state._deltalog_pool is None
    assert fp32io16_state._merge_interval == 0
    assert fp32io16_state._deltalog_phase_pool is None
    assert fp32io16_state._deltalog_pool is None


@pytest.mark.parametrize("operator", ("fp16", "fp32io16"))
def test_unified_deltalog_policy_falls_back_for_non_t1(
    operator: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if operator == "fp16":
        _require_deltalog_extension()
    else:
        _require_fp32io16_deltalog_extension()
    monkeypatch.setattr(
        wkv7_module,
        "_select_deltalog_merge_interval",
        lambda *_: 2,
    )
    case = _make_case(
        dtype=torch.float16,
        head_size=64,
        lengths=(2,) * 8,
        num_heads=64,
        state_pool_slots=8,
        with_decay_bias=True,
    )
    packed = _args(case)
    cu_seqlens = case["cu_seqlens"]
    state_indices = case["state_indices"]
    decay_bias = case["decay_bias"]
    elapsed = case["elapsed_state_pool"]
    assert isinstance(cu_seqlens, torch.Tensor)
    assert isinstance(state_indices, torch.Tensor)
    assert isinstance(decay_bias, torch.Tensor)
    assert isinstance(elapsed, torch.Tensor)
    initial_state = case["state_pool"]
    assert isinstance(initial_state, torch.Tensor)
    if operator == "fp16":
        initial_state = initial_state.half()
    ordinary_state = initial_state.clone()
    selected_state = initial_state.clone()
    if operator == "fp16":
        ordinary_elapsed = elapsed.clone()
        selected_elapsed = elapsed.clone()
        state = _bind_fp16_state(
            selected_state,
            selected_elapsed,
            sequence_capacity=8,
        )
        ordinary_output = wkv7_module._run_fp16(
            *packed,
            state_pool=ordinary_state,
            elapsed_state_pool=ordinary_elapsed,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            decay_bias=decay_bias,
            max_seqlen=2,
        )
        infer = infer_tmix_wkv7_recurrent_fp16_forward_varlen
    else:
        state = _bind_fp32io16_state(
            selected_state,
            sequence_capacity=8,
        )
        ordinary_output = wkv7_module._run_fp32io16(
            *packed,
            state=ordinary_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            scale=1.0,
            decay_bias=decay_bias,
            max_seqlen=2,
            validated_metadata=None,
        )
        infer = infer_tmix_wkv7_recurrent_fp32io16_forward_varlen
    selected_output = infer(
        *packed,
        state=state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        decay_bias=decay_bias,
        max_seqlen=2,
    )
    torch.cuda.synchronize()
    assert state._merge_interval == 2
    assert torch.count_nonzero(state._deltalog_phase_pool).item() == 0
    assert torch.count_nonzero(state._deltalog_pool).item() == 0
    assert _relative_rmse(selected_output, ordinary_output) <= 4.0e-3
    assert _relative_rmse(selected_state, ordinary_state) <= 4.0e-3


def test_unified_fp32io16_deltalog_policy_falls_back_for_bf16_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_fp32io16_deltalog_extension()
    monkeypatch.setattr(
        wkv7_module,
        "_select_deltalog_merge_interval",
        lambda *_: 2,
    )
    case = _make_case(
        dtype=torch.bfloat16,
        head_size=64,
        lengths=(1,) * 8,
        num_heads=64,
        state_pool_slots=8,
        with_decay_bias=True,
    )
    packed = _args(case)
    state_indices = case["state_indices"]
    cu_seqlens = case["cu_seqlens"]
    decay_bias = case["decay_bias"]
    initial_state = case["state_pool"]
    assert isinstance(state_indices, torch.Tensor)
    assert isinstance(cu_seqlens, torch.Tensor)
    assert isinstance(decay_bias, torch.Tensor)
    assert isinstance(initial_state, torch.Tensor)
    ordinary_state = initial_state.clone()
    selected_state = initial_state.clone()
    state = _bind_fp32io16_state(
        selected_state,
        sequence_capacity=8,
    )
    ordinary_output = wkv7_module._run_fp32io16(
        *packed,
        state=ordinary_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        scale=1.0,
        decay_bias=decay_bias,
        max_seqlen=1,
        validated_metadata=None,
    )
    selected_output = infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
        *packed,
        state=state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        decay_bias=decay_bias,
        max_seqlen=1,
    )
    torch.cuda.synchronize()
    assert state._merge_interval == 2
    assert torch.count_nonzero(state._deltalog_phase_pool).item() == 0
    assert torch.count_nonzero(state._deltalog_pool).item() == 0
    assert torch.equal(selected_output, ordinary_output)
    assert torch.equal(selected_state, ordinary_state)


@pytest.mark.cuda_graph
@pytest.mark.parametrize("operator", ("fp16", "fp32io16"))
def test_unified_live_metadata_zero_active_warmup_and_graph_replay(
    operator: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if operator == "fp16":
        _require_deltalog_extension()
    else:
        _require_fp32io16_deltalog_extension()
    monkeypatch.setattr(
        wkv7_module,
        "_select_deltalog_merge_interval",
        lambda *_: 3,
    )
    device = torch.device("cuda")
    token_capacity, sequence_capacity, slots, heads = 4, 4, 3, 2
    torch.manual_seed(20260825)
    packed = tuple(
        (torch.randn(token_capacity, heads, 64, device=device) * scale).half()
        for scale in (0.12, 3.0, 0.06, 0.06, 0.06, 0.06)
    )
    initial_state_fp32 = (
        torch.randn(slots, heads, 64, 64, device=device) * 0.03
    )
    initial_state = (
        initial_state_fp32.half() if operator == "fp16" else initial_state_fp32
    )
    base_state = initial_state.clone()
    elapsed = torch.arange(slots, device=device, dtype=torch.int32) * 7
    initial_elapsed = elapsed.clone()
    if operator == "fp16":
        state = _bind_fp16_state(
            base_state,
            elapsed,
            sequence_capacity=sequence_capacity,
        )
        infer = infer_tmix_wkv7_recurrent_fp16_forward_varlen
    else:
        state = _bind_fp32io16_state(
            base_state,
            sequence_capacity=sequence_capacity,
        )
        infer = infer_tmix_wkv7_recurrent_fp32io16_forward_varlen
    snapshot = state.clone()
    phase = state._deltalog_phase_pool
    logs = state._deltalog_pool
    assert phase is not None
    assert logs is not None
    cu = torch.tensor([0, -1, -1, -1, -1], device=device, dtype=torch.int32)
    state_indices = torch.full(
        (sequence_capacity,), 99, device=device, dtype=torch.int32
    )
    active_tokens = torch.zeros(1, device=device, dtype=torch.int32)
    active_sequences = torch.zeros(1, device=device, dtype=torch.int32)
    ticket = prepare_tmix_wkv7_recurrent_metadata(
        cu,
        state_indices,
        state_pool_size=slots,
        token_capacity=token_capacity,
        sequence_capacity=sequence_capacity,
        max_seqlen_capacity=1,
        num_active_tokens=active_tokens,
        num_active_sequences=active_sequences,
    )
    infer(
        *packed,
        state=state,
        cu_seqlens=cu,
        state_indices=state_indices,
        validated_metadata=ticket,
    )
    torch.cuda.synchronize()
    assert torch.equal(base_state, initial_state)
    assert torch.equal(elapsed, initial_elapsed)
    assert torch.count_nonzero(phase).item() == 0
    assert torch.count_nonzero(logs).item() == 0

    cu.copy_(torch.tensor([0, 1, 2, -1, -1], device=device, dtype=torch.int32))
    state_indices.copy_(
        torch.tensor([2, 0, 99, 99], device=device, dtype=torch.int32)
    )
    active_tokens.fill_(2)
    active_sequences.fill_(2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_ticket = prepare_tmix_wkv7_recurrent_metadata(
            cu,
            state_indices,
            state_pool_size=slots,
            token_capacity=token_capacity,
            sequence_capacity=sequence_capacity,
            max_seqlen_capacity=1,
            num_active_tokens=active_tokens,
            num_active_sequences=active_sequences,
        )
        graph_output = infer(
            *packed,
            state=state,
            cu_seqlens=cu,
            state_indices=state_indices,
            validated_metadata=graph_ticket,
        )

    state.copy_(snapshot)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(graph_output[:2]).all()
    assert phase.cpu().tolist() == [1, 0, 1]
    expected_elapsed = [1, 7, 15] if operator == "fp16" else [0, 7, 14]
    assert elapsed.cpu().tolist() == expected_elapsed

    # Replay with different active count, slot ownership, and an externally
    # restored per-slot phase.  Capacity tails remain invalid dummy entries.
    cu.copy_(torch.tensor([0, 1, -1, -1, -1], device=device, dtype=torch.int32))
    state_indices.copy_(
        torch.tensor([1, 99, 99, 99], device=device, dtype=torch.int32)
    )
    active_tokens.fill_(1)
    active_sequences.fill_(1)
    phase[1] = 2
    graph.replay()
    torch.cuda.synchronize()
    assert phase.cpu().tolist() == [1, 0, 1]
    expected_elapsed = [1, 8, 15] if operator == "fp16" else [0, 7, 14]
    assert elapsed.cpu().tolist() == expected_elapsed


@pytest.mark.cuda_graph
@pytest.mark.parametrize("operator", ("fp16", "fp32io16"))
def test_pending_deltalog_graph_fallback_materializes_only_active_slots(
    operator: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if operator == "fp16":
        _require_deltalog_extension()
    else:
        _require_fp32io16_deltalog_extension()
    monkeypatch.setattr(
        wkv7_module,
        "_select_deltalog_merge_interval",
        lambda *_: 2,
    )
    device = torch.device("cuda")
    token_capacity, sequence_capacity, slots, heads = 8, 4, 4, 2
    torch.manual_seed(20260828)

    def make_packed(tokens: int) -> tuple[torch.Tensor, ...]:
        return tuple(
            (torch.randn(tokens, heads, 64, device=device) * scale).half()
            for scale in (0.12, 3.0, 0.06, 0.06, 0.06, 0.06)
        )

    append_inputs = make_packed(sequence_capacity)
    fallback_inputs = make_packed(token_capacity)
    base_state_fp32 = torch.randn(
        slots, heads, 64, 64, device=device
    ) * 0.03
    base_state = (
        base_state_fp32.half() if operator == "fp16" else base_state_fp32
    )
    elapsed = torch.arange(slots, device=device, dtype=torch.int32) * 11
    if operator == "fp16":
        state = _bind_fp16_state(
            base_state,
            elapsed,
            sequence_capacity=sequence_capacity,
        )
        infer = infer_tmix_wkv7_recurrent_fp16_forward_varlen
    else:
        state = _bind_fp32io16_state(
            base_state,
            sequence_capacity=sequence_capacity,
        )
        infer = infer_tmix_wkv7_recurrent_fp32io16_forward_varlen
    decode_cu = torch.arange(
        sequence_capacity + 1, device=device, dtype=torch.int32
    )
    decode_slots = torch.arange(
        sequence_capacity, device=device, dtype=torch.int32
    )
    infer(
        *append_inputs,
        state=state,
        cu_seqlens=decode_cu,
        state_indices=decode_slots,
        max_seqlen=1,
    )
    torch.cuda.synchronize()
    assert torch.equal(
        state._deltalog_phase_pool,
        torch.ones_like(state._deltalog_phase_pool),
    )
    expected = state.clone()

    cu = torch.tensor([0, -1, -1, -1, -1], device=device, dtype=torch.int32)
    state_indices = torch.full(
        (sequence_capacity,), 99, device=device, dtype=torch.int32
    )
    active_tokens = torch.zeros(1, device=device, dtype=torch.int32)
    active_sequences = torch.zeros(1, device=device, dtype=torch.int32)
    warmup_ticket = prepare_tmix_wkv7_recurrent_metadata(
        cu,
        state_indices,
        state_pool_size=slots,
        token_capacity=token_capacity,
        sequence_capacity=sequence_capacity,
        max_seqlen_capacity=2,
        num_active_tokens=active_tokens,
        num_active_sequences=active_sequences,
    )
    infer(
        *fallback_inputs,
        state=state,
        cu_seqlens=cu,
        state_indices=state_indices,
        validated_metadata=warmup_ticket,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_ticket = prepare_tmix_wkv7_recurrent_metadata(
            cu,
            state_indices,
            state_pool_size=slots,
            token_capacity=token_capacity,
            sequence_capacity=sequence_capacity,
            max_seqlen_capacity=2,
            num_active_tokens=active_tokens,
            num_active_sequences=active_sequences,
        )
        graph_output = infer(
            *fallback_inputs,
            state=state,
            cu_seqlens=cu,
            state_indices=state_indices,
            validated_metadata=graph_ticket,
        )

    active_slots = torch.tensor([2, 0], device=device, dtype=torch.int32)
    cu.copy_(torch.tensor([0, 2, 4, -1, -1], device=device, dtype=torch.int32))
    state_indices.copy_(
        torch.tensor([2, 0, 99, 99], device=device, dtype=torch.int32)
    )
    active_tokens.fill_(4)
    active_sequences.fill_(2)
    expected_output = infer(
        *(tensor[:4].contiguous() for tensor in fallback_inputs),
        state=expected,
        cu_seqlens=torch.tensor(
            [0, 2, 4], device=device, dtype=torch.int32
        ),
        state_indices=active_slots,
        max_seqlen=2,
    )
    graph.replay()
    torch.cuda.synchronize()

    assert _relative_rmse(graph_output[:4], expected_output) <= 4.0e-3
    assert _relative_rmse(state._state_pool, expected._state_pool) <= 4.0e-3
    for observed, wanted in zip(
        state._components()[1:], expected._components()[1:], strict=True
    ):
        assert torch.equal(observed, wanted)
    assert state._deltalog_phase_pool.cpu().tolist() == [0, 1, 0, 1]
    assert torch.isfinite(graph_output[0]).all()
