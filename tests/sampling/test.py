# SPDX-License-Identifier: MIT

from __future__ import annotations

import re
import shutil
import subprocess

import pytest
import torch

import flashrwkv2
from flashrwkv2 import (
    infer_sampling_six_parameter_forward_varlen,
    infer_sampling_temperature_topk_topp_forward_varlen,
    setup_sampling_states,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.sm120,
]


def _slots(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def test_reproducible_state_pool() -> None:
    first = setup_sampling_states(1234, 4)
    second = setup_sampling_states(1234, 4)
    assert first.dtype == torch.int8 and first.shape[0] == 4 and first.ndim == 2
    assert first.is_cuda and first.is_contiguous()
    assert torch.equal(first, second)


def test_scalar_and_per_request_top1_match_with_indexed_slots() -> None:
    logits = torch.tensor(
        [[0.0, 4.0, 1.0, -2.0, 0.5, 0.0, -1.0, 2.0],
         [3.0, 0.0, 1.0, -2.0, 0.5, 5.0, -1.0, 2.0]],
        dtype=torch.float32,
        device="cuda",
    )
    slot_indices = _slots([3, 1])
    scalar_states = setup_sampling_states(7, 5)
    vector_states = scalar_states.clone()
    scalar = infer_sampling_temperature_topk_topp_forward_varlen(
        logits, scalar_states, slot_indices, top_k=1
    )
    vector = infer_sampling_temperature_topk_topp_forward_varlen(
        logits,
        vector_states,
        slot_indices,
        temperature=torch.ones(2, device="cuda"),
        top_k=torch.ones(2, dtype=torch.int32, device="cuda"),
        top_p=torch.ones(2, device="cuda"),
    )
    expected = logits.argmax(dim=-1).to(torch.int32)
    assert torch.equal(scalar, expected)
    assert torch.equal(vector, expected)
    assert torch.equal(scalar, vector)
    inactive = torch.tensor([0, 2, 4], device="cuda")
    baseline = setup_sampling_states(7, 5)
    assert torch.equal(scalar_states.index_select(0, inactive), baseline.index_select(0, inactive))
    assert torch.equal(vector_states.index_select(0, inactive), baseline.index_select(0, inactive))


def test_six_parameter_penalty_semantics_and_slot_isolation() -> None:
    logits = torch.tensor(
        [[0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0]],
        dtype=torch.float32,
        device="cuda",
    )
    states = setup_sampling_states(19, 4)
    penalties = torch.zeros(4, 8, dtype=torch.float32, device="cuda")
    slots = _slots([3, 1])
    first = infer_sampling_six_parameter_forward_varlen(
        logits,
        penalties,
        states,
        slots,
        presence_penalty=0.2,
        frequency_penalty=0.3,
        penalty_decay=0.5,
        top_k=1,
    )
    assert torch.equal(first, torch.tensor([1, 5], dtype=torch.int32, device="cuda"))
    assert penalties[3, 1].item() == pytest.approx(0.5)
    assert penalties[1, 5].item() == pytest.approx(0.5)
    assert torch.count_nonzero(penalties[0]).item() == 0
    assert torch.count_nonzero(penalties[2]).item() == 0

    second = infer_sampling_six_parameter_forward_varlen(
        logits,
        penalties,
        states,
        slots,
        presence_penalty=torch.tensor([0.2, 0.4], device="cuda"),
        frequency_penalty=torch.tensor([0.3, 0.1], device="cuda"),
        penalty_decay=torch.tensor([0.5, 0.25], device="cuda"),
        temperature=torch.ones(2, device="cuda"),
        top_k=torch.ones(2, dtype=torch.int32, device="cuda"),
        top_p=torch.ones(2, device="cuda"),
    )
    assert torch.equal(second, first)
    assert penalties[3, 1].item() == pytest.approx(0.55)
    assert penalties[1, 5].item() == pytest.approx(0.225)


def test_top_p_zero_and_nonfinite_logits_are_safe_top1() -> None:
    logits = torch.tensor(
        [[float("nan"), float("inf"), float("-inf"), 1.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
        device="cuda",
    )
    output = infer_sampling_temperature_topk_topp_forward_varlen(
        logits, setup_sampling_states(3, 1), _slots([0]), top_p=0.0
    )
    assert output.item() == 1


def test_empirical_distribution_without_scipy() -> None:
    batch_size = 4096
    row = torch.tensor([0.0, 0.5, 1.0, -0.5, -1.0, 0.25, 0.75, -0.25], device="cuda")
    logits = row.expand(batch_size, -1).contiguous()
    samples = infer_sampling_temperature_topk_topp_forward_varlen(
        logits,
        setup_sampling_states(2026, batch_size),
        torch.arange(batch_size, dtype=torch.int32, device="cuda"),
    )
    observed = torch.bincount(samples.to(torch.int64), minlength=row.numel()).float()
    observed /= batch_size
    expected = row.softmax(dim=0)
    assert torch.max(torch.abs(observed - expected)).item() < 0.035


@pytest.mark.cuda_graph
def test_sampling_cuda_graph_dynamic_active_rows_are_side_effect_free() -> None:
    capacity, vocab_size, num_slots = 4, 8, 5
    logits = torch.tensor(
        [
            [0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0],
            [0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    slots = _slots([3, 1, 99, 99])
    active = torch.tensor([2], device="cuda", dtype=torch.int32)
    states = setup_sampling_states(101, num_slots)
    penalties = torch.zeros(
        num_slots, vocab_size, device="cuda", dtype=torch.float32
    )

    # Warm the exact scalar six-parameter family before capture.
    warm_states = setup_sampling_states(101, capacity)
    infer_sampling_six_parameter_forward_varlen(
        logits,
        torch.zeros(capacity, vocab_size, device="cuda"),
        warm_states,
        torch.arange(capacity, device="cuda", dtype=torch.int32),
        presence_penalty=0.2,
        frequency_penalty=0.3,
        penalty_decay=0.5,
        top_k=1,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = infer_sampling_six_parameter_forward_varlen(
            logits,
            penalties,
            states,
            slots,
            presence_penalty=0.2,
            frequency_penalty=0.3,
            penalty_decay=0.5,
            top_k=1,
            sample_capacity=capacity,
            num_active_samples=active,
        )

    initial_states = setup_sampling_states(101, num_slots)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(
        output,
        torch.tensor([1, 5, -1, -1], device="cuda", dtype=torch.int32),
    )
    assert not torch.equal(states[3], initial_states[3])
    assert not torch.equal(states[1], initial_states[1])
    assert torch.equal(states[0], initial_states[0])
    assert torch.equal(states[2], initial_states[2])
    assert torch.equal(states[4], initial_states[4])
    assert penalties[3, 1].item() == pytest.approx(0.5)
    assert penalties[1, 5].item() == pytest.approx(0.5)
    assert torch.count_nonzero(penalties[[0, 2, 4]]).item() == 0

    slots.copy_(_slots([2, 99, 99, 99]))
    active.fill_(1)
    before_states = states.clone()
    before_penalties = penalties.clone()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(
        output,
        torch.tensor([1, -1, -1, -1], device="cuda", dtype=torch.int32),
    )
    assert not torch.equal(states[2], before_states[2])
    assert torch.equal(states[[0, 1, 3, 4]], before_states[[0, 1, 3, 4]])
    assert penalties[2, 1].item() == pytest.approx(0.5)
    assert torch.equal(penalties[[0, 1, 3, 4]], before_penalties[[0, 1, 3, 4]])

    active.zero_()
    slots.fill_(99)
    before_states = states.clone()
    before_penalties = penalties.clone()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(
        output,
        torch.full((capacity,), -1, device="cuda", dtype=torch.int32),
    )
    assert torch.equal(states, before_states)
    assert torch.equal(penalties, before_penalties)

    invalid_cases = (
        ([1, 1, 99, 99], 2),
        ([8, 1, 99, 99], 2),
        ([1, 2, 3, 4], 5),
    )
    for slot_values, active_count in invalid_cases:
        slots.copy_(_slots(slot_values))
        active.fill_(active_count)
        before_states = states.clone()
        before_penalties = penalties.clone()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(
            output,
            torch.full((capacity,), -1, device="cuda", dtype=torch.int32),
        )
        assert torch.equal(states, before_states)
        assert torch.equal(penalties, before_penalties)


def test_rejects_invalid_inputs_before_launch() -> None:
    states = setup_sampling_states(1, 2)
    slots = _slots([0, 1])
    with pytest.raises(ValueError, match="CUDA float32"):
        infer_sampling_temperature_topk_topp_forward_varlen(
            torch.zeros(2, 8), states, slots
        )
    with pytest.raises(ValueError, match="divisible by 4"):
        infer_sampling_temperature_topk_topp_forward_varlen(
            torch.zeros(2, 10, device="cuda"), states, slots
        )
    with pytest.raises(ValueError, match="shape"):
        infer_sampling_temperature_topk_topp_forward_varlen(
            torch.zeros(2, 8, device="cuda"), states, _slots([0])
        )
    with pytest.raises(ValueError, match="dtype"):
        infer_sampling_temperature_topk_topp_forward_varlen(
            torch.zeros(2, 8, device="cuda"),
            states,
            slots,
            top_k=torch.ones(2, dtype=torch.int64, device="cuda"),
        )
    with pytest.raises(ValueError, match="must be on"):
        infer_sampling_temperature_topk_topp_forward_varlen(
            torch.zeros(2, 8, device="cuda"),
            states,
            slots,
            top_p=torch.ones(2),
        )
    with pytest.raises(ValueError, match="shape"):
        infer_sampling_temperature_topk_topp_forward_varlen(
            torch.zeros(2, 8, device="cuda"),
            states,
            slots,
            temperature=torch.ones(3, device="cuda"),
        )
    with pytest.raises(ValueError, match="penalties"):
        infer_sampling_six_parameter_forward_varlen(
            torch.zeros(2, 8, device="cuda"),
            torch.zeros(2, 4, device="cuda"),
            states,
            slots,
        )


@pytest.mark.resource
def test_sm120_sampling_resource_usage() -> None:
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("resource gate requires the SM120 backend")
    tool = shutil.which("cuobjdump")
    if tool is None:
        pytest.fail("cuobjdump is required for the SM120 sampling resource gate")
    assert flashrwkv2._C is not None
    assert flashrwkv2._C.__name__ == "flashrwkv2._C_sm120"
    extension_path = flashrwkv2._C.__file__
    output = subprocess.run(
        (tool, "--dump-resource-usage", extension_path),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    blocks = re.findall(r"Function ([^:]+):\n((?:  .+\n)+)", output)
    hot = {
        name: body
        for name, body in blocks
        if "batch_sampling_" in name and "kernel" in name
    }
    assert len(hot) == 3, sorted(hot)
    for name, body in hot.items():
        fields = {key: int(value) for key, value in re.findall(r"(REG|STACK|LOCAL):(\d+)", body)}
        assert fields["REG"] <= 64, (name, fields)
        assert fields["STACK"] == 0, (name, fields)
        assert fields["LOCAL"] == 0, (name, fields)
