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
    pytest.mark.cuda,
]


def _slots(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def test_reproducible_state_pool() -> None:
    first = setup_sampling_states(1234, 4)
    second = setup_sampling_states(1234, 4)
    assert first.dtype == torch.int8 and first.shape[0] == 4 and first.ndim == 2
    assert first.is_cuda and first.is_contiguous()
    assert torch.equal(first, second)


@pytest.mark.memcheck
@pytest.mark.racecheck
def test_scalar_and_per_request_top1_match_with_indexed_slots() -> None:
    logits = torch.tensor(
        [
            [0.0, 4.0, 1.0, -2.0, 0.5, 0.0, -1.0, 2.0],
            [3.0, 0.0, 1.0, -2.0, 0.5, 5.0, -1.0, 2.0],
        ],
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
    assert torch.equal(
        scalar_states.index_select(0, inactive), baseline.index_select(0, inactive)
    )
    assert torch.equal(
        vector_states.index_select(0, inactive), baseline.index_select(0, inactive)
    )


def test_six_parameter_penalty_semantics_and_slot_isolation() -> None:
    logits = torch.tensor(
        [
            [0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0],
        ],
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


def test_presence_marker_survives_cancelled_penalties() -> None:
    logits = torch.tensor([[0.0, 8.0, 0.0, 0.0]], device="cuda")
    penalties = torch.zeros_like(logits)
    states = setup_sampling_states(29, 1)
    slots = _slots([0])
    first = infer_sampling_six_parameter_forward_varlen(
        logits,
        penalties,
        states,
        slots,
        presence_penalty=0.3,
        frequency_penalty=-0.3,
        top_k=1,
    )
    assert first.item() == 1
    assert penalties[0, 1].item() == 0.0
    assert torch.signbit(penalties[0, 1]).item()
    second = infer_sampling_six_parameter_forward_varlen(
        logits,
        penalties,
        states,
        slots,
        presence_penalty=0.3,
        frequency_penalty=-0.3,
        top_k=1,
    )
    assert second.item() == 1
    assert penalties[0, 1].item() == pytest.approx(-0.3)


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
    penalties = torch.zeros(num_slots, vocab_size, device="cuda", dtype=torch.float32)

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
def test_sampling_resource_usage() -> None:
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("resource gate requires the SM120 backend")
    tool = shutil.which("cuobjdump")
    if tool is None:
        pytest.fail("cuobjdump is required for the SM120 sampling resource gate")
    extension_path = flashrwkv2._extension().__file__
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
        if "sampling_kernel" in name or "sampling_cooperative_kernel" in name
    }
    assert len(hot) == 3, sorted(hot)
    for name, body in hot.items():
        fields = {
            key: int(value)
            for key, value in re.findall(r"(REG|STACK|LOCAL):(\d+)", body)
        }
        assert fields["REG"] <= 64, (name, fields)
        # Cooperative grid barriers use a small call stack, independently of spills.
        stack_limit = 64 if "sampling_cooperative_kernel" in name else 0
        assert fields["STACK"] <= stack_limit, (name, fields)
        assert fields["LOCAL"] == 0, (name, fields)


def _fractional_probabilities(
    logits: torch.Tensor, top_k: int, top_p: float
) -> torch.Tensor:
    probabilities = logits.double().softmax(dim=0)
    filtered = torch.zeros_like(probabilities)
    remaining_k = probabilities.numel() if top_k <= 0 else top_k
    remaining_p = top_p
    for value in probabilities.unique().sort(descending=True).values:
        equal = probabilities == value
        count = int(equal.sum())
        fraction = min(1.0, remaining_k / count, remaining_p / (float(value) * count))
        if fraction <= 0.0:
            break
        filtered[equal] = value * fraction
        remaining_k -= count * fraction
        remaining_p -= float(value) * count * fraction
    return filtered / filtered.sum()


@pytest.mark.parametrize(
    "provider,vocab_size",
    [
        ("scalar", 8),
        ("per_request", 8),
        ("six_parameter", 8),
        ("per_request", 65536),
    ],
)
def test_single_request_launches_only_the_sampling_kernel(
    provider: str, vocab_size: int
) -> None:
    logits = torch.full((1, vocab_size), -80.0, device="cuda")
    logits[0, 1] = 8.0
    states = setup_sampling_states(31, 8192)
    slots = _slots([8191])
    penalties = (
        torch.zeros(8192, vocab_size, device="cuda")
        if provider == "six_parameter"
        else None
    )
    controls = {"top_k": 1}
    if provider == "per_request":
        controls = {
            "temperature": torch.ones(1, device="cuda"),
            "top_k": _slots([1]),
            "top_p": torch.ones(1, device="cuda"),
        }
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        if penalties is None:
            output = infer_sampling_temperature_topk_topp_forward_varlen(
                logits, states, slots, **controls
            )
        else:
            output = infer_sampling_six_parameter_forward_varlen(
                logits, penalties, states, slots, **controls
            )
        torch.cuda.synchronize()
    assert output.item() == 1
    kernels = [
        event.name
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert not any("validate_sampling_metadata_kernel" in name for name in kernels), (
        kernels
    )
    assert (
        sum(
            "sampling_kernel" in name or "sampling_cooperative_kernel" in name
            for name in kernels
        )
        == 1
    ), kernels


@pytest.mark.cuda_graph
@pytest.mark.memcheck
@pytest.mark.racecheck
@pytest.mark.parametrize("provider", ["scalar", "per_request", "six_parameter"])
def test_single_request_dynamic_activity_preserves_rng_and_penalties(
    provider: str,
) -> None:
    logits = torch.tensor([[0.0, 8.0, 0.0, 0.0]], device="cuda")
    states = setup_sampling_states(41, 16)
    initial = states.clone()
    penalties = torch.zeros(16, 4, device="cuda")
    slots, active = _slots([15]), _slots([1])
    controls = {"top_k": 1}
    if provider == "per_request":
        controls = {
            "temperature": torch.ones(1, device="cuda"),
            "top_k": _slots([1]),
            "top_p": torch.ones(1, device="cuda"),
        }

    def call() -> torch.Tensor:
        if provider == "six_parameter":
            return infer_sampling_six_parameter_forward_varlen(
                logits,
                penalties,
                states,
                slots,
                presence_penalty=0.5,
                frequency_penalty=0.25,
                sample_capacity=1,
                num_active_samples=active,
                **controls,
            )
        return infer_sampling_temperature_topk_topp_forward_varlen(
            logits,
            states,
            slots,
            sample_capacity=1,
            num_active_samples=active,
            **controls,
        )

    call()
    states.copy_(initial)
    penalties.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    graph.replay()
    torch.cuda.synchronize()
    assert output.item() == 1
    assert torch.equal(states[:15], initial[:15])
    assert not torch.equal(states[15], initial[15])
    if provider == "six_parameter":
        assert penalties[15, 1].item() == pytest.approx(0.75)
    for count, slot in ((0, -1), (-1, 15), (2, 15), (1, -1), (1, 16)):
        active.fill_(count)
        slots.fill_(slot)
        before_states, before_penalties = states.clone(), penalties.clone()
        graph.replay()
        torch.cuda.synchronize()
        assert output.item() == -1
        assert torch.equal(states, before_states)
        assert torch.equal(penalties, before_penalties)


@pytest.mark.cuda_graph
@pytest.mark.memcheck
@pytest.mark.racecheck
def test_sampling_hash_collisions_and_invalid_batches_preserve_state() -> None:
    capacity, pool_size, vocab_size = 4, 8192, 8
    # These slots collide in the initial bucket of the eight-entry Fibonacci hash table.
    colliding_slots = [
        slot
        for slot in range(pool_size)
        if ((slot * 2654435761) & 0xFFFFFFFF) >> 29 == 0
    ][:capacity]
    slots = _slots(colliding_slots)
    active = _slots([capacity])
    logits = torch.zeros(capacity, vocab_size, device="cuda")
    logits[
        torch.arange(capacity, device="cuda"),
        torch.arange(1, capacity + 1, device="cuda"),
    ] = 8.0
    states = setup_sampling_states(37, pool_size)
    penalties = torch.zeros(pool_size, vocab_size, device="cuda")
    infer_sampling_six_parameter_forward_varlen(
        logits, penalties, states, slots, top_k=1
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = infer_sampling_six_parameter_forward_varlen(
            logits,
            penalties,
            states,
            slots,
            top_k=1,
            frequency_penalty=0.25,
            sample_capacity=capacity,
            num_active_samples=active,
        )
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, _slots([1, 2, 3, 4]))
    assert torch.equal(
        penalties[slots.long(), torch.arange(1, 5, device="cuda")],
        torch.full((capacity,), 0.25, device="cuda"),
    )

    first, second, third, fourth = colliding_slots
    cases = [
        ([first, second, third, first], 4),
        ([first, second, pool_size, fourth], 4),
        ([first, second, -1, fourth], 4),
        (colliding_slots, -1),
        (colliding_slots, capacity + 1),
        ([pool_size] * capacity, 0),
    ]
    for values, count in cases:
        slots.copy_(_slots(values))
        active.fill_(count)
        before_states, before_penalties = states.clone(), penalties.clone()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, _slots([-1] * capacity))
        assert torch.equal(states, before_states)
        assert torch.equal(penalties, before_penalties)

    # An inactive duplicate tail does not invalidate the active prefix.
    slots.copy_(_slots([first, second, first, second]))
    active.fill_(2)
    before = states.clone()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, _slots([1, 2, -1, -1]))
    unchanged = torch.ones(pool_size, dtype=torch.bool, device="cuda")
    unchanged[slots[:2].long()] = False
    assert torch.equal(states[unchanged], before[unchanged])


@pytest.mark.parametrize("provider", ["scalar", "per_request", "six_parameter"])
@pytest.mark.parametrize(
    "values,top_k,top_p",
    [
        ([0.0, 0.5, 1.0, -0.5, -1.0, 0.25, 0.75, -0.25], 3, 0.8),
        ([2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0], 1, 1.0),
        (
            [0.0, -0.28768207, -0.69314718, -1.38629436, -80.0, -80.0, -80.0, -80.0],
            -1,
            0.55,
        ),
    ],
)
def test_fractional_filter_distribution(
    provider: str, values: list[float], top_k: int, top_p: float
) -> None:
    batch_size = 8192
    row = torch.tensor(values, dtype=torch.float32)
    logits = row.cuda().expand(batch_size, -1).contiguous()
    slots = torch.arange(batch_size - 1, -1, -1, dtype=torch.int32, device="cuda")
    states = setup_sampling_states(31415, batch_size)
    controls = {"top_k": top_k, "top_p": top_p}
    if provider == "per_request":
        controls = {
            "temperature": torch.ones(batch_size, device="cuda"),
            "top_k": torch.full((batch_size,), top_k, device="cuda", dtype=torch.int32),
            "top_p": torch.full((batch_size,), top_p, device="cuda"),
        }
    if provider == "six_parameter":
        samples = infer_sampling_six_parameter_forward_varlen(
            logits, torch.zeros_like(logits), states, slots, **controls
        )
    else:
        samples = infer_sampling_temperature_topk_topp_forward_varlen(
            logits, states, slots, **controls
        )
    observed = torch.bincount(samples.long(), minlength=row.numel()).cpu().double()
    expected = _fractional_probabilities(row, top_k, top_p)
    assert observed[expected == 0].sum() == 0
    assert torch.max(torch.abs(observed / batch_size - expected)) < 0.025


@pytest.mark.memcheck
@pytest.mark.racecheck
@pytest.mark.cuda_graph
def test_cooperative_sampling_dynamic_activity_and_slots() -> None:
    logits = torch.full((1, 65536), -80.0, device="cuda")
    logits[0, 44000] = 8.0
    slots = _slots([3])
    active = _slots([1])
    states = setup_sampling_states(2718, 5)
    initial = states.clone()
    top_ks = _slots([1])
    temperatures = torch.ones(1, device="cuda")
    top_ps = torch.ones(1, device="cuda")

    # Warm up the cooperative specialization before stream capture.
    infer_sampling_temperature_topk_topp_forward_varlen(logits, states, slots, top_k=1)
    states.copy_(initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = infer_sampling_temperature_topk_topp_forward_varlen(
            logits,
            states,
            slots,
            temperature=temperatures,
            top_k=top_ks,
            top_p=top_ps,
            sample_capacity=1,
            num_active_samples=active,
        )
    graph.replay()
    torch.cuda.synchronize()
    assert output.item() == 44000
    assert not torch.equal(states[3], initial[3])
    assert torch.equal(states[[0, 1, 2, 4]], initial[[0, 1, 2, 4]])

    active.zero_()
    slots.fill_(99)
    before = states.clone()
    graph.replay()
    torch.cuda.synchronize()
    assert output.item() == -1
    assert torch.equal(states, before)

    # Invalid active metadata also exits before every cooperative grid barrier.
    active.fill_(1)
    graph.replay()
    torch.cuda.synchronize()
    assert output.item() == -1
    assert torch.equal(states, before)

    slots.fill_(1)
    logits[0, 44000] = -80.0
    logits[0, 60001] = 9.0
    graph.replay()
    torch.cuda.synchronize()
    assert output.item() == 60001
    assert not torch.equal(states[1], before[1])
    assert torch.equal(states[[0, 2, 3, 4]], before[[0, 2, 3, 4]])


@pytest.mark.parametrize("case", ["ties", "dense_ties", "fractional"])
@pytest.mark.cuda_graph
def test_cooperative_ties_and_candidate_cache_overflow(case: str) -> None:
    vocab_size, draws_per_replay, replays = 65536, 16, 128
    dense_ties = case == "dense_ties"
    logits = torch.full((1, vocab_size), -80.0, device="cuda")
    # Dense ties exceed the per-thread candidate cache during histogram refinement.
    candidates = (
        torch.arange(vocab_size, device="cuda")
        if dense_ties
        else torch.tensor([17, 2051, 16388, 44000, 65535], device="cuda")
    )
    logits[0, candidates] = 0.0
    top_k, top_p = 1, 1.0
    values = torch.zeros(candidates.numel())
    if case == "fractional":
        values = torch.tensor([0.0, -0.28768207, -0.69314718, -1.38629436, -80.0])
        logits[0, candidates] = values.cuda()
        top_k, top_p = -1, 0.55
    states = setup_sampling_states(1618, 4)
    slots = _slots([2])
    infer_sampling_temperature_topk_topp_forward_varlen(
        logits, states, slots, top_k=top_k, top_p=top_p
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = torch.stack(
            [
                infer_sampling_temperature_topk_topp_forward_varlen(
                    logits, states, slots, top_k=top_k, top_p=top_p
                )
                for _ in range(draws_per_replay)
            ]
        ).flatten()
    samples = torch.empty((replays, draws_per_replay), dtype=torch.int32, device="cuda")
    for index in range(replays):
        graph.replay()
        samples[index].copy_(outputs)
    torch.cuda.synchronize()
    flat = samples.flatten().long()
    if dense_ties:
        groups = flat // 2048
        expected = 1.0 / 32
        bins = 32
    else:
        assert torch.isin(flat, candidates).all()
        groups = torch.searchsorted(candidates, flat)
        expected = _fractional_probabilities(values, top_k, top_p).cuda()
        bins = candidates.numel()
    observed = torch.bincount(groups, minlength=bins).double() / flat.numel()
    assert torch.max(torch.abs(observed - expected)) < 0.045
