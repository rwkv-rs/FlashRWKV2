# SPDX-License-Identifier: MIT

"""Shared-forward invariance and an independent real-valued recurrence VJP."""

import math

import pytest
import torch

from flashrwkv2.tmix.wkv7.tilelang import _infer_recurrent, tmix_wkv7_recurrent

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def _case(dim, lengths, *, requires_grad=False, logits=-2.0):
    generator = torch.Generator(device="cuda").manual_seed(1729 + dim)
    n, heads = sum(lengths), 2

    def rand(*shape):
        return torch.randn(*shape, device="cuda", generator=generator)

    values = [rand(n, heads, dim).mul_(0.1).to(torch.bfloat16) for _ in range(6)]
    values[1] = (values[1].float() + logits).to(torch.bfloat16)
    kk = torch.nn.functional.normalize(values[2].float(), dim=-1)
    values[4] = (-kk).to(torch.bfloat16)
    values[5] = (kk * 0.4).to(torch.bfloat16)
    state = rand(len(lengths), heads, dim, dim).mul_(0.01)
    bias = rand(heads, dim).mul_(0.1)
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    for value in (*values, state, bias):
        value.requires_grad_(requires_grad)
    return values, state, cu, bias


def _reference(values, initial, lengths, bias, scale):
    """Independent FP32 PyTorch graph, including the declared BF16 boundary."""
    r, logits, k, v, a, b = (value.float() for value in values)
    if bias is not None:
        logits = (logits + bias).to(torch.bfloat16).float()
    # Use the mathematical exp/sigmoid formula, independently of the kernel's
    # exp2 implementation. The tiny activation error is below BF16 VJP noise.
    retention = torch.exp(-math.exp(-0.5) * torch.sigmoid(logits))
    outputs, final_states = [], []
    offset = 0
    for sequence, length in enumerate(lengths):
        state = initial[sequence]
        for token in range(offset, offset + length):
            sa = torch.einsum("hkv,hk->hv", state, a[token])
            state = (
                state * retention[token, :, :, None]
                + b[token, :, :, None] * sa[:, None, :]
                + k[token, :, :, None] * v[token, :, None, :]
            )
            outputs.append(
                (torch.einsum("hkv,hk->hv", state, r[token]) * scale).to(torch.bfloat16)
            )
        final_states.append(state)
        offset += length
    output = torch.stack(outputs) if outputs else torch.empty_like(values[0])
    return output, torch.stack(final_states)


@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("lengths", [(1,), (0, 1, 17, 0, 33)])
def test_recurrent_forward_reference_and_grad_mode(dim, lengths):
    values, state, cu, bias = _case(dim, lengths, requires_grad=True)
    original = state.detach().clone()
    actual = tmix_wkv7_recurrent(*values, state, cu, decay_bias=bias, scale=0.75)
    with torch.no_grad():
        inference = tmix_wkv7_recurrent(*values, state, cu, decay_bias=bias, scale=0.75)
        expected = _reference(values, state, lengths, bias, 0.75)
    assert torch.equal(state.detach(), original)
    for training, rollout in zip(actual, inference, strict=True):
        assert torch.equal(training, rollout)
    torch.testing.assert_close(actual[0], expected[0], atol=2e-4, rtol=8e-3)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("dim", [64, 128, 256])
def test_recurrent_packing_decode_and_column_config_are_bitwise_equal(dim):
    lengths = (1, 17, 33)
    values, state, cu, bias = _case(dim, lengths)
    packed, final = tmix_wkv7_recurrent(
        *values, state, cu, decay_bias=bias, block_values=64
    )
    alternate = tmix_wkv7_recurrent(
        *values, state, cu, decay_bias=bias, block_values=32
    )
    assert torch.equal(packed, alternate[0])
    assert torch.equal(final, alternate[1])

    offset = 0
    for sequence, length in enumerate(lengths):
        current = state[sequence : sequence + 1].clone()
        for token in range(offset, offset + length):
            single = [value[token : token + 1].contiguous() for value in values]
            y, current = tmix_wkv7_recurrent(
                *single,
                current,
                torch.tensor([0, 1], device="cuda", dtype=torch.int32),
                decay_bias=bias,
                block_values=32,
            )
            assert torch.equal(y[0], packed[token])
        assert torch.equal(current[0], final[sequence])
        offset += length


@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("bias_dtype", [torch.bfloat16, torch.float32])
def test_recurrent_bias_matches_materialized_bf16_logits(dim, bias_dtype):
    values, state, cu, bias = _case(dim, (17,))
    bias = bias.to(bias_dtype)
    with_bias = tmix_wkv7_recurrent(*values, state, cu, decay_bias=bias)
    prepared = list(values)
    prepared[1] = (values[1].float() + bias.float()).to(torch.bfloat16)
    without_bias = tmix_wkv7_recurrent(*prepared, state, cu)
    for actual, expected in zip(with_bias, without_bias, strict=True):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("logits", [-15.0, -2.0, 15.0])
@pytest.mark.parametrize("cotangent", ["random", "sparse", "zero"])
def test_recurrent_independent_vjp(dim, logits, cotangent):
    lengths = (0, 1, 17, 33)
    values, state, cu, bias = _case(dim, lengths, requires_grad=True, logits=logits)
    reference_inputs = [
        value.detach().clone().requires_grad_() for value in (*values, state, bias)
    ]
    actual = tmix_wkv7_recurrent(*values, state, cu, decay_bias=bias, scale=0.75)
    expected = _reference(
        reference_inputs[:6], reference_inputs[6], lengths, reference_inputs[7], 0.75
    )
    generator = torch.Generator(device="cuda").manual_seed(123)
    dy = torch.randn(
        actual[0].shape, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    ds = torch.randn(actual[1].shape, device="cuda", generator=generator).mul_(0.1)
    if cotangent == "zero":
        dy.zero_()
        ds.zero_()
    elif cotangent == "sparse":
        dy[1:] = 0
        ds[1:] = 0
    gradients = torch.autograd.grad(actual, (*values, state, bias), (dy, ds))
    references = torch.autograd.grad(expected, reference_inputs, (dy, ds))
    for actual_grad, reference_grad in zip(gradients, references, strict=True):
        assert torch.isfinite(actual_grad).all()
        if cotangent == "zero":
            assert torch.count_nonzero(actual_grad) == 0
            continue
        # BF16 gradient rounding contributes about 0.2% RMS. The 1% bound
        # includes gradient reduction and independent activation noise;
        # the independent FP32 initial-state gradient uses a tighter bound.
        relative_l2 = torch.linalg.vector_norm(
            actual_grad.float() - reference_grad.float()
        ) / (torch.linalg.vector_norm(reference_grad.float()).clamp_min(1e-12))
        threshold = 1e-2 if actual_grad.dtype == torch.bfloat16 else 3e-3
        assert relative_l2 < threshold
        torch.testing.assert_close(actual_grad, reference_grad, atol=2e-5, rtol=2e-2)
        # One SGD step checks both the gradient and its parameter binding.
        torch.testing.assert_close(
            -0.01 * actual_grad.float(),
            -0.01 * reference_grad.float(),
            atol=2e-7,
            rtol=2e-2,
        )


def test_recurrent_final_state_only_and_empty_sequences():
    values, state, cu, bias = _case(64, (0, 0), requires_grad=True)
    y, final = tmix_wkv7_recurrent(*values, state, cu, decay_bias=bias)
    assert y.numel() == 0
    assert torch.equal(final, state)
    ds = torch.randn_like(final)
    (gradient,) = torch.autograd.grad(final, (state,), ds)
    assert torch.equal(gradient, ds)


def test_recurrent_status_masks_inactive_capacity_and_invalid_metadata():
    values, state, _, bias = _case(64, (3, 1, 1))
    # Only the first sequence is active; inactive CU entries are deliberately
    # invalid and must never determine a token range.
    cu = torch.tensor([0, 3, -999, 1000000], device="cuda", dtype=torch.int32)
    status = torch.tensor([0, 3, 1], device="cuda", dtype=torch.int32)
    y, final = tmix_wkv7_recurrent(
        *values,
        state,
        cu,
        decay_bias=bias,
        _status=status,
    )
    active = tmix_wkv7_recurrent(
        *[value[:3] for value in values],
        state[:1],
        torch.tensor([0, 3], device="cuda", dtype=torch.int32),
        decay_bias=bias,
    )
    assert torch.equal(y[:3], active[0])
    assert torch.equal(final[:1], active[1])
    assert torch.isnan(y[3:]).all()
    assert torch.equal(final[1:], state[1:])

    status.copy_(torch.tensor([1, 0, 0], device="cuda", dtype=torch.int32))
    invalid_y, invalid_final = tmix_wkv7_recurrent(
        *values,
        state,
        cu,
        decay_bias=bias,
        _status=status,
    )
    assert torch.isnan(invalid_y).all()
    assert torch.equal(invalid_final, state)


@pytest.mark.memcheck
def test_recurrent_indexed_pool_matches_functional_forward():
    values, state, cu, bias = _case(64, (3, 1, 1))
    pool = torch.cat((state, state[:2]), dim=0)
    original = pool.clone()
    bias = bias.to(torch.bfloat16)
    cu.copy_(torch.tensor([0, 3, -999, 1000000], device="cuda", dtype=torch.int32))
    indices = torch.tensor([3, -1, 1000000], device="cuda", dtype=torch.int32)
    status = torch.tensor([0, 3, 1], device="cuda", dtype=torch.int32)
    expected = tmix_wkv7_recurrent(
        *[value[:3] for value in values],
        original[3:4],
        torch.tensor([0, 3], device="cuda", dtype=torch.int32),
        decay_bias=bias,
        block_values=64,
    )
    actual = _infer_recurrent(
        *values,
        pool,
        cu,
        indices,
        status,
        decay_bias=bias,
        block_values=32,
    )
    assert torch.equal(actual[:3], expected[0])
    assert torch.isnan(actual[3:]).all()
    assert torch.equal(pool[3:4], expected[1])
    assert torch.equal(pool[[0, 1, 2, 4]], original[[0, 1, 2, 4]])


@pytest.mark.cuda_graph
def test_recurrent_cuda_graph_replay_matches_eager():
    values, state, cu, bias = _case(64, (1, 17))
    # Compilation and allocator warmup happen before capture.
    for _ in range(3):
        expected = tmix_wkv7_recurrent(*values, state, cu, decay_bias=bias)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = tmix_wkv7_recurrent(*values, state, cu, decay_bias=bias)
    graph.replay()
    for replayed, eager in zip(actual, expected, strict=True):
        assert torch.equal(replayed, eager)
