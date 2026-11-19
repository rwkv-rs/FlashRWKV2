# SPDX-License-Identifier: MIT

"""Packed RL/Infctx recurrence, state-pool ownership, and final-state VJP."""

import math

import pytest
import torch

from flashrwkv2.tmix.wkv7 import rl_infctx_tmix_wkv7_recurrent_bf16

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def _case(dim, lengths, requires_grad=False):
    generator = torch.Generator(device="cuda").manual_seed(1729 + dim)
    heads = 2
    shape = (sum(lengths), heads, dim)
    values = [
        (torch.randn(shape, device="cuda", generator=generator) * 0.05).to(
            torch.bfloat16
        )
        for _ in range(6)
    ]
    values[1] = (values[1].float() - 2.0).to(torch.bfloat16)
    state = torch.randn((5, heads, dim, dim), device="cuda", generator=generator) * 0.01
    bias = torch.randn((heads, dim), device="cuda", generator=generator) * 0.1
    for value in (*values, state, bias):
        value.requires_grad_(requires_grad)
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    indices = torch.tensor([3, 1, 4][: len(lengths)], device="cuda", dtype=torch.int32)
    return values, state, bias, cu, indices


def _reference(values, state_pool, bias, lengths, indices, scale):
    r, logits, k, v, a, b = (value.float() for value in values)
    logits = (logits + bias).to(torch.bfloat16).float()
    retention = torch.exp(-math.exp(-0.5) * torch.sigmoid(logits))
    outputs, final = [], []
    offset = 0
    for sequence, length in enumerate(lengths):
        state = state_pool[indices[sequence]]
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
        final.append(state)
        offset += length
    return torch.stack(outputs), state_pool.index_copy(
        0, indices.long(), torch.stack(final)
    )


@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("lengths", [(1, 17), (3, 0, 33)])
@pytest.mark.parametrize("final_only", [False, True])
def test_rl_infctx_state_pool_and_independent_vjp(dim, lengths, final_only):
    values, state, bias, cu, indices = _case(dim, lengths, requires_grad=True)
    original = state.detach().clone()
    actual = rl_infctx_tmix_wkv7_recurrent_bf16(
        *values,
        state_pool=state,
        cu_seqlens=cu,
        state_indices=indices,
        decay_bias=bias,
        scale=0.75,
    )
    reference_inputs = [
        value.detach().clone().requires_grad_() for value in (*values, state, bias)
    ]
    expected = _reference(
        reference_inputs[:6],
        reference_inputs[6],
        reference_inputs[7],
        lengths,
        indices,
        0.75,
    )
    assert torch.equal(state.detach(), original)
    unused = torch.tensor(
        [slot for slot in range(5) if slot not in indices.tolist()], device="cuda"
    )
    assert torch.equal(actual[1][unused], original[unused])
    torch.testing.assert_close(actual[0], expected[0], atol=2e-4, rtol=8e-3)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-6, rtol=2e-5)

    generator = torch.Generator(device="cuda").manual_seed(42)
    dy = torch.randn(
        actual[0].shape, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    ds = (
        torch.randn(
            actual[1].shape, dtype=torch.float32, device="cuda", generator=generator
        )
        * 0.1
    )
    if final_only:
        gradients = torch.autograd.grad(actual[1], (*values, state, bias), ds)
        references = torch.autograd.grad(
            expected[1], reference_inputs, ds, allow_unused=True
        )
    else:
        gradients = torch.autograd.grad(actual, (*values, state, bias), (dy, ds))
        references = torch.autograd.grad(expected, reference_inputs, (dy, ds))
    for gradient, reference in zip(gradients, references, strict=True):
        assert torch.isfinite(gradient).all()
        if reference is None:  # r affects only token output, not the final state.
            assert torch.count_nonzero(gradient) == 0
            continue
        relative_l2 = torch.linalg.vector_norm(
            gradient.float() - reference.float()
        ) / torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
        assert relative_l2 < (1e-2 if gradient.dtype == torch.bfloat16 else 3e-3)
        torch.testing.assert_close(gradient, reference, atol=2e-5, rtol=2e-2)
    # Unused state slots take the identity path through the returned pool.
    assert torch.equal(gradients[6][unused], ds[unused])


def test_rl_infctx_dense_and_indexed_calls_share_forward():
    values, pool, bias, cu, indices = _case(64, (1, 17))
    selected = pool.index_select(0, indices.long())
    dense = rl_infctx_tmix_wkv7_recurrent_bf16(
        *values,
        state_pool=selected,
        cu_seqlens=cu,
        decay_bias=bias,
    )
    indexed = rl_infctx_tmix_wkv7_recurrent_bf16(
        *values,
        state_pool=pool,
        cu_seqlens=cu,
        state_indices=indices,
        decay_bias=bias,
    )
    assert torch.equal(dense[0], indexed[0])
    assert torch.equal(dense[1], indexed[1].index_select(0, indices.long()))


@pytest.mark.parametrize(
    "bad_indices,dtype,message",
    [
        ([1, 1], torch.int32, "unique"),
        ([0, 5], torch.int32, "within"),
        ([0, -1], torch.int32, "within"),
        ([0.2, 1.0], torch.float32, "int32"),
    ],
)
def test_rl_infctx_rejects_invalid_state_indices(bad_indices, dtype, message):
    values, pool, bias, cu, _ = _case(64, (1, 17))
    indices = torch.tensor(bad_indices, device="cuda", dtype=dtype)
    with pytest.raises(ValueError, match=message):
        rl_infctx_tmix_wkv7_recurrent_bf16(
            *values,
            state_pool=pool,
            cu_seqlens=cu,
            state_indices=indices,
            decay_bias=bias,
        )


def test_rl_infctx_rejects_nonmonotonic_sequence_boundaries():
    values, pool, bias, _, indices = _case(64, (1, 17))
    cu = torch.tensor([0, 19, 18], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="nondecreasing"):
        rl_infctx_tmix_wkv7_recurrent_bf16(
            *values,
            state_pool=pool,
            cu_seqlens=cu,
            state_indices=indices,
            decay_bias=bias,
        )
