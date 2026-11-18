# SPDX-License-Identifier: MIT

"""Shared-forward invariance and an independent real-valued recurrence VJP."""

import hashlib
import math

import pytest
import torch

from flashrwkv2.tmix.wkv7.tilelang import _infer_recurrent, tmix_wkv7_recurrent

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.memcheck
@pytest.mark.racecheck
def test_optional_fp16_state_preserves_inference_packing_and_training_precision(dim, has_bias):
    from flashrwkv2 import (
        infer_tmix_wkv7_recurrent_fp16_forward_varlen,
        prepare_tmix_wkv7_recurrent_fp16_state_from_tensor,
    )
    from flashrwkv2.tmix.wkv7.tilelang import _bias_tensor, _forward

    lengths = (1, 17, 33)
    values, initial, cu, bias = _case(dim, lengths)
    initial = initial.half()
    bias = bias if has_bias else None
    with pytest.raises(ValueError, match="FP32"):
        tmix_wkv7_recurrent(*values, initial, cu, decay_bias=bias)
    reference = _forward(
        *values, initial, cu, _bias_tensor(values[0], bias),
        torch.empty(0, dtype=torch.int32, device="cuda"), .75, False, 64, 1,
    )[:2]
    independent = _reference(values, initial, lengths, bias, .75, state_dtype=torch.float16)
    for actual, expected in zip(reference, independent, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=8e-3)

    slots = torch.tensor([3, 1, 0], dtype=torch.int32, device="cuda")
    pool = initial.new_zeros(5, *initial.shape[1:])
    pool.index_copy_(0, slots.long(), initial)
    original = pool.clone()
    handle = prepare_tmix_wkv7_recurrent_fp16_state_from_tensor(pool)
    status = torch.tensor([0, values[0].shape[0], len(lengths)], dtype=torch.int32, device="cuda")
    for columns in (64, 32):
        for prefetch in (1, 4):
            functional = _forward(
                *values, initial, cu, _bias_tensor(values[0], bias),
                torch.empty(0, dtype=torch.int32, device="cuda"), .75,
                False, columns, prefetch,
            )[:2]
            pool.copy_(original)
            inferred = _infer_recurrent(
                *values, pool, cu, slots, status, scale=.75, decay_bias=bias,
                block_values=columns, prefetch_tokens=prefetch,
            )
            for candidate, expected in zip(
                (*functional, inferred, pool[slots.long()]),
                (*reference, *reference), strict=True,
            ):
                assert torch.equal(candidate.view(torch.int16), expected.view(torch.int16))
            assert torch.equal(pool[[2, 4]], original[[2, 4]])

    pool.copy_(original)
    actual = infer_tmix_wkv7_recurrent_fp16_forward_varlen(
        *values, state=handle, cu_seqlens=cu, state_indices=slots,
        scale=.75, decay_bias=bias,
    )
    assert torch.equal(actual.view(torch.int16), reference[0].view(torch.int16))
    assert torch.equal(pool[slots.long()].view(torch.int16), reference[1].view(torch.int16))
    assert torch.equal(pool[[2, 4]], original[[2, 4]])
    assert pool.dtype == torch.float16 and actual.dtype == torch.bfloat16

    pool.copy_(original)
    offset = 0
    one = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    for sequence, length in enumerate(lengths):
        slot = slots[sequence:sequence + 1]
        for token in range(offset, offset + length):
            y = infer_tmix_wkv7_recurrent_fp16_forward_varlen(
                *(value[token:token + 1] for value in values), state=handle,
                cu_seqlens=one, state_indices=slot, scale=.75, decay_bias=bias,
            )
            assert torch.equal(y[0].view(torch.int16), reference[0][token].view(torch.int16))
        assert torch.equal(pool[slot.long()].view(torch.int16), reference[1][sequence:sequence + 1].view(torch.int16))
        offset += length


@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("has_bias", [False, True])
def test_optional_fp16_tuning_preserves_inference_outputs(dim, has_bias, monkeypatch, tmp_path):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.wkv7 import tilelang as implementation

    values, initial, cu, bias = _case(dim, (1, 3))
    initial = initial.half()
    original = initial.clone()
    monkeypatch.setattr(tuning, "_profile_root", lambda: tmp_path)
    monkeypatch.setattr(implementation, "_TUNED_CONFIGS", {})
    tuning._LOADED.clear()
    tuned = implementation.tune_tmix_wkv7_recurrent(
        *values, initial, cu, scale=.75, decay_bias=bias if has_bias else None,
        indexed=True,
    )
    candidates = tuned["candidates"]
    assert len(candidates) == 4
    assert all(item["bitwise_equal"] and item["milliseconds"] > 0 for item in candidates)
    assert torch.equal(initial.view(torch.int16), original.view(torch.int16))
    tuning._LOADED.clear()


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


def _reference(values, initial, lengths, bias, scale, *, state_dtype=None, dtype=torch.float32):
    """Mathematical recurrence with explicit token and state rounding boundaries."""
    r, logits, k, v, a, b = (value.to(dtype) for value in values)
    if bias is not None:
        logits = (logits + bias).to(torch.bfloat16).to(dtype)
    # Use the mathematical exp/sigmoid formula, independently of the kernel's
    # exp2 implementation. The tiny activation error is below BF16 VJP noise.
    retention = torch.exp(-math.exp(-0.5) * torch.sigmoid(logits))
    outputs, final_states = [], []
    offset = 0
    for sequence, length in enumerate(lengths):
        state = initial[sequence].to(dtype)
        for token in range(offset, offset + length):
            sa = torch.einsum("hkv,hk->hv", state, a[token])
            state = (
                state * retention[token, :, :, None]
                + b[token, :, :, None] * sa[:, None, :]
                + k[token, :, :, None] * v[token, :, None, :]
            )
            if state_dtype is not None:
                state = state.to(state_dtype).to(dtype)
            outputs.append(
                (torch.einsum("hkv,hk->hv", state, r[token]) * scale).to(torch.bfloat16)
            )
        final_states.append(state.to(state_dtype) if state_dtype is not None else state)
        offset += length
    output = torch.stack(outputs) if outputs else torch.empty_like(values[0])
    return output, torch.stack(final_states)


@pytest.mark.parametrize(("heads", "output_hash", "state_hash"), [
    (40, "b3f83f10611f881afdd181b9ec04551ca484a59c60a64ff35c789503155ec073",
         "ec30d6319d689b473527f6d82e5375182517e5aaafeacd03f025f0f980d90ce1"),
    (64, "6425684306d2564f13cc25d088e35ac4f65d7ce2fe3a43442937e20dc3a60c21",
         "ac6d8645c9ab412ba52a760f89b36d29f2f3ddc672ecf34a4bbb6a5c44feba08"),
])
def test_fp16_serving_prefill_and_decode_match_independent_reference(heads, output_hash, state_hash):
    from flashrwkv2 import (
        infer_tmix_wkv7_recurrent_fp16_forward_varlen,
        prepare_tmix_wkv7_recurrent_fp16_state_from_tensor,
    )

    lengths = (1, 17, 5)
    # Integer inputs generated on the host give both GPUs the same bytes.
    generator = torch.Generator().manual_seed(173041)

    def sample(shape, scale, dtype):
        raw = torch.randint(-256, 256, shape, dtype=torch.int32, generator=generator, device="cpu")
        return (raw.float() * scale).to(dtype).cuda()

    values = [sample((sum(lengths), heads, 64), scale, torch.bfloat16)
              for scale in (2**-11, 2**-7, 2**-11, 2**-11, 2**-11, 2**-11)]
    pool = sample((len(lengths), heads, 64, 64), 2**-13, torch.float16)
    bias = sample((heads, 64), 2**-10, torch.bfloat16)
    cu = torch.tensor([0, 1, 18, 23], device="cuda", dtype=torch.int32)
    expected = _reference(values, pool, lengths, bias, .75, state_dtype=torch.float16, dtype=torch.float64)
    handle = prepare_tmix_wkv7_recurrent_fp16_state_from_tensor(pool)
    slots = torch.arange(len(lengths), device="cuda", dtype=torch.int32)
    actual = infer_tmix_wkv7_recurrent_fp16_forward_varlen(
        *values, state=handle, cu_seqlens=cu, state_indices=slots, decay_bias=bias, scale=.75,
    )
    handle.materialize_slots_(slots)
    for result, reference, digest in zip((actual, pool), expected, (output_hash, state_hash), strict=True):
        torch.testing.assert_close(result, reference, atol=2e-4, rtol=8e-3)
        assert hashlib.sha256(result.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() == digest


@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("lengths", [(1,), (0, 1, 17, 0, 33)])
@pytest.mark.parametrize("prefetch_tokens", [1, 4])
def test_recurrent_forward_reference_and_grad_mode(dim, lengths, prefetch_tokens):
    values, state, cu, bias = _case(dim, lengths, requires_grad=True)
    original = state.detach().clone()
    actual = tmix_wkv7_recurrent(
        *values, state, cu, decay_bias=bias, scale=0.75,
        prefetch_tokens=prefetch_tokens,
    )
    with torch.no_grad():
        inference = tmix_wkv7_recurrent(
            *values, state, cu, decay_bias=bias, scale=0.75,
            prefetch_tokens=prefetch_tokens,
        )
        expected = _reference(values, state, lengths, bias, 0.75)
    assert torch.equal(state.detach(), original)
    for training, rollout in zip(actual, inference, strict=True):
        assert torch.equal(training, rollout)
    torch.testing.assert_close(actual[0], expected[0], atol=2e-4, rtol=8e-3)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("lengths", [(1, 17, 33), (0, 2, 3, 4, 5, 15, 16, 17)])
@pytest.mark.parametrize("prefetch_tokens", [1, 4])
@pytest.mark.memcheck
@pytest.mark.racecheck
def test_recurrent_packing_decode_and_column_config_are_bitwise_equal(
    dim, lengths, prefetch_tokens,
):
    values, state, cu, bias = _case(dim, lengths)
    packed, final = tmix_wkv7_recurrent(
        *values, state, cu, decay_bias=bias, block_values=64,
        prefetch_tokens=prefetch_tokens,
    )
    alternate = tmix_wkv7_recurrent(
        *values, state, cu, decay_bias=bias, block_values=32,
        prefetch_tokens=4 if prefetch_tokens == 1 else 1,
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
                prefetch_tokens=1,
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
@pytest.mark.parametrize("prefetch_tokens", [1, 4])
def test_recurrent_independent_vjp(dim, logits, cotangent, prefetch_tokens):
    lengths = (0, 1, 17, 33)
    values, state, cu, bias = _case(dim, lengths, requires_grad=True, logits=logits)
    reference_inputs = [
        value.detach().clone().requires_grad_() for value in (*values, state, bias)
    ]
    actual = tmix_wkv7_recurrent(
        *values, state, cu, decay_bias=bias, scale=0.75,
        prefetch_tokens=prefetch_tokens,
    )
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


@pytest.mark.parametrize(("lengths", "logits"), [((0, 4097, 33), -2.0), ((3, 4129), -8.0)])
def test_recurrent_direct_vjp_matches_checkpoint_replay(lengths, logits):
    from utils import assert_relative_rmse

    values, state, cu, bias = _case(64, lengths, requires_grad=True, logits=logits)
    dy = torch.randn_like(values[0])
    ds = torch.randn_like(state)
    status = torch.tensor([0, values[0].shape[0], len(lengths)], device=cu.device, dtype=cu.dtype)
    candidate = tmix_wkv7_recurrent(*values, state, cu, decay_bias=bias, scale=.75)
    reference = tmix_wkv7_recurrent(
        *values, state, cu, decay_bias=bias, scale=.75, _status=status,
    )
    for actual, expected in zip(candidate, reference, strict=True):
        assert torch.equal(actual, expected)
    inputs = (*values, state, bias)
    actual_grads = torch.autograd.grad(candidate, inputs, (dy, ds))
    expected_grads = torch.autograd.grad(reference, inputs, (dy, ds))
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        assert_relative_rmse(actual, expected)


def test_recurrent_backward_replays_state_before_large_updates():
    tokens = 4096
    values = [torch.zeros(tokens, 1, 64, device="cuda", dtype=torch.bfloat16) for _ in range(6)]
    values[1].fill_(-32.0)
    values[2][-1, 0, 0] = 2.0**20
    values[3][-1, 0, 0] = 2.0**20
    values[0].requires_grad_()
    state = torch.zeros(1, 1, 64, 64, device="cuda")
    state[0, 0, 0, 0] = 1.0
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    output, _ = tmix_wkv7_recurrent(*values, state, cu)
    dy = torch.zeros_like(output)
    dy[-2, 0, 0] = 1.0
    (dr,) = torch.autograd.grad(output, values[0], dy)
    assert dr[-2, 0, 0].item() == 1.0


def test_recurrent_segment_boundaries_propagate_actual_cotangent():
    tokens = 4224
    values = [torch.zeros(tokens, 1, 64, device="cuda", dtype=torch.bfloat16) for _ in range(6)]
    values[1].fill_(-32.0)
    values[4][-128:, 0, 0] = 1.0
    values[5][-128:, 0, 0] = 1.0
    for value in values:
        value.requires_grad_()
    state = torch.zeros(1, 1, 64, 64, device="cuda", requires_grad=True)
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    output, final = tmix_wkv7_recurrent(*values, state, cu)
    gradients = torch.autograd.grad(
        (output, final), (*values, state), (torch.zeros_like(output), torch.zeros_like(final)),
    )
    for gradient in gradients:
        assert torch.count_nonzero(gradient).item() == 0


def test_recurrent_final_state_only_and_empty_sequences():
    values, state, cu, bias = _case(64, (0, 0), requires_grad=True)
    y, final = tmix_wkv7_recurrent(*values, state, cu, decay_bias=bias)
    assert y.numel() == 0
    assert torch.equal(final, state)
    ds = torch.randn_like(final)
    (gradient,) = torch.autograd.grad(final, (state,), ds)
    assert torch.equal(gradient, ds)


@pytest.mark.memcheck
@pytest.mark.racecheck
@pytest.mark.parametrize("prefetch_tokens", [1, 4])
def test_recurrent_checkpoint_backward_sanitizer(prefetch_tokens):
    lengths = (17,)
    values, state, cu, bias = _case(64, lengths, requires_grad=True)
    reference_inputs = [
        value.detach().clone().requires_grad_() for value in (*values, state, bias)
    ]
    actual = tmix_wkv7_recurrent(
        *values, state, cu, decay_bias=bias, prefetch_tokens=prefetch_tokens,
    )
    expected = _reference(
        reference_inputs[:6], reference_inputs[6], lengths, reference_inputs[7], 1.0
    )
    cotangents = tuple(torch.ones_like(value) for value in actual)
    gradients = torch.autograd.grad(actual, (*values, state, bias), cotangents)
    references = torch.autograd.grad(expected, reference_inputs, cotangents)
    for actual_grad, reference_grad in zip(gradients, references, strict=True):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, reference_grad, atol=2e-5, rtol=2e-2)


@pytest.mark.parametrize("prefetch_tokens", [1, 4])
def test_recurrent_status_masks_inactive_capacity_and_invalid_metadata(prefetch_tokens):
    values, state, _, bias = _case(64, (3, 1, 1))
    for tensor in values:
        tensor[3:].fill_(float("nan"))
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
        prefetch_tokens=prefetch_tokens,
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
        prefetch_tokens=prefetch_tokens,
    )
    assert torch.isnan(invalid_y).all()
    assert torch.equal(invalid_final, state)


@pytest.mark.memcheck
@pytest.mark.racecheck
@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("block_values", [32, 64])
@pytest.mark.parametrize("prefetch_tokens", [1, 4])
def test_recurrent_indexed_pool_matches_functional_forward(dim, block_values, prefetch_tokens):
    values, state, cu, bias = _case(dim, (3, 1, 1))
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
        block_values=block_values,
        prefetch_tokens=prefetch_tokens,
    )
    assert torch.equal(actual[:3], expected[0])
    assert torch.isnan(actual[3:]).all()
    assert torch.equal(pool[3:4], expected[1])
    assert torch.equal(pool[[0, 1, 2, 4]], original[[0, 1, 2, 4]])


@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("indexed", [False, True])
def test_recurrent_tuning_preserves_both_forwards_and_caller_state(monkeypatch, tmp_path, dim, indexed):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.wkv7 import tilelang as implementation

    values, state, cu, bias = _case(dim, (1, 3))
    original = state.clone()
    monkeypatch.setattr(tuning, "_profile_root", lambda: tmp_path)
    tuning._LOADED.clear()
    monkeypatch.setattr(implementation, "_TUNED_CONFIGS", {})
    expected = tmix_wkv7_recurrent(*values, state, cu, decay_bias=bias, block_values=64)
    selected = implementation.tune_tmix_wkv7_recurrent(
        *values, state, cu, decay_bias=bias, indexed=indexed,
    )
    assert selected["indexed"] == indexed
    assert all(candidate["bitwise_equal"] for candidate in selected["candidates"])
    assert torch.equal(state, original)
    key = implementation._config_key(values[0], state.shape[0], bias, indexed=indexed)
    other_key = implementation._config_key(values[0], state.shape[0], bias, indexed=not indexed)
    assert implementation._TUNED_CONFIGS[key] == (
        selected["block_values"], selected["prefetch_tokens"],
    )
    assert other_key not in implementation._TUNED_CONFIGS
    functional = tmix_wkv7_recurrent(*values, state, cu, decay_bias=bias)
    pool = state.clone()
    output = _infer_recurrent(
        *values, pool, cu,
        torch.arange(state.shape[0], dtype=torch.int32, device="cuda"),
        torch.tensor([0, values[0].shape[0], state.shape[0]], dtype=torch.int32, device="cuda"),
        decay_bias=bias,
    )
    for actual, reference in zip((*functional, output, pool), (*expected, *expected), strict=True):
        assert torch.equal(actual, reference)
    tuning._LOADED.clear()


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
