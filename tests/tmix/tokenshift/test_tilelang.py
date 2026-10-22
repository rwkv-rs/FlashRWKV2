# SPDX-License-Identifier: MIT

import pytest
import torch
import tilelang

from flashrwkv2._internal.state import _transfer
from flashrwkv2.post_norm.tilelang import norm
from flashrwkv2.tmix.tokenshift.tilelang import (
    _backward, _forward, _fused_forward, fused_post_norm_tokenshift, tokenshift,
)


@pytest.mark.parametrize("arch", ("sm_89", "sm_120"))
@pytest.mark.parametrize("channels", (7, 128, 1024, 4096, 7168))
def test_tokenshift_lowering(arch, channels):
    with tilelang.tvm.target.Target({"kind": "cuda", "arch": arch}) as target:
        for block in (128, 256, 512):
            tilelang.lower(_forward.get_tir(channels, 6, 5, "int32", block), target=target)
        tilelang.lower(_forward.get_tir(channels, 6, 5, "int32", 256, True), target=target)
        tilelang.lower(_backward.get_tir(channels, 6, 5, "int64"), target=target)
        for write in (False, True):
            tilelang.lower(_transfer.get_tir(channels, 5, "bfloat16", write), target=target)
        for count in (1, 6):
            tilelang.lower(_fused_forward.get_tir(channels, count, 1e-5, 2), target=target)
        tilelang.lower(_backward.get_tir(channels, 6, 5, "int32", 256, 32, True), target=target)
        tilelang.lower(_backward.get_tir(channels, 1, 5, "int32", 256, 32, False), target=target)


def reference(x, mixes, offsets, initial):
    previous, final = [], []
    for b, (start, end) in enumerate(zip(offsets, offsets[1:])):
        if end > start:
            previous.append(torch.cat((initial[b:b + 1], x[start:end - 1])))
            final.append(x[end - 1])
        else:
            final.append(initial[b])
    previous = torch.cat(previous)
    outputs = tuple(
        (x.float() + (previous.float() - x.float()) * mix.float()).bfloat16()
        for mix in mixes
    )
    return outputs + (torch.stack(final),)


@pytest.mark.parametrize("live", (False, True))
@pytest.mark.parametrize("fault", ("functional_output", "functional_final", "status_output", "status_final"))
def test_tokenshift_tuning_checks_both_variants_with_active_metadata(monkeypatch, live, fault):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.tokenshift import tilelang as implementation

    x = torch.ones(5, 64, dtype=torch.bfloat16)
    mixes = torch.ones(6, 64, dtype=torch.bfloat16)
    initial = torch.ones(4, 64, dtype=torch.bfloat16)
    cu = torch.tensor([0, 1, 3, -999, 999] if live else [0, 1, 3, 4, 5], dtype=torch.int32)
    status = torch.tensor([0, 3, 2], dtype=torch.int32) if live else None
    launches = []

    def forward(channels, count, batch, index_dtype, block=256, use_status=False):
        def kernel(values, mixes, offsets, initial_shift, active):
            launches.append(use_status)
            if not use_status:
                assert offsets.tolist() == ([0, 1, 3, 3, 3] if live else [0, 1, 3, 4, 5])
                assert values.shape[0] == (3 if live else 5)
            output = torch.ones(count, *values.shape, dtype=values.dtype)
            final = torch.ones_like(initial_shift)
            if block == 128 and use_status == fault.startswith("status"):
                (output if fault.endswith("output") else final).add_(1)
            return output, final
        return kernel

    def select_config(candidates, run, *, check, reference):
        assert len(reference) == 4
        eligible = []
        for config in candidates:
            actual = check(config)
            if all(torch.equal(a, b) for a, b in zip(actual, reference, strict=True)):
                eligible.append(config)
        assert eligible == [256, 512]
        run(256)
        assert launches[-1] == live
        return 256, []

    monkeypatch.setattr(implementation, "tokenshift", lambda *args, **kwargs: None)
    monkeypatch.setattr(implementation, "_forward", forward)
    monkeypatch.setattr(implementation, "_BLOCK_CONFIGS", {})
    monkeypatch.setattr(tuning, "select_config", select_config)
    monkeypatch.setattr(tuning, "save_config", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "test")
    implementation.tune_tokenshift(x, mixes, cu, initial, _status=status)
    assert implementation._BLOCK_CONFIGS[implementation._config_key(x, mixes, 4)] == 256


@pytest.mark.parametrize("status", ((1, 3, 2), (0, 0, 0), (0, 6, 2), (0, 3, 5)))
def test_tokenshift_tuning_requires_valid_nonempty_active_sample(status):
    from flashrwkv2.tmix.tokenshift.tilelang import tune_tokenshift

    with pytest.raises(ValueError, match="valid nonempty active"):
        tune_tokenshift(
            torch.ones(5, 64, dtype=torch.bfloat16),
            torch.ones(6, 64, dtype=torch.bfloat16),
            torch.tensor([0, 1, 3, -999, 999], dtype=torch.int32),
            torch.ones(4, 64, dtype=torch.bfloat16),
            _status=torch.tensor(status, dtype=torch.int32),
        )


def test_fused_tuning_rejects_changed_pool_and_preserves_caller_state(monkeypatch):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.tokenshift import tilelang as implementation

    x = torch.ones(3, 64, dtype=torch.bfloat16)
    mixes = torch.ones(6, 64, dtype=torch.bfloat16)
    pool = torch.zeros(5, 64, dtype=torch.bfloat16)
    indices = torch.tensor([3, 1, 2], dtype=torch.int32)
    status = torch.tensor([0, 3, 3], dtype=torch.int32)

    def forward(channels, count, eps, rows_per_block=1):
        def kernel(x, res, weight, bias, *parameters_and_state):
            assert len(parameters_and_state) == 9
            scratch, indices, status = parameters_and_state[-3:]
            scratch[indices.long()] = 2 if rows_per_block == 2 else 1
            return torch.ones_like(x), torch.ones(count, *x.shape, dtype=x.dtype)
        return kernel

    def select_config(candidates, run, *, check):
        expected = check(1)
        assert len(expected) == 3
        expected_pool = pool.clone()
        expected_pool[indices.long()] = 1
        assert torch.equal(expected[2], expected_pool)
        eligible = []
        for config in candidates:
            actual = check(config)
            if all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True)):
                eligible.append(config)
        assert eligible == [1, 4]
        assert torch.equal(run(1)[0], expected[0])
        return 1, []

    monkeypatch.setattr(implementation, "fused_post_norm_tokenshift", lambda *args, **kwargs: None)
    monkeypatch.setattr(implementation, "_fused_forward", forward)
    monkeypatch.setattr(implementation, "_FUSED_CONFIGS", {})
    monkeypatch.setattr(tuning, "select_config", select_config)
    monkeypatch.setattr(tuning, "save_config", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "test")
    implementation.tune_fused_tokenshift(x, x, x[0], x[0], mixes, pool, indices, status)
    assert torch.count_nonzero(pool) == 0
    assert implementation._FUSED_CONFIGS[implementation._config_key(x, mixes, x.shape[0])] == 1


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("channels", (7, 128, 1024, 4096, 7168))
def test_tokenshift_bitwise_packing_chunks_and_grad_mode(channels):
    torch.manual_seed(711)
    offsets = [0, 0, 1, 5, 5, 12]
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    x = torch.randn(12, channels, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mixes = torch.randn(6, channels, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    initial = torch.randn(5, channels, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    outputs, final = tokenshift(x, mixes, cu, initial)
    with torch.no_grad():
        infer, infer_final = tokenshift(x, mixes, cu, initial)
        collected = [[] for _ in range(6)]
        states = []
        for b, (start, end) in enumerate(zip(offsets, offsets[1:])):
            state = initial[b:b + 1].contiguous()
            for left in range(start, end, 3):
                right = min(left + 3, end)
                metadata = torch.tensor([0, right - left], device="cuda", dtype=torch.int32)
                part, state = tokenshift(x[left:right].contiguous(), mixes, metadata, state)
                for values, result in zip(collected, part, strict=True):
                    values.append(result)
            states.append(state)
    for training, inference, chunks in zip(outputs, infer, collected, strict=True):
        assert torch.equal(training, inference)
        assert torch.equal(training, torch.cat(chunks))
    assert torch.equal(final, infer_final)
    assert torch.equal(final, torch.cat(states))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_tokenshift_independent_vjp_with_empty_sequences():
    torch.manual_seed(712)
    offsets = [0, 0, 1, 5, 5, 12]
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    inputs = (
        torch.randn(12, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True),
        torch.randn(6, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True),
        torch.randn(5, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True),
    )
    x, mixes, initial = inputs
    outputs, final = tokenshift(x, mixes, cu, initial)
    actual = (*outputs, final)
    refs = [t.detach().clone().requires_grad_() for t in inputs]
    expected = reference(refs[0], refs[1], offsets, refs[2])
    upstream = tuple(torch.randn_like(t) for t in actual)
    gradients = torch.autograd.grad(actual, inputs, upstream)
    expected_grads = torch.autograd.grad(expected, refs, upstream)
    for output, ref in zip(actual, expected, strict=True):
        assert torch.equal(output, ref)
    for gradient, ref in zip(gradients, expected_grads, strict=True):
        torch.testing.assert_close(gradient, ref, atol=0.125, rtol=0.02)


@pytest.mark.cuda
@pytest.mark.memcheck
@pytest.mark.racecheck
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("channels", (128, 4096, 7168))
@pytest.mark.parametrize("count", (1, 6))
@pytest.mark.parametrize("batch", (1, 5))
def test_fused_t1_matches_training_and_preserves_inactive_slots(channels, count, batch):
    torch.manual_seed(713)
    x = torch.randn(batch, channels, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    res = torch.randn_like(x)
    weight = torch.randn(channels, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn_like(weight)
    mixes = torch.randn(count, channels, device="cuda", dtype=torch.bfloat16)
    original = torch.randn(batch + 2, channels, device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(batch, device="cuda", dtype=torch.int32).flip(0) + 1
    status = torch.tensor([0, batch, batch], device="cuda", dtype=torch.int32)
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    normalized, total = norm(x, weight, bias, res=res, return_res=True)
    expected, final = tokenshift(normalized, mixes, cu, original.index_select(0, indices.long()))
    expected_pool = original.clone().index_copy_(0, indices.long(), final)
    parameters = tuple(mixes.unbind(0))
    mix_args = parameters + (parameters[-1],) * (6 - count)
    for group in (1, 2, 4):
        pool = original.clone()
        actual_total, actual = _fused_forward(channels, count, 1e-5, group)(
            x, res, weight, bias, *mix_args, pool, indices, status
        )
        assert torch.equal(total, actual_total)
        assert torch.equal(torch.stack(expected), actual)
        assert torch.equal(expected_pool, pool)
    unaligned = tuple(
        parameter.new_empty(channels + 1)[1:].copy_(parameter)
        for parameter in parameters
    )
    for supplied in (mixes, tuple(parameter.clone() for parameter in parameters), unaligned):
        pool = original.clone()
        actual_total, actual = fused_post_norm_tokenshift(
            x, res, weight, bias, supplied, pool, indices, status
        )
        assert torch.equal(total, actual_total)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
        assert torch.equal(expected_pool, pool)
    status.zero_()
    indices.fill_(99)
    pool = original.clone()
    inactive_total, inactive_outputs = fused_post_norm_tokenshift(
        x, res, weight, bias, mixes, pool, indices, status
    )
    assert torch.equal(pool, original)
    assert torch.isnan(inactive_total).all()
    assert all(torch.isnan(output).all() for output in inactive_outputs)
