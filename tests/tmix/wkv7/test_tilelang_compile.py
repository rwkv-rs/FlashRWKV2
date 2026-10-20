# SPDX-License-Identifier: MIT

"""CPU compiler and tuning coverage for functional and indexed recurrence."""

import pytest
import torch


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("fault", ["output", "pool"])
def test_recurrent_tuning_rejects_indexed_only_errors(monkeypatch, indexed, fault):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.wkv7 import tilelang as implementation

    values = [torch.ones(3, 1, 64, dtype=torch.bfloat16) for _ in range(6)]
    initial = torch.zeros(2, 1, 64, 64)
    cu = torch.tensor([0, 1, 3], dtype=torch.int32)
    calls = []

    def functional(*args, block_values, **kwargs):
        calls.append("functional")
        return torch.ones_like(args[0]), args[6] + 1

    def inference(*args, block_values, **kwargs):
        calls.append("indexed")
        pool, indices, status = args[6], args[8], args[9]
        assert pool.data_ptr() != initial.data_ptr()
        assert indices.tolist() == [0, 1]
        assert status.tolist() == [0, 3, 2]
        output = torch.ones_like(args[0])
        pool.add_(1)
        if block_values == 32:
            if fault == "output":
                output.add_(1)
            else:
                pool[0, 0, 0, 0] += 1e-5  # Preserve an FP32-only difference.
        return output

    def select_config(candidates, run, *, check, reference):
        assert len(reference) == 4 and reference[3].dtype == torch.float32
        measurements = []
        for config in candidates:
            actual = check(config)
            equal = all(torch.equal(a, b) for a, b in zip(actual, reference, strict=True))
            measurements.append({"config": config, "bitwise_equal": equal})
        assert [item["config"] for item in measurements if item["bitwise_equal"]] == [64]
        run(64)
        assert calls[-1] == ("indexed" if indexed else "functional")
        return 64, measurements

    monkeypatch.setattr(implementation, "tmix_wkv7_recurrent", functional)
    monkeypatch.setattr(implementation, "_infer_recurrent", inference)
    monkeypatch.setattr(implementation, "_TUNED_CONFIGS", {})
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "cpu-test")
    monkeypatch.setattr(tuning, "select_config", select_config)
    result = implementation.tune_tmix_wkv7_recurrent(*values, initial, cu, indexed=indexed)
    assert result["block_values"] == 64 and result["indexed"] == indexed
    assert torch.count_nonzero(initial) == 0
    assert implementation._TUNED_CONFIGS == {
        implementation._config_key(values[0], 2, None, indexed=indexed): 64,
    }


def test_recurrent_tuning_and_dispatch_keep_mode_caches_separate(monkeypatch):
    from flashrwkv2._internal import tuning
    from flashrwkv2.tmix.wkv7 import tilelang as implementation

    values = [torch.ones(3, 1, 64, dtype=torch.bfloat16) for _ in range(6)]
    initial = torch.zeros(2, 1, 64, 64)
    cu = torch.tensor([0, 1, 3], dtype=torch.int32)
    launches, timed = [], []

    def forward_kernel(heads, dim, scale, has_bias, save, columns, has_status, indexed, bias_dtype):
        def kernel(*args):
            launches.append((indexed, columns))
            args[11].fill_(1)
            if indexed:
                assert has_status
                args[6].add_(1)
            else:
                args[12].copy_(args[6] + 1)
        return kernel

    def select_config(candidates, run, *, check, reference):
        for config in candidates:
            assert all(torch.equal(a, b) for a, b in zip(check(config), reference, strict=True))
        selected = 64 if not timed else 32
        run(selected)
        timed.append(launches[-1])
        return selected, []

    monkeypatch.setattr(implementation, "_forward_kernel", forward_kernel)
    monkeypatch.setattr(implementation, "_check_inputs", lambda *args: None)
    monkeypatch.setattr(implementation, "_TUNED_CONFIGS", {})
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "cpu-test")
    monkeypatch.setattr(tuning, "select_config", select_config)
    for indexed in (False, True):
        implementation.tune_tmix_wkv7_recurrent(*values, initial, cu, indexed=indexed)
    assert timed == [(False, 64), (True, 32)]
    assert torch.count_nonzero(initial) == 0
    assert len(implementation._TUNED_CONFIGS) == 2
    implementation.tmix_wkv7_recurrent(*values, initial, cu)
    assert launches[-1] == (False, 64)
    implementation._infer_recurrent(
        *values, initial.clone(), cu, torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 3, 2], dtype=torch.int32),
    )
    assert launches[-1] == (True, 32)


@pytest.mark.parametrize("mode", ["functional", "indexed", "backward"])
@pytest.mark.parametrize("arch", ["sm_89", "sm_120"])
def test_recurrent_dynamic_shapes_lower_without_gpu(mode, arch):
    import tilelang
    from tilelang import tvm

    from flashrwkv2.tmix.wkv7.tilelang import (
        _PASSES,
        _backward_kernel,
        _forward_kernel,
    )

    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    with target, tilelang.transform.PassContext(config=_PASSES):
        if mode == "backward":
            function = _backward_kernel.get_tir(2, 64, 0.75, True, False, "bfloat16")
        else:
            indexed = mode == "indexed"
            function = _forward_kernel.get_tir(
                2, 64, 0.75, True, not indexed, 32, indexed, indexed, "bfloat16"
            )
        buffers = {buffer.name: buffer for buffer in function.buffer_map.values()}
        tokens = buffers["R"].shape[0]
        batch = buffers["DSN" if mode == "backward" else "Indices" if mode == "indexed" else "S0"].shape[0]
        assert isinstance(tokens, tvm.tirx.Var)
        assert isinstance(batch, tvm.tirx.Var)
        analyzer = tvm.arith.Analyzer()
        for total, sequences in ((0, 2), (33, 2), (65, 5)):
            bindings = {tokens: total, batch: sequences}
            cu_size = tvm.tirx.stmt_functor.substitute(buffers["CU"].shape[0], bindings)
            assert int(analyzer.simplify(cu_size)) == sequences + 1
            if mode != "indexed":
                checkpoints = tvm.tirx.stmt_functor.substitute(buffers["Boundaries"].shape[0], bindings)
                assert int(analyzer.simplify(checkpoints)) == total // 16 + sequences
        if mode == "indexed":
            pool_slots = buffers["S0"].shape[0]
            assert isinstance(pool_slots, tvm.tirx.Var)
            assert not pool_slots.same_as(batch)
        # Actual layout inference/code generation catches symbolic shape
        # binding and launch-grid failures without executing a CUDA kernel.
        result = tilelang.lower(function, target=target, enable_device_compile=False)
        assert result.kernel_source
