# SPDX-License-Identifier: MIT

"""CPU compiler coverage for runtime token, batch, and pool dimensions."""

import pytest


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
