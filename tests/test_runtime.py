# SPDX-License-Identifier: MIT

"""Observable package loading and public-surface behavior."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

PUBLIC_MODULES = (
    "flashrwkv2.cmix",
    "flashrwkv2.embedding",
    "flashrwkv2.head.l2wrap_ce",
    "flashrwkv2.head.linear",
    "flashrwkv2.loss.l2wrap_ce",
    "flashrwkv2.post_norm",
    "flashrwkv2.sampling",
    "flashrwkv2.tmix.a_gate",
    "flashrwkv2.tmix.kk_pre",
    "flashrwkv2.tmix.readout",
    "flashrwkv2.tmix.tokenshift",
    "flashrwkv2.tmix.vres_gate",
    "flashrwkv2.tmix.wkv_prepare",
    "flashrwkv2.tmix.wkv7",
    "flashrwkv2.tmix.wkv7.statetune",
)
PUBLIC_PREFIXES = (
    "get_",
    "infer_",
    "prepare_",
    "pretrain_",
    "statetune_",
    "rl_infctx_",
)


def test_cpu_import_does_not_compile_or_create_cache(tmp_path) -> None:
    cache = tmp_path / "extensions"
    environment = os.environ.copy()
    environment["TORCH_EXTENSIONS_DIR"] = str(cache)
    environment["CUDA_HOME"] = str(tmp_path / "missing-toolkit")
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import flashrwkv2,sys; "
                "assert flashrwkv2._C is None; "
                "assert 'flashrwkv2.compile' not in sys.modules"
            ),
        ],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert not cache.exists()


def test_root_extension_loads_once_and_publishes_private_alias(monkeypatch) -> None:
    import flashrwkv2
    from flashrwkv2 import compile as compiler

    sentinel = object()
    monkeypatch.setattr(flashrwkv2, "_C", None)
    monkeypatch.setattr(
        compiler, "load_extension", lambda: SimpleNamespace(module=sentinel)
    )
    try:
        assert flashrwkv2._extension() is sentinel
        assert flashrwkv2._extension() is sentinel
        assert sys.modules["flashrwkv2._C"] is sentinel
    finally:
        sys.modules.pop("flashrwkv2._C", None)


@pytest.mark.parametrize(
    ("capabilities", "message"),
    (
        ((), "visible CUDA GPU"),
        (((7, 5),), "8.0 or newer"),
        (((8, 9), (12, 0)), "same Compute Capability"),
    ),
)
def test_compile_boundary_rejects_invalid_visible_devices(
    capabilities: tuple[tuple[int, int], ...], message: str
) -> None:
    from flashrwkv2.compile import _capability

    cuda = SimpleNamespace(
        device_count=lambda: len(capabilities),
        get_device_capability=lambda index: capabilities[index],
    )
    with pytest.raises(RuntimeError, match=message):
        _capability(SimpleNamespace(cuda=cuda))


def test_compile_boundary_accepts_homogeneous_sm89() -> None:
    from flashrwkv2.compile import _capability

    cuda = SimpleNamespace(
        device_count=lambda: 2,
        get_device_capability=lambda index: (8, 9),
    )
    assert _capability(SimpleNamespace(cuda=cuda)) == (8, 9)


def _mock_runtime_build(monkeypatch, tmp_path, *, fail_once: bool = False):
    import torch
    import torch.utils.cpp_extension

    from flashrwkv2 import compile as compiler

    calls = {"build": 0}
    monkeypatch.setattr(compiler, "_RESULT", None)
    monkeypatch.setattr(compiler, "source_root", lambda: tmp_path / "sources")
    monkeypatch.setattr(compiler, "_capability", lambda torch: (8, 9))
    monkeypatch.setattr(compiler, "_toolchain", lambda torch: {})
    monkeypatch.setattr(
        compiler,
        "_cache_payload",
        lambda torch, root, capability, toolchain: {
            "target": "sm89",
            "cxx_flags": ["-O3"],
            "nvcc_flags": ["-gencode=arch=compute_89,code=sm_89"],
        },
    )
    monkeypatch.setattr(
        torch.utils.cpp_extension,
        "get_default_build_root",
        lambda: str(tmp_path / "cache"),
    )

    def build(name, root, build_dir, payload, toolchain):
        calls["build"] += 1
        if fail_once and calls["build"] == 1:
            raise RuntimeError("compiler failed")
        library = build_dir / f"{name}.so"
        library.write_bytes(b"extension")
        return library

    monkeypatch.setattr(compiler, "_build_extension", build)
    monkeypatch.setattr(
        compiler, "_load_module", lambda name, library: SimpleNamespace(__name__=name)
    )
    return compiler, calls


def test_runtime_cache_compiles_then_hits_manifest(monkeypatch, tmp_path) -> None:
    compiler, calls = _mock_runtime_build(monkeypatch, tmp_path)
    first = compiler.load_extension()
    assert first.status == "compiled"
    assert first.target == "sm89"
    monkeypatch.setattr(compiler, "_RESULT", None)
    second = compiler.load_extension()
    assert second.status == "cached"
    assert second.cache_key == first.cache_key
    assert second.library == first.library
    assert calls["build"] == 1


def test_runtime_cache_serializes_concurrent_builders(monkeypatch, tmp_path) -> None:
    compiler, calls = _mock_runtime_build(monkeypatch, tmp_path)
    results = []
    threads = [threading.Thread(target=lambda: results.append(compiler.load_extension())) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 6
    assert calls["build"] == 1


def test_runtime_cache_retries_after_failed_build(monkeypatch, tmp_path) -> None:
    compiler, calls = _mock_runtime_build(monkeypatch, tmp_path, fail_once=True)
    with pytest.raises(RuntimeError, match="compiler failed"):
        compiler.load_extension()
    result = compiler.load_extension()
    assert result.status == "compiled"
    assert calls["build"] == 2


def test_cache_key_changes_with_every_compatibility_input() -> None:
    baseline = {
        "source_sha256": "source-a",
        "flashrwkv2": "0.1.0a12",
        "python_executable": "/venv/a/bin/python",
        "sys_prefix": "/venv/a",
        "python": "3.12",
        "torch": "2.13.0",
        "torch_cuda": "13.0",
        "cxx11_abi": True,
        "target": "sm89",
        "nvcc": "nvcc-a",
        "host_compiler": "gcc-a",
        "cxx_flags": ["-O3"],
        "nvcc_flags": ["sm_89"],
        "link_flags": ["--strip-debug"],
    }

    def key(payload):
        import hashlib

        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    expected = key(baseline)
    for field in baseline:
        changed = dict(baseline)
        changed[field] = f"changed-{field}"
        assert key(changed) != expected


def test_root_exports_module_operators_by_identity() -> None:
    import flashrwkv2

    expected: dict[str, object] = {}
    for module_name in PUBLIC_MODULES:
        module = importlib.import_module(module_name)
        for name in module.__all__:
            if not name.startswith(PUBLIC_PREFIXES) and name != "setup_sampling_states":
                continue
            assert name not in expected, f"duplicate public operator: {name}"
            expected[name] = getattr(module, name)

    assert set(flashrwkv2.__all__) == set(expected)
    for name, operator in expected.items():
        assert getattr(flashrwkv2, name) is operator


def test_deltalog_policy_matches_pinned_albatross_exact_table() -> None:
    module = importlib.import_module("flashrwkv2.tmix.wkv7")
    expected = {
        (768, 16): 2,
        (768, 32): 3,
        (768, 64): 3,
        (768, 128): 3,
        (768, 256): 3,
        (768, 512): 3,
        (1024, 16): 2,
        (1024, 32): 3,
        (1024, 64): 3,
        (1024, 256): 3,
        (1024, 512): 3,
        (2048, 8): 2,
        (2048, 16): 3,
        (2048, 32): 3,
        (2048, 64): 3,
        (2048, 256): 3,
        (2048, 512): 4,
        (2560, 8): 2,
        (2560, 16): 3,
        (2560, 32): 3,
        (2560, 64): 3,
        (2560, 256): 3,
        (2560, 512): 4,
        (4096, 8): 2,
        (4096, 16): 3,
        (4096, 32): 3,
        (4096, 64): 3,
        (4096, 128): 3,
        (4096, 256): 3,
        (4096, 512): 4,
    }
    assert module._DELTALOG_POLICY_SOURCE_REVISION == (
        "3465da5070beceb4bab9e07b03abee1642a0bdf8"
    )
    assert module._DELTALOG_TUNED_M == expected
    fp16_unprofitable = {
        (768, 64),
        (768, 128),
        (1024, 64),
        (2048, 32),
        (2048, 64),
        (2560, 32),
        (4096, 32),
    }
    assert module._DELTALOG_FP16_UNPROFITABLE_SHAPES == fp16_unprofitable
    for (channels, batch_size), merge_interval in expected.items():
        assert (
            module._select_deltalog_merge_interval(
                channels, batch_size, 64, (12, 0), "fp32io16"
            )
            == merge_interval
        )
        expected_fp16 = (
            0 if (channels, batch_size) in fp16_unprofitable else merge_interval
        )
        assert (
            module._select_deltalog_merge_interval(
                channels, batch_size, 64, (12, 0), "fp16"
            )
            == expected_fp16
        )
    assert (
        module._select_deltalog_merge_interval(
            4096, 320, 64, (12, 0), "fp32io16"
        )
        == 0
    )
    assert (
        module._select_deltalog_merge_interval(
            4096, 64, 128, (12, 0), "fp32io16"
        )
        == 0
    )
    assert (
        module._select_deltalog_merge_interval(
            4096, 64, 64, (9, 0), "fp32io16"
        )
        == 0
    )


def test_wkv7_state_preparation_owns_memory_accounting() -> None:
    module = importlib.import_module("flashrwkv2.tmix.wkv7")
    assert not hasattr(module, "get_tmix_wkv7_recurrent_state_memory_layout")
    for name in (
        "prepare_tmix_wkv7_recurrent_fp16_state",
        "prepare_tmix_wkv7_recurrent_fp32io16_state",
    ):
        assert tuple(inspect.signature(getattr(module, name)).parameters) == (
            "state_pool_size",
            "channels",
            "sequence_capacity",
            "head_size",
            "device",
        )
    assert tuple(
        inspect.signature(
            module.prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor
        ).parameters
    ) == ("state",)


def test_retired_public_aliases_are_absent() -> None:
    import flashrwkv2

    retired = {
        "infer_ln_forward_varlen",
        "infer_res_forward_varlen",
        "infer_post_norm_forward_varlen",
        "infer_tmix_tokenshift_forward_varlen",
        "infer_cmix_tokenshift_forward_varlen",
        "infer_cmix_relu_square_forward_varlen",
        "infer_cmix_linear_ffn_down_forward_varlen",
        "infer_tmix_vres_gate_forward_varlen",
        "infer_tmix_linear_attention_c2c_forward_varlen",
        "infer_tmix_lowrank_wag_forward_varlen",
        "infer_tmix_lowrank_wagv_vres_forward_varlen",
        "infer_tmix_kk_a_gate_forward_varlen",
        "infer_tmix_lnx_rkvres_xg_forward_varlen",
        "infer_head_linear_forward_varlen",
        "infer_recurrent_fp16_forward_varlen",
        "infer_tmix_wkv7_recurrent_deltalog_fp16_forward_varlen",
        "infer_tmix_wkv7_recurrent_deltalog_fp32io16_forward_varlen",
        "infer_recurrent_fp32io16_forward_varlen",
        "infer_chunk_bf16_forward_varlen",
        "get_tmix_wkv7_recurrent_state_memory_layout",
        "prepare_recurrent_metadata",
        "pretrain_recurrent_bf16",
        "statetune_recurrent_fp32io16",
        "rl_infctx_chunk_fp32io16",
        "rl_infctx_chunk_fp32io16_factor_recompute",
    }
    assert retired.isdisjoint(flashrwkv2.__all__)
    for name in retired:
        assert not hasattr(flashrwkv2, name)
