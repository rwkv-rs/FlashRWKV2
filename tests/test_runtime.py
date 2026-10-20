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
    "head_",
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
                "assert 'flashrwkv2.compile' not in sys.modules; "
                "assert 'tilelang' not in sys.modules"
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


def test_compile_cli_keeps_build_diagnostics_out_of_json(monkeypatch, capfd) -> None:
    from flashrwkv2 import compile as compiler

    result = SimpleNamespace(
        json=lambda: {
            "status": "compiled",
            "target": "sm89",
            "cache_key": "key",
            "library": "/cache/extension.so",
        }
    )

    def load_extension():
        os.write(1, b"native build diagnostics\n")
        return result

    monkeypatch.setattr(compiler, "load_extension", load_extension)
    compiler.main()
    stdout, stderr = capfd.readouterr()
    assert json.loads(stdout) == result.json()
    assert "native build diagnostics" in stderr


def test_cache_key_changes_with_every_compatibility_input() -> None:
    baseline = {
        "source_sha256": "source-a",
        "flashrwkv2": "0.1.0a13",
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


def test_source_digest_tracks_shared_native_headers(tmp_path) -> None:
    from flashrwkv2.compile import _source_digest

    header = tmp_path / "validation.h"
    header.write_text("void validate_metadata();\n")
    original = _source_digest(tmp_path)
    header.write_text("void validate_metadata(int capacity);\n")
    assert _source_digest(tmp_path) != original


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




def test_wkv7_state_preparation_owns_memory_accounting() -> None:
    module = importlib.import_module("flashrwkv2.tmix.wkv7")
    assert not hasattr(module, "get_tmix_wkv7_recurrent_state_memory_layout")
    for name in (
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
