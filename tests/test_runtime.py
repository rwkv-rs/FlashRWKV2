# SPDX-License-Identifier: MIT

"""Observable package loading and public-surface behavior."""

from __future__ import annotations

import importlib
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
    "flashrwkv2.head.ce",
    "flashrwkv2.head.l2wrap_ce",
    "flashrwkv2.head.linear",
    "flashrwkv2.loss.l2wrap_ce",
    "flashrwkv2.post_norm",
    "flashrwkv2.sampling",
    "flashrwkv2.tmix.readout",
    "flashrwkv2.tmix.tokenshift",
    "flashrwkv2.tmix.wkv_prepare",
    "flashrwkv2.tmix.wkv7",
)
PUBLIC_PREFIXES = (
    "get_",
    "head_",
    "infer_",
    "prepare_",
    "pretrain_",
    "rl_infctx_",
)


def test_tuning_device_class_shares_ordinals_but_separates_products(monkeypatch):
    import torch

    from flashrwkv2._internal import execution as tuning

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(
        name="RTX 4090" if device < 2 else "RTX 4090 D",
        major=8, minor=9, multi_processor_count=128 if device < 2 else 114,
    ))
    tuning.device_class.cache_clear()
    try:
        assert tuning.device_class(0) == tuning.device_class(1)
        assert tuning.device_class(0) != tuning.device_class(2)
    finally:
        tuning.device_class.cache_clear()


@pytest.mark.parametrize("override", [None, ""])
def test_tuning_default_root_ignores_torch_and_xdg(monkeypatch, tmp_path, override):
    from pathlib import Path

    from flashrwkv2._internal import execution as tuning

    if override is None:
        monkeypatch.delenv("FLASHRWKV2_CACHE_DIR", raising=False)
    else:
        monkeypatch.setenv("FLASHRWKV2_CACHE_DIR", override)
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    tuning.native_cache_root.cache_clear()
    try:
        assert tuning.native_cache_root() == Path.home() / ".cache" / "flashrwkv2"
        assert not tmp_path.joinpath("torch").exists()
        assert not tmp_path.joinpath("xdg").exists()
    finally:
        tuning.native_cache_root.cache_clear()


def test_fixed_strategy_is_qualified_by_architecture_and_sm_count():
    from flashrwkv2._internal.execution import _fixed_strategy_device

    assert _fixed_strategy_device(("4090", 8, 9, 128))
    assert _fixed_strategy_device(("PRO 6000", 12, 0, 188))
    assert not _fixed_strategy_device(("4090 D", 8, 9, 114))
    assert not _fixed_strategy_device(("other", 12, 0, 170))


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
        timeout=30,
    )
    assert not cache.exists()


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
    from flashrwkv2 import compile as compiler
    from flashrwkv2._internal import execution as tuning

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
        tuning,
        "native_cache_root",
        lambda: tmp_path / "cache",
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


def test_candidate_batches_preserve_distinct_compiler_options(monkeypatch):
    import tilelang

    from _tuning import compile_candidates

    calls, cache = [], {}
    def compile_batch(programs, **options):
        calls.append((programs, options))
        return [object() for _ in programs]
    monkeypatch.setattr(tilelang, "par_compile", compile_batch)
    compile_candidates(((0,), (1,)), cache,
                       lambda k: (k, {"pass_configs": {"tl.enable_async_copy": bool(k)}}))
    assert len(calls) == 2 and len(cache) == 2
    assert {call[1]["pass_configs"]["tl.enable_async_copy"] for call in calls} == {False, True}
