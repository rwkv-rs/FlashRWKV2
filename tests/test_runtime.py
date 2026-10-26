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


@pytest.fixture
def tuning_cache(monkeypatch, tmp_path):
    from flashrwkv2._internal import tuning

    monkeypatch.setattr(tuning, "_profile_root", lambda: tmp_path)
    monkeypatch.setattr(tuning, "_LOADED", {})
    monkeypatch.setattr(tuning, "_WARMED_DEFAULTS", set())
    return tuning


def test_tuning_compiler_identity_without_cuda_toolkit(monkeypatch):
    from flashrwkv2._internal import tuning

    monkeypatch.setattr(importlib.import_module("tilelang.env"), "CUDA_HOME", "")
    tuning._compiler_identity.cache_clear()
    try:
        versions, _ = tuning._compiler_identity()
        assert versions[0] is None
        assert versions[1]
    finally:
        tuning._compiler_identity.cache_clear()


def test_tuning_device_class_shares_ordinals_but_separates_products(monkeypatch):
    import torch
    from flashrwkv2._internal import tuning

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
    from flashrwkv2._internal import tuning

    if override is None:
        monkeypatch.delenv("FLASHRWKV2_CACHE_DIR", raising=False)
    else:
        monkeypatch.setenv("FLASHRWKV2_CACHE_DIR", override)
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    tuning.get_tuning_cache_dir.cache_clear()
    try:
        assert tuning.get_tuning_cache_dir() == Path.home() / ".cache" / "flashrwkv2"
        assert not tmp_path.joinpath("torch").exists()
        assert not tmp_path.joinpath("xdg").exists()
    finally:
        tuning.get_tuning_cache_dir.cache_clear()


def test_tuning_missing_then_published_is_visible_in_same_process(tuning_cache):
    tuning = tuning_cache
    key = (("test", 8, 9, 128), 32, "torch.bfloat16")
    configs = {}
    call = lambda: tuning.get_config(configs, "linear", key, (16, 64, 2, 32))
    before = tuning.prepare_tuning([call], mode="inspect")
    assert not before["complete"]
    assert before["counts"]["missing"] == 1
    assert key not in configs
    tuning.save_config("linear", key, (32, 64, 2, 64))
    after = tuning.prepare_tuning([call])
    assert after["complete"] and after["counts"]["hit"] == 1
    assert call() == (32, 64, 2, 64)


def test_tuning_prepare_reloads_external_publication_and_detects_removal(tuning_cache):
    tuning = tuning_cache
    key = (("test", 8, 9, 128), 32)
    call = lambda: tuning.get_config({}, "linear", key)
    tuning.save_config("linear", key, (16, 64, 2, 32))
    tuning.prepare_tuning([call])
    path = tuning._profile_path("linear", tuning._encode(key))
    record = json.loads(path.read_text())
    record["config"] = [32, 64, 2, 64]
    path.write_text(json.dumps(record))
    refreshed = tuning.prepare_tuning([call])
    assert refreshed["records"][0]["config"] == (32, 64, 2, 64)
    path.unlink()
    with pytest.raises(RuntimeError, match="missing verified tuning profile"):
        tuning.prepare_tuning([call])


def test_tuning_require_and_capture_fail_before_using_missing_key(tuning_cache, monkeypatch):
    tuning = tuning_cache
    key = (("test", 8, 9, 128), 128)
    calls = []
    def call():
        return tuning.get_config({}, "linear", key, tune=lambda: calls.append(1))
    with pytest.raises(RuntimeError, match="missing verified tuning profile"):
        tuning.prepare_tuning([call])
    assert not calls
    monkeypatch.setattr(tuning, "_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="outside CUDA Graph"):
        tuning.prepare_tuning([call], mode="tune")
    with pytest.raises(RuntimeError, match="unprepared tuning key"):
        call()
    assert not calls


def test_tuning_prepares_once_and_reports_actual_dispatch(tuning_cache):
    tuning = tuning_cache
    key = (("test", 12, 0, 188), 128)
    count = []
    def tune():
        count.append(1)
        tuning.save_config("linear", key, (32, 64, 2, 64))
    def call():
        assert tuning.get_config({}, "linear", key, tune=tune) == (32, 64, 2, 64)
    report = tuning.prepare_tuning([call, call], mode="tune")
    assert count == [1]
    assert report["counts"]["tuned"] == 1
    assert report["records"][0]["key"][0] == ["test", 12, 0, 188]
    assert tuning.prepare_tuning([call], mode="tune")["counts"]["hit"] == 1
    assert count == [1]


def test_tuning_preparation_includes_worker_thread_dispatch(tuning_cache):
    tuning = tuning_cache
    def call():
        tuning.get_config({}, "linear", (("test", 12, 0, 188), 1))
        worker = threading.Thread(target=lambda: tuning.get_config({}, "linear", (("test", 12, 0, 188), 2)))
        worker.start()
        worker.join(timeout=10)
        assert not worker.is_alive()
    report = tuning.prepare_tuning([call], mode="inspect")
    assert report["counts"]["missing"] == 2
    assert not report["complete"]


def test_tuning_revalidates_legacy_without_timing(tuning_cache, tmp_path, monkeypatch):
    import torch
    from pathlib import Path
    tuning = tuning_cache
    key = (("test", 12, 0, 188), 128)
    encoded = tuning._encode(("test", 128))
    legacy = tmp_path / "legacy"
    path = legacy / "linear" / (tuning.hashlib.sha256(encoded.encode()).hexdigest() + ".json")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"key": json.loads(encoded), "config": [32, 64, 2, 64], "bitwise_equal": True}))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: pytest.fail("migration must not time candidates"))
    def tune():
        config, records = tuning.select_config(
            ((16, 64, 2, 32),), lambda config: torch.ones(1), reference=torch.ones(1)
        )
        assert config == (32, 64, 2, 64)
        assert records == [{"config": config, "bitwise_equal": True}]
        tuning.save_config("linear", key, config)
    report = tuning.prepare_tuning(
        [lambda: tuning.get_config({}, "linear", key, tune=tune)], mode="tune", legacy_dir=legacy,
    )
    assert report["counts"]["revalidated"] == 1
    assert Path(report["records"][0]["path"]).is_file()


def test_missing_legacy_directory_stops_before_tuning(tuning_cache, tmp_path):
    with pytest.raises(NotADirectoryError, match="legacy tuning namespace does not exist"):
        tuning_cache.prepare_tuning(
            [lambda: pytest.fail("an invalid migration path must not start model preparation")],
            mode="tune", legacy_dir=tmp_path / "missing",
        )


@pytest.mark.parametrize("contents", ["{", "[]", '{"bitwise_equal":false}', '{"config":[0]}'])
def test_tuning_corrupt_record_fails_closed(tuning_cache, contents):
    tuning = tuning_cache
    key = (("test", 8, 9, 128), 1)
    path = tuning._profile_path("linear", tuning._encode(key))
    path.parent.mkdir(parents=True)
    path.write_text(contents)
    with pytest.raises(ValueError):
        tuning.prepare_tuning([lambda: tuning.get_config({}, "linear", key)])


def test_tuning_processes_share_single_writer_and_ignore_torch_cache(tmp_path):
    script = """
import json, os, time
from pathlib import Path
from flashrwkv2 import prepare_tuning
from flashrwkv2._internal import tuning
key = (("test", 12, 0, 188), 64)
def tune():
    with (tuning.get_tuning_cache_dir() / "writers").open("a") as output:
        output.write("writer\\n")
    time.sleep(.1)
    tuning.save_config("linear", key, (32, 64, 2, 64))
report = prepare_tuning([lambda: tuning.get_config({}, "linear", key, tune=tune)], mode="tune")
print(json.dumps(report))
"""
    processes = []
    for ordinal in range(3):
        environment = dict(os.environ, FLASHRWKV2_CACHE_DIR=str(tmp_path / "shared"),
                           TORCH_HOME=str(tmp_path / f"torch-{ordinal}"),
                           XDG_CACHE_HOME=str(tmp_path / f"xdg-{ordinal}"))
        processes.append(subprocess.Popen([sys.executable, "-c", script], env=environment,
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    reports = []
    try:
        for process in processes:
            output, error = process.communicate(timeout=45)
            assert process.returncode == 0, error
            reports.append(json.loads(output))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    assert (tmp_path / "shared" / "writers").read_text() == "writer\n"
    assert sum(report["counts"]["tuned"] for report in reports) == 1
    assert sum(report["counts"]["hit"] for report in reports) == 2
    assert len({report["records"][0]["path"] for report in reports}) == 1


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
    from flashrwkv2._internal import tuning

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
        tuning,
        "get_tuning_cache_dir",
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

    from flashrwkv2._internal import tuning

    expected: dict[str, object] = {
        name: getattr(tuning, name) for name in ("get_tuning_cache_dir", "prepare_tuning")
    }
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
