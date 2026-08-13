"""Static checks for the active module-local native source contract."""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NATIVE_ROOTS = (ROOT / "csrc" / "sm90", ROOT / "csrc" / "sm120")
GLOBAL_NATIVE = {
    Path("csrc/bindings.cpp"),
    Path("csrc/registration.cpp"),
    Path("csrc/validation.cpp"),
    Path("csrc/validation/recurrent_metadata.cu"),
}
MIRRORED_MODULES = (
    "cmix/mix",
    "cmix/sparse",
    "embedding",
    "head/l2wrap_ce",
    "head/linear",
    "loss/l2wrap_ce",
    "sampling",
    "tmix/a_gate",
    "tmix/kk_a_gate",
    "tmix/kk_pre",
    "tmix/linear",
    "tmix/lnx_rkvres_xg",
    "tmix/mix6",
    "tmix/normalization",
    "tmix/vres_gate",
    "tmix/wkv7",
)
PUBLIC_INFERENCE_MODULES = (
    "flashrwkv2.embedding",
    "flashrwkv2.tmix.mix6",
    "flashrwkv2.tmix.kk_a_gate",
    "flashrwkv2.tmix.linear",
    "flashrwkv2.tmix.lnx_rkvres_xg",
    "flashrwkv2.tmix.normalization",
    "flashrwkv2.tmix.vres_gate",
    "flashrwkv2.tmix.wkv7",
    "flashrwkv2.cmix.mix",
    "flashrwkv2.cmix.sparse",
    "flashrwkv2.head.linear",
    "flashrwkv2.sampling",
)
PUBLIC_TRAINING_MODULES = (
    "flashrwkv2.tmix.wkv7",
    "flashrwkv2.tmix.wkv7.statetune",
    "flashrwkv2.tmix.a_gate",
    "flashrwkv2.tmix.vres_gate",
    "flashrwkv2.tmix.mix6",
    "flashrwkv2.tmix.kk_pre",
    "flashrwkv2.tmix.lnx_rkvres_xg",
    "flashrwkv2.cmix.mix",
    "flashrwkv2.loss.l2wrap_ce",
    "flashrwkv2.head.l2wrap_ce",
)
PUBLIC_TRAINING_PREFIXES = ("pretrain_", "statetune_", "rl_infctx_")


def _active_native_files() -> set[Path]:
    return {
        path.relative_to(ROOT)
        for native_root in NATIVE_ROOTS
        for path in native_root.rglob("*")
        if path.suffix in {".cpp", ".cu"}
    }


def _setup_sources() -> set[Path]:
    tree = ast.parse((ROOT / "setup.py").read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "CUDAExtension":
            continue
        sources = next(
            keyword.value for keyword in node.keywords if keyword.arg == "sources"
        )
        assert isinstance(sources, ast.List)
        values = set()
        for item in sources.elts:
            assert isinstance(item, ast.Constant) and isinstance(item.value, str)
            values.add(Path(item.value))
        return values
    raise AssertionError("setup.py does not define a CUDAExtension source list")


def test_active_native_sources_are_paired_and_listed() -> None:
    native_files = _active_native_files()
    module_files = native_files - GLOBAL_NATIVE
    assert module_files
    assert module_files <= _setup_sources()
    assert _setup_sources() == module_files | GLOBAL_NATIVE

    for relative in module_files:
        if relative.suffix == ".cpp":
            assert relative.with_suffix(".cu") in module_files, relative
        if relative.suffix == ".cu":
            assert relative.with_suffix(".cpp") in module_files, relative


def test_module_paths_are_mirrored() -> None:
    for module in MIRRORED_MODULES:
        assert (ROOT / "flashrwkv2" / module).exists(), module
        assert (ROOT / "tests" / module).exists(), module
        assert (ROOT / "benchmarks" / module).exists(), module
        assert any((native_root / module).exists() for native_root in NATIVE_ROOTS), (
            module
        )

    wkv7_workload_layout = {
        "pretrain": "pretrain.py",
        "rl_infctx": "rl_infctx.py",
        "statetune": "statetune.py",
    }
    for workload, python_file in wkv7_workload_layout.items():
        assert (ROOT / "flashrwkv2/tmix/wkv7" / python_file).is_file()
        assert (ROOT / "tests/tmix/wkv7" / workload / "test.py").is_file()
        assert (ROOT / "benchmarks/tmix/wkv7" / workload / "bench.py").is_file()


def test_forbidden_global_and_legacy_paths_are_absent_from_active_tree() -> None:
    assert not (ROOT / "csrc" / "common").exists()
    assert not (ROOT / "flashrwkv2" / "elementwise").exists()
    assert not (ROOT / "csrc" / "elementwise").exists()
    assert not (ROOT / "flashrwkv2" / "rl_infctx").exists()
    assert not (ROOT / "tests" / "rl_infctx").exists()
    assert not (ROOT / "benchmarks" / "rl_infctx").exists()
    for native_root in NATIVE_ROOTS:
        assert not (native_root / "rl_infctx").exists()

    active_paths = {
        path.relative_to(ROOT)
        for root in (
            ROOT / "flashrwkv2",
            ROOT / "csrc" / "sm90",
            ROOT / "csrc" / "sm120",
        )
        for path in root.rglob("*")
        if path.is_file()
    }
    rendered = "\n".join(str(path) for path in sorted(active_paths))
    for forbidden in (
        "infer_common_",
        "pretrain_common_",
        "_registration.cpp",
        "rwkv7_fast_ops_fp16",
        "rwkv7_wkv_fp16_v2",
    ):
        assert forbidden not in rendered


def test_module_cuda_files_have_provenance_headers() -> None:
    for relative in sorted(_active_native_files() - GLOBAL_NATIVE):
        if relative.suffix != ".cu":
            continue
        text = (ROOT / relative).read_text()
        assert "SPDX-License-Identifier:" in text, relative
        assert "revision" in text.lower(), relative


def test_python_surface_stays_operator_only() -> None:
    """FlashRWKV2 exposes operators; model classes and model forward APIs stay external."""

    for path in sorted((ROOT / "flashrwkv2").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                assert not re.search(r"(?:rwkv|transformer|model)", node.name, re.IGNORECASE), (
                    path
                )
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "forward"
            ):
                argument_names = {argument.arg for argument in node.args.args}
                assert not argument_names.intersection(
                    {"input_ids", "attention_mask", "position_ids"}
                ), path


def test_root_exports_all_public_inference_operators() -> None:
    import flashrwkv2

    expected = {}
    for module_name in PUBLIC_INFERENCE_MODULES:
        module = importlib.import_module(module_name)
        for name in module.__all__:
            if not name.startswith("infer_"):
                continue
            assert name not in expected, (
                f"public inference operator {name} is exported by both "
                f"{expected[name].__module__} and {module_name}"
            )
            expected[name] = getattr(module, name)

    assert len(expected) == 46
    root_inference_names = {
        name for name in flashrwkv2.__all__ if name.startswith("infer_")
    }
    assert root_inference_names == set(expected)
    for name, operator in expected.items():
        assert getattr(flashrwkv2, name) is operator


def test_root_exports_all_public_training_operators() -> None:
    import flashrwkv2

    expected = {}
    for module_name in PUBLIC_TRAINING_MODULES:
        module = importlib.import_module(module_name)
        for name in module.__all__:
            if not name.startswith(PUBLIC_TRAINING_PREFIXES):
                continue
            assert name not in expected, (
                f"public training operator {name} is exported by both "
                f"{expected[name].__module__} and {module_name}"
            )
            expected[name] = getattr(module, name)

    assert len(expected) == 14
    root_training_names = {
        name for name in flashrwkv2.__all__ if name.startswith(PUBLIC_TRAINING_PREFIXES)
    }
    assert root_training_names == set(expected)
    for name, operator in expected.items():
        assert getattr(flashrwkv2, name) is operator


def test_root_exports_sampling_state_setup() -> None:
    import flashrwkv2
    from flashrwkv2.sampling import setup_sampling_states

    assert "setup_sampling_states" in flashrwkv2.__all__
    assert flashrwkv2.setup_sampling_states is setup_sampling_states


def test_fp16_elapsed_advance_stays_in_the_wkv7_owner() -> None:
    cuda_source = (
        ROOT / "csrc/sm120/tmix/wkv7/infer_recurrent_fp16_forward_varlen.cu"
    ).read_text()
    cpp_source = (
        ROOT / "csrc/sm120/tmix/wkv7/infer_recurrent_fp16_forward_varlen.cpp"
    ).read_text()
    python_source = (ROOT / "flashrwkv2/tmix/wkv7/__init__.py").read_text()

    assert "advance_i32_varlen_kernel" in cuda_source
    assert "recurrent_fp16_advance_i32_varlen" in cpp_source
    assert "infer_recurrent_fp16_advance_i32_varlen" in python_source
    assert "elementwise" not in python_source


def test_packed_multidimensional_launches_check_the_actual_extent() -> None:
    mix6 = (ROOT / "csrc/sm120/tmix/mix6/infer_fp16_forward_varlen.cu").read_text()
    cmix = (ROOT / "csrc/sm120/cmix/mix/infer_fp16_forward_varlen.cu").read_text()
    kk_a = (ROOT / "csrc/sm120/tmix/kk_a_gate/infer_fp16_forward_varlen.cu").read_text()
    sparse = (ROOT / "csrc/sm120/cmix/sparse/infer_fp16_forward_varlen.cpp").read_text()
    linear = (ROOT / "csrc/sm120/tmix/linear/infer_fp16_forward_varlen.cpp").read_text()

    assert re.search(
        r"use_tmix_mix6_grid3d\(\s*int batch_size,\s*int total_tokens,"
        r"\s*int max_seqlen,\s*int channels\)",
        mix6,
    )
    assert "total_tokens > kMaxGridDimYZ" in mix6
    assert "total_tokens <= kMaxGridDimYZ" in cmix
    assert "total_tokens <= kMaxGridDimYZ" in kk_a
    assert sparse.count("check_sparse_grid_rows(") == 4
    assert '"grid.y"' in sparse and '"grid.y/grid.z"' in sparse
    assert "M maps to CUDA grid.y" in linear
    assert "x.size(0) <= kMaxGridDimYZ" in linear


def test_forced_only_upstream_kernels_are_explicit_disabled_references() -> None:
    lnx = (
        ROOT / "csrc/sm120/tmix/lnx_rkvres_xg/infer_fp16_forward_varlen.cu"
    ).read_text()
    fp32_wkv = (
        ROOT / "csrc/sm120/tmix/wkv7/infer_recurrent_fp32io16_forward_varlen.cu"
    ).read_text()

    assert "const bool use_grid2d = false" not in lnx
    assert "const bool use_short = false" not in fp32_wkv
    assert re.search(
        r"#if 0.*tmix_lnx_rkvres_xg_warp_2d_kernel.*#endif",
        lnx,
        re.DOTALL,
    )
    assert re.search(
        r"#if 0.*wkv_fp32_v2_short_block_kernel.*#endif",
        fp32_wkv,
        re.DOTALL,
    )
    for source in (lnx, fp32_wkv):
        assert "Upstream status:" in source
        assert "Local status:" in source
        assert "ee3308f6922e59f2166c7fac3c5a192340a2b48e" in source


def test_sm120_has_no_hardcoded_false_runtime_dispatch() -> None:
    forbidden = re.compile(
        r"(?:const|constexpr)\s+bool\s+\w+\s*=\s*false\s*;|if\s*\(\s*false\s*\)"
    )
    offenders = []
    for path in sorted((ROOT / "csrc/sm120").rglob("*")):
        if path.suffix not in {".cpp", ".cu"}:
            continue
        if forbidden.search(path.read_text()):
            offenders.append(path.relative_to(ROOT))
    assert not offenders


def test_sm120_active_kernels_have_launch_owners_and_disabled_allowlist_is_exact() -> (
    None
):
    kernel_definition = re.compile(
        r"__global__\s+(?:void\s+)?"
        r"(?:__launch_bounds__\s*\([^)]*\)\s*)?"
        r"(?:void\s+)?([A-Za-z_]\w*)\s*\("
    )

    def split_disabled_if0(source: str) -> tuple[str, list[str]]:
        active_lines = []
        disabled_regions = []
        disabled_lines = []
        depth = 0
        for line in source.splitlines(keepends=True):
            directive = line.lstrip()
            if depth:
                if directive.startswith("#if"):
                    depth += 1
                elif directive.startswith("#endif"):
                    depth -= 1
                if depth:
                    disabled_lines.append(line)
                else:
                    disabled_regions.append("".join(disabled_lines))
                    disabled_lines = []
                continue
            if re.match(r"#if\s+0\b", directive):
                depth = 1
                continue
            active_lines.append(line)
        assert depth == 0
        return "".join(active_lines), disabled_regions

    disabled_symbols = set()
    unreachable = []
    for path in sorted((ROOT / "csrc/sm120").rglob("*.cu")):
        source = path.read_text()
        active, disabled_regions = split_disabled_if0(source)
        for region in disabled_regions:
            disabled_symbols.update(kernel_definition.findall(region))
        active_without_comments = re.sub(
            r"//.*?$|/\*.*?\*/", "", active, flags=re.MULTILINE | re.DOTALL
        )
        for symbol in kernel_definition.findall(active_without_comments):
            if (
                len(re.findall(rf"\b{re.escape(symbol)}\b", active_without_comments))
                < 2
            ):
                unreachable.append((path.relative_to(ROOT), symbol))

    assert disabled_symbols == {
        "add_layer_norm_cmix_mix_f16_kernel",
        "add_layer_norm_cmix_mix_f16_generic_kernel",
        "add_layer_norm_cmix_mix_f16_scalar_stats_kernel",
        "add_layer_norm_cmix_mix_f16_welford_kernel",
        "add_layer_norm_tmix_mix6_f16_kernel",
        "add_layer_norm_tmix_mix6_f16_generic_kernel",
        "add_layer_norm_tmix_mix6_f16_scalar_stats_kernel",
        "tmix_lnx_rkvres_xg_warp_2d_kernel",
        "wkv_fp32_v2_short_block_kernel",
    }
    assert not unreachable, "\n".join(
        f"{path}: {symbol}" for path, symbol in unreachable
    )
