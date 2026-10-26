"""Plan, execute and verify source-bound incremental and full validation."""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, replace
from email.parser import BytesParser
from pathlib import Path

import tomllib

# Isolated Python execution must expose CI helpers, never the checkout package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_targets import (
    TARGETS,
    _git_changed_files,
    _release_metadata_only,
    _target_from_path,
    classify,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 7
PRIMARY_TORCH = "2.14.1"
TILELANG_VERSION = "0.1.15"
TORCH_VERSIONS = (PRIMARY_TORCH, "2.13.0", "2.11.0", "2.12.0")
BUILD_INPUT_PATHS = (
    "setup.py",
    "pyproject.toml",
    "MANIFEST.in",
    "README.md",
    "LICENSE",
    "NOTICE",
    "LICENSES",
    "flashrwkv2",
    "csrc",
)
VALIDATION_INPUT_PATHS = (
    "flashrwkv2",
    "csrc",
    "tests",
    "ci",
    ".github/workflows",
    "setup.py",
    "pyproject.toml",
    "uv.lock",
    "MANIFEST.in",
)
BLOCK_TEST = "tests/tmix/wkv7/test_block.py"
COMPATIBILITY_PATHS = (
    BLOCK_TEST,
    "tests/tmix/wkv7/statetune",
    "tests/tmix/wkv7/test_tilelang.py",
    "tests/head/l2wrap_ce",
    "tests/loss/l2wrap_ce",
    "tests/tmix/wkv_prepare",
    "tests/cmix",
    "tests/sampling",
)
COMPATIBILITY_KEYWORDS = (
    "fullgraph or prepare_tuning_covers or zero_temperature_is_deterministic_greedy "
    "or recurrent_forward_reference_and_grad_mode"
)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _json_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_tree_sha(revision: str) -> str:
    return _git("rev-parse", f"{revision}^{{tree}}")


def _inputs_hash(revision: str, paths: tuple[str, ...]) -> str:
    return _json_hash(
        [SCHEMA_VERSION, _git("ls-tree", "-r", "--full-tree", revision, "--", *paths)]
    )


def _test_paths(modules: tuple[str, ...] | list[str]) -> list[str]:
    paths = [f"tests/{module}" for module in modules]
    paths = [
        path
        for path in paths
        if not any(path.startswith(parent + "/") for parent in paths if parent != path)
    ]
    if set(modules) - {"sampling", "embedding"} and "tests/tmix/wkv7" not in paths:
        paths.append(BLOCK_TEST)
    return sorted(paths)


def build_plan(
    base_sha: str,
    head_sha: str,
    *,
    mode="incremental",
    extended=False,
    target="sm120",
) -> dict:
    changed = _git_changed_files(base_sha, head_sha)
    impact = classify(
        changed,
        base_sha=base_sha,
        head_sha=head_sha,
        release_metadata_only=_release_metadata_only(changed, base_sha, head_sha),
    )
    if mode in {"full", "performance"}:
        impact = replace(
            impact,
            affected_modules=tuple(sorted(TARGETS)),
            affected_cuda_modules=tuple(sorted(TARGETS)),
            run_gpu=True,
            run_all=True,
            run_sanitizer=True,
            run_package=True,
            run_contract=True,
            run_compatibility=extended,
        )
    plan = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "extended": extended,
        "target": target,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "source_tree_sha": source_tree_sha(head_sha),
        "build_input_hash": _inputs_hash(head_sha, BUILD_INPUT_PATHS),
        "validation_input_hash": _inputs_hash(head_sha, VALIDATION_INPUT_PATHS),
        "policy_hash": _inputs_hash(
            head_sha, ("ci", ".github/workflows", "pyproject.toml", "uv.lock")
        ),
        "impact": asdict(impact),
        "test_paths": ["tests"]
        if impact.run_all
        else _test_paths(impact.affected_modules),
        "torch_versions": list(
            TORCH_VERSIONS if impact.run_compatibility else (PRIMARY_TORCH,)
        )
        if impact.run_gpu
        else [],
        "cold_cache": extended,
    }
    plan["plan_hash"] = _json_hash(plan)
    return plan


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_outputs(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as handle:
        for key, value in payload.items():
            if isinstance(value, (dict, list, tuple, bool)):
                value = json.dumps(value, separators=(",", ":"))
            handle.write(f"{key}={value}\n")


def package_artifacts(directory: Path) -> dict:
    artifacts = {}
    for kind, pattern in (("wheel", "*.whl"), ("sdist", "*.tar.gz")):
        paths = list(directory.rglob(pattern))
        if len(paths) != 1:
            raise ValueError(f"expected one {kind}, found {len(paths)}")
        artifacts[kind] = {"name": paths[0].name, "sha256": _sha256(paths[0])}
    return artifacts


def verify_package(directory: Path) -> dict:
    artifacts = package_artifacts(directory)
    wheel = next(directory.rglob("*.whl"))
    sdist = next(directory.rglob("*.tar.gz"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    version = project["version"]
    if (
        wheel.name != f"flashrwkv2-{version}-py3-none-any.whl"
        or sdist.name != f"flashrwkv2-{version}.tar.gz"
    ):
        raise ValueError(
            "source distribution filenames do not match the project version"
        )
    python_sources = {
        p.relative_to(ROOT).as_posix(): p.read_bytes()
        for p in (ROOT / "flashrwkv2").rglob("*.py")
    }
    native_sources = {
        p.relative_to(ROOT / "csrc").as_posix(): p.read_bytes()
        for p in (ROOT / "csrc").rglob("*")
        if p.is_file()
    }
    architectures = {
        name.split("/", 1)[0]
        for name in native_sources
        if re.fullmatch(r"sm\d+", name.split("/", 1)[0])
    }
    if architectures != {"sm80"}:
        raise ValueError("native sources must use the single sm80 source tree")

    def verify_metadata(data):
        metadata = BytesParser().parsebytes(data)
        if (
            re.sub(r"[-_.]+", "-", metadata["Name"]).lower() != "flashrwkv2"
            or metadata["Version"] != version
        ):
            raise ValueError("source distribution metadata does not match the project")

    with zipfile.ZipFile(wheel) as archive:
        expected = {
            **python_sources,
            **{
                f"flashrwkv2/_csrc/{name}": data
                for name, data in native_sources.items()
            },
        }
        names = archive.namelist()
        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise ValueError("wheel must contain one package metadata record")
        verify_metadata(archive.read(metadata_names[0]))
        if any(name.endswith((".so", ".dll", ".pyd")) for name in names):
            raise ValueError("wheel contains a compiled extension")
        for name, data in expected.items():
            if archive.read(name) != data:
                raise ValueError(f"wheel source mismatch: {name}")
        if {
            name
            for name in names
            if name.startswith("flashrwkv2/")
            and (name.endswith(".py") or name.startswith("flashrwkv2/_csrc/"))
        } != set(expected):
            raise ValueError("wheel includes unexpected package sources")
    with tarfile.open(sdist) as archive:
        prefix = sdist.name.removesuffix(".tar.gz")
        verify_metadata(archive.extractfile(f"{prefix}/PKG-INFO").read())
        expected = {
            **python_sources,
            **{f"csrc/{name}": data for name, data in native_sources.items()},
        }
        if any(
            member.name.endswith((".so", ".dll", ".pyd"))
            for member in archive.getmembers()
        ):
            raise ValueError("sdist contains a compiled extension")
        for name, data in expected.items():
            if archive.extractfile(f"{prefix}/{name}").read() != data:
                raise ValueError(f"sdist source mismatch: {name}")
        packaged_sources = {
            member.name.removeprefix(prefix + "/")
            for member in archive.getmembers()
            if member.isfile()
            and (
                member.name.startswith(prefix + "/csrc/")
                or (
                    member.name.startswith(prefix + "/flashrwkv2/")
                    and member.name.endswith(".py")
                )
            )
        }
        if packaged_sources != set(expected):
            raise ValueError("sdist includes unexpected package sources")
    return {"status": "passed", "artifacts": artifacts}


def cpu_environment() -> dict:
    import importlib.metadata

    return {
        "target": "cpu",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "tilelang", "pytest", "numpy")
        },
    }


def environment(target: str) -> dict:
    import importlib.metadata

    import torch

    import flashrwkv2

    package = Path(flashrwkv2.__file__).resolve()
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError(f"GPU validation must import the installed wheel: {package}")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required for GPU validation")
    capability = torch.cuda.get_device_capability()
    actual_target = f"sm{capability[0]}{capability[1]}"
    if target != "auto" and actual_target != target:
        raise ValueError(f"expected {target}, found {actual_target}")

    def command(*args):
        return subprocess.run(
            args, check=True, capture_output=True, text=True, timeout=30
        ).stdout.strip()

    gpu = torch.cuda.get_device_properties(0)
    return {
        "target": actual_target,
        "gpu": gpu.name,
        "visible_devices": torch.cuda.device_count(),
        "multiprocessors": gpu.multi_processor_count,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "tilelang": importlib.metadata.version("tilelang"),
        "python": platform.python_version(),
        "driver": command(
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
            f"--id={os.environ.get('FLASHRWKV_PHYSICAL_GPU', '0')}",
        ),
        "nvcc": command("nvcc", "--version"),
        "cxx": command("c++", "--version"),
        "ninja": command("ninja", "--version"),
        "package_path": str(package),
    }


def _environment_identity(value: dict) -> dict:
    return {key: item for key, item in value.items() if key != "package_path"}


SUITE_MARKERS = {
    "cpu": "not cuda and not resource",
    "correctness": "cuda or resource",
    "memcheck": "cuda and memcheck",
    "racecheck": "cuda and racecheck",
    "compatibility": "cuda",
}


def _suite_paths(plan: dict, suite: str) -> list[str]:
    if suite == "cpu":
        return ["tests"]
    return list(COMPATIBILITY_PATHS) if suite == "compatibility" else plan["test_paths"]


def _paths_cover(recorded: list[str], requested: list[str]) -> bool:
    return all(
        any(path == parent or path.startswith(parent + "/") for parent in recorded)
        for path in requested
    )


def _selected_for_suite(nodeid: str, markers: list[str], suite: str) -> bool:
    cuda, resource = "cuda" in markers, "resource" in markers
    if suite == "cpu":
        return not cuda and not resource
    if suite == "correctness":
        return cuda or resource
    if suite in {"memcheck", "racecheck"}:
        return cuda and suite in markers
    if suite == "compatibility":
        return cuda and any(
            word in nodeid.lower() for word in COMPATIBILITY_KEYWORDS.split(" or ")
        )
    raise ValueError(f"unknown test suite: {suite}")


class TestRecorder:
    """Capture collection before pytest applies marker/keyword deselection."""

    def __init__(self):
        self.collected = {}
        self.selected = {}
        self.reports = {}

    def pytest_collection_modifyitems(self, items):
        self.collected = {
            item.nodeid: sorted({mark.name for mark in item.iter_markers()})
            for item in items
        }

    def pytest_collection_finish(self, session):
        self.selected = {
            item.nodeid: sorted({mark.name for mark in item.iter_markers()})
            for item in session.items
        }

    def pytest_runtest_logreport(self, report):
        self.reports.setdefault(report.nodeid, {})[report.when] = {
            "outcome": report.outcome,
            "seconds": report.duration,
            "reason": str(report.longrepr) if report.skipped or report.failed else "",
        }


def validate_test_result(result: dict, *, allow_empty=False) -> None:
    collected, selected, reports = (
        result["collected"],
        result["selected"],
        result["reports"],
    )
    if not collected:
        raise ValueError("required test collection is empty")
    if any(
        not _paths_cover(result["paths"], [nodeid.split("::")[0]])
        for nodeid in collected
    ):
        raise ValueError("test collection is outside the recorded scope")
    expected = {
        nodeid: markers
        for nodeid, markers in collected.items()
        if _selected_for_suite(nodeid, markers, result["suite"])
    }
    if selected != expected:
        raise ValueError("selected tests do not cover the complete collected suite")
    if not selected:
        if (
            allow_empty
            and result["status"] == "not-required"
            and result["exit_code"] == 5
            and not reports
        ):
            return
        raise ValueError("required test selection is empty")
    if (
        result["status"] != "passed"
        or result["exit_code"] != 0
        or set(selected) != set(reports)
    ):
        raise ValueError("test execution failed or selected tests were not executed")
    for nodeid, phases in reports.items():
        if set(phases) != {"setup", "call", "teardown"}:
            raise ValueError(f"test execution is incomplete: {nodeid}")
        allowed_skip = (
            result["environment"].get("target") == "sm89"
            and nodeid == "tests/sampling/test.py::test_sampling_resource_usage"
            and phases["setup"]["outcome"] == phases["teardown"]["outcome"] == "passed"
            and phases["call"]["outcome"] == "skipped"
            and "resource gate requires the SM120 backend" in phases["call"]["reason"]
        )
        if not allowed_skip and any(
            phase["outcome"] != "passed" for phase in phases.values()
        ):
            raise ValueError(f"test did not complete successfully: {nodeid}")


def _validate_result_identity(result: dict, plan: dict, suite: str) -> None:
    for key in (
        "schema_version",
        "source_tree_sha",
        "validation_input_hash",
        "policy_hash",
    ):
        if result[key] != plan[key]:
            raise ValueError(f"test result {key} mismatch")
    if suite == "cpu" and (
        result["environment"].get("target") != "cpu"
        or result["wheel_sha256"] is not None
    ):
        raise ValueError("CPU contracts must identify source execution")
    if result["suite"] != suite or not _paths_cover(
        result["paths"], _suite_paths(plan, suite)
    ):
        raise ValueError("test result does not cover the requested suite and paths")


def run_tests(
    plan: dict,
    suite: str,
    output: Path,
    package: Path | None,
    reuse: Path | None = None,
) -> int:
    import pytest

    env = cpu_environment() if suite == "cpu" else environment(plan["target"])
    if suite == "cpu" and reuse and reuse.is_file():
        previous = _read_json(reuse)
        if (
            reuse_identity(previous, plan)
            and previous.get("cpu")
            and previous["cpu"]["environment"] == env
        ):
            _write_json(output, previous["cpu"])
            print(
                "Reusing CPU results for identical source, policy and environment",
                flush=True,
            )
            return 0
    paths = _suite_paths(plan, suite)
    # -I plus an explicit pytest pythonpath prevents checkout shadowing.
    pythonpath = [str(ROOT / "tests")]
    if suite == "cpu":
        pythonpath.insert(0, str(ROOT))
    output.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "-q",
        "-c",
        str(ROOT / "pyproject.toml"),
        "--rootdir",
        str(ROOT),
        "-o",
        "pythonpath=" + " ".join(pythonpath),
        "--durations=20",
        "-m",
        SUITE_MARKERS[suite],
        f"--junitxml={output.with_suffix('.xml')}",
        *[str(ROOT / path) for path in paths],
    ]
    if suite == "compatibility":
        args += ["-k", COMPATIBILITY_KEYWORDS]
    pytest.hookimpl(tryfirst=True)(TestRecorder.pytest_collection_modifyitems)
    recorder = TestRecorder()
    started = time.monotonic()
    code = int(pytest.main(args, plugins=[recorder]))
    result = {
        "schema_version": SCHEMA_VERSION,
        "plan_hash": plan["plan_hash"],
        "source_tree_sha": plan["source_tree_sha"],
        "validation_input_hash": plan["validation_input_hash"],
        "policy_hash": plan["policy_hash"],
        "suite": suite,
        "paths": paths,
        "collected": recorder.collected,
        "environment": env,
        "wheel_sha256": package_artifacts(package)["wheel"]["sha256"]
        if package
        else None,
        "selected": recorder.selected,
        "reports": recorder.reports,
        "exit_code": code,
        "seconds": time.monotonic() - started,
        "status": "passed" if code == 0 else "not-required" if code == 5 else "failed",
    }
    try:
        validate_test_result(result, allow_empty=suite in {"memcheck", "racecheck"})
    except ValueError:
        result["status"] = "failed"
        raise
    finally:
        _write_json(output, result)
    return 0


def _isolated_json(python: str, *args: str, env=None, timeout=1800) -> dict:
    try:
        result = subprocess.run(
            [python, "-I", *args],
            check=False,
            capture_output=True,
            text=True,
            cwd=ROOT / "artifacts",
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        for log in (error.stdout, error.stderr):
            if log:
                sys.stderr.write(
                    log.decode(errors="replace") if isinstance(log, bytes) else log
                )
        raise
    if result.returncode:
        sys.stderr.write(result.stdout)
    sys.stderr.write(result.stderr)
    result.check_returncode()
    return json.loads(result.stdout)


def run_gpu(
    plan: dict, package: Path, output: Path, python: str, reuse: Path | None = None
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    wheel = next(package.rglob("*.whl")).resolve()
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "--no-deps",
            "--reinstall",
            str(wheel),
        ],
        check=True,
        timeout=180,
    )
    previous = _read_json(reuse) if reuse and reuse.is_file() else None
    reports = []
    for version in plan["torch_versions"]:
        if version != PRIMARY_TORCH:
            subprocess.run(
                ["uv", "pip", "install", "--python", python, f"torch=={version}"],
                check=True,
                timeout=600,
            )
        env = _isolated_json(
            python,
            str(Path(__file__).resolve()),
            "environment",
            "--target",
            plan["target"],
            timeout=60,
        )
        if env["torch"].split("+")[0] != version:
            raise ValueError(f"unexpected Torch version: {env['torch']}")
        old = (
            next(
                (
                    row
                    for row in previous.get("gpu", [])
                    if row["torch_version"] == version
                ),
                None,
            )
            if previous
            else None
        )
        reuse_suites = bool(old and reusable_gpu_row(previous, plan, version, env))
        if reuse_suites and (not plan["cold_cache"] or old["cold_cache"]):
            reports.append({**old, "reused": True})
            print(
                f"Reusing {version} {plan['target']} validation for identical inputs",
                flush=True,
            )
            continue
        current = output / version
        current.mkdir(parents=True, exist_ok=True)
        compile_env = os.environ.copy()
        # A fresh cache is a measured weekly/release test, not a per-PR penalty.
        compile_started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="cold-", dir=output) as cold:
            if plan["cold_cache"]:
                compile_env["FLASHRWKV2_CACHE_DIR"] = cold
            first = _isolated_json(python, "-m", "flashrwkv2.compile", env=compile_env)
            second = _isolated_json(python, "-m", "flashrwkv2.compile", env=compile_env)
            if (
                first["target"] != plan["target"]
                or second["status"] != "cached"
                or first["cache_key"] != second["cache_key"]
            ):
                raise ValueError("runtime compilation/cache verification failed")
            if plan["cold_cache"] and first["status"] != "compiled":
                raise ValueError("cold-cache validation did not compile")
            listing = subprocess.run(
                ["cuobjdump", "--list-elf", second["library"]],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout
            targets = sorted({f"sm{cc}" for cc in re.findall(r"\bsm_(\d+)\b", listing)})
            if targets != [plan["target"]]:
                raise ValueError(f"runtime library target mismatch: {targets}")
            compile_result = {
                "first": first,
                "second": second,
                "native_targets": targets,
            }
        compile_result["seconds"] = time.monotonic() - compile_started
        _write_json(current / "compile.json", compile_result)
        if reuse_suites:
            suite_results = old["suites"]
        else:
            suites = ["correctness"] if version == PRIMARY_TORCH else ["compatibility"]
            if version == PRIMARY_TORCH and plan["impact"]["run_sanitizer"]:
                suites += ["memcheck", "racecheck"]
            suite_results = []
            for suite in suites:
                destination = current / f"{suite}.json"
                command = [
                    python,
                    "-I",
                    str(Path(__file__).resolve()),
                    "test",
                    "--plan",
                    str(output / "plan.json"),
                    "--suite",
                    suite,
                    "--output",
                    str(destination),
                    "--package",
                    str(package),
                ]
                _write_json(output / "plan.json", plan)
                if suite in {"memcheck", "racecheck"}:
                    command = [
                        "compute-sanitizer",
                        "--error-exitcode=1",
                        "--tool",
                        suite,
                        *command,
                    ]
                subprocess.run(
                    command, check=True, timeout=3600, cwd=ROOT / "artifacts"
                )
                suite_results.append(_read_json(destination))
        reports.append(
            {
                "torch_version": version,
                "reused_suites": reuse_suites,
                "environment": env,
                "cold_cache": plan["cold_cache"],
                "compile": compile_result,
                "suites": suite_results,
            }
        )
    _write_json(output / "gpu.json", {"gpu": reports})


def _validate_gpu(rows: list[dict], plan: dict, artifacts: dict) -> None:
    by_version = {row["torch_version"]: row for row in rows}
    if (
        len(by_version) != len(rows)
        or not set(plan["torch_versions"]) <= by_version.keys()
    ):
        raise ValueError("missing or duplicate Torch validation")
    for version in plan["torch_versions"]:
        row = by_version[version]
        if (
            row["environment"]["target"] != plan["target"]
            or row["environment"]["torch"].split("+")[0] != version
        ):
            raise ValueError("GPU environment identity mismatch")
        if row["environment"]["tilelang"] != TILELANG_VERSION:
            raise ValueError("GPU TileLang version mismatch")
        compiled = row["compile"]
        first, second = compiled["first"], compiled["second"]
        if (
            first["status"] not in {"compiled", "cached"}
            or second["status"] != "cached"
            or first["target"] != plan["target"]
            or second["target"] != plan["target"]
            or first["cache_key"] != second["cache_key"]
            or compiled["native_targets"] != [plan["target"]]
            or (
                plan["cold_cache"]
                and (not row["cold_cache"] or first["status"] != "compiled")
            )
        ):
            raise ValueError("runtime build/cache evidence is missing or inconsistent")
        suites = {suite["suite"]: suite for suite in row["suites"]}
        required = {"correctness"} if version == PRIMARY_TORCH else {"compatibility"}
        if version == PRIMARY_TORCH and plan["impact"]["run_sanitizer"]:
            required |= {"memcheck", "racecheck"}
        if len(suites) != len(row["suites"]) or not required <= suites.keys():
            raise ValueError("required GPU suite is missing")
        for name in required:
            result = suites[name]
            _validate_result_identity(result, plan, name)
            validate_test_result(
                result,
                allow_empty=name in {"memcheck", "racecheck"}
                and not plan["impact"]["run_all"],
            )
            if (
                result["source_tree_sha"] != plan["source_tree_sha"]
                or result["wheel_sha256"] != artifacts["wheel"]["sha256"]
            ):
                raise ValueError(
                    "test result does not identify the tested source and wheel"
                )
            if _environment_identity(result["environment"]) != _environment_identity(
                row["environment"]
            ):
                raise ValueError("test process environment mismatch")
        if version == PRIMARY_TORCH:
            covered = {
                _target_from_path(nodeid.split("::")[0], "tests/")
                for nodeid, markers in suites["correctness"]["selected"].items()
                if "cuda" in markers
            }
            if not set(plan["impact"]["affected_cuda_modules"]) <= covered:
                raise ValueError("selected GPU tests do not cover all affected modules")
            if (
                BLOCK_TEST in plan["test_paths"] or plan["impact"]["run_all"]
            ) and not any(
                nodeid.startswith(BLOCK_TEST + "::")
                for nodeid in suites["correctness"]["selected"]
            ):
                raise ValueError("block integration tests are missing")


def finalize(plan: dict, package: Path | None, validation: Path) -> dict:
    artifacts = package_artifacts(package) if package else {}
    if plan["impact"]["run_package"] and not artifacts:
        raise ValueError("required source distributions are missing")
    cpu = (
        _read_json(validation / "cpu.json") if plan["impact"]["run_contract"] else None
    )
    if cpu:
        _validate_result_identity(cpu, plan, "cpu")
        validate_test_result(cpu)
    gpu = (
        _read_json(validation / "gpu.json")["gpu"] if plan["impact"]["run_gpu"] else []
    )
    if cpu and cpu["source_tree_sha"] != plan["source_tree_sha"]:
        raise ValueError("CPU result identity mismatch")
    _validate_gpu(gpu, plan, artifacts)
    if artifacts and _read_json(validation / "package.json") != {
        "status": "passed",
        "artifacts": artifacts,
    }:
        raise ValueError("source distribution verification is missing")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "source_tree_sha": plan["source_tree_sha"],
        "head_sha": plan["head_sha"],
        "build_input_hash": plan["build_input_hash"],
        "validation_input_hash": plan["validation_input_hash"],
        "policy_hash": plan["policy_hash"],
        "mode": plan["mode"],
        "plan": plan,
        "artifacts": artifacts,
        "cpu": cpu,
        "gpu": gpu,
    }


def reuse_identity(manifest: dict, plan: dict) -> bool:
    return (
        manifest.get("schema_version") == SCHEMA_VERSION
        and all(
            manifest.get(key) == plan[key]
            for key in (
                "source_tree_sha",
                "build_input_hash",
                "validation_input_hash",
                "policy_hash",
            )
        )
        and covers_plan(manifest, manifest.get("plan", {}))
    )


def reusable_gpu_row(manifest: dict, plan: dict, version: str, env: dict) -> bool:
    if not reuse_identity(manifest, plan):
        return False
    old = next(
        (row for row in manifest["gpu"] if row["torch_version"] == version), None
    )
    if old is None or _environment_identity(
        old["environment"]
    ) != _environment_identity(env):
        return False
    previous_paths = manifest["plan"]["test_paths"]
    if version == PRIMARY_TORCH and any(
        not any(
            path == parent or path.startswith(parent + "/") for parent in previous_paths
        )
        for path in plan["test_paths"]
    ):
        return False
    try:
        _validate_gpu(
            [old],
            {**plan, "torch_versions": [version], "cold_cache": False},
            manifest["artifacts"],
        )
    except (KeyError, TypeError, ValueError):
        return False
    return True


def covers_plan(manifest: dict, plan: dict) -> bool:
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "passed"
        or not isinstance(manifest.get("plan"), dict)
    ):
        return False
    if manifest.get("mode") != manifest["plan"]["mode"]:
        return False
    for key in (
        "source_tree_sha",
        "build_input_hash",
        "validation_input_hash",
        "policy_hash",
    ):
        if manifest.get(key) != plan[key]:
            return False
    if plan["mode"] != "incremental" and manifest.get("mode") == "incremental":
        return False
    if plan["impact"]["run_contract"] and not manifest.get("cpu"):
        return False
    if plan["impact"]["run_package"] and set(manifest.get("artifacts", {})) != {
        "wheel",
        "sdist",
    }:
        return False
    try:
        if plan["impact"]["run_contract"]:
            _validate_result_identity(manifest["cpu"], plan, "cpu")
            validate_test_result(manifest["cpu"])
            if manifest["cpu"]["source_tree_sha"] != plan["source_tree_sha"]:
                return False
        _validate_gpu(manifest.get("gpu", []), plan, manifest.get("artifacts", {}))
    except (KeyError, TypeError, ValueError):
        return False
    return True


def verify_manifest(manifest: dict, plan: dict, package: Path | None) -> None:
    if not covers_plan(manifest, plan):
        raise ValueError("quality evidence does not satisfy the requested validation")
    if manifest["artifacts"] != (package_artifacts(package) if package else {}):
        raise ValueError("source distributions differ from the tested originals")


def _api_json(path: str, repository: str, token: str) -> dict:
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}{path}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _artifact_bytes(artifact: dict, token: str) -> bytes:
    request = urllib.request.Request(artifact["archive_download_url"])
    request.add_unredirected_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _run_matches_source(
    run: dict, revision: str, tree: str, repository: str, token: str
) -> bool:
    if run.get("head_repository", {}).get("full_name") != repository:
        return False
    commit = _api_json(f"/git/commits/{revision}", repository, token)
    if commit["tree"]["sha"] != tree:
        return False
    # PR checkout uses GitHub's merge commit; its second parent is the run head.
    return revision == run["head_sha"] or (
        run["event"] == "pull_request"
        and len(commit["parents"]) == 2
        and commit["parents"][1]["sha"] == run["head_sha"]
    )


def resolve_reuse(plan: dict, repository: str, token: str, current_run: int) -> dict:
    name = f"flashrwkv2-quality-v{SCHEMA_VERSION}-{plan['source_tree_sha']}"
    query = urllib.parse.urlencode({"name": name, "per_page": 100})
    artifacts = _api_json(f"/actions/artifacts?{query}", repository, token).get(
        "artifacts", []
    )
    partial = None
    for artifact in sorted(
        artifacts, key=lambda item: item["created_at"], reverse=True
    ):
        run_id = artifact.get("workflow_run", {}).get("id")
        if artifact.get("expired") or not run_id or run_id == current_run:
            continue
        run = _api_json(f"/actions/runs/{run_id}", repository, token)
        if run.get("path") != ".github/workflows/pro6000-gpu.yml":
            continue
        jobs = _api_json(
            f"/actions/runs/{run_id}/jobs?per_page=100", repository, token
        )["jobs"]
        if not any(
            job["name"] == "Quality Gate" and job["conclusion"] == "success"
            for job in jobs
        ):
            continue
        with zipfile.ZipFile(io.BytesIO(_artifact_bytes(artifact, token))) as archive:
            manifest = json.loads(archive.read("quality-manifest.json"))
            if not reuse_identity(manifest, plan) or not _run_matches_source(
                run, manifest["head_sha"], plan["source_tree_sha"], repository, token
            ):
                continue
            if plan["impact"]["run_package"] and set(manifest["artifacts"]) != {
                "wheel",
                "sdist",
            }:
                continue
            complete = covers_plan(manifest, plan)
            result = {
                "reuse": True,
                "complete": complete,
                "run_id": str(run_id),
                "artifact": name,
            }
            if complete:
                return result
            if partial is None:
                partial = result
    if partial:
        return partial
    return {"reuse": False, "complete": False, "run_id": "", "artifact": name}


def _performance_covers(
    result: dict, plan: dict, artifacts: dict, target: str, env: dict
) -> bool:
    from benchmark_gate import BENCHMARKS, _contract_hash
    from benchmark_gate import SCHEMA_VERSION as BENCHMARK_SCHEMA

    if (
        result.get("schema_version") != BENCHMARK_SCHEMA
        or result.get("source_tree_sha") != plan["source_tree_sha"]
        or result.get("wheel_sha256") != artifacts["wheel"]["sha256"]
        or result.get("profile") != "full"
        or sorted(result.get("completed_modules", [])) != sorted(TARGETS)
        or result.get("status") != "completed"
        or any(
            result.get("environment", {}).get(key) != value
            for key, value in _environment_identity(env).items()
        )
    ):
        return False
    measurements = result.get("measurements", {})
    if set(measurements) != set(TARGETS):
        return False
    for module, record in measurements.items():
        if (
            record.get("source_tree_sha") != plan["source_tree_sha"]
            or record.get("wheel_sha256") != artifacts["wheel"]["sha256"]
            or record.get("target") != target
            or record.get("profile") != "full"
            or record.get("benchmark_contract_sha256") != _contract_hash(module)
            or record.get("profile_count", 0) <= 0
            or sorted(record.get("workloads", []))
            != sorted(case.workload for case in BENCHMARKS[module])
        ):
            return False
    return True


def release_payload(
    plan: dict, bundle: Path, repository: str, tag: str, output: Path
) -> None:
    manifest = _read_json(bundle / "quality-manifest.json")
    verify_manifest(manifest, plan, bundle)
    verify_package(bundle)
    if plan["mode"] != "full" or not plan["extended"] or plan["target"] != "sm120":
        raise ValueError("release requires full extended PRO6000 acceptance")
    if not all(
        "PRO6000" in "".join(row["environment"]["gpu"].upper().split())
        for row in manifest["gpu"]
    ):
        raise ValueError("SM120 release acceptance requires PRO6000 hardware")
    token = os.environ["GH_TOKEN"]
    release = _api_json(
        f"/releases/tags/{urllib.parse.quote(tag, safe='')}", repository, token
    )
    assets = {asset["name"]: asset for asset in release["assets"]}
    external_reports = {}
    for name in ("sm89-quality-manifest.json", "sm89-performance-summary.json"):
        asset = assets.get(name)
        if asset is None:
            raise ValueError(f"release requires the independent RTX4090 asset {name}")
        request = urllib.request.Request(
            asset["url"],
            headers={
                "Accept": "application/octet-stream",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        request.add_unredirected_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(request, timeout=120) as response:
            external_reports[name] = json.load(response)
    external = external_reports["sm89-quality-manifest.json"]
    if (
        not covers_plan(external, {**plan, "target": "sm89"})
        or external["artifacts"] != manifest["artifacts"]
    ):
        raise ValueError(
            "RTX4090 full validation does not match the release distributions"
        )
    if not all("4090" in row["environment"]["gpu"] for row in external["gpu"]):
        raise ValueError("SM89 release acceptance requires RTX4090 hardware")
    if not _performance_covers(
        external_reports["sm89-performance-summary.json"],
        plan,
        manifest["artifacts"],
        "sm89",
        next(
            row["environment"]
            for row in external["gpu"]
            if row["torch_version"] == PRIMARY_TORCH
        ),
    ):
        raise ValueError("complete RTX4090 performance acceptance is missing")
    name = f"flashrwkv2-performance-v7-{plan['source_tree_sha']}"
    artifacts = _api_json(
        f"/actions/artifacts?{urllib.parse.urlencode({'name': name, 'per_page': 100})}",
        repository,
        token,
    )["artifacts"]
    performance = None
    for artifact in sorted(artifacts, key=lambda row: row["created_at"], reverse=True):
        if artifact["expired"]:
            continue
        run_id = artifact["workflow_run"]["id"]
        run = _api_json(f"/actions/runs/{run_id}", repository, token)
        if run.get("path") != ".github/workflows/pro6000-gpu.yml":
            continue
        jobs = _api_json(
            f"/actions/runs/{run_id}/jobs?per_page=100", repository, token
        )["jobs"]
        if not any(
            job["name"] == "Independent performance report"
            and job["conclusion"] == "success"
            for job in jobs
        ):
            continue
        with zipfile.ZipFile(io.BytesIO(_artifact_bytes(artifact, token))) as archive:
            candidate = json.loads(archive.read("summary.json"))
            if _run_matches_source(
                run, candidate["revision"], plan["source_tree_sha"], repository, token
            ) and _performance_covers(
                candidate,
                plan,
                manifest["artifacts"],
                "sm120",
                next(
                    row["environment"]
                    for row in manifest["gpu"]
                    if row["torch_version"] == PRIMARY_TORCH
                ),
            ):
                performance = candidate
                break
    if performance is None:
        raise ValueError(
            "complete PRO6000 performance acceptance is missing for this release"
        )
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(bundle / "dist", output / "dist")
    _write_json(output / "quality-manifest.json", manifest)
    _write_json(output / "performance-summary.json", performance)
    for name, report in external_reports.items():
        _write_json(output / name, report)
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    expected = {
        f"flashrwkv2-{version}-py3-none-any.whl",
        f"flashrwkv2-{version}.tar.gz",
    }
    if (
        tag != f"v{version}"
        or {row["name"] for row in manifest["artifacts"].values()} != expected
    ):
        raise ValueError("release tag, version and distribution filenames differ")


def validation_summary(manifest: dict, repository: str, output: Path) -> None:
    with output.open("a", encoding="utf-8") as handle:
        handle.write(f"### Validation ({manifest['mode']})\n\n")
        handle.write(f"Source tree: `{manifest['source_tree_sha']}`\n\n")
        handle.write(
            "| Torch | Target | Suite | Tests | Seconds |\n|---|---|---|---:|---:|\n"
        )
        for row in manifest["gpu"]:
            for suite in row["suites"]:
                handle.write(
                    f"| {row['torch_version']} | {row['environment']['target']} | {suite['suite']} | {len(suite['selected'])} | {suite['seconds']:.1f} |\n"
                )
        if manifest["cpu"]:
            handle.write(
                f"\nCPU tests: {len(manifest['cpu']['selected'])}; {manifest['cpu']['seconds']:.1f} seconds.\n"
            )
        token = os.environ["GH_TOKEN"]
        runs = _api_json(
            "/actions/workflows/pro6000-gpu.yml/runs?event=pull_request&per_page=30",
            repository,
            token,
        )["workflow_runs"]
        handle.write("\n### Five-PR timing observation\n\n")
        handle.write(
            "| Run | Queue/dependencies (s) | Environment (s) | GPU validation (s) |\n|---|---:|---:|---:|\n"
        )
        count = 0
        seen_prs = set()
        for run in runs:
            prs = {pull["number"] for pull in run.get("pull_requests", [])}
            if not prs or prs & seen_prs:
                continue
            jobs = _api_json(
                f"/actions/runs/{run['id']}/jobs?per_page=100", repository, token
            )["jobs"]
            gpu = next(
                (
                    job
                    for job in jobs
                    if job["name"] == "Affected GPU correctness and sanitizer"
                    and job["conclusion"] == "success"
                ),
                None,
            )
            if gpu is None:
                continue
            steps = {step["name"]: step for step in gpu["steps"]}

            def elapsed(start, end):
                return (
                    datetime.datetime.fromisoformat(end.replace("Z", "+00:00"))
                    - datetime.datetime.fromisoformat(start.replace("Z", "+00:00"))
                ).total_seconds()

            def step_seconds(name, steps=steps):
                step = steps.get(name)
                return elapsed(step["started_at"], step["completed_at"]) if step else 0

            waiting = elapsed(run["created_at"], gpu["started_at"])
            setup = step_seconds("Synchronize the project environment")
            validation = step_seconds(
                "Validate the installed wheel or reuse matching GPU results"
            )
            handle.write(
                f"| [Run {run['id']}]({run['html_url']}) | {waiting:.0f} | {setup:.0f} | {validation:.0f} |\n"
            )
            count += 1
            seen_prs.update(prs)
            if count == 5:
                break
        handle.write(
            f"\nObserved {count}/5 distinct PRs with completed GPU validation. The initial single-module target is 20 minutes excluding queue/dependencies.\n"
        )


def _self_test() -> None:
    import copy
    from unittest.mock import patch

    head = _git("rev-parse", "HEAD")
    with patch(
        __name__ + "._git_changed_files",
        return_value=["flashrwkv2/post_norm/tilelang.py"],
    ):
        plan = build_plan(head, head)
    assert set(plan["impact"]["affected_modules"]) == {
        "post_norm",
        "embedding",
        "cmix",
        "tmix/tokenshift",
    }
    assert BLOCK_TEST in plan["test_paths"] and not plan["impact"]["run_all"]
    full = build_plan(head, head, mode="full", extended=True)
    assert full["torch_versions"] == list(TORCH_VERSIONS) and full["test_paths"] == [
        "tests"
    ]
    with patch(__name__ + "._git_changed_files", return_value=["README.md"]):
        doc = build_plan(head, head)
    assert not doc["impact"]["run_gpu"] and doc["impact"]["run_package"]

    env = {"target": "sm120", "torch": PRIMARY_TORCH, "tilelang": TILELANG_VERSION}
    collected = {
        f"tests/{owner}/test.py::test_example[0]": ["cuda", "memcheck", "racecheck"]
        for owner in plan["impact"]["affected_modules"]
    }
    collected[BLOCK_TEST + "::test_fullgraph[0]"] = [
        "cuda",
        "cuda_graph",
        "memcheck",
        "racecheck",
    ]
    collected["tests/test_runtime.py::test_example"] = []
    artifacts = {
        "wheel": {"name": "candidate.whl", "sha256": "a" * 64},
        "sdist": {"name": "candidate.tar.gz", "sha256": "b" * 64},
    }

    def result(suite):
        selected = {
            node: marks
            for node, marks in collected.items()
            if (suite == "cpu") == (not marks)
        }
        return {
            **{
                key: plan[key]
                for key in (
                    "schema_version",
                    "source_tree_sha",
                    "validation_input_hash",
                    "policy_hash",
                )
            },
            "suite": suite,
            "paths": ["tests"],
            "environment": {"target": "cpu"} if suite == "cpu" else env,
            "wheel_sha256": None if suite == "cpu" else artifacts["wheel"]["sha256"],
            "collected": collected,
            "selected": selected,
            "reports": {
                node: {
                    phase: {"outcome": "passed", "seconds": 0, "reason": ""}
                    for phase in ("setup", "call", "teardown")
                }
                for node in selected
            },
            "exit_code": 0,
            "seconds": 0,
            "status": "passed",
        }

    first = {"target": "sm120", "status": "cached", "cache_key": "key"}
    row = {
        "torch_version": PRIMARY_TORCH,
        "environment": env,
        "cold_cache": False,
        "compile": {"first": first, "second": first, "native_targets": ["sm120"]},
        "suites": [result(kind) for kind in ("correctness", "memcheck", "racecheck")],
    }
    manifest = {
        **{
            key: plan[key]
            for key in (
                "schema_version",
                "source_tree_sha",
                "build_input_hash",
                "validation_input_hash",
                "policy_hash",
            )
        },
        "status": "passed",
        "mode": "incremental",
        "plan": plan,
        "artifacts": artifacts,
        "cpu": result("cpu"),
        "gpu": [row],
    }
    assert covers_plan(manifest, plan) and reuse_identity(manifest, plan)
    assert reusable_gpu_row(manifest, plan, PRIMARY_TORCH, env)
    assert not reusable_gpu_row(
        manifest, plan, PRIMARY_TORCH, {**env, "torch": "different"}
    )
    assert not covers_plan(manifest, {**plan, "test_paths": ["tests/new-owner"]})
    assert not covers_plan(manifest, {**plan, "source_tree_sha": "different"})
    assert not covers_plan(manifest, {**plan, "cold_cache": True})
    assert not covers_plan(manifest, {**plan, "torch_versions": list(TORCH_VERSIONS)})
    assert not reusable_gpu_row(manifest, full, PRIMARY_TORCH, env)
    for fault in (
        "deselected",
        "missing-report",
        "skipped",
        "failed-teardown",
        "wheel",
        "paths",
        "cache",
        "native-target",
        "duplicate-suite",
    ):
        invalid = copy.deepcopy(manifest)
        suite = invalid["gpu"][0]["suites"][0]
        node = next(iter(suite["selected"]))
        if fault == "deselected":
            suite["selected"].pop(node)
            suite["reports"].pop(node)
        elif fault == "missing-report":
            suite["reports"].pop(node)
        elif fault == "skipped":
            suite["reports"][node]["call"]["outcome"] = "skipped"
        elif fault == "failed-teardown":
            suite["reports"][node]["teardown"]["outcome"] = "failed"
        elif fault == "wheel":
            suite["wheel_sha256"] = "different"
        elif fault == "paths":
            suite["paths"] = ["tests/sampling"]
        elif fault == "cache":
            invalid["gpu"][0]["compile"]["second"]["status"] = "compiled"
        elif fault == "native-target":
            invalid["gpu"][0]["compile"]["native_targets"] = ["sm89"]
        else:
            invalid["gpu"][0]["suites"].append(copy.deepcopy(suite))
        assert not covers_plan(invalid, plan), fault
    for key in (
        "schema_version",
        "source_tree_sha",
        "build_input_hash",
        "validation_input_hash",
        "policy_hash",
    ):
        assert not covers_plan({**manifest, key: "different"}, plan), key
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        wheel = root / artifacts["wheel"]["name"]
        sdist = root / artifacts["sdist"]["name"]
        wheel.write_bytes(b"wheel")
        sdist.write_bytes(b"sdist")
        manifest["artifacts"] = package_artifacts(root)
        for suite in manifest["gpu"][0]["suites"]:
            suite["wheel_sha256"] = manifest["artifacts"]["wheel"]["sha256"]
        verify_manifest(manifest, plan, root)
        wheel.write_bytes(b"tampered")
        try:
            verify_manifest(manifest, plan, root)
        except ValueError:
            pass
        else:
            raise AssertionError("tampered wheel was accepted")
    # GPU skip decorators must remain visible to the CPU/GPU selection boundary.
    for path in (ROOT / "tests").rglob("test*.py"):
        text = path.read_text()
        tree = ast.parse(text)
        module_cuda = any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
            )
            and "pytest.mark.cuda" in ast.get_source_segment(text, node)
            for node in tree.body
        )
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                decorators = [
                    ast.get_source_segment(text, mark) for mark in node.decorator_list
                ]
                uses_gpu = any(
                    mark == "cuda"
                    or ("skipif" in mark and "torch.cuda.is_available()" in mark)
                    for mark in decorators
                )
                assert (
                    not uses_gpu or module_cuda or "pytest.mark.cuda" in decorators
                ), (path, node.name)
    print("quality_gate self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan")
    p.add_argument("--base", required=True)
    p.add_argument("--head", required=True)
    p.add_argument(
        "--mode", choices=("incremental", "full", "performance"), default="incremental"
    )
    p.add_argument("--extended", action="store_true")
    p.add_argument("--target", default="sm120", choices=("sm120", "sm89"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--github-output", type=Path)
    p = commands.add_parser("package")
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("environment")
    p.add_argument("--target", required=True)
    p = commands.add_parser("test")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument(
        "--suite",
        choices=("cpu", "correctness", "compatibility", "memcheck", "racecheck"),
        required=True,
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--package", type=Path)
    p.add_argument("--reuse", type=Path)
    p = commands.add_parser("gpu")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--reuse", type=Path)
    p = commands.add_parser("finalize")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--package", type=Path)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("verify")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--package", type=Path)
    p = commands.add_parser("resolve")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--repository", required=True)
    p.add_argument("--current-run", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--github-output", type=Path)
    p = commands.add_parser("release")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--repository", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("summary")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--repository", required=True)
    p.add_argument("--output", type=Path, required=True)
    commands.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "self-test":
        _self_test()
    elif args.command == "plan":
        plan = build_plan(
            args.base,
            args.head,
            mode=args.mode,
            extended=args.extended,
            target=args.target,
        )
        _write_json(args.output, plan)
        _write_outputs(
            args.github_output,
            {
                **plan["impact"],
                "source_tree_sha": plan["source_tree_sha"],
                "mode": plan["mode"],
                "extended": plan["extended"],
            },
        )
    elif args.command == "package":
        _write_json(args.output, verify_package(args.directory))
    elif args.command == "environment":
        print(json.dumps(environment(args.target)))
    elif args.command == "test":
        return run_tests(
            _read_json(args.plan),
            args.suite,
            args.output.resolve(),
            args.package,
            args.reuse,
        )
    elif args.command == "gpu":
        run_gpu(
            _read_json(args.plan),
            args.package.resolve(),
            args.output.resolve(),
            str(Path(args.python).absolute()),
            args.reuse,
        )
    elif args.command == "finalize":
        _write_json(
            args.output, finalize(_read_json(args.plan), args.package, args.validation)
        )
    elif args.command == "verify":
        verify_manifest(_read_json(args.manifest), _read_json(args.plan), args.package)
    elif args.command == "resolve":
        result = resolve_reuse(
            _read_json(args.plan),
            args.repository,
            os.environ["GH_TOKEN"],
            args.current_run,
        )
        _write_json(args.output, result)
        _write_outputs(args.github_output, result)
    elif args.command == "release":
        release_payload(
            _read_json(args.plan), args.bundle, args.repository, args.tag, args.output
        )
    elif args.command == "summary":
        validation_summary(_read_json(args.manifest), args.repository, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
