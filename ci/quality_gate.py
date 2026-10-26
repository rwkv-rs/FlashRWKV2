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
from pathlib import Path

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
SCHEMA_VERSION = 6
PRIMARY_TORCH = "2.14.1"
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
    scheduled=False,
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
        "scheduled": scheduled,
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
    if not wheel.name.endswith("-py3-none-any.whl"):
        raise ValueError("expected a universal source wheel")
    python_sources = {
        p.relative_to(ROOT).as_posix(): p.read_bytes()
        for p in (ROOT / "flashrwkv2").rglob("*.py")
    }
    native_sources = {
        p.relative_to(ROOT / "csrc").as_posix(): p.read_bytes()
        for p in (ROOT / "csrc").rglob("*")
        if p.is_file()
    }
    with zipfile.ZipFile(wheel) as archive:
        expected = {
            **python_sources,
            **{
                f"flashrwkv2/_csrc/{name}": data
                for name, data in native_sources.items()
            },
        }
        names = archive.namelist()
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
    if actual_target != target:
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


class TestRecorder:
    """Record the collection and every execution phase in the same pytest process."""

    def __init__(self):
        self.selected = {}
        self.reports = {}

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
    selected, reports = result["selected"], result["reports"]
    if not selected:
        if allow_empty and result["exit_code"] == 5 and not reports:
            return
        raise ValueError("required test selection is empty")
    if result["exit_code"] != 0 or set(selected) != set(reports):
        raise ValueError("test execution failed or selected tests were not executed")
    for nodeid, phases in reports.items():
        allowed_skip = (
            result["environment"].get("target") == "sm89"
            and nodeid == "tests/sampling/test.py::test_sampling_resource_usage"
            and "resource gate requires the SM120 backend"
            in phases.get("call", {}).get("reason", "")
        )
        if allowed_skip and all(
            phase["outcome"] != "failed" for phase in phases.values()
        ):
            continue
        if set(phases) != {"setup", "call", "teardown"} or any(
            phase["outcome"] != "passed" for phase in phases.values()
        ):
            raise ValueError(f"test did not complete successfully: {nodeid}")


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
            (reuse_identity(previous, plan) or scheduled_equivalent(previous, plan))
            and previous.get("cpu")
            and previous["cpu"]["environment"] == env
        ):
            _write_json(output, previous["cpu"])
            print(
                "Reusing CPU results for identical source, policy and environment",
                flush=True,
            )
            return 0
    marker = {
        "cpu": "not cuda and not resource",
        "correctness": "cuda or resource",
        "memcheck": "cuda and memcheck",
        "racecheck": "cuda and racecheck",
        "compatibility": "cuda",
    }[suite]
    paths = ["tests"] if suite == "cpu" else plan["test_paths"]
    if suite == "compatibility":
        paths = list(COMPATIBILITY_PATHS)
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
        marker,
        f"--junitxml={output.with_suffix('.xml')}",
        *[str(ROOT / path) for path in paths],
    ]
    if suite == "compatibility":
        args += ["-k", COMPATIBILITY_KEYWORDS]
    recorder = TestRecorder()
    started = time.monotonic()
    code = int(pytest.main(args, plugins=[recorder]))
    result = {
        "schema_version": SCHEMA_VERSION,
        "plan_hash": plan["plan_hash"],
        "source_tree_sha": plan["source_tree_sha"],
        "suite": suite,
        "environment": env,
        "wheel_sha256": package_artifacts(package)["wheel"]["sha256"]
        if package
        else None,
        "selected": recorder.selected,
        "reports": recorder.reports,
        "exit_code": code,
        "seconds": time.monotonic() - started,
        "status": "failed",
    }
    try:
        validate_test_result(result, allow_empty=suite in {"memcheck", "racecheck"})
        result["status"] = "passed" if recorder.selected else "not-required"
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
            targets = {
                word.strip(".:")
                for word in listing.replace(".", " ").split()
                if word.startswith("sm_")
            }
            if f"sm_{plan['target'][2:]}" not in listing or any(
                f"sm_{cc}" in listing
                for cc in (80, 89, 90, 120)
                if f"sm{cc}" != plan["target"]
            ):
                raise ValueError(
                    f"runtime library does not exclusively target {plan['target']}: {targets}"
                )
            _write_json(current / "compile.json", {"first": first, "second": second})
        compile_seconds = time.monotonic() - compile_started
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
                "compile": {
                    "status": "passed",
                    "cache_key": second["cache_key"],
                    "seconds": compile_seconds,
                },
                "suites": suite_results,
            }
        )
    result = {"gpu": reports}
    if previous and scheduled_equivalent(previous, plan):
        result["reference"] = previous
    _write_json(output / "gpu.json", result)


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
        if row["compile"]["status"] != "passed" or (
            plan["cold_cache"] and not row["cold_cache"]
        ):
            raise ValueError("runtime build validation is missing")
        suites = {suite["suite"]: suite for suite in row["suites"]}
        required = {"correctness"} if version == PRIMARY_TORCH else {"compatibility"}
        if version == PRIMARY_TORCH and plan["impact"]["run_sanitizer"]:
            required |= {"memcheck", "racecheck"}
        if not required <= suites.keys():
            raise ValueError("required GPU suite is missing")
        for name in required:
            result = suites[name]
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
        validate_test_result(cpu)
    gpu_payload = (
        _read_json(validation / "gpu.json")
        if plan["impact"]["run_gpu"]
        else {"gpu": []}
    )
    gpu = gpu_payload["gpu"]
    reference = gpu_payload.get("reference")
    unchanged = reference is not None and scheduled_equivalent(reference, plan)
    if unchanged:
        _validate_scheduled(reference, plan, artifacts, cpu, gpu)
    else:
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
        "status": "unchanged" if unchanged else "passed",
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
        "reference": reference if unchanged else None,
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
        and covers_plan(manifest, manifest["plan"])
    )


def scheduled_equivalent(manifest: dict, plan: dict) -> bool:
    # A scheduled no-op refers to one complete prior run. It never creates
    # release evidence for a different Git tree, or an inherited module chain.
    return bool(
        plan.get("scheduled")
        and plan["mode"] == "full"
        and all(
            manifest.get(key) == plan[key]
            for key in ("build_input_hash", "validation_input_hash", "policy_hash")
        )
        and covers_plan(
            manifest, {**plan, "source_tree_sha": manifest.get("source_tree_sha")}
        )
    )


def _validate_scheduled(
    reference: dict, plan: dict, artifacts: dict, cpu: dict, gpu: list[dict]
) -> None:
    if not scheduled_equivalent(reference, plan) or artifacts != reference["artifacts"]:
        raise ValueError("scheduled no-op does not match a complete prior validation")
    allowed = {plan["source_tree_sha"], reference["source_tree_sha"]}
    validate_test_result(cpu)
    if cpu["source_tree_sha"] not in allowed:
        raise ValueError("scheduled CPU result identity mismatch")
    if sorted(row["torch_version"] for row in gpu) != sorted(plan["torch_versions"]):
        raise ValueError("scheduled GPU version coverage mismatch")
    for row in gpu:
        trees = {suite["source_tree_sha"] for suite in row["suites"]}
        if len(trees) != 1 or not trees <= allowed:
            raise ValueError("scheduled GPU result identity mismatch")
        _validate_gpu(
            [row],
            {
                **plan,
                "source_tree_sha": trees.pop(),
                "torch_versions": [row["torch_version"]],
            },
            artifacts,
        )


def reusable_gpu_row(manifest: dict, plan: dict, version: str, env: dict) -> bool:
    equivalent = scheduled_equivalent(manifest, plan)
    if not (reuse_identity(manifest, plan) or equivalent):
        return False
    if equivalent:
        plan = {**plan, "source_tree_sha": manifest["source_tree_sha"]}
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
    ):
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
            validate_test_result(manifest["cpu"])
            if manifest["cpu"]["source_tree_sha"] != plan["source_tree_sha"]:
                return False
        _validate_gpu(manifest.get("gpu", []), plan, manifest.get("artifacts", {}))
    except (KeyError, TypeError, ValueError):
        return False
    return True


def verify_manifest(manifest: dict, plan: dict, package: Path | None) -> None:
    if manifest.get("status") == "unchanged":
        if manifest["source_tree_sha"] != plan["source_tree_sha"]:
            raise ValueError("scheduled source identity mismatch")
        _validate_scheduled(
            manifest["reference"],
            plan,
            manifest["artifacts"],
            manifest["cpu"],
            manifest["gpu"],
        )
    elif not covers_plan(manifest, plan):
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
            if not reuse_identity(manifest, plan):
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
    if plan.get("scheduled"):
        recent = _api_json("/actions/artifacts?per_page=100", repository, token).get(
            "artifacts", []
        )
        for artifact in recent:
            if artifact["expired"] or not artifact["name"].startswith(
                "flashrwkv2-quality-v6-"
            ):
                continue
            run_id = artifact["workflow_run"]["id"]
            if run_id == current_run:
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
            with zipfile.ZipFile(
                io.BytesIO(_artifact_bytes(artifact, token))
            ) as archive:
                manifest = json.loads(archive.read("quality-manifest.json"))
                if scheduled_equivalent(manifest, plan):
                    return {
                        "reuse": True,
                        "complete": False,
                        "run_id": str(run_id),
                        "artifact": artifact["name"],
                    }
    return {"reuse": False, "complete": False, "run_id": "", "artifact": name}


def _performance_covers(result: dict, plan: dict, artifacts: dict, target: str) -> bool:
    return (
        result.get("source_tree_sha") == plan["source_tree_sha"]
        and result.get("wheel_sha256") == artifacts["wheel"]["sha256"]
        and result.get("profile") == "full"
        and sorted(result.get("completed_modules", [])) == sorted(TARGETS)
        and result.get("status") == "completed"
        and result.get("environment", {}).get("capability")
        == [int(target[2:-1]), int(target[-1])]
    )


def release_payload(
    plan: dict, bundle: Path, repository: str, tag: str, output: Path
) -> None:
    manifest = _read_json(bundle / "quality-manifest.json")
    verify_manifest(manifest, plan, bundle)
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
    ):
        raise ValueError("complete RTX4090 performance acceptance is missing")
    name = f"flashrwkv2-performance-v6-{plan['source_tree_sha']}"
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
            if _performance_covers(candidate, plan, manifest["artifacts"], "sm120"):
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
    import tomllib

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
        if manifest.get("status") == "unchanged":
            handle.write(
                f"Validation inputs and package are unchanged; reused full run for `{manifest['reference']['source_tree_sha']}` after checking the active environment.\n\n"
            )
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
    from unittest.mock import patch

    head = _git("rev-parse", "HEAD")
    with patch(
        __name__ + "._git_changed_files",
        return_value=["flashrwkv2/post_norm/tilelang.py"],
    ):
        plan = build_plan(head, head)
    assert not plan["impact"]["run_all"] and BLOCK_TEST in plan["test_paths"]
    assert set(plan["impact"]["affected_modules"]) == {
        "post_norm",
        "cmix",
        "tmix/tokenshift",
    }
    with patch(__name__ + "._git_changed_files", return_value=["README.md"]):
        doc = build_plan(head, head)
    assert not doc["impact"]["run_gpu"] and doc["impact"]["run_package"]
    full = build_plan(head, head, mode="full", extended=True)
    assert full["torch_versions"] == list(TORCH_VERSIONS) and full["test_paths"] == [
        "tests"
    ]
    sample = {
        "selected": {"test::one": ["cuda"]},
        "reports": {
            "test::one": {
                phase: {"outcome": "passed"} for phase in ("setup", "call", "teardown")
            }
        },
        "environment": {"target": "sm120"},
        "exit_code": 0,
    }
    validate_test_result(sample)
    invalid = [
        {**sample, "selected": {}},
        {**sample, "reports": {}},
        {**sample, "exit_code": 1},
        {**sample, "reports": {"test::one": {"setup": {"outcome": "skipped"}}}},
        {
            **sample,
            "reports": {
                "test::one": {
                    "setup": {"outcome": "passed"},
                    "call": {"outcome": "passed"},
                    "teardown": {"outcome": "failed"},
                }
            },
        },
    ]
    for result in invalid:
        try:
            validate_test_result(result)
        except ValueError:
            pass
        else:
            raise AssertionError("incomplete test evidence was accepted")
    assert not covers_plan({"schema_version": 5, "status": "passed"}, plan)
    assert not covers_plan({}, plan)
    # Exercise real acceptance boundaries: a passed flag is insufficient without
    # the selected executions, original wheel, environment and requested scope.
    import copy

    env = {
        "target": "sm120",
        "torch": PRIMARY_TORCH,
        "package_path": "/venv/flashrwkv2/__init__.py",
    }
    nodes = {
        f"tests/{owner}/test.py::test_example": ["cuda"]
        for owner in plan["impact"]["affected_modules"]
    }
    nodes[BLOCK_TEST + "::test_example"] = ["cuda", "cuda_graph"]
    completed = {
        nodeid: copy.deepcopy(sample["reports"]["test::one"]) for nodeid in nodes
    }
    artifacts = {
        "wheel": {"name": "candidate.whl", "sha256": "a" * 64},
        "sdist": {"name": "candidate.tar.gz", "sha256": "b" * 64},
    }
    suites = [
        {
            **sample,
            "suite": kind,
            "source_tree_sha": plan["source_tree_sha"],
            "wheel_sha256": artifacts["wheel"]["sha256"],
            "environment": env,
            "selected": nodes,
            "reports": completed,
        }
        for kind in ("correctness", "memcheck", "racecheck")
    ]
    cpu = {**sample, "source_tree_sha": plan["source_tree_sha"]}
    manifest = {
        **{
            key: plan[key]
            for key in (
                "source_tree_sha",
                "build_input_hash",
                "validation_input_hash",
                "policy_hash",
            )
        },
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "mode": "incremental",
        "plan": plan,
        "cpu": cpu,
        "artifacts": artifacts,
        "gpu": [
            {
                "torch_version": PRIMARY_TORCH,
                "environment": env,
                "compile": {"status": "passed"},
                "cold_cache": False,
                "suites": suites,
            }
        ],
    }
    assert covers_plan(manifest, plan)
    assert reuse_identity(manifest, plan)
    assert reusable_gpu_row(manifest, plan, PRIMARY_TORCH, env)
    assert not reusable_gpu_row(
        manifest, plan, PRIMARY_TORCH, {**env, "torch": "different"}
    )
    extended = {**plan, "torch_versions": list(TORCH_VERSIONS), "cold_cache": True}
    assert not covers_plan(manifest, extended)
    assert reuse_identity(manifest, extended)
    assert reusable_gpu_row(manifest, extended, PRIMARY_TORCH, env)
    assert not reusable_gpu_row(manifest, full, PRIMARY_TORCH, env)
    full_plan = {
        **plan,
        "mode": "full",
        "test_paths": ["tests"],
        "impact": {
            **plan["impact"],
            "run_all": True,
            "affected_cuda_modules": list(TARGETS),
        },
    }
    full_manifest = copy.deepcopy(manifest)
    full_manifest.update(mode="full", plan=full_plan)
    full_nodes = {f"tests/{owner}/test.py::test_example": ["cuda"] for owner in TARGETS}
    full_nodes[BLOCK_TEST + "::test_example"] = ["cuda", "cuda_graph"]
    for suite in full_manifest["gpu"][0]["suites"]:
        suite["selected"] = full_nodes
        suite["reports"] = {
            node: copy.deepcopy(sample["reports"]["test::one"]) for node in full_nodes
        }
    assert covers_plan(full_manifest, full_plan)
    scheduled = {**full_plan, "scheduled": True, "source_tree_sha": "d" * 40}
    assert scheduled_equivalent(full_manifest, scheduled)
    assert not reuse_identity(full_manifest, scheduled)
    assert reusable_gpu_row(full_manifest, scheduled, PRIMARY_TORCH, env)
    _validate_scheduled(
        full_manifest, scheduled, artifacts, full_manifest["cpu"], full_manifest["gpu"]
    )
    assert not scheduled_equivalent(full_manifest, {**scheduled, "scheduled": False})
    assert not scheduled_equivalent(
        full_manifest, {**scheduled, "validation_input_hash": "changed"}
    )
    unchanged = {
        **full_manifest,
        "status": "unchanged",
        "source_tree_sha": scheduled["source_tree_sha"],
        "plan": scheduled,
        "reference": full_manifest,
    }
    assert not covers_plan(unchanged, scheduled)
    assert not scheduled_equivalent(unchanged, scheduled)
    try:
        verify_manifest(unchanged, {**scheduled, "scheduled": False}, None)
    except ValueError:
        pass
    else:
        raise AssertionError("scheduled no-op was accepted as release evidence")
    for key in (
        "source_tree_sha",
        "build_input_hash",
        "validation_input_hash",
        "policy_hash",
    ):
        assert not covers_plan({**manifest, key: "mismatch"}, plan), key
    for fault in (
        "wheel",
        "environment",
        "missing-sanitizer",
        "missing-test",
        "skipped-test",
    ):
        invalid = copy.deepcopy(manifest)
        result = invalid["gpu"][0]["suites"][0]
        if fault == "wheel":
            result["wheel_sha256"] = "mismatch"
        elif fault == "environment":
            result["environment"]["torch"] = "mismatch"
        elif fault == "missing-sanitizer":
            invalid["gpu"][0]["suites"].pop()
        elif fault == "missing-test":
            result["reports"].pop(next(iter(result["reports"])))
        else:
            result["reports"][next(iter(result["reports"]))]["call"]["outcome"] = (
                "skipped"
            )
        assert not covers_plan(invalid, plan), fault
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
            raise AssertionError("tampered source distribution was accepted")
    performance = {
        "status": "completed",
        "profile": "full",
        "source_tree_sha": plan["source_tree_sha"],
        "wheel_sha256": manifest["artifacts"]["wheel"]["sha256"],
        "completed_modules": list(TARGETS),
        "environment": {"capability": [12, 0]},
    }
    assert _performance_covers(performance, plan, manifest["artifacts"], "sm120")
    assert not _performance_covers(
        {**performance, "completed_modules": ["cmix"]},
        plan,
        manifest["artifacts"],
        "sm120",
    )
    assert not _performance_covers(performance, plan, manifest["artifacts"], "sm89")
    # Prevent an unmarked CUDA test from silently landing in CPU-only CI.
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
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith(
                "test_"
            ):
                continue
            decorators = [
                ast.get_source_segment(text, mark) for mark in node.decorator_list
            ]
            uses_gpu = any(
                mark == "cuda"
                or ("skipif" in mark and "torch.cuda.is_available()" in mark)
                for mark in decorators
            )
            assert not uses_gpu or module_cuda or "pytest.mark.cuda" in decorators, (
                path,
                node.name,
            )
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        text = path.read_text()
        assert "timeout-minutes:" in text and "--clear" not in text
    workflow = (ROOT / ".github/workflows/pro6000-gpu.yml").read_text()
    assert "name: Quality Gate" in workflow and "queue: max" in workflow
    assert "17 8 * * *" in workflow and "--inherit-manifest" not in workflow
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
    p.add_argument("--scheduled", action="store_true")
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
            scheduled=args.scheduled,
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
