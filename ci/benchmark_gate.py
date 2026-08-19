"""Run affected benchmarks once and report advisory regressions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(
    os.environ.get("FLASH_RWKV_SOURCE_ROOT", Path(__file__).resolve().parents[1])
).resolve()
SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    workload: str
    path: str
    arguments: tuple[str, ...] = ()
    output_argument: str | None = None


BENCHMARKS = {
    "cmix": (
        BenchmarkCase("infer", "benchmarks/cmix/bench.py", ("--operator", "infer")),
        BenchmarkCase(
            "pretrain_forward_backward",
            "benchmarks/cmix/bench.py",
            ("--operator", "pretrain", "--backward"),
        ),
        BenchmarkCase(
            "statetune_forward_backward",
            "benchmarks/cmix/bench.py",
            ("--operator", "stateful", "--backward"),
        ),
    ),
    "embedding": (BenchmarkCase("infer", "benchmarks/embedding/bench.py"),),
    "head/l2wrap_ce": (
        BenchmarkCase("pretrain", "benchmarks/head/l2wrap_ce/bench.py"),
    ),
    "head/linear": (BenchmarkCase("infer", "benchmarks/head/linear/bench.py"),),
    "loss/l2wrap_ce": (
        BenchmarkCase(
            "pretrain_forward_backward", "benchmarks/loss/l2wrap_ce/bench.py"
        ),
    ),
    "sampling": (BenchmarkCase("infer", "benchmarks/sampling/bench.py"),),
    "tmix/a_gate": (
        BenchmarkCase("pretrain", "benchmarks/tmix/a_gate/bench.py"),
    ),
    "tmix/kk_pre": (
        BenchmarkCase("pretrain", "benchmarks/tmix/kk_pre/bench.py"),
    ),
    "tmix/readout": (
        BenchmarkCase(
            "infer", "benchmarks/tmix/readout/bench.py", ("--operator", "infer")
        ),
        BenchmarkCase(
            "pretrain_forward_backward",
            "benchmarks/tmix/readout/bench.py",
            ("--operator", "pretrain", "--backward"),
        ),
    ),
    "tmix/tokenshift": (
        BenchmarkCase("infer", "benchmarks/tmix/tokenshift/bench.py"),
        BenchmarkCase(
            "pretrain_forward_backward",
            "benchmarks/tmix/tokenshift/pretrain/bench.py",
            ("--operator", "pretrain", "--backward"),
        ),
        BenchmarkCase(
            "statetune_forward_backward",
            "benchmarks/tmix/tokenshift/pretrain/bench.py",
            ("--operator", "stateful", "--backward"),
        ),
    ),
    "post_norm": (BenchmarkCase("infer", "benchmarks/post_norm/bench.py"),),
    "tmix/vres_gate": (
        BenchmarkCase("pretrain", "benchmarks/tmix/vres_gate/bench.py"),
    ),
    "tmix/wkv_prepare": (
        BenchmarkCase("infer", "benchmarks/tmix/wkv_prepare/bench.py"),
    ),
    "tmix/wkv7": (
        BenchmarkCase(
            "infer_recurrent_fp32io16",
            "benchmarks/tmix/wkv7/bench.py",
            ("--shapes", "h32d64", "--dtype", "bfloat16", "--seed", "20260804"),
            "--output",
        ),
        BenchmarkCase("infer_chunk_bf16", "benchmarks/tmix/wkv7/chunk/bench.py"),
        BenchmarkCase(
            "pretrain_forward_backward", "benchmarks/tmix/wkv7/pretrain/bench.py"
        ),
        BenchmarkCase(
            "statetune", "benchmarks/tmix/wkv7/statetune/bench.py"
        ),
    ),
    "tmix/wkv7/rl_infctx": (
        BenchmarkCase(
            "forward_materialized",
            "benchmarks/tmix/wkv7/rl_infctx/bench.py",
            ("--stage", "forward", "--strategy", "materialized"),
        ),
        BenchmarkCase(
            "forward_recompute",
            "benchmarks/tmix/wkv7/rl_infctx/bench.py",
            ("--stage", "forward", "--strategy", "recompute"),
        ),
        BenchmarkCase(
            "backward_replay",
            "benchmarks/tmix/wkv7/rl_infctx/bench.py",
            ("--stage", "backward_replay"),
        ),
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_executable(python: str) -> str:
    executable = shutil.which(python)
    if executable is None:
        raise SystemExit(f"benchmark Python executable not found: {python}")
    return str(Path(executable).absolute())


def _environment(python: str) -> dict[str, Any]:
    program = r"""
import json, platform, torch
from pathlib import Path
extension = getattr(__import__('flashrwkv2'), '_C', None)
extension_path = Path(getattr(extension, '__file__', ''))
print(json.dumps({
  'python': platform.python_version(),
  'torch': torch.__version__,
  'torch_cuda': torch.version.cuda,
  'gpu': torch.cuda.get_device_name(0),
  'capability': list(torch.cuda.get_device_capability(0)),
  'extension_path': str(extension_path),
}))
"""
    payload = json.loads(
        subprocess.run(
            (_python_executable(python), "-c", program),
            cwd=tempfile.gettempdir(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    extension_path = Path(payload.pop("extension_path"))
    if not extension_path.is_file():
        raise SystemExit("flashrwkv2._C is not loaded from an installed wheel")
    payload["extension_sha256"] = _sha256(extension_path)
    payload["wheel_sha256"] = os.environ.get("FLASH_RWKV_WHEEL_SHA256")
    query = "driver_version,temperature.gpu,power.draw,clocks.current.sm,clocks_event_reasons.active"
    result = subprocess.run(
        (
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
            "--id=0",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    values = [value.strip() for value in result.stdout.strip().split(",")]
    payload["driver"] = values[0] if values else None
    payload["nvidia_smi"] = result.stdout.strip()
    payload["throttle_reason"] = values[4] if len(values) > 4 else None
    return payload


def _assert_gpu_idle() -> None:
    result = subprocess.run(
        (
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
            "--id=0",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    pids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if pids:
        raise SystemExit(f"benchmark GPU is not idle; active compute PIDs: {pids}")


def _last_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeError("benchmark did not emit a JSON object")


def _contract_hash(module: str) -> str:
    digest = hashlib.sha256()
    for case in BENCHMARKS[module]:
        digest.update(case.workload.encode())
        digest.update(b"\0")
        digest.update(json.dumps(case.arguments, separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(str(case.output_argument).encode())
        digest.update(b"\0")
        path = ROOT / case.path
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    timing = ROOT / "benchmarks/_timing.py"
    digest.update(str(timing.relative_to(ROOT)).encode())
    digest.update(b"\0")
    digest.update(timing.read_bytes())
    digest.update(b"\0")
    return digest.hexdigest()


def _positive(values: list[float], context: str) -> list[float]:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise RuntimeError(f"{context} emitted invalid latency values")
    return values


def _profiles(payload: dict[str, Any], module: str) -> list[dict[str, Any]]:
    if isinstance(payload.get("profiles"), list):
        profiles = []
        for row in payload["profiles"]:
            raw = _positive(
                [float(value) for value in row["raw_batch_mean_us"]],
                f"{module}/{row.get('profile')}",
            )
            profiles.append(
                {
                    "profile": str(row["profile"]),
                    "unit": "us",
                    "direction": "lower",
                    "mean": float(row.get("mean_us", sum(raw) / len(raw))),
                    "raw_batch_mean_us": raw,
                    "total_launches": int(row.get("total_launches", 10000)),
                }
            )
        if profiles:
            return profiles

    if module == "sampling":
        profiles = []
        for row in payload.get("results", []):
            raw = _positive(
                [float(value) for value in row.get("raw_batch_mean_us", ())],
                f"sampling/{row.get('profile')}",
            )
            profiles.append(
                {
                    "profile": str(row["profile"]),
                    "unit": "us",
                    "direction": "lower",
                    "mean": float(row.get("mean_us", sum(raw) / len(raw))),
                    "raw_batch_mean_us": raw,
                    "total_launches": int(row.get("total_launches", 10000)),
                }
            )
        if profiles:
            return profiles

    if module == "tmix/wkv7":
        profiles = []
        for row in payload.get("results", []):
            timing = row.get("timing")
            if not isinstance(timing, dict):
                raise TypeError(f"WKV7 profile lacks timing: {row}")
            raw = _positive(
                [float(value) for value in timing["raw_batch_mean_us"]],
                "tmix/wkv7",
            )
            profiles.append(
                {
                    "profile": "/".join(
                        str(row[key])
                        for key in ("operator_shape", "case", "token_dtype")
                    ),
                    "unit": "us",
                    "direction": "lower",
                    "mean": float(timing["mean_us"]),
                    "raw_batch_mean_us": raw,
                    "total_launches": int(timing["total_launches"]),
                }
            )
        if profiles:
            return profiles

    timing = payload.get("timing")
    if isinstance(timing, dict):
        raw = _positive(
            [float(value) for value in timing.get("raw_batch_mean_us", ())],
            module,
        )
        return [
            {
                "profile": str(payload.get("profile", payload.get("operator", "ci"))),
                "unit": "us",
                "direction": "lower",
                "mean": float(timing.get("mean_us", sum(raw) / len(raw))),
                "raw_batch_mean_us": raw,
                "total_launches": int(timing.get("total_launches", 10000)),
            }
        ]

    raw_value = payload.get(
        "raw_batch_mean_us",
        payload.get("raw_latency_us", payload.get("latency_us")),
    )
    if not isinstance(raw_value, list):
        raise TypeError(f"{module} benchmark lacks latency samples")
    raw = _positive([float(value) for value in raw_value], module)
    return [
        {
            "profile": str(payload.get("profile", "ci")),
            "unit": "us",
            "direction": "lower",
            "mean": float(payload.get("mean_us", sum(raw) / len(raw))),
            "raw_batch_mean_us": raw,
            "total_launches": int(payload.get("total_launches", 10000)),
        }
    ]


def _run_case(
    module: str,
    case: BenchmarkCase,
    *,
    python: str,
) -> list[dict[str, Any]]:
    script = ROOT / case.path
    command = [_python_executable(python), str(script), *case.arguments]
    output_path: Path | None = None
    if case.output_argument is not None:
        output_path = Path(tempfile.gettempdir()) / (
            f"flashrwkv2-{module.replace('/', '-')}-{case.workload}-{os.getpid()}.json"
        )
        command.extend((case.output_argument, str(output_path)))
    environment_variables = os.environ.copy()
    environment_variables["PYTHONPATH"] = str(ROOT / "benchmarks")
    result = subprocess.run(
        command,
        cwd=tempfile.gettempdir(),
        env=environment_variables,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = (
        json.loads(output_path.read_text(encoding="utf-8"))
        if output_path is not None
        else _last_json(result.stdout)
    )
    if output_path is not None:
        output_path.unlink(missing_ok=True)
    profiles = _profiles(payload, module)
    for profile in profiles:
        profile["profile"] = f"{case.workload}/{profile['profile']}"
    return profiles


def run_module(
    module: str,
    *,
    python: str,
    revision: str,
    status: str,
    environment: dict[str, Any],
) -> dict[str, Any]:
    profiles = []
    for case in BENCHMARKS[module]:
        profiles.extend(_run_case(module, case, python=python))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "module": module,
        "target": "sm120",
        "revision": revision,
        "benchmark_contract_sha256": _contract_hash(module),
        "environment": environment,
        "profiles": profiles,
    }


def run_suite(
    modules: list[str],
    *,
    python: str,
    revision: str,
    status: str,
    output: Path,
) -> dict[str, Any]:
    unknown = sorted(set(modules) - BENCHMARKS.keys())
    if unknown:
        raise SystemExit(f"no canonical benchmark configured for: {unknown}")
    _assert_gpu_idle()
    environment = _environment(python)
    output.mkdir(parents=True, exist_ok=True)
    warnings: list[dict[str, Any]] = []
    completed: list[str] = []
    for module in modules:
        safe = module.replace("/", "-")
        try:
            payload = run_module(
                module,
                python=python,
                revision=revision,
                status=status,
                environment=environment,
            )
        except (
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            subprocess.CalledProcessError,
        ) as error:  # Benchmark failures are advisory by design.
            warning = {
                "kind": "benchmark-unavailable",
                "module": module,
                "message": f"{type(error).__name__}: {error}",
            }
            warnings.append(warning)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "status": "unavailable",
                "module": module,
                "target": "sm120",
                "revision": revision,
                "warning": warning,
            }
        else:
            completed.append(module)
        (output / f"{safe}-{status}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed-with-warnings" if warnings else "completed",
        "modules": modules,
        "completed_modules": completed,
        "warnings": warnings,
    }


def _compatible(baseline: dict[str, Any], candidate: dict[str, Any]) -> None:
    for key in ("schema_version", "module", "target", "benchmark_contract_sha256"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"baseline {key} mismatch")
    for key in ("gpu", "capability", "torch", "torch_cuda", "driver"):
        if baseline["environment"].get(key) != candidate["environment"].get(key):
            raise ValueError(f"baseline environment mismatch for {key}")


def compare_module(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> list[dict[str, Any]]:
    _compatible(baseline, candidate)
    baseline_profiles = {row["profile"]: row for row in baseline["profiles"]}
    candidate_profiles = {row["profile"]: row for row in candidate["profiles"]}
    if baseline_profiles.keys() != candidate_profiles.keys():
        raise ValueError("baseline/candidate profile sets differ")
    warnings = []
    for profile in sorted(candidate_profiles):
        base_mean = float(baseline_profiles[profile]["mean"])
        candidate_mean = float(candidate_profiles[profile]["mean"])
        if candidate_mean > base_mean:
            warnings.append(
                {
                    "kind": "performance-regression",
                    "module": candidate["module"],
                    "profile": profile,
                    "baseline_mean_us": base_mean,
                    "candidate_mean_us": candidate_mean,
                    "absolute_us": candidate_mean - base_mean,
                    "percent": (candidate_mean - base_mean) / base_mean * 100.0,
                    "baseline_wheel_sha256": baseline["environment"].get(
                        "wheel_sha256"
                    ),
                    "candidate_wheel_sha256": candidate["environment"].get(
                        "wheel_sha256"
                    ),
                }
            )
    return warnings


def compare_suite(
    modules: list[str], baseline_dir: Path, candidate_dir: Path
) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    for module in modules:
        safe = module.replace("/", "-")
        candidate_path = candidate_dir / f"{safe}-candidate.json"
        baseline_path = baseline_dir / f"{safe}-baseline.json"
        if not candidate_path.is_file():
            warnings.append(
                {"kind": "benchmark-unavailable", "module": module, "message": "candidate missing"}
            )
            continue
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        if candidate.get("status") == "unavailable":
            warnings.append(candidate["warning"])
            continue
        if not baseline_path.is_file():
            warnings.append(
                {"kind": "baseline-unavailable", "module": module}
            )
            continue
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline.get("status") == "unavailable":
            warnings.append(
                {"kind": "baseline-unavailable", "module": module}
            )
            continue
        try:
            warnings.extend(compare_module(baseline, candidate))
        except (KeyError, TypeError, ValueError) as error:
            warnings.append(
                {
                    "kind": "baseline-incompatible",
                    "module": module,
                    "message": str(error),
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "warning" if warnings else "clean",
        "modules": modules,
        "warnings": warnings,
    }


def _api_json(url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def fetch_main_baselines(
    repository: str,
    token: str,
    modules: list[str],
    output: Path,
    current_environment: dict[str, Any],
) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    runs_url = (
        f"https://api.github.com/repos/{repository}/actions/workflows/"
        "pro6000-gpu.yml/runs?branch=main&status=success&per_page=20"
    )
    runs = _api_json(runs_url, token).get("workflow_runs", [])
    wanted = {module.replace("/", "-"): module for module in modules}
    found: set[str] = set()
    for run in runs:
        if run.get("name") != "Quality Gate":
            continue
        artifacts_url = (
            f"https://api.github.com/repos/{repository}/actions/runs/"
            f"{run['id']}/artifacts?per_page=100"
        )
        artifacts = _api_json(artifacts_url, token).get("artifacts", [])
        evidence = next(
            (
                row
                for row in artifacts
                if row.get("name", "").startswith("flashrwkv2-quality-v2-")
                and not row.get("expired")
            ),
            None,
        )
        if evidence is None:
            continue
        request = urllib.request.Request(
            evidence["archive_download_url"],
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(request) as response, tempfile.NamedTemporaryFile() as archive:
            archive.write(response.read())
            archive.flush()
            with zipfile.ZipFile(archive.name) as bundle:
                names = set(bundle.namelist())
                for safe, module in wanted.items():
                    if module in found:
                        continue
                    candidates = [
                        name
                        for name in names
                        if name.endswith(f"/{safe}-candidate.json")
                        or name == f"{safe}-candidate.json"
                    ]
                    if len(candidates) != 1:
                        continue
                    payload = json.loads(bundle.read(candidates[0]))
                    if payload.get("status") != "candidate":
                        continue
                    if payload.get("benchmark_contract_sha256") != _contract_hash(
                        module
                    ):
                        continue
                    if any(
                        payload.get("environment", {}).get(key)
                        != current_environment.get(key)
                        for key in (
                            "gpu",
                            "capability",
                            "torch",
                            "torch_cuda",
                            "driver",
                        )
                    ):
                        continue
                    payload["status"] = "baseline"
                    (output / f"{safe}-baseline.json").write_text(
                        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
                    )
                    found.add(module)
        if len(found) == len(modules):
            break
    return [module for module in modules if module not in found]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _parse_modules(value: str) -> list[str]:
    modules = json.loads(value)
    if not isinstance(modules, list) or not all(
        isinstance(module, str) for module in modules
    ):
        raise SystemExit("--modules must be a JSON string list")
    return modules


def _self_test() -> None:
    environment = {
        "gpu": "gpu",
        "capability": [12, 0],
        "torch": "torch",
        "torch_cuda": "cuda",
        "driver": "driver",
        "wheel_sha256": "wheel",
    }

    def payload(value: float) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "candidate",
            "module": "cmix",
            "target": "sm120",
            "benchmark_contract_sha256": "contract",
            "environment": environment,
            "profiles": [
                {
                    "profile": "ci",
                    "unit": "us",
                    "direction": "lower",
                    "mean": value,
                    "raw_batch_mean_us": [value] * 10,
                    "total_launches": 10000,
                }
            ],
        }

    assert compare_module(payload(100.0), payload(100.0)) == []
    warnings = compare_module(payload(100.0), payload(100.000001))
    assert len(warnings) == 1 and warnings[0]["kind"] == "performance-regression"
    assert _contract_hash("cmix") == _contract_hash("cmix")
    configured_paths = {
        case.path for cases in BENCHMARKS.values() for case in cases
    }
    discovered_paths = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "benchmarks").glob("**/bench.py")
    }
    assert configured_paths == discovered_paths
    for module, cases in BENCHMARKS.items():
        workloads = [case.workload for case in cases]
        assert len(workloads) == len(set(workloads)), module
        assert all((ROOT / case.path).is_file() for case in cases), module
    assert {case.workload for case in BENCHMARKS["cmix"]} == {
        "infer",
        "pretrain_forward_backward",
        "statetune_forward_backward",
    }
    assert {case.workload for case in BENCHMARKS["tmix/tokenshift"]} == {
        "infer",
        "pretrain_forward_backward",
        "statetune_forward_backward",
    }
    assert {case.workload for case in BENCHMARKS["tmix/wkv7"]} == {
        "infer_recurrent_fp32io16",
        "infer_chunk_bf16",
        "pretrain_forward_backward",
        "statetune",
    }
    assert {case.workload for case in BENCHMARKS["tmix/wkv7/rl_infctx"]} == {
        "forward_materialized",
        "forward_recompute",
        "backward_replay",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run-suite")
    run_parser.add_argument("--modules", required=True)
    run_parser.add_argument("--python", default=sys.executable)
    run_parser.add_argument("--revision", required=True)
    run_parser.add_argument("--status", choices=("candidate", "baseline"), required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--summary", type=Path, required=True)

    fetch_parser = commands.add_parser("fetch-baselines")
    fetch_parser.add_argument("--repository", required=True)
    fetch_parser.add_argument("--modules", required=True)
    fetch_parser.add_argument("--output", type=Path, required=True)
    fetch_parser.add_argument("--python", default=sys.executable)
    fetch_parser.add_argument("--github-output", type=Path)

    compare_parser = commands.add_parser("compare-suite")
    compare_parser.add_argument("--modules", required=True)
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)

    commands.add_parser("self-test")
    args = parser.parse_args()

    if args.command == "self-test":
        _self_test()
        print("benchmark_gate self-test passed")
        return 0
    modules = _parse_modules(args.modules)
    if args.command == "run-suite":
        summary = run_suite(
            modules,
            python=args.python,
            revision=args.revision,
            status=args.status,
            output=args.output,
        )
        _write_json(args.summary, summary)
        print(json.dumps(summary, separators=(",", ":")))
        return 0
    if args.command == "compare-suite":
        payload = compare_suite(modules, args.baseline, args.candidate)
        _write_json(args.output, payload)
        print(json.dumps(payload, separators=(",", ":")))
        return 0

    token = os.environ.get("GH_TOKEN", "")
    if not token:
        raise SystemExit("GH_TOKEN is required")
    missing = fetch_main_baselines(
        args.repository,
        token,
        modules,
        args.output,
        _environment(args.python),
    )
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"missing_modules={json.dumps(missing, separators=(',', ':'))}\n")
    print(json.dumps({"missing_modules": missing}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
