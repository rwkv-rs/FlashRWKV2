"""Run pytest against the installed wheel and benchmark every operator owner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from importlib.metadata import distribution
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TORCH_VERSIONS = ("2.11.0", "2.12.0", "2.13.0", "2.14.1")


def run(args, **kwargs):
    return subprocess.run(args, check=True, timeout=7200, **kwargs)


def revision():
    return run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def packages(directory):
    wheel, = directory.glob("*.whl")
    source, = directory.glob("*.tar.gz")
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (wheel, source)}


def benchmarks(output):
    for script in sorted((ROOT / "benchmarks").rglob("bench.py")):
        owner = script.parent.relative_to(ROOT / "benchmarks").as_posix()
        if owner == "tmix/wkv7":
            for dtype in ("float32", "float16"):
                yield owner + "/" + dtype, script, [
                    "--state-dtype", dtype, "--shapes", "h40d64", "h64d64", "--timing-mode", "graph",
                    "--output", str(output / ("wkv7-" + dtype + ".json")),
                ]
            yield owner + "/float16-live-mixed", script, [
                "--state-dtype", "float16", "--shapes", "h64d64", "--cases", "255x1+1x256",
                "--metadata", "live", "--timing-mode", "graph", "--slot-order", "random",
                "--output", str(output / "wkv7-float16-live-mixed.json"),
            ]
        else:
            yield owner, script, []


def pytest_cases(output, env, name, *, marks="cuda", keywords=None, prefix=()):
    junit = output / f"{name}.xml"
    options = [] if keywords is None else ["-k", keywords]
    run([*prefix, sys.executable, "-I", "-m", "pytest", "-q", "-m", marks, *options,
         "-o", f"pythonpath={ROOT / 'tests'} {ROOT / 'benchmarks'}",
         f"--junitxml={junit}", str(ROOT / "tests")], cwd=output, env=env)
    count = sum(case.find("skipped") is None for case in ET.parse(junit).iter("testcase"))
    if count == 0:
        raise ValueError(f"{name} did not execute any tests with marker expression {marks!r}")
    return count


def release_compatibility(output, env):
    # Use child interpreters so each test loads only its own Torch libraries.
    cases = {}
    for version in TORCH_VERSIONS:
        run(["uv", "pip", "install", "--python", sys.executable, f"torch=={version}"], cwd=ROOT)
        cases[version] = pytest_cases(
            output, env, f"torch-{version}",
            keywords="fullgraph or zero_temperature_is_deterministic_greedy or recurrent_forward_reference_and_grad_mode",
        )
    return cases


def validate(args):
    output = args.output.resolve()
    output.mkdir(exist_ok=True, parents=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "benchmarks"))
    compatibility = {}
    if args.release:
        try:
            compatibility = release_compatibility(output, env)
        finally:
            run(["uv", "sync", "--frozen", "--no-install-project", "--group", "dev"], cwd=ROOT)
            wheel, = args.dist.glob("*.whl")
            run(["uv", "pip", "install", "--python", sys.executable, "--no-deps",
                 "--reinstall", str(wheel.resolve())], cwd=ROOT)
    import torch

    props = torch.cuda.get_device_properties(0)
    arch = f"sm{props.major}{props.minor}"
    if (arch, props.multi_processor_count) not in {("sm89", 128), ("sm120", 188)}:
        raise ValueError(f"unexpected acceptance device: {props.name}")
    report = output / f"{arch}.json"
    report.unlink(missing_ok=True)
    hashes = packages(args.dist)
    wheel, = args.dist.glob("*.whl")
    installed = distribution("flashrwkv2")
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if (name.startswith("flashrwkv2/") and not name.endswith("/")
                    and installed.locate_file(name).read_bytes() != archive.read(name)):
                raise ValueError(f"installed file differs from {wheel.name}: {name}")
    passed = pytest_cases(output, env, "pytest")
    resource_passed = pytest_cases(output, env, "pytest-resource", marks="resource and not cuda")
    sanitizers = {}
    if args.release:
        for tool in ("memcheck", "racecheck"):
            sanitizers[tool] = pytest_cases(
                output, env, tool, marks=f"cuda and {tool}",
                prefix=("compute-sanitizer", "--tool", tool, "--error-exitcode", "1"),
            )
    measured = []
    for case, script, flags in benchmarks(output):
        with (output / (case.replace("/", "-") + ".log")).open("w") as log:
            run([sys.executable, str(script), *flags], cwd=output, env=env, stdout=log)
        measured.append(case)
    report.write_text(json.dumps({"commit": revision(), "packages": hashes, "gpu": props.name,
                                 "architecture": arch, "sm_count": props.multi_processor_count,
                                 "torch": str(torch.__version__), "passed": passed,
                                 "resource_passed": resource_passed, "sanitizers": sanitizers,
                                 "compatibility": compatibility,
                                 "benchmarks": measured}, indent=2) + "\n")


def verify(args):
    expected = packages(args.dist)
    cases = [case for case, _, _ in benchmarks(args.output)]
    for arch, sms in (("sm89", 128), ("sm120", 188)):
        report = json.loads((args.output / f"{arch}.json").read_text())
        if not (report["commit"] == revision() and report["packages"] == expected
                and (report["architecture"], report["sm_count"]) == (arch, sms)
                and report["passed"] > 0
                and report["resource_passed"] > 0
                and set(report["sanitizers"]) == {"memcheck", "racecheck"}
                and all(count > 0 for count in report["sanitizers"].values())
                and set(report["compatibility"]) == set(TORCH_VERSIONS)
                and all(count > 0 for count in report["compatibility"].values())
                and report["benchmarks"] == cases):
            raise ValueError(f"{arch} acceptance does not match the release commit and distributions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "verify"))
    parser.add_argument("--release", action="store_true", help="run the sanitizer cases for release acceptance")
    parser.add_argument("--dist", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    (validate if args.command == "validate" else verify)(args)
