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
        else:
            yield owner, script, []


def release_compatibility(output, env):
    # Use child interpreters so each test loads only its own Torch libraries.
    for version in TORCH_VERSIONS:
        run(["uv", "pip", "install", "--python", sys.executable, f"torch=={version}"], cwd=ROOT)
        run([sys.executable, "-I", "-m", "pytest", "-q", "-m", "cuda", "-k",
             "fullgraph or zero_temperature_is_deterministic_greedy or recurrent_forward_reference_and_grad_mode",
             "-o", f"pythonpath={ROOT / 'tests'} {ROOT / 'benchmarks'}",
             f"--junitxml={output / ('torch-' + version + '.xml')}", str(ROOT / "tests")], cwd=output, env=env)


def validate(args):
    output = args.output.resolve()
    output.mkdir(exist_ok=True, parents=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "benchmarks"))
    if args.release:
        try:
            release_compatibility(output, env)
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
    junit = output / "pytest.xml"
    run([sys.executable, "-I", "-m", "pytest", "-q", "-o",
         f"pythonpath={ROOT / 'tests'} {ROOT / 'benchmarks'}", f"--junitxml={junit}",
         str(ROOT / "tests")], cwd=output, env=env)
    suite = ET.parse(junit).getroot().find("testsuite")
    passed = int(suite.attrib["tests"]) - int(suite.attrib["skipped"])
    if passed <= 0:
        raise ValueError("pytest must execute tests on the acceptance device")
    if args.release:
        for tool in ("memcheck", "racecheck"):
            run(["compute-sanitizer", "--tool", tool, "--error-exitcode", "1",
                 sys.executable, "-I", "-m", "pytest", "-q", "-o",
                 f"pythonpath={ROOT / 'tests'} {ROOT / 'benchmarks'}", "-m", tool, str(ROOT / "tests")],
                cwd=output, env=env)
    measured = []
    for case, script, flags in benchmarks(output):
        with (output / (case.replace("/", "-") + ".log")).open("w") as log:
            run([sys.executable, str(script), *flags], cwd=output, env=env, stdout=log)
        measured.append(case)
    report.write_text(json.dumps({"commit": revision(), "packages": hashes, "gpu": props.name,
                                 "architecture": arch, "sm_count": props.multi_processor_count,
                                 "torch": str(torch.__version__), "passed": passed, "sanitizers": args.release,
                                 "compatibility": TORCH_VERSIONS if args.release else [],
                                 "benchmarks": measured}, indent=2) + "\n")


def verify(args):
    expected = packages(args.dist)
    cases = [case for case, _, _ in benchmarks(args.output)]
    for arch, sms in (("sm89", 128), ("sm120", 188)):
        report = json.loads((args.output / f"{arch}.json").read_text())
        if not (report["commit"] == revision() and report["packages"] == expected
                and (report["architecture"], report["sm_count"]) == (arch, sms)
                and report["sanitizers"] is True and report["passed"] > 0
                and report["compatibility"] == list(TORCH_VERSIONS) and report["benchmarks"] == cases):
            raise ValueError(f"{arch} acceptance does not match the release commit and distributions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "verify"))
    parser.add_argument("--release", action="store_true", help="run the sanitizer cases for release acceptance")
    parser.add_argument("--dist", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    (validate if args.command == "validate" else verify)(args)
