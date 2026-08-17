"""Select the smallest fail-closed FlashRWKV2 CI validation set."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "cmix",
    "embedding",
    "head/l2wrap_ce",
    "head/linear",
    "loss/l2wrap_ce",
    "sampling",
    "tmix/a_gate",
    "tmix/kk_pre",
    "tmix/readout",
    "tmix/tokenshift",
    "post_norm",
    "tmix/vres_gate",
    "tmix/wkv_prepare",
    "tmix/wkv7",
)
WORKLOAD_TARGETS = ("tmix/wkv7/rl_infctx",)
TARGETS = MODULES + WORKLOAD_TARGETS
BACKENDS_BY_TARGET = {
    "cmix": ("sm90", "sm120"),
    "embedding": ("sm120",),
    "head/l2wrap_ce": ("sm90",),
    "head/linear": ("sm120",),
    "loss/l2wrap_ce": ("sm90",),
    "sampling": ("sm120",),
    "tmix/a_gate": ("sm90",),
    "tmix/kk_pre": ("sm90",),
    "tmix/readout": ("sm90", "sm120"),
    "tmix/tokenshift": ("sm90", "sm120"),
    "post_norm": ("sm120",),
    "tmix/vres_gate": ("sm90",),
    "tmix/wkv_prepare": ("sm120",),
    "tmix/wkv7": ("sm90", "sm120"),
    "tmix/wkv7/rl_infctx": ("sm90",),
}
SHARED_FILES = {
    "setup.py",
    "pyproject.toml",
    "uv.lock",
    "flashrwkv2/__init__.py",
    "tests/fixtures/tolerances-v1.json",
}
SHARED_PREFIXES = (
    ".github/workflows/",
    "ci/",
    "csrc/bindings.cpp",
    "csrc/registration.cpp",
    "csrc/validation.cpp",
    "csrc/validation/",
)
DOC_PREFIXES = ("docs/",)
DOC_FILES = {
    "AGENTS.md",
    "LICENSE",
    "NOTICE",
    "README.md",
    "RTK.md",
}
EXECUTABLE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".lock",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class Impact:
    base_sha: str
    head_sha: str
    change_class: str
    affected_modules: tuple[str, ...]
    affected_sm90_modules: tuple[str, ...]
    affected_sm120_modules: tuple[str, ...]
    run_gpu: bool
    run_benchmark: bool
    run_sanitizer: bool
    run_all: bool
    package_smoke_only: bool
    changed_files: tuple[str, ...]
    reasons: tuple[str, ...]


def _target_from_path(path: str, prefix: str) -> str | None:
    if not path.startswith(prefix):
        return None
    relative = path.removeprefix(prefix)
    for target in sorted(TARGETS, key=len, reverse=True):
        if (
            relative in {target, f"{target}.py"}
            or relative.startswith((f"{target}/", f"{target}_"))
        ):
            return target
    return None


def validate_layout(root: Path = ROOT) -> None:
    missing: list[str] = []
    backend_mismatches: list[str] = []
    for module in MODULES:
        for owner, filename in (
            ("flashrwkv2", "__init__.py"),
            ("tests", "test.py"),
            ("benchmarks", "bench.py"),
        ):
            candidate = root / owner / module / filename
            if not candidate.is_file():
                missing.append(str(candidate.relative_to(root)))
        for architecture in ("sm90", "sm120"):
            source_root = root / "csrc" / architecture / module
            has_sources = source_root.is_dir() and any(
                path.suffix in {".cpp", ".cu"}
                for path in source_root.rglob("*")
                if path.is_file()
            )
            expected = architecture in BACKENDS_BY_TARGET[module]
            if has_sources != expected:
                backend_mismatches.append(
                    f"{module}/{architecture}: expected_sources={expected}, "
                    f"actual_sources={has_sources}"
                )

    for candidate in (
        root / "flashrwkv2/tmix/wkv7/rl_infctx.py",
        root / "tests/tmix/wkv7/rl_infctx/test.py",
        root / "benchmarks/tmix/wkv7/rl_infctx/bench.py",
        root / "csrc/sm90/tmix/wkv7/rl_infctx_chunk_fp32io16_forward.cpp",
        root / "csrc/sm90/tmix/wkv7/rl_infctx_chunk_fp32io16_forward.cu",
        root / "csrc/sm90/tmix/wkv7/rl_infctx_chunk_fp32io16_backward.cpp",
        root / "csrc/sm90/tmix/wkv7/rl_infctx_chunk_fp32io16_backward.cu",
    ):
        if not candidate.is_file():
            missing.append(str(candidate.relative_to(root)))

    source_stems = {
        path.relative_to(root / "csrc").with_suffix("")
        for path in (root / "csrc").rglob("*")
        if path.is_file() and path.suffix in {".cpp", ".cu"}
    }
    docs_root = root / "docs/dev/csrc"
    document_stems = {
        path.relative_to(docs_root).with_suffix("")
        for path in docs_root.rglob("*.md")
    }
    missing_docs = sorted(source_stems - document_stems)
    extra_docs = sorted(document_stems - source_stems)

    errors = []
    if missing:
        errors.append("missing paths: " + ", ".join(missing))
    if backend_mismatches:
        errors.append("backend ownership: " + ", ".join(backend_mismatches))
    if missing_docs:
        errors.append(
            "missing csrc docs: " + ", ".join(map(str, missing_docs))
        )
    if extra_docs:
        errors.append("orphan csrc docs: " + ", ".join(map(str, extra_docs)))
    if errors:
        raise SystemExit("active module layout is incomplete: " + "; ".join(errors))




def classify(
    paths: list[str],
    *,
    base_sha: str = "",
    head_sha: str = "",
    release_metadata_only: bool = False,
) -> Impact:
    changed = tuple(
        sorted({path.strip().removeprefix("./") for path in paths if path.strip()})
    )
    modules: set[str] = set()
    reasons: list[str] = []
    run_gpu = False
    run_benchmark = False
    run_sanitizer = False
    run_all = False
    only_docs = bool(changed)

    if release_metadata_only:
        return Impact(
            base_sha=base_sha,
            head_sha=head_sha,
            change_class="release_metadata",
            affected_modules=(),
            affected_sm90_modules=(),
            affected_sm120_modules=(),
            run_gpu=True,
            run_benchmark=False,
            run_sanitizer=False,
            run_all=False,
            package_smoke_only=True,
            changed_files=changed,
            reasons=("release-metadata-only",),
        )

    for path in changed:
        suffix = Path(path).suffix.lower()
        if path in DOC_FILES or path.startswith(DOC_PREFIXES):
            continue
        only_docs = False
        if path == "tests/test_runtime.py":
            reasons.append("runtime-behavior-tests")
            continue
        if path == "tests/utils.py":
            run_all = run_gpu = run_sanitizer = True
            reasons.append("shared-test-behavior")
            continue
        if path in SHARED_FILES or path.startswith(SHARED_PREFIXES):
            run_all = run_gpu = run_benchmark = True
            if path.startswith(("csrc/", "ci/", ".github/workflows/")):
                run_sanitizer = True
            reasons.append(f"shared:{path}")
            continue

        module = None
        owner = None
        for candidate_owner in ("flashrwkv2", "tests", "benchmarks"):
            module = _target_from_path(path, f"{candidate_owner}/")
            if module:
                owner = candidate_owner
                break
        if module is None and path.startswith("csrc/sm"):
            parts = path.split("/", 2)
            if len(parts) == 3:
                module = _target_from_path(parts[2], "")
                owner = "csrc"

        if module:
            modules.add(module)
            run_gpu = True
            if owner in {"flashrwkv2", "benchmarks", "csrc"}:
                run_benchmark = True
            if owner == "csrc" and suffix in {".cpp", ".cu", ".cuh", ".h", ".hpp"}:
                run_sanitizer = True
            reasons.append(f"{owner}:{module}")
            continue

        if suffix in EXECUTABLE_SUFFIXES or path.startswith(
            ("flashrwkv2/", "tests/", "benchmarks/", "csrc/")
        ):
            run_all = run_gpu = run_benchmark = run_sanitizer = True
            reasons.append(f"unknown-executable:{path}")
        else:
            reasons.append(f"metadata:{path}")

    if run_all:
        modules = set(TARGETS)
    affected_modules = tuple(sorted(modules))
    affected_sm90_modules = tuple(
        module
        for module in affected_modules
        if "sm90" in BACKENDS_BY_TARGET[module]
    )
    affected_sm120_modules = tuple(
        module
        for module in affected_modules
        if "sm120" in BACKENDS_BY_TARGET[module]
    )
    if not changed:
        change_class = "empty"
    elif only_docs and not run_gpu:
        change_class = "documentation"
    elif run_all:
        change_class = "shared_or_unknown"
    elif run_sanitizer:
        change_class = "native"
    elif run_benchmark:
        change_class = "runtime_or_benchmark"
    elif run_gpu:
        change_class = "tests"
    else:
        change_class = "metadata"

    return Impact(
        base_sha=base_sha,
        head_sha=head_sha,
        change_class=change_class,
        affected_modules=affected_modules,
        affected_sm90_modules=affected_sm90_modules,
        affected_sm120_modules=affected_sm120_modules,
        run_gpu=run_gpu,
        run_benchmark=run_benchmark,
        run_sanitizer=run_sanitizer,
        run_all=run_all,
        package_smoke_only=False,
        changed_files=changed,
        reasons=tuple(sorted(set(reasons))),
    )


def _git_changed_files(base_sha: str, head_sha: str) -> list[str]:
    result = subprocess.run(
        ("git", "diff", "--name-only", "--find-renames", base_sha, head_sha, "--"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def _release_metadata_only(
    paths: list[str], base_sha: str, head_sha: str
) -> bool:
    allowed = {"pyproject.toml", "flashrwkv2/__init__.py", "uv.lock"}
    changed = {path.strip().removeprefix("./") for path in paths if path.strip()}
    if not base_sha or not head_sha or not changed or not changed <= allowed:
        return False
    versions: dict[str, str] = {}

    def revision_text(revision: str, path: str) -> str:
        return subprocess.run(
            ("git", "show", f"{revision}:{path}"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    if "pyproject.toml" in changed:
        before = tomllib.loads(revision_text(base_sha, "pyproject.toml"))
        after = tomllib.loads(revision_text(head_sha, "pyproject.toml"))
        versions["pyproject.toml"] = str(after["project"]["version"])
        before_compare = copy.deepcopy(before)
        after_compare = copy.deepcopy(after)
        before_compare["project"]["version"] = "<release-version>"
        after_compare["project"]["version"] = "<release-version>"
        if before_compare != after_compare:
            return False

    if "uv.lock" in changed:
        before = tomllib.loads(revision_text(base_sha, "uv.lock"))
        after = tomllib.loads(revision_text(head_sha, "uv.lock"))

        def normalize_lock(payload: dict[str, object]) -> tuple[dict[str, object], str]:
            normalized = copy.deepcopy(payload)
            roots = [
                package
                for package in normalized.get("package", [])
                if package.get("name") == "flashrwkv2"
                and package.get("source") == {"editable": "."}
            ]
            if len(roots) != 1:
                raise ValueError("uv.lock must contain one editable flashrwkv2 package")
            version = str(roots[0]["version"])
            roots[0]["version"] = "<release-version>"
            return normalized, version

        try:
            before_compare, _ = normalize_lock(before)
            after_compare, version = normalize_lock(after)
        except (KeyError, TypeError, ValueError):
            return False
        versions["uv.lock"] = version
        if before_compare != after_compare:
            return False

    result = subprocess.run(
        (
            "git",
            "diff",
            "--unified=0",
            base_sha,
            head_sha,
            "--",
            *sorted(changed),
        ),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    current = ""
    saw_change = False
    patterns = {
        "pyproject.toml": re.compile(r'version\s*=\s*"[^"]+"'),
        "flashrwkv2/__init__.py": re.compile(r'__version__\s*=\s*"[^"]+"'),
        "uv.lock": re.compile(r'version\s*=\s*"[^"]+"'),
    }
    for line in result.stdout.splitlines():
        if line.startswith("+++ b/"):
            current = line.removeprefix("+++ b/")
            continue
        if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
            continue
        content = line[1:].strip()
        pattern = patterns.get(current)
        if pattern is None or pattern.fullmatch(content) is None:
            return False
        match = re.search(r'"([^"]+)"', content)
        if line.startswith("+") and current == "flashrwkv2/__init__.py" and match:
            versions[current] = match.group(1)
        saw_change = True
    return saw_change and len(set(versions.values())) == 1


def _write_github_outputs(path: Path, impact: Impact, artifact_path: Path) -> None:
    values = {
        "base_sha": impact.base_sha,
        "head_sha": impact.head_sha,
        "change_class": impact.change_class,
        "affected_modules": json.dumps(impact.affected_modules, separators=(",", ":")),
        "affected_sm90_modules": json.dumps(
            impact.affected_sm90_modules, separators=(",", ":")
        ),
        "affected_sm120_modules": json.dumps(
            impact.affected_sm120_modules, separators=(",", ":")
        ),
        "benchmark_matrix": json.dumps(
            [
                {"module": module, "safe": module.replace("/", "-")}
                for module in impact.affected_sm120_modules
            ],
            separators=(",", ":"),
        ),
        "run_gpu": str(impact.run_gpu).lower(),
        "run_benchmark": str(impact.run_benchmark).lower(),
        "run_sanitizer": str(impact.run_sanitizer).lower(),
        "run_all": str(impact.run_all).lower(),
        "package_smoke_only": str(impact.package_smoke_only).lower(),
        "impact_artifact": str(artifact_path),
    }
    with path.open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            handle.write(f"{key}={value}\n")


def _self_test() -> None:
    validate_layout()
    doc = classify(["README.md"])
    assert doc.change_class == "documentation" and not doc.run_gpu
    test = classify(["tests/tmix/wkv_prepare/test.py"])
    assert (
        test.affected_modules == ("tmix/wkv_prepare",)
        and test.run_gpu
        and not test.run_benchmark
    )
    assert test.affected_sm90_modules == ()
    assert test.affected_sm120_modules == ("tmix/wkv_prepare",)
    native = classify(["csrc/sm120/tmix/wkv7/infer_recurrent_fp16_forward_varlen.cu"])
    assert (
        native.affected_modules == ("tmix/wkv7",)
        and native.run_sanitizer
        and native.run_benchmark
    )
    benchmark = classify(["benchmarks/cmix/bench.py"])
    assert benchmark.affected_modules == ("cmix",) and benchmark.run_benchmark
    rl_infctx = classify(
        ["csrc/sm90/tmix/wkv7/rl_infctx_chunk_fp32io16_forward.cu"]
    )
    assert rl_infctx.affected_modules == ("tmix/wkv7/rl_infctx",)
    assert rl_infctx.affected_sm90_modules == ("tmix/wkv7/rl_infctx",)
    assert rl_infctx.affected_sm120_modules == ()
    assert rl_infctx.run_benchmark and rl_infctx.run_sanitizer
    shared = classify(["setup.py"])
    assert shared.run_all and shared.affected_modules == tuple(sorted(TARGETS))
    assert "embedding" not in shared.affected_sm90_modules
    assert "head/l2wrap_ce" not in shared.affected_sm120_modules
    test_utils = classify(["tests/utils.py"])
    assert test_utils.run_all and test_utils.run_gpu and test_utils.run_sanitizer
    assert not test_utils.run_benchmark
    runtime = classify(["tests/test_runtime.py"])
    assert not runtime.run_gpu and runtime.change_class == "metadata"
    unknown = classify(["flashrwkv2/new_family.py"])
    assert unknown.run_all and unknown.run_sanitizer
    release = classify(
        ["pyproject.toml", "flashrwkv2/__init__.py", "uv.lock"],
        release_metadata_only=True,
    )
    assert release.package_smoke_only and release.run_gpu
    assert not release.run_benchmark and not release.affected_modules


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="")
    parser.add_argument("--head", default="")
    parser.add_argument("--files", nargs="*")
    parser.add_argument("--output", type=Path, default=Path("artifacts/impact.json"))
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print("select_targets self-test passed")
        return 0
    if args.files is None and not (args.base and args.head):
        parser.error("provide --files or both --base and --head")
    paths = (
        args.files
        if args.files is not None
        else _git_changed_files(args.base, args.head)
    )
    release_metadata_only = (
        args.files is None and _release_metadata_only(paths, args.base, args.head)
    )
    impact = classify(
        paths,
        base_sha=args.base,
        head_sha=args.head,
        release_metadata_only=release_metadata_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(impact), indent=2) + "\n", encoding="utf-8"
    )
    if args.github_output:
        _write_github_outputs(args.github_output, impact, args.output)
    print(json.dumps(asdict(impact), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
