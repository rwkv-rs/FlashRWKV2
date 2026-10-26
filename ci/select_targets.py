"""Select validation by semantic owner and its explicit consumers."""

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
RELEASE_METADATA_FILES = {
    "pyproject.toml",
    "uv.lock",
    "flashrwkv2/__init__.py",
}
SHARED_FILES = {"setup.py", *RELEASE_METADATA_FILES}
SHARED_PREFIXES = ("csrc/bindings.cpp", "csrc/registration.cpp", "csrc/validation")
# Edges point from a provider to every semantic consumer, not directory parents.
MODULE_CONSUMERS = {
    "post_norm": ("cmix", "tmix/tokenshift"),
    "head/linear": ("head/l2wrap_ce",),
    "tmix/kk_pre": ("tmix/wkv_prepare",),
    "tmix/vres_gate": ("tmix/wkv_prepare",),
    "tmix/wkv7": ("tmix/wkv7/rl_infctx", "tmix/tokenshift", "cmix"),
}
PACKAGE_FILES = {
    "setup.py",
    "pyproject.toml",
    "MANIFEST.in",
    "README.md",
    "LICENSE",
    "NOTICE",
}
COMPATIBILITY_FILES = {
    *SHARED_FILES,
    "flashrwkv2/compile.py",
    "flashrwkv2/_internal/tuning.py",
}
BENCHMARK_SHARED_FILES = {
    "benchmarks/_timing.py",
    "ci/benchmark_gate.py",
}
SHARED_PRIMITIVE_CALLERS = {
    "flashrwkv2/_internal/linear.py": (
        "cmix",
        "head/linear",
        "head/l2wrap_ce",
        "tmix/readout",
        "tmix/wkv_prepare",
    ),
    "flashrwkv2/_internal/elementwise.py": (
        "cmix",
        "tmix/a_gate",
        "tmix/kk_pre",
        "tmix/vres_gate",
        "tmix/wkv_prepare",
    ),
    "flashrwkv2/_internal/state.py": ("cmix", "tmix/tokenshift", "tmix/wkv7"),
    "flashrwkv2/_internal/logprob.py": (
        "head/linear",
        "head/l2wrap_ce",
        "loss/l2wrap_ce",
    ),
    "flashrwkv2/_internal/tuning.py": tuple(
        module for module in MODULES if module != "sampling"
    ),
}
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
    affected_cuda_modules: tuple[str, ...]
    run_gpu: bool
    run_benchmark: bool
    run_sanitizer: bool
    run_all: bool
    changed_files: tuple[str, ...]
    reasons: tuple[str, ...]
    run_package: bool = False
    run_contract: bool = True
    run_compatibility: bool = False


def _target_from_path(path: str, prefix: str) -> str | None:
    if not path.startswith(prefix):
        return None
    relative = path.removeprefix(prefix)
    for target in sorted(TARGETS, key=len, reverse=True):
        if relative in {target, f"{target}.py"} or relative.startswith(
            (f"{target}/", f"{target}_")
        ):
            return target
    return None


def validate_layout(root: Path = ROOT) -> None:
    missing: list[str] = []
    source_mismatches: list[str] = []
    for module in MODULES:
        for owner, filename in (
            ("flashrwkv2", "__init__.py"),
            ("tests", "test.py"),
            ("benchmarks", "bench.py"),
        ):
            candidate = root / owner / module / filename
            if not candidate.is_file():
                missing.append(str(candidate.relative_to(root)))
        source_root = root / "csrc/sm80" / module
        if module == "sampling" and (
            not source_root.is_dir()
            or not any(
                path.suffix in {".cpp", ".cu"}
                for path in source_root.rglob("*")
                if path.is_file()
            )
        ):
            source_mismatches.append(f"{module}/sm80")

    for candidate in (
        root / "flashrwkv2/tmix/wkv7/tilelang.py",
        root / "tests/tmix/wkv7/rl_infctx/test.py",
        root / "benchmarks/tmix/wkv7/rl_infctx/bench.py",
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
        path.relative_to(docs_root).with_suffix("") for path in docs_root.rglob("*.md")
    }
    missing_docs = sorted(source_stems - document_stems)
    extra_docs = sorted(document_stems - source_stems)

    errors = []
    if missing:
        errors.append("missing paths: " + ", ".join(missing))
    if source_mismatches:
        errors.append("CUDA source ownership: " + ", ".join(source_mismatches))
    if missing_docs:
        errors.append("missing csrc docs: " + ", ".join(map(str, missing_docs)))
    if extra_docs:
        errors.append("orphan csrc docs: " + ", ".join(map(str, extra_docs)))
    if errors:
        raise SystemExit("active module layout is incomplete: " + "; ".join(errors))


def expand_modules(modules: set[str]) -> tuple[str, ...]:
    expanded = set(modules)
    pending = list(modules)
    while pending:
        for consumer in MODULE_CONSUMERS.get(pending.pop(), ()):
            if consumer not in expanded:
                expanded.add(consumer)
                pending.append(consumer)
    return tuple(sorted(expanded))


def classify(
    paths: list[str],
    *,
    base_sha: str = "",
    head_sha: str = "",
    release_metadata_only: bool = False,
) -> Impact:
    changed = tuple(sorted({path.removeprefix("./") for path in paths if path}))
    modules: set[str] = set()
    reasons: set[str] = set()
    run_all = run_sanitizer = run_benchmark = run_compatibility = False
    run_package = any(
        path in PACKAGE_FILES or path.startswith("LICENSES/") for path in changed
    )
    run_contract = False
    benchmark_only = True
    for path in changed:
        if path in DOC_FILES or path.startswith((*DOC_PREFIXES, "LICENSES/")):
            continue
        run_contract = True
        if release_metadata_only and path in RELEASE_METADATA_FILES:
            modules.add("sampling")
            reasons.add("release-metadata")
            benchmark_only = False
            continue
        if path in BENCHMARK_SHARED_FILES:
            modules.update(TARGETS)
            run_benchmark = True
            reasons.add("shared-benchmark")
            continue
        if path.startswith(("ci/", ".github/workflows/")):
            modules.update(("sampling", "tmix/wkv7"))
            reasons.add("gpu-pipeline-smoke")
            benchmark_only = False
            continue
        if path == "tests/test_runtime.py":
            reasons.add("cpu-runtime-tests")
            benchmark_only = False
            continue
        if path in COMPATIBILITY_FILES or path.startswith("csrc/"):
            run_compatibility = True
        if (
            path in SHARED_FILES
            or path.startswith(SHARED_PREFIXES)
            or path == "tests/utils.py"
        ):
            run_all = run_sanitizer = True
            run_benchmark |= not path.startswith("tests/")
            reasons.add(f"shared:{path}")
            benchmark_only = False
            continue
        if path in SHARED_PRIMITIVE_CALLERS:
            modules.update(SHARED_PRIMITIVE_CALLERS[path])
            run_sanitizer = run_benchmark = True
            reasons.add(f"primitive:{path}")
            benchmark_only = False
            continue
        module = owner = None
        for candidate in ("flashrwkv2", "tests", "benchmarks"):
            module = _target_from_path(path, f"{candidate}/")
            if module:
                owner = candidate
                break
        if module is None and path.startswith("csrc/sm80/"):
            module = _target_from_path(path, "csrc/sm80/")
            owner = "csrc"
        if module:
            modules.add(module)
            run_sanitizer |= owner in {"flashrwkv2", "csrc"}
            run_benchmark |= owner != "tests"
            benchmark_only &= owner == "benchmarks"
            reasons.add(f"{owner}:{module}")
        elif Path(path).suffix in EXECUTABLE_SUFFIXES or path.startswith(
            ("flashrwkv2/", "tests/", "benchmarks/", "csrc/")
        ):
            run_all = run_sanitizer = True
            run_benchmark |= not path.startswith("tests/")
            benchmark_only = False
            reasons.add(f"shared-or-unknown:{path}")
    affected = tuple(sorted(TARGETS)) if run_all else expand_modules(modules)
    run_gpu = bool(affected)
    if run_all:
        change_class = "shared_or_unknown"
    elif run_gpu:
        change_class = "benchmark" if benchmark_only else "module"
    else:
        change_class = "cpu" if run_contract else "documentation"
    return Impact(
        base_sha=base_sha,
        head_sha=head_sha,
        change_class=change_class,
        affected_modules=affected,
        affected_cuda_modules=affected,
        run_gpu=run_gpu,
        run_benchmark=run_benchmark,
        run_sanitizer=run_sanitizer,
        run_all=run_all,
        changed_files=changed,
        reasons=tuple(sorted(reasons)),
        run_package=run_package or run_gpu,
        run_contract=run_contract,
        run_compatibility=run_compatibility,
    )


def _git_changed_files(base_sha: str, head_sha: str) -> list[str]:
    # Disabling rename detection exposes both the removed and added owner.
    result = subprocess.run(
        ("git", "diff", "--name-only", "--no-renames", "-z", base_sha, head_sha, "--"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return [path for path in result.stdout.split("\0") if path]


def _release_metadata_paths(paths: list[str]) -> tuple[str, ...]:
    changed = {path.strip().removeprefix("./") for path in paths if path.strip()}
    runtime_changes = {
        path
        for path in changed
        if path not in DOC_FILES and not path.startswith(DOC_PREFIXES)
    }
    if not runtime_changes or not runtime_changes <= RELEASE_METADATA_FILES:
        return ()
    return tuple(sorted(runtime_changes))


def _release_metadata_only(paths: list[str], base_sha: str, head_sha: str) -> bool:
    changed = _release_metadata_paths(paths)
    if not base_sha or not head_sha or not changed:
        return False
    versions: dict[str, str] = {}

    def revision_text(revision: str, path: str) -> str:
        return subprocess.run(
            ("git", "show", f"{revision}:{path}"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
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
        timeout=30,
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
    values = {**asdict(impact), "impact_artifact": str(artifact_path)}
    with path.open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            if isinstance(value, (bool, tuple)):
                value = json.dumps(value, separators=(",", ":"))
            handle.write(f"{key}={value}\n")


def _self_test() -> None:
    validate_layout()
    assert not classify(["docs/kernel_api.md"]).run_gpu
    assert not classify(["docs/kernel_api.md"]).run_package
    assert classify(["README.md"]).run_package
    assert not classify(["README.md"]).run_contract
    assert not classify(["tests/test_runtime.py"]).run_gpu
    assert classify(["tests/test_runtime.py"]).run_contract
    for path in (
        "setup.py",
        "flashrwkv2/compile.py",
        "flashrwkv2/__init__.py",
        "flashrwkv2/new.py",
        "tests/utils.py",
    ):
        assert classify([path]).run_all, path
    assert classify(["pyproject.toml"]).run_compatibility
    assert classify(
        ["csrc/sm80/sampling/infer_fp32_forward_varlen.cu"]
    ).run_compatibility
    assert classify(["ci/quality_gate.py"]).run_gpu
    assert not classify(["ci/quality_gate.py"]).run_all
    assert not classify(["tests/cmix/test.py"]).run_sanitizer
    assert classify(["flashrwkv2/cmix/__init__.py"]).run_sanitizer
    assert classify(["benchmarks/cmix/bench.py"]).change_class == "benchmark"
    assert not classify(["benchmarks/cmix/bench.py"]).run_all
    assert set(classify(["flashrwkv2/post_norm/tilelang.py"]).affected_modules) == {
        "post_norm",
        "cmix",
        "tmix/tokenshift",
    }
    assert set(classify(["flashrwkv2/tmix/kk_pre/tilelang.py"]).affected_modules) == {
        "tmix/kk_pre",
        "tmix/wkv_prepare",
    }
    assert set(classify(["flashrwkv2/head/linear/__init__.py"]).affected_modules) == {
        "head/linear",
        "head/l2wrap_ce",
    }
    primitive = classify(["flashrwkv2/_internal/linear.py"])
    assert set(primitive.affected_modules) == set(
        SHARED_PRIMITIVE_CALLERS["flashrwkv2/_internal/linear.py"]
    )
    # A rename is represented by old and new paths, preserving both owners.
    moved = classify(
        ["flashrwkv2/tmix/kk_pre/tilelang.py", "flashrwkv2/tmix/vres_gate/tilelang.py"]
    )
    assert set(moved.affected_modules) == {
        "tmix/kk_pre",
        "tmix/vres_gate",
        "tmix/wkv_prepare",
    }
    release = classify(
        ["pyproject.toml", "flashrwkv2/__init__.py", "uv.lock"],
        release_metadata_only=True,
    )
    assert release.run_gpu and not release.run_all and not release.run_compatibility
    assert _release_metadata_paths(["pyproject.toml", "README.md"]) == (
        "pyproject.toml",
    )
    assert not _release_metadata_paths(["setup.py", "pyproject.toml"])
    # Replay recent real changes through the owner map, including removals.
    revisions = subprocess.run(
        ("git", "rev-list", "--max-count=8", "--min-parents=1", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.splitlines()
    for revision in revisions:
        paths = _git_changed_files(f"{revision}^", revision)
        impact = classify(paths)
        owners = set()
        for path in paths:
            for prefix in ("flashrwkv2/", "tests/", "benchmarks/", "csrc/sm80/"):
                owner = _target_from_path(path, prefix)
                if owner:
                    owners.add(owner)
        assert set(expand_modules(owners)) <= set(impact.affected_modules), revision


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
    release_metadata_only = args.files is None and _release_metadata_only(
        paths, args.base, args.head
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
