"""Build and verify tree-bound FlashRWKV2 quality evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from select_targets import (
    BACKENDS_BY_TARGET,
    CUDA_GRAPH_TARGETS,
    RACECHECK_TARGETS,
    TARGETS,
    _git_changed_files,
    _release_metadata_only,
    classify,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 5
WORKFLOW_NAME = "Quality Gate"
BUILD_INPUT_PATHS = (
    "setup.py",
    "pyproject.toml",
    "MANIFEST.in",
    "uv.lock",
    "README.md",
    "LICENSE",
    "NOTICE",
    "LICENSES",
    "flashrwkv2",
    "csrc",
)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha(revision: str) -> str:
    return _git("rev-parse", f"{revision}^{{tree}}")


def build_input_hash(revision: str) -> str:
    listing = subprocess.run(
        (
            "git",
            "ls-tree",
            "-r",
            "--full-tree",
            "-z",
            revision,
            "--",
            *BUILD_INPUT_PATHS,
        ),
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    digest = hashlib.sha256()
    digest.update(f"flashrwkv2-build-v{SCHEMA_VERSION}\0".encode())
    digest.update(listing)
    return digest.hexdigest()


FUNCTION_TARGETS = {
    (
        "csrc/sm120/tmix/wkv7/infer_recurrent_deltalog_fp16_forward_varlen.cu",
        "materialize_slots_kernel",
    ): {
        "module": "tmix/wkv7",
        "backend": "sm120",
        "nodeid": (
            "tests/tmix/wkv7/test.py::"
            "test_unified_state_materialize_and_pending_log_fallback[fp16]"
        ),
        "memcheck": True,
        "racecheck": True,
    },
    (
        (
            "csrc/sm120/tmix/wkv7/"
            "infer_recurrent_deltalog_fp32io16_forward_varlen.cu"
        ),
        "materialize_slots_kernel",
    ): {
        "module": "tmix/wkv7",
        "backend": "sm120",
        "nodeid": (
            "tests/tmix/wkv7/test.py::"
            "test_unified_state_materialize_and_pending_log_fallback[fp32io16]"
        ),
        "memcheck": True,
        "racecheck": True,
    },
}


def _changed_function_contexts(
    base_sha: str, head_sha: str, paths: list[str]
) -> list[tuple[str, str]]:
    runtime_paths = [
        path
        for path in paths
        if path.startswith(("csrc/", "flashrwkv2/", "tests/", "benchmarks/"))
    ]
    if not runtime_paths:
        return []
    diff = subprocess.run(
        (
            "git",
            "diff",
            "--unified=0",
            "--find-renames",
            base_sha,
            head_sha,
            "--",
            *runtime_paths,
        ),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    current = ""
    contexts: set[tuple[str, str]] = set()
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = line.removeprefix("+++ b/")
            continue
        if not line.startswith("@@"):
            continue
        context = line.split("@@", 2)[-1].strip()
        matches = re.findall(r"(?:^|\s)(?:def\s+)?([A-Za-z_]\w*)\s*\(", context)
        if not matches:
            raise SystemExit(f"function ledger cannot resolve changed hunk: {current}: {context}")
        contexts.add((current, matches[-1]))
    return sorted(contexts)


def _build_function_targets(
    base_sha: str, head_sha: str, changed: list[str]
) -> list[dict[str, Any]] | None:
    contexts = _changed_function_contexts(base_sha, head_sha, changed)
    native = [context for context in contexts if context[0].startswith("csrc/")]
    if not native:
        return None
    targets: dict[tuple[str, str], dict[str, Any]] = {}
    for context in native:
        target = FUNCTION_TARGETS.get(context)
        if target is None:
            raise SystemExit(
                "native function is absent from the CI function ledger: "
                f"{context[0]}::{context[1]}"
            )
        key = (target["backend"], target["nodeid"])
        targets[key] = {"source": context[0], "function": context[1], **target}
    test_contexts = {context for context in contexts if context[0].startswith("tests/")}
    expected_tests = {
        (target["nodeid"].split("::", 1)[0], target["nodeid"].split("::", 1)[1].split("[", 1)[0])
        for target in targets.values()
    }
    unknown_tests = test_contexts - expected_tests
    if unknown_tests:
        rendered = ", ".join(f"{path}::{function}" for path, function in sorted(unknown_tests))
        raise SystemExit(f"changed tests are absent from the native function ledger: {rendered}")
    return [targets[key] for key in sorted(targets)]


def build_plan(base_sha: str, head_sha: str) -> dict[str, Any]:
    changed = _git_changed_files(base_sha, head_sha)
    release_only = _release_metadata_only(changed, base_sha, head_sha)
    impact = classify(
        changed,
        base_sha=base_sha,
        head_sha=head_sha,
        release_metadata_only=release_only,
    )
    function_targets = (
        None
        if impact.run_all
        else _build_function_targets(base_sha, head_sha, changed)
    )
    if function_targets:
        modules = tuple(sorted({target["module"] for target in function_targets}))
        sm90_modules = tuple(
            sorted({target["module"] for target in function_targets if target["backend"] == "sm90"})
        )
        sm120_modules = tuple(
            sorted({target["module"] for target in function_targets if target["backend"] == "sm120"})
        )
        impact = replace(
            impact,
            change_class="native_function",
            affected_modules=modules,
            affected_sm90_modules=sm90_modules,
            affected_sm120_modules=sm120_modules,
            run_gpu=True,
            run_benchmark=False,
            run_sanitizer=True,
            run_all=False,
            package_smoke_only=False,
            reasons=tuple(sorted({*impact.reasons, "function-ledger"})),
        )
    elif impact.run_all:
        pass
    elif changed and all(
        path in {"AGENTS.md", "README.md", "RTK.md"}
        or path.startswith(("ci/", ".github/workflows/", "docs/"))
        for path in changed
    ):
        impact = replace(
            impact,
            change_class="control_plane",
            affected_modules=(),
            affected_sm90_modules=(),
            affected_sm120_modules=(),
            run_gpu=False,
            run_benchmark=False,
            run_sanitizer=False,
            run_all=False,
            reasons=tuple(sorted({*impact.reasons, "contract-only-control-plane"})),
        )
    elif impact.run_gpu and not impact.package_smoke_only:
        raise SystemExit(
            "GPU change has no exact function target; add its source function, "
            "pytest node IDs, sanitizers, and benchmark cases to FUNCTION_TARGETS"
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "source_tree_sha": source_tree_sha(head_sha),
        "base_tree_sha": source_tree_sha(base_sha),
        "build_input_hash": build_input_hash(head_sha),
        "impact": asdict(impact),
        "contract_required": impact.change_class != "documentation",
        "package_required": impact.run_gpu,
        "scope": (
            "function"
            if function_targets
            else "control"
            if impact.change_class == "control_plane"
            else "tree"
        ),
        "function_targets": function_targets or [],
        "sm90_correctness_nodeids": [
            target["nodeid"] for target in function_targets or [] if target["backend"] == "sm90"
        ],
        "sm120_correctness_nodeids": [
            target["nodeid"] for target in function_targets or [] if target["backend"] == "sm120"
        ],
        "sm90_memcheck_nodeids": [
            target["nodeid"] for target in function_targets or []
            if target["backend"] == "sm90" and target["memcheck"]
        ],
        "sm120_memcheck_nodeids": [
            target["nodeid"] for target in function_targets or []
            if target["backend"] == "sm120" and target["memcheck"]
        ],
        "sm90_racecheck_nodeids": [
            target["nodeid"] for target in function_targets or []
            if target["backend"] == "sm90" and target["racecheck"]
        ],
        "sm120_racecheck_nodeids": [
            target["nodeid"] for target in function_targets or []
            if target["backend"] == "sm120" and target["racecheck"]
        ],
        "benchmark_modules": (
            list(impact.affected_sm120_modules) if impact.run_benchmark else []
        ),
        "cpu_test_paths": (
            []
            if function_targets
            else ["tests"]
            if impact.run_all
            else [
                (
                    "tests/tmix/wkv7"
                    if module == "tmix/wkv7"
                    else f"tests/{module}/test.py"
                )
                for module in impact.affected_modules
            ]
            + (["tests/test_runtime.py"] if "tests/test_runtime.py" in changed else [])
        ),
        "cuda_graph_modules": [
            module
            for module in impact.affected_sm120_modules
            if not function_targets and module in CUDA_GRAPH_TARGETS
        ],
        "sm90_racecheck_modules": [
            module
            for module in impact.affected_sm90_modules
            if not function_targets and module in RACECHECK_TARGETS
        ],
        "sm120_racecheck_modules": [
            module
            for module in impact.affected_sm120_modules
            if not function_targets and module in RACECHECK_TARGETS
        ],
        "sm90_execution_mode": "PTX-on-SM120",
        "sm120_execution_mode": "native-sm120",
    }
    payload["coverage_requires_parent"] = payload["scope"] == "tree" and not (
        impact.run_all and impact.run_sanitizer
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["plan_hash"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _api_json(path: str, repository: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}{path}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def _artifact_request(url: str, token: str) -> urllib.request.Request:
    request = urllib.request.Request(url)
    request.add_unredirected_header("Authorization", f"Bearer {token}")
    return request


def _artifact_run(
    repository: str,
    token: str,
    artifact_name: str,
    *,
    current_run: int,
    require_success: bool,
    plan: dict[str, Any] | None = None,
) -> str:
    query = urllib.parse.urlencode({"name": artifact_name, "per_page": 100})
    artifacts = _api_json(
        f"/actions/artifacts?{query}", repository, token
    ).get("artifacts", [])
    for artifact in sorted(
        artifacts, key=lambda row: row.get("created_at", ""), reverse=True
    ):
        if artifact.get("expired"):
            continue
        run_id = artifact.get("workflow_run", {}).get("id")
        if not run_id or int(run_id) == current_run:
            continue
        run = _api_json(f"/actions/runs/{run_id}", repository, token)
        if run.get("name") != WORKFLOW_NAME or run.get("status") != "completed":
            continue
        if require_success and run.get("conclusion") != "success":
            continue
        if plan is not None and not _artifact_covers_plan(
            artifact, plan, token
        ):
            continue
        return str(run_id)
    return ""


def _artifact_covers_plan(
    artifact: dict[str, Any], plan: dict[str, Any], token: str
) -> bool:
    request = _artifact_request(artifact["archive_download_url"], token)
    try:
        with urllib.request.urlopen(request) as response, tempfile.NamedTemporaryFile() as tmp:
            tmp.write(response.read())
            tmp.flush()
            with zipfile.ZipFile(tmp.name) as bundle:
                names = [name for name in bundle.namelist() if name.endswith("quality-manifest.json")]
                if len(names) != 1:
                    return False
                manifest = json.loads(bundle.read(names[0]))
    except (KeyError, OSError, ValueError, zipfile.BadZipFile):
        return False
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "passed":
        return False
    source = manifest.get("source", {})
    if source.get("source_tree_sha") != plan["source_tree_sha"]:
        return False
    if source.get("build_input_hash") != plan["build_input_hash"]:
        return False
    if plan.get("scope") == "function":
        observed = {
            (entry.get("backend"), entry.get("nodeid"))
            for entry in manifest.get("function_coverage", [])
            if entry.get("correctness") == "passed"
        }
        required_functions = {
            (target["backend"], target["nodeid"])
            for target in plan["function_targets"]
        }
        if not required_functions <= observed:
            return False
    else:
        coverage = manifest.get("coverage", {})
        for backend in ("sm90", "sm120"):
            for module in plan["impact"][f"affected_{backend}_modules"]:
                if coverage.get(backend, {}).get(module, {}).get("correctness") != "passed":
                    return False
    if plan["package_required"] and not {
        "wheel",
        "sdist",
    } <= manifest.get("artifacts", {}).keys():
        return False
    checks = manifest.get("checks", {})
    required = {
        "contract": "passed" if plan["contract_required"] else None,
        "package": "passed" if plan["package_required"] else None,
        "sm90": "PTX-on-SM120" if plan["impact"]["affected_sm90_modules"] else None,
        "sm120": "native-sm120" if plan["impact"]["affected_sm120_modules"] else None,
        "sanitizer": "passed" if plan["impact"]["run_sanitizer"] else None,
    }
    return all(expected is None or checks.get(key) == expected for key, expected in required.items())


def resolve_reuse(
    plan: dict[str, Any], repository: str, token: str, current_run: int
) -> dict[str, Any]:
    evidence_name = f"flashrwkv2-quality-v5-{plan['source_tree_sha']}"
    wheel_name = f"flashrwkv2-wheel-v5-{plan['build_input_hash']}"
    evidence_run_id = _artifact_run(
        repository,
        token,
        evidence_name,
        current_run=current_run,
        require_success=True,
        plan=plan,
    )
    wheel_run_id = _artifact_run(
        repository,
        token,
        wheel_name,
        current_run=current_run,
        require_success=False,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "reuse_evidence": bool(evidence_run_id),
        "reuse_wheel": bool(wheel_run_id),
        "evidence_name": evidence_name,
        "evidence_run_id": evidence_run_id,
        "wheel_name": wheel_name,
        "wheel_run_id": wheel_run_id,
    }


def resolve_tree_evidence(
    tree: str, repository: str, token: str, current_run: int
) -> dict[str, Any]:
    artifact_name = f"flashrwkv2-quality-v5-{tree}"
    run_id = _artifact_run(
        repository,
        token,
        artifact_name,
        current_run=current_run,
        require_success=True,
    )
    return {
        "evidence_name": artifact_name,
        "evidence_run_id": run_id,
        "found": bool(run_id),
    }


def finalize(
    plan: dict[str, Any],
    *,
    wheel: Path | None,
    sdist: Path | None,
    benchmark: Path | None,
    validation: Path | None,
    inherit_manifest: Path | None,
    requirements: Path | None,
    audit: Path | None,
) -> dict[str, Any]:
    impact = plan["impact"]
    artifacts: dict[str, Any] = {}
    if wheel is not None:
        artifacts["wheel"] = {
            "name": wheel.name,
            "sha256": _sha256(wheel),
            "artifact": f"flashrwkv2-wheel-v5-{plan['build_input_hash']}",
        }
    if sdist is not None:
        artifacts["sdist"] = {"name": sdist.name, "sha256": _sha256(sdist)}
    if requirements is not None:
        artifacts["requirements"] = {
            "name": requirements.name,
            "sha256": _sha256(requirements),
        }
    if audit is not None:
        artifacts["audit"] = {"name": audit.name, "sha256": _sha256(audit)}
    required_artifacts = {"wheel", "sdist", "requirements", "audit"}
    if plan["package_required"] and required_artifacts - artifacts.keys():
        raise SystemExit(
            "package-required evidence needs wheel, sdist, requirements, and audit"
        )

    benchmark_payload: dict[str, Any] = {
        "status": "not-required",
        "warnings": [],
    }
    if benchmark is not None:
        benchmark_payload = json.loads(benchmark.read_text(encoding="utf-8"))
    if impact.get("run_benchmark"):
        if benchmark is None:
            raise SystemExit("benchmark-required evidence lacks a comparison summary")
        unavailable = [
            warning
            for warning in benchmark_payload.get("warnings", [])
            if warning.get("kind") == "benchmark-unavailable"
        ]
        if unavailable:
            raise SystemExit("benchmark-required evidence lacks complete candidate results")

    validation_payload: dict[str, Any] = {
        "environment": {},
        "checks": {},
        "results": [],
    }
    if validation is not None:
        validation_payload = json.loads(validation.read_text(encoding="utf-8"))
    recorded_checks = validation_payload.get("checks", {})
    recorded_results = validation_payload.get("results", [])

    def require_result(
        kind: str, backend: str, expected: list[str]
    ) -> None:
        matches = [
            result
            for result in recorded_results
            if result.get("kind") == kind and result.get("backend") == backend
        ]
        if len(matches) != 1 or matches[0].get("status") != "passed":
            raise SystemExit(f"validation lacks passed {backend} {kind} result")
        coverage_key = "nodeids" if plan.get("scope") == "function" else "modules"
        if sorted(matches[0].get(coverage_key, [])) != sorted(expected):
            raise SystemExit(
                f"validation {backend} {kind} {coverage_key} coverage mismatch"
            )
    if plan["package_required"]:
        for check in ("package_identity", "binary_contents"):
            if recorded_checks.get(check) != "passed":
                raise SystemExit(f"package-required evidence lacks passed {check}")
        stable_abi = validation_payload.get("stable_abi", {})
        if stable_abi.get("target") != "0x020b000000000000":
            raise SystemExit("package-required evidence has the wrong Stable ABI target")
        if stable_abi.get("build_torch") != "2.13":
            raise SystemExit("package-required evidence was not built with Torch 2.13")
        runtimes = stable_abi.get("runtimes", {})
        if runtimes != {version: "passed" for version in ("2.11", "2.12", "2.13")}:
            raise SystemExit("package-required evidence lacks the complete runtime matrix")
        if stable_abi.get("wheel_sha256") != artifacts["wheel"]["sha256"]:
            raise SystemExit("Stable ABI runtime matrix did not use the final wheel")
    if (
        impact["affected_sm120_modules"]
        and not impact["package_smoke_only"]
        and recorded_checks.get("sm120") != "native-sm120"
    ):
        raise SystemExit("SM120 validation was not recorded as native-sm120")
    if (
        impact["affected_sm90_modules"]
        and not impact["package_smoke_only"]
        and recorded_checks.get("sm90") != "PTX-on-SM120"
    ):
        raise SystemExit("SM90 validation was not recorded as PTX-on-SM120")
    if not impact["package_smoke_only"]:
        for backend in ("sm90", "sm120"):
            expected = (
                plan[f"{backend}_correctness_nodeids"]
                if plan.get("scope") == "function"
                else impact[f"affected_{backend}_modules"]
            )
            if expected:
                require_result("correctness", backend, expected)
    if impact["run_sanitizer"]:
        if recorded_checks.get("memcheck") != "passed":
            raise SystemExit("sanitizer evidence lacks passed memcheck")
        racecheck_required = bool(
            plan.get("sm90_racecheck_nodeids")
            or plan.get("sm120_racecheck_nodeids")
            or plan["sm90_racecheck_modules"]
            or plan["sm120_racecheck_modules"]
        )
        if racecheck_required and recorded_checks.get("racecheck") != "passed":
            raise SystemExit("sanitizer evidence lacks required racecheck")
        for backend in ("sm90", "sm120"):
            memcheck = (
                plan[f"{backend}_memcheck_nodeids"]
                if plan.get("scope") == "function"
                else impact[f"affected_{backend}_modules"]
            )
            if memcheck:
                require_result("memcheck", backend, memcheck)
            racecheck = (
                plan[f"{backend}_racecheck_nodeids"]
                if plan.get("scope") == "function"
                else plan[f"{backend}_racecheck_modules"]
            )
            if racecheck:
                require_result("racecheck", backend, racecheck)
    if plan["cuda_graph_modules"] and recorded_checks.get("cuda_graph") != "passed":
        raise SystemExit("CUDA Graph evidence was not recorded as passed")

    inherited: dict[str, Any] | None = None
    coverage: dict[str, dict[str, Any]] = {"sm90": {}, "sm120": {}}
    if inherit_manifest is not None:
        inherited = json.loads(inherit_manifest.read_text(encoding="utf-8"))
        if inherited.get("status") != "passed":
            raise SystemExit("inherited quality evidence did not pass")
        if inherited.get("source", {}).get("source_tree_sha") != plan["base_tree_sha"]:
            raise SystemExit("inherited quality evidence does not match the base tree")
        inherited_coverage = inherited.get("coverage", {})
        for backend in coverage:
            coverage[backend] = dict(inherited_coverage.get(backend, {}))
    elif plan["coverage_requires_parent"]:
        raise SystemExit("this incremental plan requires base-tree quality evidence")

    current_tree = plan["source_tree_sha"]
    for backend in ("sm90", "sm120"):
        for module in impact[f"affected_{backend}_modules"]:
            previous = coverage[backend].get(module, {})
            entry = {
                "source_tree_sha": current_tree,
                "correctness": "passed",
                "execution_mode": (
                    "PTX-on-SM120" if backend == "sm90" else "native-sm120"
                ),
                "memcheck": previous.get("memcheck"),
                "racecheck": previous.get("racecheck", "not-required"),
            }
            if impact["run_sanitizer"]:
                entry["memcheck"] = "passed"
                entry["racecheck"] = (
                    "passed" if module in RACECHECK_TARGETS else "not-required"
                )
            coverage[backend][module] = entry

    if plan.get("scope") not in {"function", "control"}:
        for module in TARGETS:
            for backend in BACKENDS_BY_TARGET[module]:
                entry = coverage[backend].get(module)
                if not entry or entry.get("correctness") != "passed":
                    raise SystemExit(f"quality coverage lacks {backend} correctness for {module}")
                if entry.get("memcheck") != "passed":
                    raise SystemExit(f"quality coverage lacks {backend} memcheck for {module}")
                if module in RACECHECK_TARGETS and entry.get("racecheck") != "passed":
                    raise SystemExit(f"quality coverage lacks {backend} racecheck for {module}")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "source": {
            "head_sha": plan["head_sha"],
            "source_tree_sha": plan["source_tree_sha"],
            "build_input_hash": plan["build_input_hash"],
        },
        "plan": {
            "hash": plan["plan_hash"],
            "change_class": impact["change_class"],
            "affected_modules": impact["affected_modules"],
            "affected_sm90_modules": impact["affected_sm90_modules"],
            "affected_sm120_modules": impact["affected_sm120_modules"],
        },
        "checks": {
            **recorded_checks,
            "contract": "passed" if plan["contract_required"] else "not-required",
            "package": "passed" if plan["package_required"] else "not-required",
            "sm90": recorded_checks.get("sm90", "not-required"),
            "sm120": recorded_checks.get("sm120", "not-required"),
            "sanitizer": "passed" if impact["run_sanitizer"] else "not-required",
        },
        "coverage": coverage,
        "function_coverage": [
            {
                **target,
                "source_tree_sha": current_tree,
                "correctness": "passed",
                "memcheck": "passed" if target["memcheck"] else "not-required",
                "racecheck": "passed" if target["racecheck"] else "not-required",
            }
            for target in plan.get("function_targets", [])
        ],
        "environment": validation_payload.get("environment", {}),
        "stable_abi": validation_payload.get("stable_abi", {}),
        "results": validation_payload.get("results", []),
        "artifacts": artifacts,
        "benchmark": benchmark_payload,
        "limitations": {
            "sm90": (
                "compute_90 PTX JIT executed on SM120; this is not native H100 "
                "cubin or SM90 performance evidence"
            )
        },
        "inherited_from": (
            None
            if inherited is None
            else inherited.get("source", {}).get("source_tree_sha")
        ),
    }


def verify_manifest(
    manifest: dict[str, Any],
    *,
    expected_tree: str,
    wheel: Path | None,
    sdist: Path | None,
) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit("unsupported quality evidence schema")
    if manifest.get("status") != "passed":
        raise SystemExit("quality evidence did not pass")
    if manifest.get("source", {}).get("source_tree_sha") != expected_tree:
        raise SystemExit("quality evidence source tree mismatch")
    artifacts = manifest.get("artifacts", {})
    for kind, path in (("wheel", wheel), ("sdist", sdist)):
        expected = artifacts.get(kind)
        if expected is None:
            if path is not None:
                raise SystemExit(f"unexpected {kind} for evidence without {kind}")
            continue
        if path is None:
            raise SystemExit(f"quality evidence is missing its {kind} file")
        if expected.get("name") != path.name:
            raise SystemExit(f"quality evidence {kind} filename mismatch")
        if expected.get("sha256") != _sha256(path):
            raise SystemExit(f"quality evidence {kind} SHA256 mismatch")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_outputs(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for key, value in payload.items():
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, separators=(",", ":"))
            elif isinstance(value, bool):
                value = str(value).lower()
            handle.write(f"{key}={value}\n")


def _extract_artifact(
    repository: str, token: str, artifact_name: str, run_id: str, output: Path
) -> None:
    artifacts = _api_json(
        f"/actions/runs/{run_id}/artifacts?per_page=100", repository, token
    ).get("artifacts", [])
    matches = [row for row in artifacts if row.get("name") == artifact_name]
    if len(matches) != 1:
        raise SystemExit(
            f"run {run_id} has {len(matches)} artifacts named {artifact_name}"
        )
    request = _artifact_request(matches[0]["archive_download_url"], token)
    output.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request) as response, tempfile.NamedTemporaryFile() as tmp:
        tmp.write(response.read())
        tmp.flush()
        with zipfile.ZipFile(tmp.name) as bundle:
            bundle.extractall(output)


def _self_test() -> None:
    head = _git("rev-parse", "HEAD")
    assert len(source_tree_sha(head)) == 40
    assert build_input_hash(head) == build_input_hash(head)
    request = _artifact_request("https://api.github.com/artifact", "test-token")
    assert request.get_header("Authorization") == "Bearer test-token"
    redirected = urllib.request.HTTPRedirectHandler().redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://example.blob.core.windows.net/artifact.zip?signature=test",
    )
    assert redirected is not None
    assert redirected.get_header("Authorization") is None
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        wheel = root / "candidate.whl"
        sdist = root / "candidate.tar.gz"
        requirements = root / "requirements.txt"
        audit = root / "audit.txt"
        wheel.write_bytes(b"wheel")
        sdist.write_bytes(b"sdist")
        requirements.write_text("torch==test\n", encoding="utf-8")
        audit.write_text("manylinux\n", encoding="utf-8")
        plan = {
            "schema_version": SCHEMA_VERSION,
            "head_sha": head,
            "source_tree_sha": source_tree_sha(head),
            "build_input_hash": "a" * 64,
            "plan_hash": "b" * 64,
            "contract_required": True,
            "package_required": True,
            "benchmark_modules": [],
            "cuda_graph_modules": [],
            "sm90_racecheck_modules": ["cmix"],
            "sm120_racecheck_modules": ["cmix"],
            "base_tree_sha": "c" * 40,
            "coverage_requires_parent": False,
            "impact": {
                "change_class": "native",
                "affected_modules": list(TARGETS),
                "affected_sm90_modules": [
                    module for module in TARGETS if "sm90" in BACKENDS_BY_TARGET[module]
                ],
                "affected_sm120_modules": [
                    module for module in TARGETS if "sm120" in BACKENDS_BY_TARGET[module]
                ],
                "package_smoke_only": False,
                "run_benchmark": False,
                "run_sanitizer": True,
                "run_all": True,
            },
        }
        validation = root / "validation.json"
        validation.write_text(
            json.dumps(
                {
                    "checks": {
                        "package_identity": "passed",
                        "binary_contents": "passed",
                        "sm90": "PTX-on-SM120",
                        "sm120": "native-sm120",
                        "memcheck": "passed",
                        "racecheck": "passed",
                    },
                    "stable_abi": {
                        "target": "0x020b000000000000",
                        "build_torch": "2.13",
                        "runtimes": {
                            version: "passed"
                            for version in ("2.11", "2.12", "2.13")
                        },
                        "wheel_sha256": _sha256(wheel),
                    },
                    "results": [
                        {
                            "kind": "correctness",
                            "backend": backend,
                            "status": "passed",
                            "modules": plan["impact"][f"affected_{backend}_modules"],
                        }
                        for backend in ("sm90", "sm120")
                        if plan["impact"][f"affected_{backend}_modules"]
                    ]
                    + [
                        {
                            "kind": "memcheck",
                            "backend": backend,
                            "status": "passed",
                            "modules": plan["impact"][f"affected_{backend}_modules"],
                        }
                        for backend in ("sm90", "sm120")
                        if plan["impact"][f"affected_{backend}_modules"]
                    ]
                    + [
                        {
                            "kind": "racecheck",
                            "backend": backend,
                            "status": "passed",
                            "modules": plan[f"{backend}_racecheck_modules"],
                        }
                        for backend in ("sm90", "sm120")
                        if plan[f"{backend}_racecheck_modules"]
                    ],
                }
            ),
            encoding="utf-8",
        )
        manifest = finalize(
            plan,
            wheel=wheel,
            sdist=sdist,
            benchmark=None,
            validation=validation,
            inherit_manifest=None,
            requirements=requirements,
            audit=audit,
        )
        verify_manifest(
            manifest,
            expected_tree=plan["source_tree_sha"],
            wheel=wheel,
            sdist=sdist,
        )
        old_manifest = {**manifest, "schema_version": 2}
        try:
            verify_manifest(
                old_manifest,
                expected_tree=plan["source_tree_sha"],
                wheel=wheel,
                sdist=sdist,
            )
        except SystemExit:
            pass
        else:
            raise AssertionError("v2 quality evidence was accepted")
        missing_results = root / "missing-results.json"
        missing_results.write_text(
            json.dumps({"checks": json.loads(validation.read_text())["checks"]}),
            encoding="utf-8",
        )
        try:
            finalize(
                plan,
                wheel=wheel,
                sdist=sdist,
                benchmark=None,
                validation=missing_results,
                inherit_manifest=None,
                requirements=requirements,
                audit=audit,
            )
        except SystemExit:
            pass
        else:
            raise AssertionError("validation without structured results was accepted")
        benchmark = root / "benchmark.json"
        benchmark.write_text(
            json.dumps(
                {
                    "status": "warning",
                    "warnings": [{"kind": "benchmark-unavailable", "module": "cmix"}],
                }
            ),
            encoding="utf-8",
        )
        benchmark_plan = {
            **plan,
            "impact": {**plan["impact"], "run_benchmark": True},
        }
        try:
            finalize(
                benchmark_plan,
                wheel=wheel,
                sdist=sdist,
                benchmark=benchmark,
                validation=validation,
                inherit_manifest=None,
                requirements=requirements,
                audit=audit,
            )
        except SystemExit:
            pass
        else:
            raise AssertionError("incomplete candidate benchmark was accepted")
        inherited_path = root / "inherited.json"
        inherited_path.write_text(json.dumps(manifest), encoding="utf-8")
        incremental_plan = {
            **plan,
            "source_tree_sha": "d" * 40,
            "base_tree_sha": plan["source_tree_sha"],
            "package_required": False,
            "coverage_requires_parent": True,
            "impact": {
                **plan["impact"],
                "change_class": "tests",
                "affected_modules": ["cmix"],
                "affected_sm90_modules": [],
                "affected_sm120_modules": ["cmix"],
                "run_sanitizer": False,
                "run_all": False,
            },
        }
        incremental_validation = root / "incremental-validation.json"
        incremental_validation.write_text(
            json.dumps(
                {
                    "checks": {"sm120": "native-sm120"},
                    "results": [
                        {
                            "kind": "correctness",
                            "backend": "sm120",
                            "status": "passed",
                            "modules": ["cmix"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        incremental = finalize(
            incremental_plan,
            wheel=None,
            sdist=None,
            benchmark=None,
            validation=incremental_validation,
            inherit_manifest=inherited_path,
            requirements=None,
            audit=None,
        )
        assert incremental["coverage"]["sm120"]["cmix"][
            "source_tree_sha"
        ] == "d" * 40
        assert incremental["coverage"]["sm90"]["cmix"][
            "source_tree_sha"
        ] == plan["source_tree_sha"]
        wheel.write_bytes(b"tampered")
        try:
            verify_manifest(
                manifest,
                expected_tree=plan["source_tree_sha"],
                wheel=wheel,
                sdist=sdist,
            )
        except SystemExit:
            pass
        else:
            raise AssertionError("tampered wheel was accepted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument("--base", required=True)
    plan_parser.add_argument("--head", required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    plan_parser.add_argument("--github-output", type=Path)

    resolve_parser = commands.add_parser("resolve")
    resolve_parser.add_argument("--plan", type=Path, required=True)
    resolve_parser.add_argument("--repository", required=True)
    resolve_parser.add_argument("--current-run", type=int, default=0)
    resolve_parser.add_argument("--output", type=Path, required=True)
    resolve_parser.add_argument("--github-output", type=Path)

    tree_parser = commands.add_parser("resolve-tree")
    tree_parser.add_argument("--tree", required=True)
    tree_parser.add_argument("--repository", required=True)
    tree_parser.add_argument("--current-run", type=int, default=0)
    tree_parser.add_argument("--output", type=Path, required=True)
    tree_parser.add_argument("--github-output", type=Path)

    extract_parser = commands.add_parser("extract")
    extract_parser.add_argument("--repository", required=True)
    extract_parser.add_argument("--artifact", required=True)
    extract_parser.add_argument("--run-id", required=True)
    extract_parser.add_argument("--output", type=Path, required=True)

    finalize_parser = commands.add_parser("finalize")
    finalize_parser.add_argument("--plan", type=Path, required=True)
    finalize_parser.add_argument("--wheel", type=Path)
    finalize_parser.add_argument("--sdist", type=Path)
    finalize_parser.add_argument("--benchmark", type=Path)
    finalize_parser.add_argument("--validation", type=Path)
    finalize_parser.add_argument("--inherit-manifest", type=Path)
    finalize_parser.add_argument("--requirements", type=Path)
    finalize_parser.add_argument("--audit", type=Path)
    finalize_parser.add_argument("--output", type=Path, required=True)

    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.add_argument("--expected-tree", required=True)
    verify_parser.add_argument("--wheel", type=Path)
    verify_parser.add_argument("--sdist", type=Path)

    commands.add_parser("self-test")
    args = parser.parse_args()

    if args.command == "self-test":
        _self_test()
        print("quality_gate self-test passed")
        return 0
    if args.command == "plan":
        payload = build_plan(args.base, args.head)
        _write_json(args.output, payload)
        if args.github_output:
            impact = payload["impact"]
            _write_outputs(
                args.github_output,
                {
                    "source_tree_sha": payload["source_tree_sha"],
                    "base_tree_sha": payload["base_tree_sha"],
                    "build_input_hash": payload["build_input_hash"],
                    "plan_hash": payload["plan_hash"],
                    "change_class": impact["change_class"],
                    "affected_modules": impact["affected_modules"],
                    "affected_sm90_modules": impact["affected_sm90_modules"],
                    "affected_sm120_modules": impact["affected_sm120_modules"],
                    "benchmark_modules": payload["benchmark_modules"],
                    "scope": payload["scope"],
                    "function_targets": payload["function_targets"],
                    "sm90_correctness_nodeids": payload["sm90_correctness_nodeids"],
                    "sm120_correctness_nodeids": payload["sm120_correctness_nodeids"],
                    "sm90_memcheck_nodeids": payload["sm90_memcheck_nodeids"],
                    "sm120_memcheck_nodeids": payload["sm120_memcheck_nodeids"],
                    "sm90_racecheck_nodeids": payload["sm90_racecheck_nodeids"],
                    "sm120_racecheck_nodeids": payload["sm120_racecheck_nodeids"],
                    "cpu_test_paths": payload["cpu_test_paths"],
                    "cuda_graph_modules": payload["cuda_graph_modules"],
                    "sm90_racecheck_modules": payload["sm90_racecheck_modules"],
                    "sm120_racecheck_modules": payload["sm120_racecheck_modules"],
                    "run_gpu": impact["run_gpu"],
                    "run_benchmark": impact["run_benchmark"],
                    "run_sanitizer": impact["run_sanitizer"],
                    "package_smoke_only": impact["package_smoke_only"],
                    "contract_required": payload["contract_required"],
                    "coverage_requires_parent": payload["coverage_requires_parent"],
                },
            )
        print(json.dumps(payload, separators=(",", ":")))
        return 0

    token = os.environ.get("GH_TOKEN", "")
    if args.command in {"resolve", "resolve-tree", "extract"} and not token:
        raise SystemExit("GH_TOKEN is required")
    if args.command == "resolve":
        payload = resolve_reuse(
            json.loads(args.plan.read_text(encoding="utf-8")),
            args.repository,
            token,
            args.current_run,
        )
        _write_json(args.output, payload)
        if args.github_output:
            _write_outputs(args.github_output, payload)
        print(json.dumps(payload, separators=(",", ":")))
        return 0
    if args.command == "resolve-tree":
        payload = resolve_tree_evidence(
            args.tree, args.repository, token, args.current_run
        )
        _write_json(args.output, payload)
        if args.github_output:
            _write_outputs(args.github_output, payload)
        print(json.dumps(payload, separators=(",", ":")))
        return 0
    if args.command == "extract":
        _extract_artifact(
            args.repository, token, args.artifact, args.run_id, args.output
        )
        return 0
    if args.command == "finalize":
        payload = finalize(
            json.loads(args.plan.read_text(encoding="utf-8")),
            wheel=args.wheel,
            sdist=args.sdist,
            benchmark=args.benchmark,
            validation=args.validation,
            inherit_manifest=args.inherit_manifest,
            requirements=args.requirements,
            audit=args.audit,
        )
        _write_json(args.output, payload)
        print(json.dumps(payload, separators=(",", ":")))
        return 0
    verify_manifest(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        expected_tree=args.expected_tree,
        wheel=args.wheel,
        sdist=args.sdist,
    )
    print("quality evidence verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
