# SPDX-License-Identifier: MIT

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

_CXX_FLAGS = ("-O3", "-Wno-psabi")
_LINK_FLAGS = ("-Wl,--strip-debug",)
_NVCC_FLAGS = (
    "-O3",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-lineinfo",
    "-Xptxas=-v",
    "-D_N_=64",
    "-D_CHUNK_LEN_=16",
    "-std=c++17",
)
_SOURCE_SUFFIXES = {".cpp", ".cu", ".cuh", ".h", ".hpp"}
_PROCESS_LOCK = threading.Lock()
_RESULT: CompileResult | None = None


@dataclass(frozen=True)
class CompileResult:
    status: str
    target: str
    cache_key: str
    library: str
    module: ModuleType

    def json(self) -> dict[str, str]:
        return {
            "status": self.status,
            "target": self.target,
            "cache_key": self.cache_key,
            "library": self.library,
        }


def source_root() -> Path:
    repository = Path(__file__).resolve().parent.parent / "csrc"
    if (repository / "sm80").is_dir():
        return repository
    packaged = Path(__file__).resolve().parent / "_csrc"
    if (packaged / "sm80").is_dir():
        return packaged
    raise RuntimeError("FlashRWKV2 native CUDA sources are not installed")


def _source_files(root: Path) -> list[Path]:
    files = [
        root / "bindings.cpp",
        root / "registration.cpp",
        root / "validation.cpp",
        root / "validation" / "recurrent_metadata.cu",
    ]
    files.extend(
        path
        for path in (root / "sm80").rglob("*")
        if path.is_file() and path.suffix in _SOURCE_SUFFIXES
    )
    return sorted(files)


def _compile_sources(root: Path) -> list[str]:
    return [
        str(path)
        for path in _source_files(root)
        if path.suffix in {".cpp", ".cu"}
    ]


def _source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"runtime-builder\0")
    digest.update(Path(__file__).read_bytes())
    digest.update(b"\0")
    for path in _source_files(root):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _identity(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout.strip()


def _capability(torch) -> tuple[int, int]:
    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError(
            "FlashRWKV2 runtime compilation requires a visible CUDA GPU"
        )
    capabilities = {tuple(torch.cuda.get_device_capability(index)) for index in range(count)}
    if len(capabilities) != 1:
        rendered = ", ".join(f"sm{major}{minor}" for major, minor in sorted(capabilities))
        raise RuntimeError(
            "FlashRWKV2 requires all visible GPUs to have the same Compute "
            f"Capability, got {rendered}"
        )
    capability = capabilities.pop()
    if capability < (8, 0):
        raise RuntimeError(
            "FlashRWKV2 requires Compute Capability 8.0 or newer, got "
            f"sm{capability[0]}{capability[1]}"
        )
    return capability


def _toolchain(torch) -> dict[str, str]:
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        raise RuntimeError(
            "FlashRWKV2 runtime compilation requires a system CUDA Toolkit"
        )
    nvcc = Path(CUDA_HOME) / "bin" / "nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"FlashRWKV2 could not find NVCC at {nvcc}")
    ninja = shutil.which("ninja")
    if ninja is None:
        raise RuntimeError("FlashRWKV2 runtime compilation requires Ninja")
    host = os.environ.get("CXX") or sysconfig.get_config_var("CXX") or "c++"
    host_command = shlex.split(host)
    toolkit = _identity([str(nvcc), "--version"])
    host_identity = _identity([*host_command, "--version"])
    torch_cuda = torch.version.cuda
    if torch_cuda is None:
        raise RuntimeError("FlashRWKV2 requires a CUDA-enabled PyTorch build")
    toolkit_release = next(
        (
            token.rstrip(",")
            for line in toolkit.splitlines()
            for index, token in enumerate(line.split())
            if index > 0 and line.split()[index - 1] == "release"
        ),
        None,
    )
    if toolkit_release is None:
        raise RuntimeError("FlashRWKV2 could not determine the NVCC release")
    if toolkit_release.split(".", 1)[0] != torch_cuda.split(".", 1)[0]:
        raise RuntimeError(
            "FlashRWKV2 requires the system CUDA Toolkit major version to "
            f"match PyTorch CUDA: toolkit={toolkit_release}, torch={torch_cuda}"
        )
    return {
        "cuda_home": str(CUDA_HOME),
        "nvcc": str(nvcc),
        "nvcc_identity": toolkit,
        "host": host,
        "host_identity": host_identity,
        "ninja": ninja,
    }


def _cache_payload(torch, root: Path, capability: tuple[int, int], toolchain: dict[str, str]) -> dict[str, object]:
    from flashrwkv2 import __version__

    target = f"sm{capability[0]}{capability[1]}"
    target_define = f"-DFLASHRWKV_TARGET_SM={capability[0]}{capability[1]}"
    cxx_flags = (*_CXX_FLAGS, target_define)
    nvcc_flags = (
        *_NVCC_FLAGS,
        target_define,
        f"-gencode=arch=compute_{capability[0]}{capability[1]},code=sm_{capability[0]}{capability[1]}",
    )
    return {
        "source_sha256": _source_digest(root),
        "flashrwkv2": __version__,
        "python_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "target": target,
        "nvcc": toolchain["nvcc_identity"],
        "host_compiler": toolchain["host_identity"],
        "cxx_flags": cxx_flags,
        "nvcc_flags": nvcc_flags,
        "link_flags": _LINK_FLAGS,
    }


def _load_module(name: str, library: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, library)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load FlashRWKV2 extension from {library}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def _build_extension(
    name: str,
    root: Path,
    build_dir: Path,
    payload: dict[str, object],
    toolchain: dict[str, str],
) -> Path:
    from setuptools import Distribution
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    class UniqueObjectBuildExtension(BuildExtension):
        """Disambiguate equal-basename C++ and CUDA objects by source path."""

        def build_extensions(self):
            compiler = self.compiler
            original_setup_compile = compiler._setup_compile

            def setup_compile(*args, **kwargs):
                macros, objects, extra_postargs, pp_opts, build = original_setup_compile(
                    *args, **kwargs
                )
                sources = kwargs.get("sources")
                if sources is None and len(args) >= 4:
                    sources = args[3]
                unique = []
                for source, object_path in zip(sources, objects, strict=True):
                    stem, suffix = os.path.splitext(str(object_path))
                    source_key = hashlib.sha256(str(source).encode()).hexdigest()[:12]
                    unique.append(f"{stem}_{source_key}{suffix}")
                return macros, unique, extra_postargs, pp_opts, build

            compiler._setup_compile = setup_compile
            try:
                super().build_extensions()
            finally:
                compiler._setup_compile = original_setup_compile

    extension = CUDAExtension(
        name=name,
        include_dirs=[str(root)],
        sources=_compile_sources(root),
        extra_compile_args={
            "cxx": list(payload["cxx_flags"]),
            "nvcc": list(payload["nvcc_flags"]),
        },
        extra_link_args=list(payload["link_flags"]),
    )
    distribution = Distribution(
        {
            "name": name,
            "ext_modules": [extension],
            "cmdclass": {"build_ext": UniqueObjectBuildExtension},
        }
    )
    command = UniqueObjectBuildExtension(distribution)
    command.build_lib = str(build_dir)
    command.build_temp = str(build_dir / "objects")
    command.inplace = False
    command.force = True
    command.use_ninja = True
    command.ensure_finalized()
    os.environ["CUDA_HOME"] = toolchain["cuda_home"]
    command.run()
    library = Path(command.get_ext_fullpath(name)).resolve()
    if not library.is_file():
        raise RuntimeError(f"FlashRWKV2 build did not produce {library}")
    return library


def load_extension() -> CompileResult:
    global _RESULT
    with _PROCESS_LOCK:
        if _RESULT is not None:
            return _RESULT

        import torch
        from torch.utils.cpp_extension import get_default_build_root

        root = source_root()
        capability = _capability(torch)
        toolchain = _toolchain(torch)
        payload = _cache_payload(torch, root, capability, toolchain)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        cache_key = hashlib.sha256(encoded).hexdigest()
        target = str(payload["target"])
        name = f"_flashrwkv2_{cache_key[:16]}"
        cache_root = Path(get_default_build_root())
        build_dir = cache_root / name
        manifest = build_dir / "manifest.json"
        lock_path = cache_root / f".{name}.lock"
        cache_root.mkdir(parents=True, exist_ok=True)

        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if manifest.is_file():
                record = json.loads(manifest.read_text())
                library = Path(record["library"])
                if record.get("cache_key") == cache_key and library.is_file():
                    try:
                        module = _load_module(name, library)
                    except Exception:
                        manifest.unlink(missing_ok=True)
                        raise
                    _RESULT = CompileResult("cached", target, cache_key, str(library), module)
                    return _RESULT

            if build_dir.exists():
                shutil.rmtree(build_dir)
            build_dir.mkdir(parents=True)
            library = _build_extension(name, root, build_dir, payload, toolchain)
            module = _load_module(name, library)
            record = {
                "cache_key": cache_key,
                "library": str(library),
                "payload": payload,
            }
            temporary = manifest.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
            os.replace(temporary, manifest)
            _RESULT = CompileResult("compiled", target, cache_key, str(library), module)
            return _RESULT


def main() -> None:
    sys.stdout.flush()
    saved_stdout = os.dup(1)
    try:
        os.dup2(2, 1)
        result = load_extension()
    finally:
        sys.stdout.flush()
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)
    print(json.dumps(result.json(), sort_keys=True))


if __name__ == "__main__":
    main()
