# SPDX-License-Identifier: MIT

import os
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


class UniqueObjectBuildExtension(BuildExtension):
    """Keep objects unique by source suffix and private backend."""

    def build_extensions(self):
        compiler = self.compiler
        original_setup_compile = compiler._setup_compile
        original_build_extension = self.build_extension

        def setup_compile(*args, **kwargs):
            macros, objects, extra_postargs, pp_opts, build = original_setup_compile(
                *args, **kwargs
            )
            sources = kwargs.get("sources")
            if sources is None and len(args) >= 4:
                sources = args[3]
            counts: dict[str, int] = {}
            for object_path in objects:
                key = str(object_path)
                counts[key] = counts.get(key, 0) + 1

            used: set[str] = set()
            unique_objects = []
            for source, object_path in zip(sources, objects):
                object_path = str(object_path)
                suffix = Path(source).suffix.lstrip(".") or "src"
                backend = self._flashrwkv_backend
                stem, extension = os.path.splitext(object_path)
                candidate = f"{stem}_{backend}"
                if counts[object_path] > 1:
                    candidate += f"_{suffix}"
                candidate += extension
                index = 2
                while candidate in used:
                    candidate = f"{stem}_{backend}_{suffix}_{index}{extension}"
                    index += 1
                used.add(candidate)
                unique_objects.append(candidate)
            return macros, unique_objects, extra_postargs, pp_opts, build

        def build_extension(extension):
            self._flashrwkv_backend = extension.name.rsplit("_", 1)[-1]
            return original_build_extension(extension)

        compiler._setup_compile = setup_compile
        self.build_extension = build_extension
        try:
            super().build_extensions()
        finally:
            self.build_extension = original_build_extension
            compiler._setup_compile = original_setup_compile


BUILD_COMMANDS = {
    "bdist",
    "bdist_egg",
    "bdist_rpm",
    "bdist_wheel",
    "build",
    "build_clib",
    "build_ext",
    "build_py",
    "build_scripts",
    "develop",
    "editable_wheel",
    "install",
    "install_lib",
}
NATIVE_BUILD = bool(BUILD_COMMANDS.intersection(sys.argv[1:]))


def _native_sources(architecture: str) -> list[str]:
    shared = [
        "csrc/bindings.cpp",
        "csrc/registration.cpp",
        "csrc/validation.cpp",
    ]
    if architecture == "sm120":
        shared.append("csrc/validation/recurrent_metadata.cu")
    architecture_root = Path("csrc") / architecture
    return shared + sorted(
        str(path)
        for path in architecture_root.rglob("*")
        if path.suffix in {".cpp", ".cu"}
    )


def _compile_args(architecture: str) -> dict[str, list[str]]:
    capability = {"sm90": "90", "sm120": "120"}[architecture]
    return {
        "cxx": ["-O3", "-Wno-psabi"],
        "nvcc": [
            "-O3",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-lineinfo",
            "-Xptxas=-v",
            "-D_N_=64",
            "-D_CHUNK_LEN_=16",
            # CUDA 13 nvcc ICEs on the unmodified canonical Albatross v3a
            # body under C++20. Keep the translation unit on C++17.
            "-std=c++17",
            f"-gencode=arch=compute_{capability},code=sm_{capability}",
        ],
    }


EXT_MODULES = (
    [
        CUDAExtension(
            name="flashrwkv2._C_sm90",
            define_macros=[("FLASHRWKV_BACKEND_SM90", "1")],
            sources=_native_sources("sm90"),
            extra_compile_args=_compile_args("sm90"),
        ),
        CUDAExtension(
            name="flashrwkv2._C_sm120",
            define_macros=[("FLASHRWKV_BACKEND_SM120", "1")],
            sources=_native_sources("sm120"),
            extra_compile_args=_compile_args("sm120"),
        ),
    ]
    if NATIVE_BUILD
    else []
)

setup(
    ext_modules=EXT_MODULES,
    cmdclass={"build_ext": UniqueObjectBuildExtension} if NATIVE_BUILD else {},
    zip_safe=False,
)
