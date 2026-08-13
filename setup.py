# SPDX-License-Identifier: MIT

import os
import re
import sys
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_ARCHITECTURE_PATTERN = re.compile(
    r"(?P<major>[0-9]+)\.(?P<minor>[0-9]+)(?:a)?(?:\+PTX)?",
    re.IGNORECASE,
)


def _parse_torch_cuda_arch_list(value: str) -> tuple[tuple[int, int], ...]:
    tokens = tuple(token for token in re.split(r"[;,\s]+", value.strip()) if token)
    if not tokens:
        raise ValueError("TORCH_CUDA_ARCH_LIST must name at least one architecture")
    capabilities: list[tuple[int, int]] = []
    for token in tokens:
        match = _ARCHITECTURE_PATTERN.fullmatch(token)
        if match is None:
            raise ValueError(
                "FlashRWKV2 requires numeric CUDA architecture tokens such as "
                f"12.0; got {token!r}"
            )
        capabilities.append((int(match["major"]), int(match["minor"])))
    return tuple(capabilities)


def _validate_wheel_architectures(
    requested: str | None,
    *,
    detected: tuple[int, int] | None = None,
) -> tuple[tuple[int, int], ...]:
    if requested is not None:
        capabilities = _parse_torch_cuda_arch_list(requested)
    elif detected is not None:
        capabilities = (detected,)
    else:
        raise ValueError(
            "FlashRWKV2 native builds require TORCH_CUDA_ARCH_LIST=12.0 or an "
            "SM120 CUDA device"
        )
    unsupported = tuple(
        capability for capability in capabilities if capability < (12, 0)
    )
    if unsupported:
        rendered = ", ".join(f"{major}.{minor}" for major, minor in unsupported)
        raise RuntimeError(
            "this migration slice requires compute capability >= 12.0 (SM120); "
            f"unsupported target(s): {rendered}"
        )
    return capabilities


requested_architectures = os.environ.get("TORCH_CUDA_ARCH_LIST")
detected_architecture = (
    tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
)


class UniqueObjectBuildExtension(BuildExtension):
    """Keep same-stem C++/CUDA translation units in distinct object files."""

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
            counts: dict[str, int] = {}
            for object_path in objects:
                key = str(object_path)
                counts[key] = counts.get(key, 0) + 1

            used: set[str] = set()
            unique_objects = []
            for source, object_path in zip(sources, objects):
                object_path = str(object_path)
                candidate = object_path
                if counts[object_path] > 1:
                    suffix = Path(source).suffix.lstrip(".") or "src"
                    stem, extension = os.path.splitext(object_path)
                    candidate = f"{stem}_{suffix}{extension}"
                    index = 2
                    while candidate in used:
                        candidate = f"{stem}_{suffix}_{index}{extension}"
                        index += 1
                used.add(candidate)
                unique_objects.append(candidate)
            return macros, unique_objects, extra_postargs, pp_opts, build

        compiler._setup_compile = setup_compile
        try:
            super().build_extensions()
        finally:
            compiler._setup_compile = original_setup_compile


build_commands = {
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
native_build = bool(build_commands.intersection(sys.argv[1:]))
if native_build:
    _validate_wheel_architectures(
        requested_architectures,
        detected=detected_architecture,
    )

ext_modules = (
    [
        CUDAExtension(
            name="flashrwkv2._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/registration.cpp",
                "csrc/validation.cpp",
                "csrc/validation/recurrent_metadata.cu",
                "csrc/sm120/tmix/wkv7/infer_recurrent_fp32io16_forward_varlen.cpp",
                "csrc/sm120/tmix/wkv7/infer_recurrent_fp32io16_forward_varlen.cu",
                "csrc/sm120/tmix/wkv7/infer_recurrent_fp16_forward_varlen.cpp",
                "csrc/sm120/tmix/wkv7/infer_recurrent_fp16_forward_varlen.cu",
                "csrc/sm120/tmix/wkv7/infer_chunk_bf16_forward_varlen.cpp",
                "csrc/sm120/tmix/wkv7/infer_chunk_bf16_forward_varlen.cu",
                "csrc/sm120/tmix/mix6/infer_fp16_forward_varlen.cpp",
                "csrc/sm120/tmix/mix6/infer_fp16_forward_varlen.cu",
                "csrc/sm120/tmix/kk_a_gate/infer_fp16_forward_varlen.cpp",
                "csrc/sm120/tmix/kk_a_gate/infer_fp16_forward_varlen.cu",
                "csrc/sm120/tmix/lnx_rkvres_xg/infer_fp16_forward_varlen.cpp",
                "csrc/sm120/tmix/lnx_rkvres_xg/infer_fp16_forward_varlen.cu",
                "csrc/sm120/tmix/vres_gate/infer_fp16_forward_varlen.cpp",
                "csrc/sm120/tmix/vres_gate/infer_fp16_forward_varlen.cu",
                "csrc/sm120/cmix/mix/infer_fp16_forward_varlen.cpp",
                "csrc/sm120/cmix/mix/infer_fp16_forward_varlen.cu",
                "csrc/sm120/cmix/sparse/infer_fp16_forward_varlen.cpp",
                "csrc/sm120/cmix/sparse/infer_fp16_forward_varlen.cu",
                "csrc/sm120/tmix/linear/infer_fp16_forward_varlen.cpp",
                "csrc/sm120/tmix/linear/infer_fp16_forward_varlen.cu",
                "csrc/sm120/tmix/normalization/infer_fp16_forward_varlen.cpp",
                "csrc/sm120/tmix/normalization/infer_fp16_forward_varlen.cu",
                "csrc/sm120/embedding/infer_fp16_forward_varlen.cpp",
                "csrc/sm120/embedding/infer_fp16_forward_varlen.cu",
                "csrc/sm120/head/linear/infer_fp16_forward_varlen.cpp",
                "csrc/sm120/head/linear/infer_fp16_forward_varlen.cu",
                "csrc/sm120/sampling/infer_fp32_forward_varlen.cpp",
                "csrc/sm120/sampling/infer_fp32_forward_varlen.cu",
                "csrc/sm90/loss/l2wrap_ce/pretrain_bf16_forward.cpp",
                "csrc/sm90/loss/l2wrap_ce/pretrain_bf16_forward.cu",
                "csrc/sm90/loss/l2wrap_ce/pretrain_bf16_backward.cpp",
                "csrc/sm90/loss/l2wrap_ce/pretrain_bf16_backward.cu",
                "csrc/sm90/tmix/wkv7/pretrain_recurrent_bf16_forward.cpp",
                "csrc/sm90/tmix/wkv7/pretrain_recurrent_bf16_forward.cu",
                "csrc/sm90/tmix/a_gate/pretrain_bf16_forward.cpp",
                "csrc/sm90/tmix/a_gate/pretrain_bf16_forward.cu",
                "csrc/sm90/tmix/a_gate/pretrain_bf16_backward.cpp",
                "csrc/sm90/tmix/a_gate/pretrain_bf16_backward.cu",
                "csrc/sm90/tmix/vres_gate/pretrain_bf16_forward.cpp",
                "csrc/sm90/tmix/vres_gate/pretrain_bf16_forward.cu",
                "csrc/sm90/tmix/vres_gate/pretrain_bf16_backward.cpp",
                "csrc/sm90/tmix/vres_gate/pretrain_bf16_backward.cu",
                "csrc/sm90/tmix/mix6/pretrain_bf16_forward.cpp",
                "csrc/sm90/tmix/mix6/pretrain_bf16_forward.cu",
                "csrc/sm90/tmix/mix6/pretrain_bf16_backward.cpp",
                "csrc/sm90/tmix/mix6/pretrain_bf16_backward.cu",
                "csrc/sm90/tmix/mix6/statetune_bf16_forward.cpp",
                "csrc/sm90/tmix/mix6/statetune_bf16_forward.cu",
                "csrc/sm90/tmix/mix6/statetune_bf16_backward.cpp",
                "csrc/sm90/tmix/mix6/statetune_bf16_backward.cu",
                "csrc/sm90/tmix/kk_pre/pretrain_bf16_forward.cpp",
                "csrc/sm90/tmix/kk_pre/pretrain_bf16_forward.cu",
                "csrc/sm90/tmix/kk_pre/pretrain_bf16_backward.cpp",
                "csrc/sm90/tmix/kk_pre/pretrain_bf16_backward.cu",
                "csrc/sm90/tmix/lnx_rkvres_xg/pretrain_bf16_forward.cpp",
                "csrc/sm90/tmix/lnx_rkvres_xg/pretrain_bf16_forward.cu",
                "csrc/sm90/tmix/lnx_rkvres_xg/pretrain_bf16_backward.cpp",
                "csrc/sm90/tmix/lnx_rkvres_xg/pretrain_bf16_backward.cu",
                "csrc/sm90/head/l2wrap_ce/pretrain_bf16_forward.cpp",
                "csrc/sm90/head/l2wrap_ce/pretrain_bf16_forward.cu",
                "csrc/sm90/tmix/wkv7/statetune_recurrent_fp32io16_forward.cpp",
                "csrc/sm90/tmix/wkv7/statetune_recurrent_fp32io16_forward.cu",
                "csrc/sm90/tmix/wkv7/statetune_recurrent_fp32io16_backward.cpp",
                "csrc/sm90/tmix/wkv7/statetune_recurrent_fp32io16_backward.cu",
                "csrc/sm90/tmix/wkv7/rl_infctx_chunk_fp32io16_forward.cpp",
                "csrc/sm90/tmix/wkv7/rl_infctx_chunk_fp32io16_forward.cu",
                "csrc/sm90/tmix/wkv7/rl_infctx_chunk_fp32io16_backward.cpp",
                "csrc/sm90/tmix/wkv7/rl_infctx_chunk_fp32io16_backward.cu",
                "csrc/sm90/cmix/mix/pretrain_bf16_forward.cpp",
                "csrc/sm90/cmix/mix/pretrain_bf16_forward.cu",
                "csrc/sm90/cmix/mix/pretrain_bf16_backward.cpp",
                "csrc/sm90/cmix/mix/pretrain_bf16_backward.cu",
                "csrc/sm90/cmix/mix/statetune_bf16_forward.cpp",
                "csrc/sm90/cmix/mix/statetune_bf16_forward.cu",
                "csrc/sm90/cmix/mix/statetune_bf16_backward.cpp",
                "csrc/sm90/cmix/mix/statetune_bf16_backward.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-Wno-psabi"],
                "nvcc": [
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-lineinfo",
                    "-Xptxas=-v",
                    "-D_N_=64",
                    "-D_CHUNK_LEN_=16",
                    # CUDA 13 nvcc ICEs on the unmodified canonical Albatross
                    # v3a body under C++20.  The exact upstream translation
                    # unit compiles under C++17; keep the body unchanged.
                    "-std=c++17",
                ],
            },
        )
    ]
    if native_build
    else []
)

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": UniqueObjectBuildExtension} if native_build else {},
    zip_safe=False,
)
