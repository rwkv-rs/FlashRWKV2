# SPDX-License-Identifier: MIT

import ast
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

ROOT = Path(__file__).resolve().parent


def _check_runtime_version(distribution_version: str) -> None:
    init = ROOT / "flashrwkv2" / "__init__.py"
    module = ast.parse(init.read_text(encoding="utf-8"), filename=str(init))
    (value,) = (
        node.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
    )
    runtime_version = ast.literal_eval(value)
    if runtime_version != distribution_version:
        raise ValueError(
            f"flashrwkv2.__version__ is {runtime_version}, "
            f"but the distribution version is {distribution_version}"
        )


def _copy_native_sources(destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(ROOT / "csrc", destination)


class SourceBuildPy(build_py):
    """Include the native source tree without importing Torch or running NVCC."""

    def run(self):
        _check_runtime_version(self.distribution.get_version())
        super().run()
        _copy_native_sources(Path(self.build_lib) / "flashrwkv2" / "_csrc")


class SourceSdist(sdist):
    """Keep the same native source tree in source distributions."""

    def make_release_tree(self, base_dir, files):
        _check_runtime_version(self.distribution.get_version())
        super().make_release_tree(base_dir, files)
        _copy_native_sources(Path(base_dir) / "csrc")


setup(
    cmdclass={"build_py": SourceBuildPy, "sdist": SourceSdist},
    zip_safe=False,
)
