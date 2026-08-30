# SPDX-License-Identifier: MIT

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

ROOT = Path(__file__).resolve().parent


def _copy_native_sources(destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(ROOT / "csrc", destination)


class SourceBuildPy(build_py):
    """Include the native source tree without importing Torch or running NVCC."""

    def run(self):
        super().run()
        _copy_native_sources(Path(self.build_lib) / "flashrwkv2" / "_csrc")


class SourceSdist(sdist):
    """Keep the same native source tree in source distributions."""

    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        destination = Path(base_dir) / "csrc"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(ROOT / "csrc", destination)


setup(
    cmdclass={"build_py": SourceBuildPy, "sdist": SourceSdist},
    zip_safe=False,
)
