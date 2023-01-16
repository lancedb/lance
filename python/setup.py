#!/usr/bin/env python3
#
# Copyright 2022 Lance Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import platform
from pathlib import Path

import numpy as np
import pyarrow as pa
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

extra_libs = []
# TODO: ciwheelbuild can not find / dont need arrow_python.
if platform.system() == "Linux":
    extra_libs.append("arrow_python")

pa.create_library_symlinks()
arrow_includes = pa.get_include()
arrow_library_dirs = pa.get_library_dirs()
numpy_includes = np.get_include()

# TODO allow for custom liblance directory
lance_cpp = Path(__file__).resolve().parent.parent / "cpp"
lance_includes = str(lance_cpp / "include")
lance_libs = str(lance_cpp / "build")

extensions = [
    Extension(
        "lance.lib",
        ["lance/_lib.pyx"],
        include_dirs=[lance_includes, arrow_includes, numpy_includes],
        libraries=["lance"] + extra_libs,
        library_dirs=[lance_libs] + arrow_library_dirs,
        language="c++",
        extra_compile_args=["-Wall", "-std=c++20", "-O3"],
        extra_link_args=[
            "-Wl,-rpath",
            lance_libs,
        ],  # , "-Wl,-rpath", arrow_library_dirs[0]],
    )
]

about = {}
with open(Path("lance") / "version.py", "r") as fh:
    exec(fh.read(), about)

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="pylance",
    version=about["__version__"],
    author="Lance Developers",
    author_email="contact@eto.ai",
    description="Python extension for lance",
    license="Apache Software License",
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=cythonize(extensions, language_level="3"),
    zip_safe=False,
    install_requires=["numpy", "pillow", "pyarrow>=10,<11", "requests", "pandas"],
    extras_require={
        "test": ["pytest>=6.0", "duckdb==0.6.1", "click", "requests_mock", "hypothesis"]
    },
    python_requires=">=3.8",
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 3 - Alpha",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Developers",
    ],
)
