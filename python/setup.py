from pathlib import Path
from setuptools import Extension, find_packages, setup

import numpy as np
import pyarrow as pa
from Cython.Build import cythonize

arrow_includes = pa.get_include()
numpy_includes = np.get_include()

# TODO allow for custom liblance directory
lance_cpp = Path(__file__).resolve().parent.parent / 'cpp'
lance_includes = str(lance_cpp / 'include')
lance_libs = str(lance_cpp / 'build')


extensions = [Extension(
    "lance.lib",
    ["lance/_lib.pyx"],
    include_dirs=[lance_includes, arrow_includes, numpy_includes],
    libraries=['lance'],
    library_dirs=[lance_libs],
    language="c++",
    extra_compile_args=["-Wall", "-std=c++20", "-O3"],
    extra_link_args=["-Wl,-rpath", lance_libs]
)]


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="pylance",
    version="0.0.1",
    author="Lance Developers",
    author_email="contact@eto.ai",
    description="Python extension for lance",
    long_description="",
    ext_modules=cythonize(extensions, language_level="3"),
    zip_safe=False,
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.8",
    packages=find_packages()
)
