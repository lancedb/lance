# Release process

1. Run `./build_wheel.sh`
2. twine upload wheels/*.whl

## Details

Because of the C++ extensions, we're building [manylinux wheels](https://github.com/pypa/manylinux).
The main entry point is the `build_wheel.sh` script.
This script:
1. Builds a docker image using [Dockerfile.manylinux2014](Dockerfile.manylinux2014) based on
   `quay.io/pypa/manylinux2014_x86_64` and installs critical C++ dependencies like Arrow, Parquet, and Protobuf.
2. Runs the [build_manylinux_wheels.sh](build_manylinux_wheels.sh) which does the following:
- build lance C++
- install pylance dependencies like pyarrow, numpy, and cython
- build pylance python wheels for python 3.8 - 3.10 (under `dist` directory)
- use `auditwheel` to convert wheels into platform-specific wheels that's uploadable to pypi (under `wheels` directory)
