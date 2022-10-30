#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/lib
export PATH=/usr/local/cmake/bin:/opt/rh/devtoolset-11/root/usr/bin:/opt/python/cp310-cp310/bin:$PATH
gcc --version

pushd /code
rm -rf /code/cuda-build
cmake -B cuda-build -DLANCE_BUILD_CUDA=TRUE
make -C cuda-build -j