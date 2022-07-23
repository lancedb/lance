#!/bin/bash

set -ex

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib

pushd /code/cpp
rm -rf build
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
popd

pushd /code/python
rm -rf wheels dist build
for py in cp38 cp39 cp310
do
  /opt/python/${py}-${py}/bin/pip install numpy pyarrow cython
  /opt/python/${py}-${py}/bin/python setup.py bdist_wheel
done

for whl in dist/*.whl; do
    /code/python/tools/auditwheel repair "$whl" -w wheels
done
