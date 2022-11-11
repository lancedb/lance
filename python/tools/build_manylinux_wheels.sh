#!/bin/bash
#

set -ex

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib

if [[ "$#" -ne 1 ]]; then
  py_versions=(cp38 cp39 cp310)
else
  py_versions=($1)
fi

pushd /code/cpp
rm -rf build
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
popd

pushd /code/python
rm -rf wheels dist build
for py in "${py_versions[@]}"
do
  /opt/python/${py}-${py}/bin/pip install numpy "pyarrow>=10,<11" cython
  /opt/python/${py}-${py}/bin/python setup.py bdist_wheel
done

for whl in dist/*.whl; do
  /code/python/tools/auditwheel repair "$whl" -w wheels
done
