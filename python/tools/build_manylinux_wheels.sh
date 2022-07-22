#!/bin/bash

set -ex

yum update -y
yum install -y epel-release || yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-$(cut -d: -f5 /etc/system-release-cpe | cut -d. -f1).noarch.rpm
yum install -y https://apache.jfrog.io/artifactory/arrow/centos/$(cut -d: -f5 /etc/system-release-cpe | cut -d. -f1)/apache-arrow-release-latest.rpm
yum install -y --enablerepo=epel arrow-devel \
  arrow-dataset-devel parquet-devel wget

# Build protobuf manually
pushd /tmp
wget -q https://github.com/protocolbuffers/protobuf/releases/download/v3.20.1/protobuf-cpp-3.20.1.tar.gz -O - | tar -xz
cd protobuf-*
./configure
make -j
make install
popd

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib

pushd /code/cpp
rm -rf build
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
popd

pushd /code/python
rm -rf wheels dist
for py in cp38 cp39 cp310
do
  /opt/python/${py}-${py}/bin/pip install numpy pyarrow cython
  /opt/python/${py}-${py}/bin/python setup.py bdist_wheel
done

for whl in dist/*.whl; do
    auditwheel repair "$whl" -w wheels
done
