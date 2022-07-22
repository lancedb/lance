#!/usr/bin/env bash
#
# Build pylance manylinux wheels

set -x

TOOLS_DIR=$(dirname $(realpath $0))
LANCE_ROOT=$(realpath ${TOOLS_DIR}/../..)

docker build -t pylance_manylinux -f Dockerfile.manylinux2014 .

docker run -v ${TOOLS_DIR}:/opt/lance/tools \
  -v ${LANCE_ROOT}:/code --rm \
  -w /code \
  quay.io/pypa/manylinux2014_x86_64 \
  /opt/lance/tools/build_manylinux_wheels.sh