#!/usr/bin/env bash
#
# Build pylance manylinux wheels

set -x

TOOLS_DIR=$(dirname $(realpath $0))
LANCE_ROOT=$(realpath ${TOOLS_DIR}/../..)

docker build -t pylance_manylinux -f ${TOOLS_DIR}/Dockerfile.manylinux2014 ${TOOLS_DIR}

docker run -v ${TOOLS_DIR}:/opt/lance/tools \
  -v ${LANCE_ROOT}:/code --rm \
  -w /code \
  pylance_manylinux \
  /opt/lance/tools/build_manylinux_wheels.sh