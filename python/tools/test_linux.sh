#!/usr/bin/env bash

set -e

MINOR_VERSION=${1:-cp310}

sudo rm -rf wheels
./tools/build_wheel.sh $MINOR_VERSION
pip install --force-reinstall wheels/*.whl
pytest lance/tests