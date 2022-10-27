#!/usr/bin/env bash

MINOR_VERSION=${1:-cp310}

pushd ..
rm -rf wheelhouse
CIBW_TEST_SKIP=* CIBW_BUILD=${MINOR_VERSION}* python -m cibuildwheel --platform macos python && \
pip install --force-reinstall wheelhouse/*.whl && \
popd && \
pytest lance/tests