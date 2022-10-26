#!/usr/bin/env bash

pushd ..
CIBW_TEST_SKIP=* CIBW_BUILD=cp310* python -m cibuildwheel --platform macos python && \
pip install --force-reinstall wheelhouse/pylance-0.1.5.dev0-cp310-cp310-macosx_11_0_arm64.whl && \
popd && \
pytest lance/tests