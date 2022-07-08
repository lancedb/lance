#!/bin/sh

set -ex

APACHE_ARROW_VERSION=release-8.0.0

# Build apache arrow
function build_arrow {
    git clone git@github.com:apache/arrow
    pushd arrow
    git checkout ${APACHE_ARROW_VERSION}
    git submodule update --init
    pushd python
    pip install -r requirements-build.txt

    export OPENSSL_ROOT_DIR=/opt/homebrew/opt/openssl@1.1
    python setup.py build_ext --inplace --with-dataset --with-parquet --with-s3
    python setup.py develop
}

build_arrow