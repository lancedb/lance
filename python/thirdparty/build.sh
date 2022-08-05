#!/bin/bash

set -ex

APACHE_ARROW_VERSION=apache-arrow-8.0.1

# Build apache arrow
build_arrow() {
    git clone git@github.com:apache/arrow
    pushd arrow
    git pull --tags    
    git checkout ${APACHE_ARROW_VERSION}
    git submodule update --init
    pushd python
    pip install -r requirements-build.txt

    export OPENSSL_ROOT_DIR=/opt/homebrew/opt/openssl@1.1
    python setup.py build_ext --inplace --with-dataset --with-parquet --with-s3
    python setup.py develop
}

build_arrow
