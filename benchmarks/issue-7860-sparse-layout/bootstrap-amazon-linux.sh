#!/usr/bin/env bash
set -euo pipefail

sudo dnf install -y \
    clang-devel \
    cmake \
    gcc \
    gcc-c++ \
    git \
    make \
    openssl-devel \
    perl \
    pkgconf-pkg-config \
    protobuf-compiler \
    protobuf-devel \
    python3-devel

if ! command -v rustup >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --profile minimal --default-toolchain 1.97.0
fi

source "${HOME}/.cargo/env"
rustup toolchain install 1.97.0 --profile minimal
rustup default 1.97.0

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="${HOME}/.local/bin:${PATH}"
git --version
rustc --version
cargo --version
uv --version
python3 --version
