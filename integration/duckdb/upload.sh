#!/usr/bin/env bash
# ./upload.sh s3://eto-public 0.1 or ./upload.sh s3://eto-public 0.1 cuda
set -ex

# artifact root uri like "s3://eto-public"
ARTIFACT_ROOT=$1
# the extension version (should this match lance? or i guess we need a "latest" pointer?)
VER=$2
# any string means we want cuda build
CUDA=$3

OS=$(python3 -c "import platform; system = platform.uname().system; print('osx' if system == 'Darwin' else system)")
ARCH=$(python3 -c "import platform; uname = platform.uname(); print(uname.machine)")

if [[ $OS = "osx" ]]; then
  BUILD_DIR="osx-build"
elif [[ -n "$CUDA" ]]; then
  BUILD_DIR="cuda-build"
else
  BUILD_DIR="manylinux-build"
fi

if [[ -n "$CUDA" ]]; then
  CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda.replace('.', ''))")
  ZIP_NAME="lance.duckdb_extension.${OS}.${ARCH}.cu${CUDA_VERSION}.zip"
else
  ZIP_NAME="lance.duckdb_extension.${OS}.${ARCH}.cpu.zip"
fi

# Create the local zip file
zip "${ZIP_NAME}" -xi "${BUILD_DIR}/lance.duckdb_extension"

ROOT_URI="${ARTIFACT_ROOT}/artifacts/lance/lance_duckdb"
REMOTE_URI="${ROOT_URI}/${VER}/${ZIP_NAME}"

# copy local zip file to s3
aws s3 cp "${ZIP_NAME}" "${REMOTE_URI}"

# verify the upload was successful
aws s3 ls "${REMOTE_URI}"

# cleanup
rm "${ZIP_NAME}"


# update latest
LATEST_POINTER="${ARTIFACT_ROOT}/artifacts/lance/lance_duckdb/latest"
LATEST_VER=$(aws s3 cp "${LATEST_POINTER}" - || echo "0.0.0")

if [[ "${VER}" > "${LATEST_VER}" ]]; then
  echo "${VER}" >> ./latest
  aws s3 cp latest "${LATEST_POINTER}"
  rm latest # cleanup
fi
