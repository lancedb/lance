#!/usr/bin/env bash
set -euo pipefail

version="${PROTOC_VERSION:-24.4}"
install_dir="${PROTOC_INSTALL_DIR:-/usr/local}"
machine="${1:-$(uname -m)}"

case "${machine}" in
  aarch64 | arm64)
    asset_arch="aarch_64"
    ;;
  x86_64)
    asset_arch="x86_64"
    ;;
  *)
    echo "Unsupported protoc architecture: ${machine}" >&2
    exit 1
    ;;
esac

zip_path="/tmp/protoc-${version}-linux-${asset_arch}.zip"
url="https://github.com/protocolbuffers/protobuf/releases/download/v${version}/protoc-${version}-linux-${asset_arch}.zip"

for attempt in 1 2 3 4 5; do
  rm -f "${zip_path}"

  if curl -fsSL --connect-timeout 15 --max-time 120 -o "${zip_path}" "${url}" \
    && unzip -tq "${zip_path}" >/dev/null; then
    break
  fi

  if [[ "${attempt}" == "5" ]]; then
    echo "Failed to download a valid protoc archive from ${url}" >&2
    exit 1
  fi

  sleep "$((attempt * 2))"
done

unzip -q -o "${zip_path}" -d "${install_dir}"
rm -f "${zip_path}"

if [[ "$(uname -s)" == "Linux" ]]; then
  "${install_dir}/bin/protoc" --version
else
  test -f "${install_dir}/bin/protoc"
fi
