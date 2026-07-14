#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "usage: DATASET_ROOT=/absolute/path RESULT_ROOT=/absolute/path EXPECTED_COMMIT=<40-hex> [PYTHON=/usr/bin/python3.11] [CARGO_TARGET_DIR=/absolute/path] [UNIT_NAME=stable-row-address-smoke] $0 {install|status}" >&2
  exit 2
}

[[ $# -eq 1 ]] || usage
action=$1
UNIT_NAME=${UNIT_NAME:-stable-row-address-smoke}
[[ ${UNIT_NAME} =~ ^[a-zA-Z0-9_.@-]+$ ]] || {
  echo "UNIT_NAME contains unsupported characters" >&2
  exit 2
}

status_service() {
  sudo systemctl status "${UNIT_NAME}.service" --no-pager
}

install_service() {
  : "${DATASET_ROOT:?DATASET_ROOT is required}"
  : "${RESULT_ROOT:?RESULT_ROOT is required}"
  : "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
  PYTHON=${PYTHON:-/usr/bin/python3.11}
  CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-}
  [[ ${EXPECTED_COMMIT} =~ ^[0-9a-f]{40}$ ]] || {
    echo "EXPECTED_COMMIT must be a lowercase full Git SHA" >&2
    exit 2
  }
  for value in "${DATASET_ROOT}" "${RESULT_ROOT}" "${EXPECTED_COMMIT}" "${PYTHON}"; do
    [[ ${value} =~ ^[-_./:a-zA-Z0-9]+$ ]] || {
      echo "service environment contains unsupported characters: ${value}" >&2
      exit 2
    }
  done

  local repository_root
  local actual_commit
  local service_user
  local service_home
  local cargo_bin
  local environment_file
  local unit_file
  local environment_tmp
  local unit_tmp
  repository_root=$(git rev-parse --show-toplevel)
  actual_commit=$(git -C "${repository_root}" rev-parse HEAD)
  [[ ${actual_commit} == "${EXPECTED_COMMIT}" ]] || {
    echo "checkout ${actual_commit} does not match EXPECTED_COMMIT ${EXPECTED_COMMIT}" >&2
    exit 2
  }
  [[ -x ${PYTHON} ]] || {
    echo "PYTHON is not executable: ${PYTHON}" >&2
    exit 2
  }
  service_user=$(id -un)
  service_home=$(getent passwd "${service_user}" | cut -d: -f6)
  cargo_bin=$(dirname "$(command -v cargo)")
  mkdir -p "${RESULT_ROOT}"
  RESULT_ROOT=$(cd "${RESULT_ROOT}" && pwd -P)
  if [[ -n ${CARGO_TARGET_DIR} ]]; then
    [[ ${CARGO_TARGET_DIR} =~ ^[-_./:a-zA-Z0-9]+$ ]] || {
      echo "CARGO_TARGET_DIR contains unsupported characters" >&2
      exit 2
    }
    [[ ${CARGO_TARGET_DIR} == /* ]] || {
      echo "CARGO_TARGET_DIR must be absolute" >&2
      exit 2
    }
    mkdir -p "${CARGO_TARGET_DIR}"
    CARGO_TARGET_DIR=$(cd "${CARGO_TARGET_DIR}" && pwd -P)
  fi
  environment_file="/etc/systemd/system/${UNIT_NAME}.environment"
  unit_file="/etc/systemd/system/${UNIT_NAME}.service"
  environment_tmp="${RESULT_ROOT}/.${UNIT_NAME}.environment.tmp-$$"
  unit_tmp="${RESULT_ROOT}/.${UNIT_NAME}.tmp-$$.service"

  {
    printf 'DATASET_ROOT=%s\n' "${DATASET_ROOT}"
    printf 'RESULT_ROOT=%s\n' "${RESULT_ROOT}"
    printf 'EXPECTED_COMMIT=%s\n' "${EXPECTED_COMMIT}"
    printf 'PYTHON=%s\n' "${PYTHON}"
    if [[ -n ${CARGO_TARGET_DIR} ]]; then
      printf 'CARGO_TARGET_DIR=%s\n' "${CARGO_TARGET_DIR}"
    fi
    printf 'HOME=%s\n' "${service_home}"
    printf 'PATH=%s:/usr/local/bin:/usr/bin:/bin\n' "${cargo_bin}"
  } >"${environment_tmp}"
  chmod 0600 "${environment_tmp}"

  {
    echo '[Unit]'
    echo 'Description=Lance stable logical row-address smoke protocol'
    printf 'RequiresMountsFor=%s %s %s\n' "${repository_root}" "${DATASET_ROOT}" "${RESULT_ROOT}"
    echo
    echo '[Service]'
    echo 'Type=oneshot'
    printf 'User=%s\n' "${service_user}"
    printf 'WorkingDirectory=%s\n' "${repository_root}"
    printf 'EnvironmentFile=%s\n' "${environment_file}"
    printf 'ExecStart=/usr/bin/env bash %s/benchmarks/stable_row_address/smoke_remote.sh smoke-all\n' "${repository_root}"
    echo 'RemainAfterExit=yes'
    echo
    echo '[Install]'
    echo 'WantedBy=multi-user.target'
  } >"${unit_tmp}"
  systemd-analyze verify "${unit_tmp}"
  sudo install -m 0600 "${environment_tmp}" "${environment_file}.tmp-$$"
  sudo mv "${environment_file}.tmp-$$" "${environment_file}"
  sudo install -m 0644 "${unit_tmp}" "${unit_file}.tmp-$$"
  sudo mv "${unit_file}.tmp-$$" "${unit_file}"
  rm -f "${environment_tmp}" "${unit_tmp}"
  sudo systemctl daemon-reload
  sudo systemctl enable --now "${UNIT_NAME}.service"
}

case ${action} in
  install)
    install_service
    ;;
  status)
    status_service
    ;;
  *)
    usage
    ;;
esac
