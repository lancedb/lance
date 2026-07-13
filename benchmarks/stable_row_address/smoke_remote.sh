#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "usage: DATASET_ROOT=/absolute/path RESULT_ROOT=/absolute/path EXPECTED_COMMIT=<40-hex> [PYTHON=python3] $0 {run-all|report-all|smoke-all}" >&2
  exit 2
}

[[ $# -eq 1 ]] || usage
action=$1
: "${DATASET_ROOT:?DATASET_ROOT is required}"
: "${RESULT_ROOT:?RESULT_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
PYTHON=${PYTHON:-python3}
[[ ${EXPECTED_COMMIT} =~ ^[0-9a-f]{40}$ ]] || {
  echo "EXPECTED_COMMIT must be a lowercase full Git SHA" >&2
  exit 2
}
[[ ${DATASET_ROOT} == /* ]] || {
  echo "DATASET_ROOT must be an absolute local path" >&2
  exit 2
}
[[ ${RESULT_ROOT} == /* ]] || {
  echo "RESULT_ROOT must be an absolute local path" >&2
  exit 2
}

repository_root=$(git rev-parse --show-toplevel)
cd "${repository_root}"
mkdir -p "${DATASET_ROOT}" "${RESULT_ROOT}"

verify_checkout() {
  local checkout_commit
  local checkout_status
  checkout_commit=$(git rev-parse HEAD) || return 2
  if [[ ${checkout_commit} != "${EXPECTED_COMMIT}" ]]; then
    echo "checkout ${checkout_commit} does not match EXPECTED_COMMIT ${EXPECTED_COMMIT}" >&2
    return 2
  fi
  checkout_status=$(git status --porcelain=v1 --untracked-files=all) || return 2
  if [[ -n ${checkout_status} ]]; then
    echo "smoke protocol requires a clean checkout" >&2
    return 2
  fi
}

verify_checkout || exit $?
output="${RESULT_ROOT}/stable-row-address-smoke.jsonl"
report_json="${RESULT_ROOT}/stable-row-address-smoke.report.json"
report_markdown="${RESULT_ROOT}/stable-row-address-smoke.md"
pass_marker="${RESULT_ROOT}/stable-row-address-smoke.pass"
execution_marker="${RESULT_ROOT}/stable-row-address-smoke.execution-complete"
protocol=(
  "${PYTHON}" benchmarks/stable_row_address/protocol.py
  --dataset-root "${DATASET_ROOT}"
  --storage ebs
  --output "${output}"
  --profile smoke
  --track matrix
  --track sustained
  --track adversarial_natural
  --track adversarial_aligned
)

run_all() {
  exec 9>"${RESULT_ROOT}/stable-row-address-smoke.lock" || return 2
  flock -n 9 || {
    echo "another smoke protocol owns ${RESULT_ROOT}" >&2
    return 2
  }
  rm -f "${pass_marker}" "${execution_marker}" || return 2
  local -a command=("${protocol[@]}")
  if [[ -e ${output} || -e ${output}.protocol.json ]]; then
    command+=(--resume)
  fi
  "${command[@]}" || return $?
  local marker="${RESULT_ROOT}/.stable-row-address-smoke.execution-complete.tmp-$$"
  printf '%s\n' "${EXPECTED_COMMIT}" >"${marker}" || return 2
  mv "${marker}" "${execution_marker}" || return 2
}

report_all() {
  rm -f "${pass_marker}" "${report_json}" "${report_markdown}" || return 2
  "${PYTHON}" benchmarks/stable_row_address/protocol_report.py \
    "${output}" \
    --markdown "${report_markdown}" \
    --json "${report_json}" || return $?
  verify_checkout || return 2
  local report_commit
  local report_verdict
  local report_hash
  report_commit=$("${PYTHON}" -c 'import json, pathlib, sys; print(json.loads(pathlib.Path(sys.argv[1]).read_text())["commit"])' "${report_json}") || return 2
  report_verdict=$("${PYTHON}" -c 'import json, pathlib, sys; print(json.loads(pathlib.Path(sys.argv[1]).read_text())["verdict"])' "${report_json}") || return 2
  [[ ${report_commit} == "${EXPECTED_COMMIT}" ]] || {
    echo "smoke report commit ${report_commit} does not match ${EXPECTED_COMMIT}" >&2
    return 2
  }
  [[ ${report_verdict} == PASS ]] || {
    echo "smoke report verdict is ${report_verdict}, expected PASS" >&2
    return 1
  }
  report_hash=$("${PYTHON}" -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "${report_json}") || return 2
  local marker="${RESULT_ROOT}/.stable-row-address-smoke.pass.tmp-$$"
  {
    echo "commit=${EXPECTED_COMMIT}"
    echo "report_sha256=${report_hash}"
  } >"${marker}" || return 2
  mv "${marker}" "${pass_marker}" || return 2
}

smoke_all() {
  run_all
  report_all
}

case ${action} in
  run-all)
    run_all
    ;;
  report-all)
    report_all
    ;;
  smoke-all)
    smoke_all
    ;;
  *)
    usage
    ;;
esac
