#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "usage: DATASET_ROOT=s3://bucket/prefix RESULT_ROOT=/path AWS_REGION=region EXPECTED_COMMIT=<40-hex> [PYTHON=python3] $0 {run-all|report-all|release-all}" >&2
  exit 2
}

[[ $# -eq 1 ]] || usage
action=$1
: "${DATASET_ROOT:?DATASET_ROOT is required}"
: "${RESULT_ROOT:?RESULT_ROOT is required}"
: "${AWS_REGION:?AWS_REGION is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
PYTHON=${PYTHON:-python3}
[[ ${EXPECTED_COMMIT} =~ ^[0-9a-f]{40}$ ]] || {
  echo "EXPECTED_COMMIT must be a lowercase full Git SHA" >&2
  exit 2
}
[[ ${DATASET_ROOT} == s3://* ]] || {
  echo "DATASET_ROOT must be an s3:// URI" >&2
  exit 2
}

repository_root=$(git rev-parse --show-toplevel)
cd "${repository_root}"
mkdir -p "${RESULT_ROOT}"

verify_checkout() {
  local checkout_commit
  local checkout_status
  if ! checkout_commit=$(git rev-parse HEAD); then
    echo "failed to resolve checkout HEAD" >&2
    return 2
  fi
  if [[ ${checkout_commit} != "${EXPECTED_COMMIT}" ]]; then
    echo "checkout ${checkout_commit} does not match EXPECTED_COMMIT ${EXPECTED_COMMIT}" >&2
    return 2
  fi
  if ! checkout_status=$(git status --porcelain=v1 --untracked-files=all); then
    echo "failed to inspect checkout status" >&2
    return 2
  fi
  if [[ -n ${checkout_status} ]]; then
    echo "release protocol requires a clean checkout" >&2
    return 2
  fi
}

verify_checkout || exit $?
actual_commit=${EXPECTED_COMMIT}

bucket_and_prefix=${DATASET_ROOT#s3://}
bucket=${bucket_and_prefix%%/*}
bucket_region=$(aws s3api get-bucket-location \
  --bucket "${bucket}" \
  --query LocationConstraint \
  --output text)
if [[ -z ${bucket_region} || ${bucket_region} == None ]]; then
  bucket_region=us-east-1
fi
if [[ ${bucket_region} != "${AWS_REGION}" ]]; then
  echo "bucket region ${bucket_region} does not match AWS_REGION ${AWS_REGION}" >&2
  exit 2
fi

export AWS_DEFAULT_REGION=${AWS_REGION}
output_base="${RESULT_ROOT}/stable-row-address-release.jsonl"
protocol=(
  "${PYTHON}" benchmarks/stable_row_address/protocol.py
  --dataset-root "${DATASET_ROOT}"
  --storage s3
  --output "${output_base}"
  --profile release
  --track matrix
  --track sustained
  --track adversarial_natural
  --track adversarial_aligned
  --shard-count 9
)

run_all() {
  exec 9>"${RESULT_ROOT}/stable-row-address-release.lock" || return 2
  flock -n 9 || {
    echo "another release protocol owns ${RESULT_ROOT}" >&2
    return 2
  }
  local status=0
  local shard_id
  local shard_output
  local shard_status
  local execution_incomplete=0
  local -a command
  rm -f "${RESULT_ROOT}/stable-row-address-release.pass" || return 2
  rm -f "${RESULT_ROOT}/stable-row-address-release.execution-complete" || return 2
  for shard_index in {0..8}; do
    shard_id=$(printf 'shard-%03d-of-009' "${shard_index}")
    shard_output="${RESULT_ROOT}/stable-row-address-release.${shard_id}.jsonl"
    command=("${protocol[@]}" --shard-index "${shard_index}")
    if [[ -e ${shard_output} || -e ${shard_output}.protocol.json ]]; then
      command+=(--resume)
    fi
    echo "starting ${shard_id}" >&2
    if "${command[@]}"; then
      :
    else
      shard_status=$?
      (( shard_status > status )) && status=${shard_status}
      (( shard_status > 1 )) && execution_incomplete=1
      echo "${shard_id} did not complete successfully (status ${shard_status})" >&2
    fi
  done
  if [[ ${execution_incomplete} -eq 0 ]]; then
    local marker
    marker="${RESULT_ROOT}/.stable-row-address-release.execution-complete.tmp-$$"
    git rev-parse HEAD >"${marker}" || return 2
    mv "${marker}" "${RESULT_ROOT}/stable-row-address-release.execution-complete" || return 2
  fi
  return "${status}"
}

report_all() {
  local status=0
  local shard_id
  local input
  local report_status
  local aggregate_commit
  rm -f "${RESULT_ROOT}/stable-row-address-release.pass" || return 2
  for shard_index in {0..8}; do
    shard_id=$(printf 'shard-%03d-of-009' "${shard_index}")
    input="${RESULT_ROOT}/stable-row-address-release.${shard_id}.jsonl"
    if "${PYTHON}" benchmarks/stable_row_address/protocol_report.py \
      "${input}" \
      --markdown "${input%.jsonl}.md" \
      --json "${input%.jsonl}.report.json"; then
      :
    else
      report_status=$?
      (( report_status > status )) && status=${report_status}
    fi
  done
  rm -f \
    "${RESULT_ROOT}/stable-row-address-release.aggregate.md" \
    "${RESULT_ROOT}/stable-row-address-release.aggregate.json" || return 2
  if "${PYTHON}" benchmarks/stable_row_address/protocol_aggregate.py \
    "${RESULT_ROOT}"/stable-row-address-release.shard-*-of-009.jsonl \
    --expected-commit "${actual_commit}" \
    --execution-marker "${RESULT_ROOT}/stable-row-address-release.execution-complete" \
    --markdown "${RESULT_ROOT}/stable-row-address-release.aggregate.md" \
    --json "${RESULT_ROOT}/stable-row-address-release.aggregate.json"; then
    :
  else
    report_status=$?
    (( report_status > status )) && status=${report_status}
  fi
  if [[ -f ${RESULT_ROOT}/stable-row-address-release.aggregate.json ]]; then
    aggregate_commit=$("${PYTHON}" -c 'import json, pathlib, sys; print(json.loads(pathlib.Path(sys.argv[1]).read_text())["commit"])' "${RESULT_ROOT}/stable-row-address-release.aggregate.json" 2>/dev/null) || status=2
    [[ ${aggregate_commit:-} == "${actual_commit}" ]] || status=2
  else
    status=2
  fi
  if [[ ${status} -eq 0 ]]; then
    local aggregate_hash
    local marker
    verify_checkout || return 2
    aggregate_hash=$("${PYTHON}" -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "${RESULT_ROOT}/stable-row-address-release.aggregate.json") || return 2
    marker="${RESULT_ROOT}/.stable-row-address-release.pass.tmp-$$"
    {
      echo "commit=${actual_commit}"
      echo "aggregate_sha256=${aggregate_hash}"
    } >"${marker}" || return 2
    mv "${marker}" "${RESULT_ROOT}/stable-row-address-release.pass" || return 2
  fi
  return "${status}"
}

release_all() {
  local run_status=0
  local report_status=0
  local aggregate_verdict
  run_all || run_status=$?
  report_all || report_status=$?
  if [[ ${report_status} -eq 1 ]]; then
    verify_checkout || return 2
    aggregate_verdict=$("${PYTHON}" -c 'import json, pathlib, sys; print(json.loads(pathlib.Path(sys.argv[1]).read_text())["verdict"])' "${RESULT_ROOT}/stable-row-address-release.aggregate.json" 2>/dev/null) || return 2
    [[ ${aggregate_verdict} == FAIL ]] && return 65
    return 2
  fi
  if [[ ${report_status} -ne 0 || ${run_status} -ne 0 ]]; then
    return 2
  fi
  return 0
}

case ${action} in
  run-all)
    run_all
    ;;
  report-all)
    report_all
    ;;
  release-all)
    release_all
    ;;
  *)
    usage
    ;;
esac
