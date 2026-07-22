#!/usr/bin/env bash
set -euo pipefail

EXPECTED_SHA=${EXPECTED_SHA:?EXPECTED_SHA must name the committed benchmark revision}
RESULT_ROOT=${RESULT_ROOT:?RESULT_ROOT must name an existing benchmark result directory}
ATTRS=${ATTRS:-50}
TARGET_DIR=${CARGO_TARGET_DIR:-${HOME}/lance-issue-7860-target}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
ACTUAL_SHA=$(git -C "${REPO_ROOT}" rev-parse HEAD)
if [[ ${ACTUAL_SHA} != "${EXPECTED_SHA}" ]]; then
    echo "expected git SHA ${EXPECTED_SHA}, found ${ACTUAL_SHA}" >&2
    exit 1
fi

for name in \
    v2_0_default \
    v2_1_default \
    v2_3_default \
    v2_3_miniblock \
    v2_3_sparse; do
    output="${RESULT_ROOT}/layout-${name}.json"
    temporary="${output}.tmp"
    CARGO_TARGET_DIR="${TARGET_DIR}" cargo bench \
        --quiet \
        --manifest-path "${REPO_ROOT}/Cargo.toml" \
        -p lance \
        --bench s3_file_reader_diagnostics \
        -- \
        --uri "${RESULT_ROOT}/${name}" \
        --columns all \
        --describe-layout > "${temporary}"
    cd "${REPO_ROOT}/python"
    uv run python -m json.tool "${temporary}" >/dev/null
    mv "${temporary}" "${output}"
done

cd "${REPO_ROOT}/python"
uv run python "${SCRIPT_DIR}/validate_layouts.py" \
    --root "${RESULT_ROOT}" \
    --attrs "${ATTRS}"
