#!/usr/bin/env bash
set -euo pipefail

EXPECTED_SHA=${EXPECTED_SHA:?EXPECTED_SHA must name the committed benchmark revision}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
REPEATS=${REPEATS:-11}
RESULT_ROOT=${RESULT_ROOT:-${HOME}/lance-issue-7860-results/${RUN_ID}}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
ACTUAL_SHA=$(git -C "${REPO_ROOT}" rev-parse HEAD)
if [[ ${ACTUAL_SHA} != "${EXPECTED_SHA}" ]]; then
    echo "expected git SHA ${EXPECTED_SHA}, found ${ACTUAL_SHA}" >&2
    exit 1
fi

source "${HOME}/.cargo/env"
export PATH="${HOME}/.local/bin:${PATH}"

cd "${REPO_ROOT}/python"
make install
uv run python "${REPO_ROOT}/benchmarks/issue-7860-sparse-layout/benchmark.py" \
    --root "${RESULT_ROOT}" \
    --rows 50000 \
    --attrs 50 \
    --repeats "${REPEATS}" \
    --seed 0 \
    --expected-sha "${EXPECTED_SHA}"

RESULT_ROOT="${RESULT_ROOT}" ATTRS=50 \
    "${REPO_ROOT}/benchmarks/issue-7860-sparse-layout/inspect-layouts.sh"
