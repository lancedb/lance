#!/usr/bin/env bash
set -euo pipefail

EXPECTED_SHA=${EXPECTED_SHA:?EXPECTED_SHA must name the committed benchmark revision}
STAGE=${STAGE:-all}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
S3_BUCKET=${S3_BUCKET:-lance-bench-054483968661-us-east-2}
S3_PREFIX=${S3_PREFIX:-bench/sparse-structural-pr5/${RUN_ID}}
ROWS=${ROWS:-100000000}
BATCH_ROWS=${BATCH_ROWS:-65536}
FORMAT_BATCH_ROWS=${FORMAT_BATCH_ROWS:-1000000}
TAKE_COUNT=${TAKE_COUNT:-1024}
SMOKE_ROWS=${SMOKE_ROWS:-200000}

REPO_ROOT=$(git rev-parse --show-toplevel)
MANIFEST=${REPO_ROOT}/benchmarks/sparse-structural-final/Cargo.toml
TARGET_DIR=${CARGO_TARGET_DIR:-${REPO_ROOT}/target/sparse-structural-final}
BIN=${TARGET_DIR}/release-with-debug/sparse-structural-final-bench
RESULT_DIR=${RESULT_DIR:-${HOME}/sparse-structural-pr5-results/${RUN_ID}}

ACTUAL_SHA=$(git rev-parse HEAD)
if [[ ${ACTUAL_SHA} != "${EXPECTED_SHA}" ]]; then
    echo "expected git SHA ${EXPECTED_SHA}, found ${ACTUAL_SHA}" >&2
    exit 1
fi

mkdir -p "${RESULT_DIR}"

build_bench() {
    CARGO_TARGET_DIR="${TARGET_DIR}" cargo build \
        --profile release-with-debug \
        --manifest-path "${MANIFEST}"
}

validate_csv() {
    local path=$1
    local header=$2
    local lines=$3
    local actual_header
    local actual_lines
    actual_header=$(head -n 1 "${path}")
    actual_lines=$(wc -l < "${path}")
    if [[ ${actual_header} != "${header}" ]]; then
        echo "unexpected CSV header in ${path}: ${actual_header}" >&2
        exit 1
    fi
    if [[ ${actual_lines} -ne ${lines} ]]; then
        echo "unexpected CSV line count in ${path}: expected ${lines}, got ${actual_lines}" >&2
        exit 1
    fi
}

upload_artifact() {
    local path=$1
    local destination=$2
    aws s3 cp "${path}" "s3://${S3_BUCKET}/${S3_PREFIX}/${destination}"
}

run_lance() {
    local run_rows=$1
    local label=$2
    local prefix=${S3_PREFIX}/lance-${label}-${RUN_ID}
    local csv=${RESULT_DIR}/lance-${label}-${RUN_ID}.csv
    local tmp=${csv}.tmp
    local err=${RESULT_DIR}/lance-${label}-${RUN_ID}.err
    local failures=${RESULT_DIR}/lance-${label}-${RUN_ID}-failures.csv
    local header=case,mode,rows,bytes,objects,data_files,pages,op,phase,ms,out_rows,uri
    local case_name
    local mode

    printf '%s\n' "${header}" > "${tmp}"
    printf 'case,mode,exit_code\n' > "${failures}"
    : > "${err}"
    for case_name in hnsw uniform deep; do
        for mode in sparse miniblock fullzip; do
            local part=${RESULT_DIR}/lance-${label}-${RUN_ID}-${case_name}-${mode}.csv
            local part_err=${RESULT_DIR}/lance-${label}-${RUN_ID}-${case_name}-${mode}.err
            local exit_code=0
            if RUST_BACKTRACE=1 \
                ACTION=lance \
                ROWS="${run_rows}" \
                BATCH_ROWS="${BATCH_ROWS}" \
                TAKE_COUNT="${TAKE_COUNT}" \
                CASES="${case_name}" \
                MODES="${mode}" \
                S3_BUCKET="${S3_BUCKET}" \
                S3_PREFIX="${prefix}" \
                    "${BIN}" > "${part}" 2> "${part_err}"; then
                :
            else
                exit_code=$?
                printf '%s,%s,%s\n' "${case_name}" "${mode}" "${exit_code}" >> "${failures}"
            fi
            if [[ $(head -n 1 "${part}") != "${header}" ]]; then
                echo "unexpected CSV header in ${part}" >&2
                exit 1
            fi
            if [[ ${exit_code} -eq 0 && $(wc -l < "${part}") -ne 10 ]]; then
                echo "unexpected successful CSV line count in ${part}" >&2
                exit 1
            fi
            tail -n +2 "${part}" >> "${tmp}"
            printf 'case=%s mode=%s exit_code=%s\n' \
                "${case_name}" "${mode}" "${exit_code}" >> "${err}"
            cat "${part_err}" >> "${err}"
        done
    done

    mv "${tmp}" "${csv}"
    upload_artifact "${csv}" "artifacts/$(basename "${csv}")"
    upload_artifact "${err}" "artifacts/$(basename "${err}")"
    upload_artifact "${failures}" "artifacts/$(basename "${failures}")"
}

run_formats() {
    local run_rows=$1
    local label=$2
    local out_dir=${RESULT_DIR}/formats-${label}-${RUN_ID}
    local csv=${RESULT_DIR}/formats-${label}-${RUN_ID}.csv
    local tmp=${csv}.tmp
    local err=${RESULT_DIR}/formats-${label}-${RUN_ID}.err

    mkdir -p "${out_dir}"
    ACTION=formats \
    ROWS="${run_rows}" \
    FORMAT_BATCH_ROWS="${FORMAT_BATCH_ROWS}" \
    CASES=hnsw,uniform,deep \
    OUT_DIR="${out_dir}" \
        "${BIN}" > "${tmp}" 2> "${err}"

    validate_csv "${tmp}" "case,format,rows,bytes,seconds,path" 7
    mv "${tmp}" "${csv}"
    upload_artifact "${csv}" "artifacts/$(basename "${csv}")"
    upload_artifact "${err}" "artifacts/$(basename "${err}")"
    for output in "${out_dir}"/*; do
        upload_artifact "${output}" "formats/${label}-${RUN_ID}/$(basename "${output}")"
    done
}

case "${STAGE}" in
    build)
        build_bench
        ;;
    smoke)
        build_bench
        run_lance "${SMOKE_ROWS}" smoke
        run_formats "${SMOKE_ROWS}" smoke
        ;;
    lance)
        build_bench
        run_lance "${ROWS}" 100m
        ;;
    formats)
        build_bench
        run_formats "${ROWS}" 100m
        ;;
    all)
        build_bench
        run_lance "${ROWS}" 100m
        run_formats "${ROWS}" 100m
        ;;
    *)
        echo "unknown STAGE ${STAGE}" >&2
        exit 1
        ;;
esac
