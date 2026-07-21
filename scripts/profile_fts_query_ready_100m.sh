#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 BASELINE_BINARY CANDIDATE_BINARY OUTPUT_DIR" >&2
  exit 2
fi

BASELINE_BINARY=$1
CANDIDATE_BINARY=$2
OUTPUT_DIR=$3
REPO_ROOT=$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)
QUERY_FILE=$REPO_ROOT/rust/examples/fts_100m_queries.txt
DATASET_URI=s3://mmlb-us-east-1/repros/fts-global-stats-20260715T103302Z/bench/user_case.lance

mkdir -p "$OUTPUT_DIR"

run_profile() {
  local label=$1
  local binary=$2
  sudo perf stat \
    --all-cpus \
    --delay 90000 \
    --event task-clock,cycles,instructions,branches,branch-misses,context-switches,cpu-migrations,page-faults \
    --output "$OUTPUT_DIR/$label.perf.txt" \
    -- \
    "$binary" \
      --uri "$DATASET_URI" \
      --column full_content \
      --k 10 \
      --expected-rows 100000000 \
      --expected-index full_content_idx \
      --prewarm-index full_content_idx \
      --index-cache-size-gib 192 \
      --label "$label" \
      --query-file "$QUERY_FILE" \
      --warmup-rounds 0 \
      --measured-rounds 1000 \
      --concurrency 1 \
      > "$OUTPUT_DIR/$label.jsonl"
}

run_profile baseline "$BASELINE_BINARY"
run_profile candidate "$CANDIDATE_BINARY"
