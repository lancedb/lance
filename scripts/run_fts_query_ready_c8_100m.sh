#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 BASELINE_COMMIT CANDIDATE_COMMIT OUTPUT_ROOT" >&2
  exit 2
fi

BASELINE_COMMIT=$(git rev-parse "$1^{commit}")
CANDIDATE_COMMIT=$(git rev-parse "$2^{commit}")
OUTPUT_ROOT=$3
REPO_ROOT=$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)
BINARY_CACHE_ROOT=/home/ec2-user/bench-results/query-ready-binary-cache/commits
BASELINE_BINARY=$BINARY_CACHE_ROOT/$BASELINE_COMMIT/fts_s3_benchmark
CANDIDATE_BINARY=$BINARY_CACHE_ROOT/$CANDIDATE_COMMIT/fts_s3_benchmark
QUERY_FILE=$REPO_ROOT/rust/examples/fts_100m_queries.txt
DATASET_URI=s3://mmlb-us-east-1/repros/fts-global-stats-20260715T103302Z/bench/user_case.lance
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=$OUTPUT_ROOT/$RUN_ID
RUNS=${LANCE_FTS_BENCH_RUNS:-2}
ROUNDS=${LANCE_FTS_BENCH_ROUNDS:-1000}
CONCURRENCY=${LANCE_FTS_BENCH_CONCURRENCY:-8}

for binary in "$BASELINE_BINARY" "$CANDIDATE_BINARY"; do
  if [[ ! -x "$binary" ]]; then
    echo "benchmark binary is missing or not executable: $binary" >&2
    exit 2
  fi
done
if [[ -e "$RUN_DIR" ]]; then
  echo "refusing to reuse existing run directory: $RUN_DIR" >&2
  exit 2
fi

mkdir -p "$RUN_DIR/results" "$RUN_DIR/timing"
cp "$QUERY_FILE" "$RUN_DIR/queries.txt"

{
  echo "run_id=$RUN_ID"
  echo "dataset_uri=$DATASET_URI"
  echo "baseline_commit=$BASELINE_COMMIT"
  echo "candidate_commit=$CANDIDATE_COMMIT"
  echo "runs=$RUNS"
  echo "rounds=$ROUNDS"
  echo "query_count=$(grep -Ec '^[^#[:space:]]' "$RUN_DIR/queries.txt")"
  echo "concurrency=$CONCURRENCY"
  echo "prewarm_index=full_content_idx"
  echo "index_cache_size_gib=192"
  uname -a
  lscpu
  free -h
} > "$RUN_DIR/environment.txt"

run_variant() {
  local variant=$1
  local run=$2
  local binary
  if [[ "$variant" == "baseline" ]]; then
    binary=$BASELINE_BINARY
  else
    binary=$CANDIDATE_BINARY
  fi

  /usr/bin/time -v -a -o "$RUN_DIR/timing/throughput-$variant-run-$run.txt" \
    "$binary" \
      --uri "$DATASET_URI" \
      --column full_content \
      --k 10 \
      --expected-rows 100000000 \
      --expected-index full_content_idx \
      --prewarm-index full_content_idx \
      --index-cache-size-gib 192 \
      --label "throughput-$variant-run-$run" \
      --query-file "$RUN_DIR/queries.txt" \
      --warmup-rounds 0 \
      --measured-rounds "$ROUNDS" \
      --concurrency "$CONCURRENCY" \
      >> "$RUN_DIR/results/throughput_$variant.jsonl" \
      2>> "$RUN_DIR/results/stderr.log"
}

for ((run = 0; run < RUNS; run++)); do
  if ((run % 2 == 0)); then
    order=(candidate baseline)
  else
    order=(baseline candidate)
  fi
  for variant in "${order[@]}"; do
    run_variant "$variant" "$run"
  done
done

python3 "$REPO_ROOT/scripts/analyze_fts_s3_benchmark.py" \
  "$RUN_DIR/results" \
  --phases throughput \
  --output "$RUN_DIR/summary.json"

echo "$RUN_DIR"
