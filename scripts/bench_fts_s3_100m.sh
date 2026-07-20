#!/usr/bin/env bash

# Reproducible baseline/candidate FTS benchmark over the existing 100M S3
# corpus. Source is checked out exclusively through Git worktrees; every run
# gets a new directory and is preserved for later inspection.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 BASELINE_REF CANDIDATE_REF [OUTPUT_ROOT]" >&2
  exit 2
fi

BASELINE_REF=$1
CANDIDATE_REF=$2
OUTPUT_ROOT=${3:-/mnt/lance-fts-document-benchmark}
DATASET_URI=${LANCE_FTS_BENCH_URI:-s3://mmlb-us-east-1/repros/fts-global-stats-20260715T103302Z/bench/user_case.lance}
TEXT_COLUMN=${LANCE_FTS_BENCH_COLUMN:-full_content}
INDEX_NAME=${LANCE_FTS_BENCH_INDEX:-full_content_idx}
EXPECTED_ROWS=${LANCE_FTS_BENCH_EXPECTED_ROWS:-100000000}
COLD_TRIALS=${LANCE_FTS_BENCH_COLD_TRIALS:-2}
WARM_RUNS=${LANCE_FTS_BENCH_WARM_RUNS:-2}
WARMUP_ROUNDS=${LANCE_FTS_BENCH_WARMUP_ROUNDS:-1}
WARM_ROUNDS=${LANCE_FTS_BENCH_WARM_ROUNDS:-5}
THROUGHPUT_CONCURRENCY=${LANCE_FTS_BENCH_CONCURRENCY:-8}
THROUGHPUT_ROUNDS=${LANCE_FTS_BENCH_THROUGHPUT_ROUNDS:-5}

REPO_ROOT=$(git rev-parse --show-toplevel)
QUERY_FILE=${LANCE_FTS_BENCH_QUERY_FILE:-$REPO_ROOT/rust/examples/fts_100m_queries.txt}
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=$OUTPUT_ROOT/$RUN_ID

if [[ -e "$RUN_DIR" ]]; then
  echo "refusing to reuse existing run directory: $RUN_DIR" >&2
  exit 2
fi
if [[ ! -f "$QUERY_FILE" ]]; then
  echo "query file does not exist: $QUERY_FILE" >&2
  exit 2
fi

mkdir -p "$RUN_DIR/bin" "$RUN_DIR/results" "$RUN_DIR/timing" "$RUN_DIR/target"

BASELINE_COMMIT=$(git rev-parse "$BASELINE_REF^{commit}")
CANDIDATE_COMMIT=$(git rev-parse "$CANDIDATE_REF^{commit}")
if [[ "$BASELINE_COMMIT" == "$CANDIDATE_COMMIT" ]]; then
  echo "baseline and candidate resolve to the same commit" >&2
  exit 2
fi

git worktree add --detach "$RUN_DIR/source-baseline" "$BASELINE_COMMIT"
git worktree add --detach "$RUN_DIR/source-candidate" "$CANDIDATE_COMMIT"

cp "$QUERY_FILE" "$RUN_DIR/queries.txt"

{
  echo "run_id=$RUN_ID"
  echo "dataset_uri=$DATASET_URI"
  echo "text_column=$TEXT_COLUMN"
  echo "index_name=$INDEX_NAME"
  echo "expected_rows=$EXPECTED_ROWS"
  echo "cold_trials=$COLD_TRIALS"
  echo "warm_runs=$WARM_RUNS"
  echo "warmup_rounds=$WARMUP_ROUNDS"
  echo "warm_rounds=$WARM_ROUNDS"
  echo "throughput_concurrency=$THROUGHPUT_CONCURRENCY"
  echo "throughput_rounds=$THROUGHPUT_ROUNDS"
  echo "baseline_commit=$BASELINE_COMMIT"
  echo "candidate_commit=$CANDIDATE_COMMIT"
  echo "aws_region=${AWS_REGION:-${AWS_DEFAULT_REGION:-unset}}"
  uname -a
  lscpu
  free -h
} > "$RUN_DIR/environment.txt"

build_binary() {
  local source_dir=$1
  local output_name=$2
  (
    cd "$source_dir"
    CARGO_TARGET_DIR="$RUN_DIR/target" cargo build \
      -p lance-examples \
      --example fts_s3_benchmark \
      --profile release-with-debug
  )
  cp "$RUN_DIR/target/release-with-debug/examples/fts_s3_benchmark" \
    "$RUN_DIR/bin/$output_name"
}

build_binary "$RUN_DIR/source-baseline" baseline
build_binary "$RUN_DIR/source-candidate" candidate

export AWS_REGION=${AWS_REGION:-us-east-1}
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-$AWS_REGION}

COMMON_ARGS=(
  --uri "$DATASET_URI"
  --column "$TEXT_COLUMN"
  --k 10
  --expected-rows "$EXPECTED_ROWS"
  --expected-index "$INDEX_NAME"
)

run_benchmark() {
  local binary=$1
  local label=$2
  local output=$3
  shift 3
  /usr/bin/time -v -a -o "$RUN_DIR/timing/${label}.txt" \
    "$binary" "${COMMON_ARGS[@]}" --label "$label" "$@" \
    >> "$output" 2>> "$RUN_DIR/results/stderr.log"
}

mapfile -t QUERIES < <(sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' "$RUN_DIR/queries.txt" | sed -e '/^$/d' -e '/^#/d')
if [[ ${#QUERIES[@]} -eq 0 ]]; then
  echo "query panel is empty" >&2
  exit 2
fi

for ((trial = 0; trial < COLD_TRIALS; trial++)); do
  if (( trial % 2 == 0 )); then
    ORDER=(baseline candidate)
  else
    ORDER=(candidate baseline)
  fi
  for query in "${QUERIES[@]}"; do
    for variant in "${ORDER[@]}"; do
      run_benchmark \
        "$RUN_DIR/bin/$variant" \
        "cold-${variant}-trial-${trial}" \
        "$RUN_DIR/results/cold_${variant}.jsonl" \
        --query "$query" \
        --warmup-rounds 0 \
        --measured-rounds 1 \
        --concurrency 1
    done
  done
done

for ((run = 0; run < WARM_RUNS; run++)); do
  if (( run % 2 == 0 )); then
    ORDER=(baseline candidate)
  else
    ORDER=(candidate baseline)
  fi
  for variant in "${ORDER[@]}"; do
    run_benchmark \
      "$RUN_DIR/bin/$variant" \
      "warm-${variant}-run-${run}" \
      "$RUN_DIR/results/warm_${variant}.jsonl" \
      --query-file "$RUN_DIR/queries.txt" \
      --warmup-rounds "$WARMUP_ROUNDS" \
      --measured-rounds "$WARM_ROUNDS" \
      --concurrency 1
  done
done

for ((run = 0; run < WARM_RUNS; run++)); do
  if (( run % 2 == 0 )); then
    ORDER=(candidate baseline)
  else
    ORDER=(baseline candidate)
  fi
  for variant in "${ORDER[@]}"; do
    run_benchmark \
      "$RUN_DIR/bin/$variant" \
      "throughput-${variant}-run-${run}" \
      "$RUN_DIR/results/throughput_${variant}.jsonl" \
      --query-file "$RUN_DIR/queries.txt" \
      --warmup-rounds "$WARMUP_ROUNDS" \
      --measured-rounds "$THROUGHPUT_ROUNDS" \
      --concurrency "$THROUGHPUT_CONCURRENCY"
  done
done

python3 "$RUN_DIR/source-candidate/scripts/analyze_fts_s3_benchmark.py" \
  "$RUN_DIR/results" \
  --output "$RUN_DIR/summary.json"

echo "$RUN_DIR"
