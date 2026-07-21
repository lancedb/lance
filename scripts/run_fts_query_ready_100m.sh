#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 BASELINE_REF CANDIDATE_REF OUTPUT_ROOT" >&2
  exit 2
fi

export LANCE_FTS_BENCH_COLD_TRIALS=0
export LANCE_FTS_BENCH_WARM_RUNS=2
export LANCE_FTS_BENCH_WARMUP_ROUNDS=0
export LANCE_FTS_BENCH_WARM_ROUNDS=100
export LANCE_FTS_BENCH_CONCURRENCY=8
export LANCE_FTS_BENCH_THROUGHPUT_ROUNDS=100
export LANCE_FTS_BENCH_PREWARM_INDEX=full_content_idx
export LANCE_FTS_BENCH_PREWARM_POSITIONS=false
export LANCE_FTS_BENCH_INDEX_CACHE_SIZE_GIB=192
export LANCE_FTS_BENCH_BINARY_CACHE_ROOT=/home/ec2-user/bench-results/query-ready-binary-cache

exec scripts/bench_fts_s3_100m.sh "$1" "$2" "$3"
