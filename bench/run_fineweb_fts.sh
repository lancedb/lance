#!/usr/bin/env bash
# Driver for the fineweb FTS benchmark.
#
# Runs the 12 configs (3 memtable sizes × durable yes/no × FTS yes/no), saves
# each result.json locally and to S3, and at the end prints a small summary.
#
# Usage:
#   ./bench/run_fineweb_fts.sh [run_id]
#
# Env vars (optional):
#   DATASET_PREFIX   default: s3://jack-devland-build/bench/mem-fts-fineweb
#   BENCH_BASE_ROWS  default: 1000000
#   BENCH_INGEST_ROWS default: 1000000
#   BENCH_BATCH_SIZE default: 1000
#   AWS_DEFAULT_REGION default: us-east-1

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
DATASET_PREFIX="${DATASET_PREFIX:-s3://jack-devland-build/bench/mem-fts-fineweb}"
BENCH_BASE_ROWS="${BENCH_BASE_ROWS:-1000000}"
BENCH_INGEST_ROWS="${BENCH_INGEST_ROWS:-1000000}"
BENCH_BATCH_SIZE="${BENCH_BATCH_SIZE:-1000}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

LOCAL_DIR="bench/results/${RUN_ID}"
mkdir -p "$LOCAL_DIR"

BIN="target/release/mem_wal_fineweb_fts"
if [ ! -x "$BIN" ]; then
    echo "building bench binary..."
    cargo build --release -p lance --bench mem_wal_fineweb_fts
    # criterion-style bench output goes to deps/; resolve it.
    BIN="$(ls -t target/release/deps/mem_wal_fineweb_fts-* | grep -v '\.d$' | head -1)"
fi
echo "using bench binary: $BIN"

CONFIGS=(
    "100000 0 0"
    "100000 0 1"
    "100000 1 0"
    "100000 1 1"
    "500000 0 0"
    "500000 0 1"
    "500000 1 0"
    "500000 1 1"
    "1000000 0 0"
    "1000000 0 1"
    "1000000 1 0"
    "1000000 1 1"
)

echo "=== Run $RUN_ID =="
echo "  prefix: $DATASET_PREFIX"
echo "  base_rows: $BENCH_BASE_ROWS  ingest_rows: $BENCH_INGEST_ROWS  batch_size: $BENCH_BATCH_SIZE"
echo ""

for cfg in "${CONFIGS[@]}"; do
    read -r MT D F <<< "$cfg"
    if [ "$MT" = "1000000" ]; then SZ="1M"; elif [ "$MT" = "500000" ]; then SZ="500k"; else SZ="100k"; fi
    NAME="mt${SZ}_durable${D}_fts${F}"
    OUT="$LOCAL_DIR/${NAME}.json"
    LOG="$LOCAL_DIR/${NAME}.log"
    echo ">>> $NAME"
    if [ -f "$OUT" ]; then
        echo "    result already exists, skipping"
        continue
    fi
    set +e
    BENCH_RUN_ID="$RUN_ID" \
    DATASET_PREFIX="$DATASET_PREFIX" \
    BENCH_MAX_MEMTABLE_ROWS="$MT" \
    DURABLE_WRITE="$D" \
    FTS_ENABLED="$F" \
    BENCH_BASE_ROWS="$BENCH_BASE_ROWS" \
    BENCH_INGEST_ROWS="$BENCH_INGEST_ROWS" \
    BENCH_BATCH_SIZE="$BENCH_BATCH_SIZE" \
    BENCH_CACHE_DIR="${BENCH_CACHE_DIR:-/mnt/data/fineweb}" \
    RESULT_FILE="$OUT" \
    "$BIN" --bench --nocapture 2>&1 | tee "$LOG"
    RC=${PIPESTATUS[0]}
    set -e
    if [ "$RC" -ne 0 ]; then
        echo "    !!! config failed (rc=$RC); see $LOG"
    fi
    # Upload to S3 alongside the dataset.
    if [ -f "$OUT" ]; then
        aws s3 cp "$OUT" "$DATASET_PREFIX/$RUN_ID/results/${NAME}.json" || true
        aws s3 cp "$LOG" "$DATASET_PREFIX/$RUN_ID/results/${NAME}.log" || true
    fi
done

echo ""
echo "=== summary ==="
python3 - <<PY
import glob, json, os
results = []
for p in sorted(glob.glob(os.path.join("$LOCAL_DIR", "*.json"))):
    try:
        with open(p) as f: r = json.load(f)
        results.append(r)
    except Exception as e:
        print(f"  failed to read {p}: {e}")

print(f"{'config':30s} {'rows/s':>10} {'p95_ms':>7} {'mt_p95_ms':>10} {'cons_mean':>10}")
for r in results:
    name = r["config_name"]
    tp = r["ingest"]["rows_per_sec"]
    p95 = r["ingest"]["put_p95_ms"]
    rd = r.get("read")
    mt = rd["mt_latency_p95_ms"] if rd else 0
    cm = rd["consistency_mean"] if rd else 0
    print(f"{name:30s} {tp:>10.0f} {p95:>7.2f} {mt:>10.2f} {cm:>10.3f}")
PY

echo ""
echo "Results:"
echo "  local: $LOCAL_DIR"
echo "  s3:    $DATASET_PREFIX/$RUN_ID/results/"
