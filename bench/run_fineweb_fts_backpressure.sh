#!/usr/bin/env bash
# Backpressure sweep for async MemWAL writes on real FineWeb text.
#
# Mirrors the HNSW vector-index backpressure sweep
# (analysis/lance/jack-mem-wal-hnsw/native-shard-writer-backpressure-*):
# pace the ingest at increasing target rows/s and find the highest rate
# the flush/index pipeline sustains without accumulating backpressure
# (no puts >= 1s, small leftover backlog, bounded drain). Also runs one
# unpaced cell per mode for the (non-sustainable) peak.
#
# Usage: ./bench/run_fineweb_fts_backpressure.sh [run_id]

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_ID="${1:-bp-$(date -u +%Y%m%dT%H%M%SZ)}"
DATASET_PREFIX="${DATASET_PREFIX:-s3://jack-devland-build/bench/mem-fts-fineweb}"
SEED_ROWS="${SEED_ROWS:-10000}"      # small seed: backpressure test, seed size is irrelevant
BATCH_ROWS="${BATCH_ROWS:-1000}"
CALLS="${CALLS:-120}"                # 120k rows/cell — long enough for steady state
CACHE_DIR="${CACHE_DIR:-/mnt/data/fineweb}"
CONFIG_TIMEOUT="${CONFIG_TIMEOUT:-2400}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

LOCAL_DIR="bench/results/${RUN_ID}"
mkdir -p "$LOCAL_DIR"

BIN="$(find target/release/deps -maxdepth 1 -type f -perm -111 -name 'mem_wal_fineweb_fts-*' ! -name '*.d' 2>/dev/null | sort | tail -1)"
if [ -z "$BIN" ]; then
    cargo bench -p lance --bench mem_wal_fineweb_fts --no-run
    BIN="$(find target/release/deps -maxdepth 1 -type f -perm -111 -name 'mem_wal_fineweb_fts-*' ! -name '*.d' 2>/dev/null | sort | tail -1)"
fi
echo "bench binary: $BIN"
echo "run id:       $RUN_ID"

run_cell() {
    local name="$1"; shift
    local out="$LOCAL_DIR/${name}.json"
    local log="$LOCAL_DIR/${name}.log"
    echo ">>> $name"
    if [ -f "$out" ]; then echo "    already done"; return; fi
    timeout "$CONFIG_TIMEOUT" "$BIN" --bench --phase write "$@" --output "$out" > "$log" 2>&1
    local rc=$?
    if [ "$rc" -eq 124 ]; then echo "    !!! TIMED OUT"
    elif [ "$rc" -ne 0 ]; then echo "    !!! failed rc=$rc"
    else echo "    ok"; fi
    [ -f "$out" ] && aws s3 cp "$out" "$DATASET_PREFIX/$RUN_ID/results/${name}.json" >/dev/null 2>&1
    aws s3 cp "$log" "$DATASET_PREFIX/$RUN_ID/results/${name}.log" >/dev/null 2>&1
}

# mode -> list of paced targets (rows/s); 0 = unpaced peak
sweep() {
    local mode="$1"; shift
    for tgt in "$@"; do
        local extra=() label
        if [ "$tgt" = "0" ]; then
            label="${mode}_unpaced"
        else
            label="${mode}_t${tgt}"
            extra=(--target-rows-per-sec "$tgt")
        fi
        run_cell "$label" \
            --mode "$mode" \
            --uri "$DATASET_PREFIX/$RUN_ID/bp_${label}" \
            --seed-rows "$SEED_ROWS" --batch-rows "$BATCH_ROWS" --calls "$CALLS" \
            --cache-dir "$CACHE_DIR" "${extra[@]}"
    done
}

# FTS-indexed async: expected sustainable rate is modest, sweep low.
sweep async_idx 500 1000 1500 2000 2500 3000 0
# No-index async: much higher ceiling.
sweep async_noidx 2000 4000 6000 8000 0

echo ""
echo "=== backpressure summary ==="
python3 - "$LOCAL_DIR" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
rows = []
for p in sorted(glob.glob(os.path.join(d, "*.json"))):
    try: r = json.load(open(p))
    except Exception: continue
    rows.append((os.path.basename(p)[:-5], r))
print(f"{'cell':26s} {'target':>8s} {'rows/s':>9s} {'puts_r/s':>10s} {'drain_s':>8s} {'slow>=1s':>9s} {'backlog_rows':>13s}")
for name, r in rows:
    tgt = r.get('target_rows_per_sec') or 0
    print(f"{name:26s} {tgt:>8.0f} {r['throughput_rows_per_sec']:>9.0f} "
          f"{r['throughput_puts_rows_per_sec']:>10.0f} {r['elapsed_close_seconds']:>8.1f} "
          f"{r['slow_puts_ge_1s']:>9d} {str(r.get('backlog_memtable_rows')):>13s}")
PY
echo ""
echo "results: $LOCAL_DIR  +  $DATASET_PREFIX/$RUN_ID/results/"
