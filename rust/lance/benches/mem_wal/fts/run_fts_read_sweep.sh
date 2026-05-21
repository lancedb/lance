#!/usr/bin/env bash
# Driver for the LSM FTS read benchmark panel across storage backends.
#
# Sweeps the `mem_wal_fts_read_bench` over:
#   - storage backend : local NVMe path and an s3:// prefix
#   - base table size  : configurable list (default 100k, 1M)
#   - top-k            : configurable list (default 10, 100)
#
# For each (backend, base_rows) the bench's `prepare` phase runs once to
# write the base dataset + FTS index + MemWAL; then for each k the `search`
# phase ingests flushed generations + an active memtable through ShardWriter
# and times the FTS query panel under both Local and LocalWithGlobalRescore
# scoring modes.
#
# Each config runs under a `timeout` watchdog so a hang costs one window.
#
# Usage:
#   rust/lance/benches/mem_wal/fts/run_fts_read_sweep.sh [run_id]
#
# Env:
#   NVME_PREFIX     local scratch dir on fast disk (default <tmpdir>/lsm-fts-nvme)
#   S3_PREFIX       s3:// dataset prefix (default s3://jack-devland-build/lsm-fts)
#   CACHE_DIR       FineWeb shard download cache (default <tmpdir>/lance-fineweb-cache)
#   BASE_ROWS_LIST  space-separated base sizes (default "100000 1000000")
#   K_LIST          space-separated top-k values (default "10 100")
#   MAX_MEMTABLE_ROWS  active/flushed memtable cap (default 100000)
#   FLUSHED_GENERATIONS number of flushed gens (default 2)
#   QUERIES         queries per mode (default 200)
#   RESCORE_FACTOR  K' multiplier for rescore (default 10)
#   BACKENDS        space-separated subset of "nvme s3" (default both)
#   CONFIG_TIMEOUT  per-config seconds (default 5400)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
NVME_PREFIX="${NVME_PREFIX:-${TMPDIR:-/tmp}/lsm-fts-nvme}"
S3_PREFIX="${S3_PREFIX:-s3://jack-devland-build/lsm-fts}"
CACHE_DIR="${CACHE_DIR:-${TMPDIR:-/tmp}/lance-fineweb-cache}"
BASE_ROWS_LIST="${BASE_ROWS_LIST:-100000 1000000}"
K_LIST="${K_LIST:-10 100}"
MAX_MEMTABLE_ROWS="${MAX_MEMTABLE_ROWS:-100000}"
FLUSHED_GENERATIONS="${FLUSHED_GENERATIONS:-2}"
QUERIES="${QUERIES:-200}"
RESCORE_FACTOR="${RESCORE_FACTOR:-10}"
BACKENDS="${BACKENDS:-nvme s3}"
CONFIG_TIMEOUT="${CONFIG_TIMEOUT:-5400}"

LOCAL_DIR="$REPO_ROOT/target/lsm-fts-read-results/${RUN_ID}"
mkdir -p "$LOCAL_DIR" "$CACHE_DIR" "$NVME_PREFIX"

BENCH=mem_wal_fts_read_bench
BIN="$(find target/release/deps -maxdepth 1 -type f -perm -111 -name "${BENCH}-*" ! -name '*.d' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
if [ -z "$BIN" ]; then
    echo "building bench binary..."
    cargo bench -p lance --bench "$BENCH" --no-run
    BIN="$(find target/release/deps -maxdepth 1 -type f -perm -111 -name "${BENCH}-*" ! -name '*.d' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
fi
echo "bench binary: $BIN"
echo "run id:       $RUN_ID"
echo "backends:     $BACKENDS"
echo ""

backend_prefix() {
    case "$1" in
        nvme) echo "$NVME_PREFIX/$RUN_ID" ;;
        s3)   echo "$S3_PREFIX/$RUN_ID" ;;
        *)    echo "ERROR unknown backend $1" >&2; exit 1 ;;
    esac
}

run_phase() {
    local name="$1"; shift
    local log="$LOCAL_DIR/${name}.log"
    echo ">>> $name"
    timeout "$CONFIG_TIMEOUT" "$BIN" --bench "$@" > "$log" 2>&1
    local rc=$?
    if [ "$rc" -eq 124 ]; then
        echo "    !!! TIMED OUT after ${CONFIG_TIMEOUT}s (see $log)"
        return 1
    elif [ "$rc" -ne 0 ]; then
        echo "    !!! failed rc=$rc (see $log)"
        return 1
    fi
    echo "    ok"
    return 0
}

for backend in $BACKENDS; do
    prefix="$(backend_prefix "$backend")"
    for base_rows in $BASE_ROWS_LIST; do
        case "$base_rows" in
            1000000) btag=1M ;;
            100000)  btag=100k ;;
            *)       btag="${base_rows}" ;;
        esac
        uri="$prefix/base_${btag}"

        # prepare once per (backend, base_rows)
        run_phase "prepare_${backend}_${btag}" \
            --phase prepare --uri "$uri" \
            --base-rows "$base_rows" --batch-rows 1000 \
            --cache-dir "$CACHE_DIR" || continue

        # search for each k
        for k in $K_LIST; do
            name="search_${backend}_${btag}_k${k}"
            out="$LOCAL_DIR/${name}.json"
            if [ -f "$out" ]; then
                echo ">>> $name (already done, skipping)"
                continue
            fi
            run_phase "$name" \
                --phase search --uri "$uri" \
                --base-rows "$base_rows" \
                --max-memtable-rows "$MAX_MEMTABLE_ROWS" \
                --flushed-generations "$FLUSHED_GENERATIONS" \
                --batch-rows 1000 \
                --queries "$QUERIES" --k "$k" \
                --rescore-factor "$RESCORE_FACTOR" \
                --cache-dir "$CACHE_DIR" \
                --output "$out"
            # mirror result to s3 for durability regardless of backend
            [ -f "$out" ] && aws s3 cp "$out" "$S3_PREFIX/$RUN_ID/results/${name}.json" >/dev/null 2>&1
        done
    done
done

echo ""
echo "=== summary ==="
python3 - "$LOCAL_DIR" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
hdr = f"{'config':32s} {'local_p50':>10} {'rescore_p50':>12} {'local_qps':>10} {'rescore_qps':>12} {'jaccard':>8}"
print(hdr)
for p in sorted(glob.glob(os.path.join(d, "*.json"))):
    try:
        r = json.load(open(p))
    except Exception as e:
        print(f"  bad {p}: {e}"); continue
    name = os.path.basename(p)[:-5]
    lo = r.get("local", {}); re_ = r.get("rescore", {})
    print(f"{name:32s} {lo.get('p50_us','-'):>10} {re_.get('p50_us','-'):>12} "
          f"{lo.get('qps','-'):>10} {re_.get('qps','-'):>12} {r.get('jaccard_local_vs_rescore',0):>8.4f}")
PY
echo ""
echo "results: $LOCAL_DIR  +  $S3_PREFIX/$RUN_ID/results/"
