#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <clone-dir> <dataset-uri> <result-jsonl>" >&2
    exit 2
fi

clone_dir=$1
dataset_uri=$2
result_jsonl=$3
run_root=$(dirname "$result_jsonl")
base_dir="$run_root/base"
head_dir="$run_root/head"
target_dir="$run_root/target"
binary_dir="$run_root/bin"
case_dir="$run_root/cases"
base_branch=origin/xuanwo/bench-pr7792-row-stream-base-20260723
head_branch=origin/xuanwo/bench-pr7792-row-stream-head-20260723
base_upstream=c685e4a6d1524c62ce11ae3f225a3016d65a5d6a
head_upstream=91767f5d311972b690ce427d240f8c99f34bf235

mkdir -p "$run_root" "$binary_dir" "$case_dir"
git -C "$clone_dir" fetch origin \
    xuanwo/bench-pr7792-row-stream-base-20260723 \
    xuanwo/bench-pr7792-row-stream-head-20260723

if [[ ! -d "$base_dir/.git" && ! -f "$base_dir/.git" ]]; then
    git -C "$clone_dir" worktree add --detach "$base_dir" "$base_branch"
fi
if [[ ! -d "$head_dir/.git" && ! -f "$head_dir/.git" ]]; then
    git -C "$clone_dir" worktree add --detach "$head_dir" "$head_branch"
fi

[[ $(git -C "$base_dir" rev-parse HEAD~2) == "$base_upstream" ]]
[[ $(git -C "$head_dir" rev-parse HEAD~2) == "$head_upstream" ]]

build_binary() {
    local source_dir=$1
    local output_binary=$2
    local build_log=$3
    local artifact_file
    artifact_file=$(mktemp "$run_root/artifact.XXXXXX")
    CARGO_TARGET_DIR="$target_dir" cargo test \
        --manifest-path "$source_dir/Cargo.toml" \
        -p lance \
        --profile release-with-debug \
        --test row_stream_pr7792_bench \
        --no-run \
        --message-format=json \
        2> >(tee "$build_log" >&2) \
        | tee "$artifact_file" >/dev/null
    local executable
    executable=$(jq -r \
        'select(.reason == "compiler-artifact" and .target.name == "row_stream_pr7792_bench" and .executable != null) | .executable' \
        "$artifact_file" | tail -n 1)
    [[ -n "$executable" ]]
    cp "$executable" "$output_binary"
    chmod +x "$output_binary"
}

build_binary "$base_dir" "$binary_dir/base" "$run_root/build-base.log"
CARGO_TARGET_DIR="$target_dir" cargo clean \
    --manifest-path "$head_dir/Cargo.toml" \
    -p lance
build_binary "$head_dir" "$binary_dir/head" "$run_root/build-head.log"

result_tmp=$(mktemp "$run_root/results.XXXXXX")
jq -cn \
    --arg base_upstream "$base_upstream" \
    --arg head_upstream "$head_upstream" \
    --arg base_bench_commit "$(git -C "$base_dir" rev-parse HEAD)" \
    --arg head_bench_commit "$(git -C "$head_dir" rev-parse HEAD)" \
    --arg rustc "$(rustc --version)" \
    --arg kernel "$(uname -srmo)" \
    --arg cpu "$(lscpu | awk -F: '/Model name/{gsub(/^[[:space:]]+/, "", $2); print $2; exit}')" \
    '{event:"benchmark_metadata", base_upstream:$base_upstream, head_upstream:$head_upstream, base_bench_commit:$base_bench_commit, head_bench_commit:$head_bench_commit, rustc:$rustc, kernel:$kernel, cpu:$cpu}' \
    >> "$result_tmp"

dataset_created=0
run_case() {
    local revision=$1
    local case_name=$2
    local selected=$3
    local batches=$4
    local concurrency=$5
    local samples=$6
    local unknown=$7
    local head_delay_us=$8
    local create_dataset=0
    if [[ $dataset_created -eq 0 ]]; then
        create_dataset=1
        dataset_created=1
    fi
    local label="${revision}-${case_name}"
    local case_log="$case_dir/${label}.log"
    BENCH_LABEL="$label" \
    BENCH_DATASET_URI="$dataset_uri" \
    BENCH_CREATE_DATASET="$create_dataset" \
    BENCH_FRAGMENTS=100 \
    BENCH_ROWS_PER_FRAGMENT=4096 \
    BENCH_SELECTED_FRAGMENTS="$selected" \
    BENCH_INPUT_BATCHES="$batches" \
    BENCH_CONCURRENCY="$concurrency" \
    BENCH_WARMUPS=3 \
    BENCH_SAMPLES="$samples" \
    BENCH_UNKNOWN_FILE_SIZE="$unknown" \
    BENCH_HEAD_DELAY_US="$head_delay_us" \
    "$binary_dir/$revision" --ignored --nocapture bench_row_stream_fragment_reconstruction \
        | tee "$case_log"
    local result_line
    result_line=$(grep '"event":"row_stream_pr7792_bench"' "$case_log")
    [[ -n "$result_line" ]]
    printf '%s\n' "$result_line" >> "$result_tmp"
}

run_case base modern-single 1 1 1 30 0 0
run_case head modern-single 1 1 1 30 0 0

run_case head modern-many-batches 1 100 1 30 0 0
run_case base modern-many-batches 1 100 1 30 0 0

run_case base modern-wide 100 10 1 20 0 0
run_case head modern-wide 100 10 1 20 0 0

run_case head modern-concurrent 20 10 64 10 0 0
run_case base modern-concurrent 20 10 64 10 0 0

run_case base unknown-count 1 20 1 20 1 0
run_case head unknown-count 1 20 1 20 1 0

run_case head unknown-wide-count 20 20 1 15 1 0
run_case base unknown-wide-count 20 20 1 15 1 0

run_case base unknown-2ms-synthetic 1 20 1 10 1 2000
run_case head unknown-2ms-synthetic 1 20 1 10 1 2000

mv "$result_tmp" "$result_jsonl"
echo "benchmark results: $result_jsonl"
