#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <clone-dir> <run-root>" >&2
    exit 2
fi

clone_dir=$1
run_root=$2
base_dir="$run_root/base"
head_dir="$run_root/head"
base_target_dir="$run_root/target-base"
head_target_dir="$run_root/target-head"
binary_dir="$run_root/bin"
case_dir="$run_root/cases"
result_jsonl="$run_root/results.jsonl"
base_branch=origin/xuanwo/bench-pr7792-positive-base-20260723
head_branch=origin/xuanwo/bench-pr7792-positive-head-20260723
base_upstream=c685e4a6d1524c62ce11ae3f225a3016d65a5d6a
head_upstream=91767f5d311972b690ce427d240f8c99f34bf235

mkdir -p "$run_root" "$binary_dir" "$case_dir"
git -C "$clone_dir" fetch origin \
    xuanwo/bench-pr7792-positive-base-20260723 \
    xuanwo/bench-pr7792-positive-head-20260723

if [[ ! -d "$base_dir/.git" && ! -f "$base_dir/.git" ]]; then
    git -C "$clone_dir" worktree add --detach "$base_dir" "$base_branch"
fi
if [[ ! -d "$head_dir/.git" && ! -f "$head_dir/.git" ]]; then
    git -C "$clone_dir" worktree add --detach "$head_dir" "$head_branch"
fi

[[ $(git -C "$base_dir" rev-parse HEAD~1) == "$base_upstream" ]]
[[ $(git -C "$head_dir" rev-parse HEAD~1) == "$head_upstream" ]]
[[ $(git -C "$base_dir" hash-object rust/lance/tests/filtered_read_pr7792_bench.rs) == \
   $(git -C "$head_dir" hash-object rust/lance/tests/filtered_read_pr7792_bench.rs) ]]
[[ $(git -C "$base_dir" hash-object rust/lance/tests/run_pr7792_positive_bench.sh) == \
   $(git -C "$head_dir" hash-object rust/lance/tests/run_pr7792_positive_bench.sh) ]]
[[ $(git -C "$clone_dir" diff --name-only "$base_upstream" "$head_upstream") == \
   "rust/lance/src/io/exec/filtered_read.rs" ]]

build_binary() {
    local revision=$1
    local source_dir=$2
    local output_binary=$3
    local build_log=$4
    local target_dir=$5
    local upstream_sha=$6
    local artifact_file
    artifact_file=$(mktemp "$run_root/artifact.XXXXXX")
    (cd "$source_dir" && \
        PR7792_BENCH_REVISION="$revision" \
        PR7792_UPSTREAM_SHA="$upstream_sha" \
        CARGO_TARGET_DIR="$target_dir" \
        cargo test \
            -p lance \
            --profile release-with-debug \
            --test filtered_read_pr7792_bench \
            --no-run \
            --message-format=json) \
        2> >(tee "$build_log" >&2) \
        | tee "$artifact_file" >/dev/null
    local executable
    executable=$(jq -r \
        'select(.reason == "compiler-artifact" and .target.name == "filtered_read_pr7792_bench" and .executable != null) | .executable' \
        "$artifact_file" | tail -n 1)
    [[ -n "$executable" ]]
    cp "$executable" "$output_binary"
    chmod +x "$output_binary"
}

build_binary base "$base_dir" "$binary_dir/base" "$run_root/build-base.log" \
    "$base_target_dir" "$base_upstream"
build_binary head "$head_dir" "$binary_dir/head" "$run_root/build-head.log" \
    "$head_target_dir" "$head_upstream"
if cmp -s "$binary_dir/base" "$binary_dir/head"; then
    echo "base and head benchmark binaries are unexpectedly identical" >&2
    exit 1
fi

result_tmp=$(mktemp "$run_root/results.XXXXXX")
jq -cn \
    --arg base_upstream "$base_upstream" \
    --arg head_upstream "$head_upstream" \
    --arg base_bench_commit "$(git -C "$base_dir" rev-parse HEAD)" \
    --arg head_bench_commit "$(git -C "$head_dir" rev-parse HEAD)" \
    --arg harness_blob "$(git -C "$base_dir" hash-object rust/lance/tests/filtered_read_pr7792_bench.rs)" \
    --arg base_binary_sha256 "$(sha256sum "$binary_dir/base" | awk '{print $1}')" \
    --arg head_binary_sha256 "$(sha256sum "$binary_dir/head" | awk '{print $1}')" \
    --arg rustc "$(rustc --version)" \
    --arg kernel "$(uname -srmo)" \
    --arg cpu "$(lscpu | awk -F: '/Model name/{gsub(/^[[:space:]]+/, "", $2); print $2; exit}')" \
    '{event:"benchmark_metadata", base_upstream:$base_upstream, head_upstream:$head_upstream, base_bench_commit:$base_bench_commit, head_bench_commit:$head_bench_commit, harness_blob:$harness_blob, base_binary_sha256:$base_binary_sha256, head_binary_sha256:$head_binary_sha256, rustc:$rustc, kernel:$kernel, cpu:$cpu}' \
    >> "$result_tmp"

run_binary() {
    local revision=$1
    local label=$2
    local dataset_uri=$3
    local fragments=$4
    local rows_per_fragment=$5
    local selected=$6
    local concurrency=$7
    local samples=$8
    local mode=$9
    local create_dataset=${10}
    local case_log="$case_dir/${label}-${revision}.log"
    BENCH_LABEL="$label-$revision" \
    BENCH_DATASET_URI="$dataset_uri" \
    BENCH_CREATE_DATASET="$create_dataset" \
    BENCH_MODE="$mode" \
    BENCH_FRAGMENTS="$fragments" \
    BENCH_ROWS_PER_FRAGMENT="$rows_per_fragment" \
    BENCH_SELECTED_FRAGMENTS="$selected" \
    BENCH_CONCURRENCY="$concurrency" \
    BENCH_WARMUPS=10 \
    BENCH_SAMPLES="$samples" \
    "$binary_dir/$revision" --ignored --nocapture bench_filtered_read_metadata_reuse \
        | tee "$case_log"
    grep '"event":"filtered_read_pr7792_' "$case_log" >> "$result_tmp"
}

prepare_dataset() {
    local name=$1
    local fragments=$2
    local rows_per_fragment=$3
    local dataset_uri="$run_root/$name"
    local create_dataset=0
    if [[ ! -d "$dataset_uri" ]]; then
        create_dataset=1
    fi
    run_binary base "prepare-$name" "$dataset_uri" "$fragments" \
        "$rows_per_fragment" 1 1 1 prepare "$create_dataset"
}

run_case() {
    local case_name=$1
    local dataset_name=$2
    local fragments=$3
    local rows_per_fragment=$4
    local selected=$5
    local concurrency=$6
    local samples=$7
    local mode=$8
    local dataset_uri="$run_root/$dataset_name"
    local trial
    for trial in 1 2 3; do
        if (( trial % 2 == 1 )); then
            run_binary base "$case_name-t$trial" "$dataset_uri" "$fragments" \
                "$rows_per_fragment" "$selected" "$concurrency" "$samples" "$mode" 0
            run_binary head "$case_name-t$trial" "$dataset_uri" "$fragments" \
                "$rows_per_fragment" "$selected" "$concurrency" "$samples" "$mode" 0
        else
            run_binary head "$case_name-t$trial" "$dataset_uri" "$fragments" \
                "$rows_per_fragment" "$selected" "$concurrency" "$samples" "$mode" 0
            run_binary base "$case_name-t$trial" "$dataset_uri" "$fragments" \
                "$rows_per_fragment" "$selected" "$concurrency" "$samples" "$mode" 0
        fi
    done
}

prepare_dataset dataset-100x1m 100 1000000
prepare_dataset dataset-1000x4096 1000 4096

run_case representative-direct dataset-100x1m 100 1000000 100 1 500 direct
run_case representative-staged dataset-100x1m 100 1000000 100 1 500 staged
run_case representative-concurrent dataset-100x1m 100 1000000 100 64 20 direct
run_case sparse-staged dataset-100x1m 100 1000000 1 1 1000 staged
run_case scale1000-direct dataset-1000x4096 1000 4096 100 1 500 direct
run_case scale1000-staged dataset-1000x4096 1000 4096 100 1 500 staged
run_case scale1000-sparse-staged dataset-1000x4096 1000 4096 1 1 1000 staged

mv "$result_tmp" "$result_jsonl"
echo "benchmark results: $result_jsonl"
