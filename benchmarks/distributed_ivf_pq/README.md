# Distributed IVF_PQ Benchmark

This benchmark measures the local end-to-end flow of Lance distributed `IVF_PQ`
index building:

1. Create a dataset with multiple fragments.
2. Train shared IVF/PQ parameters once.
3. Split fragments into shard groups.
4. Build shard-local partial indices concurrently.
5. Finalize the distributed merge.
6. Commit the merged index.

The benchmark prints a JSON result with timing fields such as
`train_shared_params_ms`, `shard_build_ms`, and `finalize_ms`.

## Run

```bash
cargo run --manifest-path benchmarks/distributed_ivf_pq/Cargo.toml --release -- \
  --fragments 8 \
  --rows-per-fragment 262144 \
  --shards 8 \
  --dim 128 \
  --num-partitions 32768 \
  --num-sub-vectors 16 \
  --max-iters 10 \
  --sample-rate 8 \
  --cleanup
```

## Notes

- This crate is intentionally outside the main Rust workspace.
- Use it when you want a repeatable distributed-index benchmark without
  modifying the workspace examples or criterion benches.
- For large runs, you may need to increase `RLIMIT_NOFILE`.
