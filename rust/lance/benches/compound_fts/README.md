# Exact compound FTS benchmark

This benchmark exercises compound full-text queries and checks every optimized
top-k result against an exhaustive oracle. The oracle runs the same query
without a limit, orders all matches by `_score DESC, _rowid ASC`, and then
truncates to `k`. Row ids must match exactly; scores use `1e-5` absolute and
relative tolerances. Synthetic term labels are contiguous alphanumeric tokens,
so the default tokenizer does not split a clause label into shared subtokens.

The full matrix contains:

- Boolean `SHOULD` with 8, 32, and 128 high- and low-document-frequency clauses.
- Boolean `MUST` with 2, 4, and 8 clauses.
- `MUST` plus `SHOULD` with common and rare `MUST_NOT` clauses.
- Nested Boolean queries at depths 2 and 4, including a phrase child.
- Boost queries with common and rare negative clauses.
- Multi-match queries across 4, 32, 256, and 500 fields.
- `k=10` and `k=100`, single and multiple index segments, multiple index
  partitions, cold and warm sessions, a selective scalar-index prefilter, and
  an appended unindexed fragment where the query supports flat fallback.

The deterministic rich-text corpus puts each high-DF term in about 75% of
documents and each low-DF term in about 1/64 of documents. The common and rare
negative terms occur in 1/2 and 1/97 of documents; the prefilter retains 1/16.

Run a small correctness check while developing:

```bash
cargo bench -p lance --profile release-with-debug --bench compound_fts -- \
  --profile smoke --verify-only
```

Run the acceptance matrix with optimized code and retained debug symbols:

```bash
cargo bench -p lance --profile release-with-debug --bench compound_fts -- \
  --profile full \
  --iterations 20 \
  --dataset-root /absolute/path/to/compound-fts-data \
  --run-id OSS-1597-machine-a \
  --run-label current \
  --output /absolute/path/to/current.jsonl
```

Pass `--verify-only` for one measured iteration per matrix point. Pass
`--rebuild` only when intentionally replacing an existing benchmark dataset.

## Comparable measurements

Baseline and current runs are comparable only when all of these are identical:

- dataset root and recorded dataset fingerprint;
- machine, operating system, and Rust build profile;
- workload, `k`, segment/partition shape, cache state, and iteration count;
- tokenizer, index parameters, prefilter, and fresh-overlay configuration.

Use the same `--run-id` for the pair and different `--run-label` values. The
benchmark emits raw JSONL records and intentionally computes no speedup. Do not
claim a benefit unless the recorded metadata is comparable.

“Cold” means a fresh `Dataset` session for every measured query; it does not
drop the operating-system page cache. “Warm” means one session, one unmeasured
warm-up, and then repeated measured queries. The output records this definition
so results do not imply a stronger cache reset than was actually performed.

Each result reports p50 and p95 latency, process peak RSS and peak RSS growth,
candidate documents visited and fully scored, compressed posting blocks
decoded, phrase-position checks, rows materialized, index-cache hits and misses,
and the observed segment/partition counts. Counter values are medians across the
measured iterations. `rows_materialized` is the sum of DataFusion
`output_rows` counters in the scan plan.
