# Reuse IVF Coarse-Quantizer Routing Across Index Segments

Draft PR: [#8966](https://github.com/lance-format/lance/pull/8966)

## Proposal

Add a content-derived coarse-quantizer fingerprint to IVF index segment metadata. When every physical segment of a logical index has the same well-formed fingerprint, rank the IVF centroids once per query and reuse the selected partition IDs and query-to-centroid distances across all segments.

If any segment has a missing, malformed, or different fingerprint, Lance keeps the existing per-segment routing behavior.

This implements the compatibility-based shared routing described in [Lance Vector Index: Multi-Segment Final State](https://github.com/lance-format/lance/discussions/6189).

## Motivation

Today, a multi-segment IVF query independently runs `find_partitions` for every segment:

```text
query
  +-- segment 0: find_partitions -> search partitions
  +-- segment 1: find_partitions -> search partitions
  `-- segment N: find_partitions -> search partitions
```

Distributed index builds can produce multiple segments from the same ordered IVF centroids. For these segments, `find_partitions` receives identical inputs and returns identical partition IDs and centroid distances. Repeating it once per segment performs identical work.

The proposed path is:

```text
                         +-- segment 0: search partitions
query -> find_partitions +-- segment 1: search partitions
                         `-- segment N: search partitions
                                      -> merge top-k
```

Only coarse routing is shared. Each segment still searches its own inverted lists before the existing global top-k merge.

## Design

The fingerprint is a 32-byte SHA-256 digest over the `lance.coarse_quantizer.v1` domain separator, distance metric, centroid count, vector dimension, centroid scalar type, and ordered centroid values in canonical little-endian form. Centroid order is included because partition IDs are positions in that array. The within-partition codec is excluded because it does not affect routing.

Lance derives the fingerprint when a segment is built with final, externally supplied IVF centroids and retraining is disabled. Independently trained or retrained segments omit it. An operation that retrains the IVF model clears an inherited fingerprint unless it can derive a replacement from the final model.

For a query over two or more segments, Lance compares their fingerprints. If all are equal and well-formed, it opens one representative segment, normalizes the query, calls `find_partitions` once, and fans out the result. Otherwise it uses the existing path. The first implementation is all-or-nothing and does not group matching subsets within a mixed set of segments.

## Format change

Add one optional field to `VectorIndexDetails`:

```protobuf
optional bytes coarse_quantizer_fingerprint = 10;
```

The field is an optimization hint with fail-closed semantics:

- New readers query legacy indexes through the existing path.
- Old readers ignore the new optional field.
- Invalid lengths, invalid metadata, and mismatches disable reuse.
- Losing the field loses only the optimization, not query correctness.

No dataset migration or reader/writer feature flag is required because the field does not affect index readability or query semantics.

## Benchmark plan

The benchmark isolates coarse-routing reuse from index construction and storage effects. Two Lance branches are created from the same LAION 100M dataset version and indexed with identical fragment groups, six physical segments, 24,414 IVF centroids, and a shared 5-bit RaBitQ model. The only A/B difference is whether the segments carry the coarse-quantizer fingerprint.

Before measuring performance, query-plan metrics verify that the baseline calls `find_partitions` once per segment while the optimized path calls it once and reuses the result for the other five segments. They also verify that no payload `LanceRead` is introduced.

Both indexes are pre-warmed into the index cache before timing. Timed queries return only `_rowid` and `_distance` and cover `k={10,100}`, `nprobes={16,64,256,1024}`, and request concurrency `{1,8,16}`. Recall is calculated separately, outside the timed interval, from the actual ANN results for a fixed query set. Each configuration is repeated three times with alternating A/B order.

The benchmark reports QPS, P50/P95/P99 latency, Recall@k, errors, and query-plan routing counters. Detailed results will be posted as a follow-up in this Discussion.

## Implementation summary

The draft implementation adds the optional metadata field, fingerprint generation and lifecycle handling, fail-closed validation, rank-once/fan-out execution, and query-plan counters for routing calls and reused segments.

The execution pattern is similar to Faiss [`IndexShardsIVF`](https://github.com/facebookresearch/faiss/blob/main/faiss/IndexShardsIVF.cpp), but Lance needs a persisted identity because its segments are independently stored and versioned.
