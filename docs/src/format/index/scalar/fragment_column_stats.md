# Fragment Column Statistics Index

This index stores one column summary per fragment. Readers can load these small
summaries independently of zone-level statistics and use a filter to select
candidate fragments before planning or executing scans and index searches.

It does not store row addresses or identify matching rows. A retained fragment
still needs predicate evaluation. A reader may exclude a fragment only when its
statistics prove no live row can match; missing, stale, or unsupported
statistics always retain it, and an inexact candidate set must never be
complemented to implement `NOT`.

## Index metadata and storage

Each logical index covers one field, identified by `IndexMetadata.fields`.
`index_details` contains `lance.index.pb.FragmentColumnStatsIndexDetails`.
This proposal defines `IndexMetadata.index_version = 0`.

Each segment has its own UUID and a `fragment_stats.lance` file in its index
directory. A logical index may have multiple segments, each summarizing a
different set of fragments. The manifest's `fragment_bitmap` identifies the
fragments with usable summaries for that snapshot; it is not row-index coverage.

The bitmap must be a subset of the fragments that hold records in the segment's
file: claiming a fragment without a record would let a reader interpret "covered
but no candidate ranges" as "no matching rows" and silently drop results.
Records outside the bitmap are ignored.

A summary is valid only for the column data it summarized: replacing that
column's data file invalidates it even when the fragment ID is unchanged, and
the commit must remove the fragment from the affected coverage bitmaps. A
relevant column overlay newer than the segment's `dataset_version` also makes
the summary unusable. Deletion-vector-only changes retain summaries, because
the remaining rows are a subset of the summarized physical rows.

Records are ordered by `fragment_id`, one record per fragment. Active segments
of the same logical index must not claim overlapping fragments.

## File schema

`value_type` below is the indexed column's Arrow type.

```python
pa.schema([
    pa.field("fragment_id", pa.uint64(), nullable=False),
    pa.field("min", value_type, nullable=True),
    pa.field("max", value_type, nullable=True),
    pa.field("row_count", pa.uint64(), nullable=False),
    pa.field("null_count", pa.uint64(), nullable=False),
    pa.field("nan_count", pa.uint64(), nullable=False),
])
```

- `min` and `max` bound non-null, non-NaN values using Arrow Rust ordering.
  Bounds may be wider than the exact extrema, but must not exclude any such
  value. Infinities are ordinary values; negative zero sorts before positive zero.
  Widening includes deliberate truncation of long values: a writer may store a
  shortened `min` (a prefix of the exact minimum) and a shortened `max` that is
  strictly greater than every value the exact maximum bounds. Shortened bounds
  must be valid values of the column's type: strings truncate on character
  boundaries, and the upper bound is a valid successor (for example, the final
  character replaced by its next code point, dropping trailing characters that
  have none). The raw byte recipe (increment the final byte, dropping trailing
  0xFF bytes) applies to binary values, not strings. When no valid shortened
  upper bound exists, keep the exact bound or store a null bound. Truncation
  lengths are an implementation choice.
- Null bounds mean that value bounds are unavailable, not that the fragment is
  empty. Types without supported bounds may still have exact counts.
- `row_count` is the number of physical rows summarized, including deleted rows.
- `null_count` counts nulls exactly in those physical rows.
- `nan_count` counts all NaNs exactly; zero for non-floating types. NaNs do not
  appear in bounds: an extremum that is NaN is stored as a null bound.

This index has no zone offsets, zone lengths, or null-row bitmap. Counts describe
the summarized physical data, not the number of live rows after deletions.
