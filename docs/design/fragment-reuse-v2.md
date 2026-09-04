# Mixed FRI V1/V2 implementation draft

This is experimental implementation work stacked on
[#8972](https://github.com/lance-format/lance/pull/8972), replacing
[#8978](https://github.com/lance-format/lance/pull/8978). The format changes here
are provisional. Formal format review comes after the implementation and its
semantics are agreed.

## Stable partition

A stable partition is a value-preserving rewrite. It visits ordered source
fragments in physical row order, skips deleted rows, and routes every live row
to exactly one destination. Within each destination, relative source order is
preserved. It does not declare a permanent partition key: subsequent appends
can create ordinary fragments. Sorting within a destination is not representable
by this encoding.

    Sources:      A [a, b, deleted, c], B [d, e]
    Labels:         0  1    NULL    0     1  0
    Destinations: C [a, c, e], D [b, d]

The row-map implementation is reused directly from #8972:

- `RowMapWriter` interleaves deleted positions from source deletion vectors into
  the stream of live-row destination labels.
- `RowMapReader` opens the counts matrix without reading labels. A point or
  batch lookup reads only the relevant logical blocks. Mixed translation sorts
  requested offsets and limits each read to sixteen blocks to bound label memory.
- A non-NULL label selects an entry in the ordered destination list. Its rank
  among earlier equal labels is the destination offset.

Counts check conservation; they cannot prove value identity or output order.
The rewrite worker must write destination data and labels from the same routing
decisions and preserve order within each destination.

## Persisted state

    Manifest
    +-- index section
    |   +-- __lance_frag_reuse (existing V1 representation)
    +-- stable_partition_transitions[]
        +-- source_dataset_version
        +-- committed_version
        +-- ordered source fragment digests
        +-- ordered destination fragment digests
        +-- row_map_id, row_map_size_bytes, optional base_id

    _row_maps/<row_map_id>/
    +-- row_map.lance                 (#8972 labels + counts)

Each transition owns one immutable row-map file. `row_map_id` is an artifact
identity, not the UUID of a searchable index or of the mutable V1 catalog.
Updating V1 history or rebuilding a column index does not relocate or rewrite
this file. The dedicated namespace also protects maps from older cleanup
implementations that scan `_indices` without checking historical feature flags.
The new cleaner explicitly marks and sweeps row-map directories from retained
manifest references, with the usual protection for recent uncommitted files.

There is no external copy of the complete transition catalog and no V2
`details.binpb`. Appends copy the small manifest descriptors, not mapping labels.
V1 continues to use its existing inline/external details representation and
its existing rewrite-on-update writer.

The experimental reader and writer feature bit is `1 << 9`. The reserved
mixed-data-file-version bit (`1 << 8`) remains unsupported. Every manifest with
V2 references requires bit 9 in both the reader and writer words to prevent
older readers or writers from ignoring the mapping lifecycle. The new reader accepts ordinary V1-only
datasets without this bit.

## Writing and committing

`commit_stable_partition` in `lance::index::frag_reuse_v2` commits a prepared
rewrite through the existing transaction machinery:

1. Read source fragments from one snapshot. Produce destination data and a row
   map using #8972's writer with Lance data-file version 2.1, which encodes
   destination labels compactly. Reserve fresh destination fragment IDs.
2. Open the completed row map and validate its source length, destination
   counts, and declared file size.
3. Commit `Operation::Rewrite` with its stable-partition descriptor. The commit
   validates the exact source metadata, destination reservations, row counts,
   ordering of fragment lists, and absence of conflicting index rewrites.
4. Publish destination fragments and the descriptor in the same manifest. Stamp
   the actual installing version at commit; retain the worker's source version
   separately. A failed commit publishes neither.

This API accepts prepared fragments; it is not a clustering scheduler or a new
Python/Java routing API. Low-level transaction callers have the same obligation
to finish and validate their referenced files before committing.

Disjoint stable partitions can rebase: each appends its descriptor to the
latest manifest. A concurrent delete, update, or rewrite touching a source
requires rebuilding from a new snapshot. This draft deliberately does not fold
new source deletions into an already-written row map.

## Reading mixed history

`MixedFragReuseIndex` composes existing V1 compaction groups and V2 descriptors.
An edge connects a rewrite producing a fragment to the rewrite consuming it.
Topological order follows physical fragment lineage, not V1's recorded builder
version, which is not guaranteed to equal its installing version. Duplicate
producers, duplicate consumers, and cycles are rejected.

For each requested address:

- A V1 node applies the existing compact bitmap-rank remapper.
- A V2 node resolves the concatenated source offset with #8972's reader and
  returns the destination fragment and offset, or deletion.
- Addresses outside a node's sources pass through unchanged.

The existing Python `Dataset.remap_row_addrs` method uses the same Rust mixed
translation and preserves NULL input addresses.

Current snapshot deletion vectors still apply after translation. The mapping
records deletions at rewrite time, not future tombstones.

B-tree page reads translate physical addresses before predicate evaluation.
Translation also applies to pages streamed for index maintenance. Cache identity
includes the mixed history and target snapshot so cached pages cannot return
addresses from an earlier layout. Row-map file metadata uses the dataset's
shared metadata cache.

Stored source bitmaps remain provenance across appends and other commits.
Query coverage is derived separately: the union of a logical index's segment
bitmaps must cover all sources before its destinations count as covered. Every
contributing segment is then probed. Derived probe bitmaps can overlap; they
are never persisted as source ownership. A new segment naming a destination
directly takes priority, and old segments discard translated entries for that
destination before evaluating predicates.

A destination mixing indexed and unindexed rows must be scanned. Index types without asynchronous V2 translation are excluded
from query selection when their payloads require it; their metadata remains
in the manifest. They can be rebuilt on current fragments.

## Append, delete, and compaction

Append creates ordinary fragments and preserves both histories. Existing indices
do not claim the new rows. A later stable partition can include these fragments,
subject to the same conservative coverage rule. The prepared-commit API requires
full index coverage unless the caller explicitly selects
`StablePartitionCoverage::AllowUnindexed`. An unresolved destination cannot be
repartitioned again while an index still depends on its translated entries;
rebuild direct coverage first. This restriction also follows intervening V1
compactions and is checked at the commit boundary.

Delete updates current fragment tombstones without rewriting either mapping
format. A later rewrite records deleted physical positions as NULL labels or
uses the existing V1 survivor bitmap.

Compaction continues writing V1 groups. In a mixed dataset it records the V1
mapping even when current index coverage is empty, since retained V2 mappings
may still point at the compacted sources. Both eager and deferred B-tree index
maintenance use the composed mapping. No V1-to-V2 migration is required.

V2 may eventually acquire an order-preserving representation. This PR does not
need that encoding to coexist correctly with V1 compaction.

## Retention and current limits

Cleanup marks row-map directories from retained manifests. V1 history cleanup
is suspended while V2 references remain, because pruning V1 independently can
break a mixed chain. Coordinated history draining and descriptor pruning are
follow-up work; retaining extra history is deliberate.

This draft supports physical row addresses and one storage base. Stable logical
row IDs, shallow/deep clone of mixed datasets, and cross-base row-map reads are
rejected explicitly. The protobuf reserves optional base ownership so it can
be implemented without confusing local cleanup with external ownership.

Initial query integration supports B-tree indices. Other affected index types
use scan fallback until they gain asynchronous translation. Arbitrary output
permutations, automatic clustering, concurrent deletion folding, and complete
history draining remain outside this implementation.

## Validation

Tests exercise V1 → V2 → V1 with eager and deferred compaction, indexed reads,
deletions, cleanup, reopening, repartitioning after direct rebuild, rejection of
unresolved repartitioning, joint segment coverage, direct-destination priority,
append-induced partial coverage, unsupported-index fallback, disjoint commits, and stale source
rejection. Table tests cover descriptor serialization, malformed descriptors,
and the paired reader/writer fence.
