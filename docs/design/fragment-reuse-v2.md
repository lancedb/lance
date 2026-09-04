# Mixed FRI V1/V2: stable-partition integration draft

Status: informal design for discussion. This is not a format specification or
an activation proposal. We will start the format-change process after the design
is agreed. No protobuf field numbers, feature bits, or new file contracts are
allocated by this document.

This is the proposed replacement direction for [#8978](https://github.com/lance-format/lance/pull/8978),
stacked on [#8972](https://github.com/lance-format/lance/pull/8972).
PR #8972 already supplies the stable-partition row-map writer, reader, counts,
and translation algorithms. Keep that foundation. This proposal addresses the
dataset lifecycle around it; it does not repeat the codec work.

## 1. What we want

People can continue compacting into the existing FRI V1 format. Stable partition
writes V2 metadata referencing the row maps from #8972. Our new reader and writer
handle the two paths together. No up-front rewrite of all existing indices or
mandatory conversion of V1 history is required.

    Existing compaction -> V1 FRI details --+
                                          +-> shared rewrite handling
    Stable partition -> V2 transitions ---+

Eventually V2 should also represent order-preserving compaction. That is a later
extension, not a prerequisite for this integration. The two physical formats
must already behave as one logical history before mixed operations are enabled.

The initial implementation can use conservative coverage and conflict rules.
The format should not depend on pretending an old client understands new
maintenance semantics merely because it can parse the manifest.

## 2. Stable partition, precisely

Inputs are ordered source fragments from one snapshot. Within a source, visit
rows in ascending physical offset. Each surviving row is routed to one of an
ordered list of destination fragments. Every destination preserves the relative
source order of its rows; values do not change.

This is not a persistent partition-key declaration. Later appends may add
ordinary fragments without routing through the same destinations. It is also
not arbitrary sorting: sorting within a destination needs a permutation encoding.

    Source A: [a, b, deleted, c]
    Source B: [d, e]
    Labels:   [0, 1, NULL, 0, 1, 0]
    Dest C:   [a, c, e]
    Dest D:   [b, d]

The label is a destination-list position, not a fragment ID. For physical source
position g, its new offset is the number of earlier occurrences of label[g].
A NULL label means that position was deleted in the rewrite's input snapshot.

Destinations have fresh, reserved fragment IDs and no deletion vectors at
creation. Parallel writers must restore source order within each destination.
Counts validate conservation, but cannot prove row identity or order: the writer
must emit data and labels from the same routing decisions.

## 3. Proposed V2 metadata

Use immutable mapping payloads and a small catalog of transition references.
The preferred home for that catalog is a dedicated manifest field/section.
It describes table structure, not a searchable index. Its exact placement is
still a review decision; a system-index envelope is an alternative only if it
has explicit preservation and file-reachability rules.

    Manifest
    +-- existing V1 FRI entry, unchanged wire format
    +-- V2 transition catalog
        +-- catalog format version
        +-- transition references[]
            +-- immutable transition ID
            +-- actual commit version
            +-- descriptor file reference

    _row_maps/<transition-id>/       # proposed placement, not finalized
    +-- transition.binpb
    +-- row_map.lance                # #8972 format

The descriptor's proposed fields are:

| Field | Meaning |
| --- | --- |
| format_version | Descriptor version, independent of the Lance data-file version |
| transition_id | Immutable identity; also used for idempotent publication |
| read_version | Snapshot used to produce the outputs |
| sources | Ordered fragment IDs and physical row counts |
| destinations | Ordered fragment IDs and physical row counts |
| mapping | Versioned discriminator; initially only stable partition |
| row_map | Explicit storage base, relative path, and byte length |

The exact source snapshot, including deletion and overlay metadata, is also
carried by the rewrite transaction for OCC validation. Do not introduce a second
incomplete source identity based only on physical/deleted row counts.

The catalog supplies actual commit version because workers write immutable files
before the successful commit version is known. Do not mutate a descriptor after
publication to stamp a version. Unknown mapping variants fail closed.

Creating a new transition writes its descriptor and row map once. Publishing a
new manifest copies the small references, not the historical mapping payloads.
Old snapshots retain their own reference lists. This still costs O(retained
transitions) catalog bytes per manifest; pagination/segmentation can wait for
measurements. It avoids V1's full-history payload rewrite on the V2 path.

PR #8972's row_map.lance remains one nullable UInt16 label per physical source
row, with cumulative per-destination counts in its global buffer. Default
logical blocks contain 65,536 rows. NULL positions preserve source deletions.
Point lookup reads one logical block; batches coalesce block reads; sweeps stream
blocks. Logical block boundaries need not be physical Lance page boundaries.

## 4. One dependency graph over both formats

Adapt each V1 group and V2 transition into an in-memory rewrite node with ordered
sources, destinations, and its mapping implementation. This adapter does not
rewrite V1's persisted protobuf or introduce labels into V1.

    A,B --V1--> C --V2--> D,E --V1--> F

A candidate address in A must follow all three edges. Running all V1 mappings
and then all V2 mappings is incorrect.

Use fragment lineage to establish dependencies. If a node produces a fragment
that another node consumes, the producer precedes the consumer. Validate that a
source has at most one consuming rewrite in a snapshot's retained history,
destinations are uniquely produced, and the graph has no cycles. Independent
nodes may execute in either order. Fragment IDs are never reused.

This avoids relying on V1 dataset_version as a successful commit timestamp:
that field records the builder's snapshot. Preserve it for existing purposes;
do not reinterpret it. V2 records actual commit version for diagnostics and
maintenance, but graph dependencies establish address translation order.

For point/batch translation, dispatch by the current address's fragment ID,
translate through that consuming node, and repeat until the fragment is live
or the mapping reports deletion. Full-fragment removals are terminal deletions;
current fragment existence and deletion vectors still determine visibility.
Missing historical edges cannot silently be treated as deletion: mixed writers
and retention must guarantee chain completeness. Where completeness cannot be
established, disable the affected index and scan, or return a corruption error.

## 5. Coverage is separate from stored addresses

An index bitmap may already have been advanced by V1 coverage maintenance even
though its payload still stores old row addresses. Do not relabel such a bitmap
as the payload's original address space. Likewise, deriving current coverage
does not mean the index bytes have been remapped.

The proposed integration maintains an explicit coverage frontier for each index
segment: fragments it can completely answer at a known snapshot. On first V2
use, obtain that frontier through the existing V1 coverage path at the pinned
snapshot; this does not require remapping index bytes. A persisted coverage
frontier/version may be added to IndexMetadata, or updated atomically as ordinary
coverage metadata. That schema choice needs agreement before implementation.

For each later rewrite in dependency order, let S be its sources and D its
destinations. Remove S from the frontier. Add D only if the index covered all
of S before removal. Otherwise add none of D. Intersect with current live
fragments at the end. This conservative rule works for either encoding.

Example: an index covers A but not B; a partition mixes both into C and D.
Neither C nor D is claimed as fully indexed. Queries scan them. We can later
add source/destination incidence metadata or finer-grained coverage, but the
first integration need not solve that optimization. Planning rewrite groups
with equal coverage signatures preserves more index reuse.

Retain lineage needed by the payload independently of effective coverage. A
bitmap reaching current fragments is not evidence that an old V1 record can be
trimmed. Only rebuilding/remapping the bytes, withdrawing the segment, or a
sound dependency analysis can establish that.

Index results must be translated and restricted to the selected segment's
effective coverage. A partly usable index must not leak candidates into the
scan portion and produce duplicates. Overlapping segments/generations still
require the existing query planner's selection/merge rules.

For index kinds whose decoding relies on physical ordering or fragment-level
summaries, require explicit integration or rebuild. A generic address callback
is not proof that every index format supports reordered output. Physical
addresses and stable logical row IDs must stay distinct; logical IDs use their
existing lookup path.

## 6. Write and commit

1. Pin snapshot V, choose ordered sources, reserve destination IDs, and retain
   the complete source metadata used by the scan.
2. Stream live rows to destination writers and labels to #8972's RowMapWriter.
   It interleaves deleted source positions from the source deletion vectors.
3. Finish files. Validate per-destination totals, source row count, label bounds,
   and source deletion positions. Write the immutable descriptor.
4. Submit a serializable Rewrite delta containing the descriptor reference and
   exact source/destination lists. Do not submit a replacement accumulated catalog
   built from an earlier snapshot. Do not drop the transition on serialization.
5. Against the latest manifest, validate unchanged source snapshots, matching
   descriptor/group order and reserved IDs, and completed files. Merge the delta
   into the latest catalog and assign actual commit version.
6. Atomically publish new fragments, catalog, and coverage changes in one manifest.

Disjoint rewrites can rebase and merge deltas. Overlapping sources conflict.
Source deletes, updates, overlays, and other changes affecting the prepared
outputs require retry from a new snapshot. Reusing the same read version while
replacing the source metadata with current metadata is not a valid retry.

Failed publication leaves immutable orphan files for normal age-protected
cleanup. Retrying an already committed transaction must not append a duplicate
transition. Existing transaction idempotency plus transition identity should
enforce that contract.

Ordinary compaction keeps its V1 wire format. Its orchestration must change on
mixed datasets: it must record the next V1 edge when any old index or V2 lineage
can depend on it, including destinations that appear unindexed in today's
bitmap. A simple initial policy is to record every address-changing compaction
while retained V2 transitions exist. Eager remapping cannot erase the needs of
other segments or in-flight index builds.

Concurrent V1 catalog replacement/trim must follow existing conflict handling
plus the mixed-retention checks. A V2 delta being disjoint from another rewrite
does not authorize it to overwrite a stale V1 or V2 catalog snapshot.

## 7. Reads, appends, and deletes

### Reads

A full scan reads current fragments and deletion vectors without historical
label I/O. An indexed read derives coverage, decodes candidate addresses, follows
both mapping kinds, filters to coverage, and applies current row visibility.
Uncovered fragments use the normal scan path.

At a stable-partition hop, group candidate addresses by source label block and
asynchronously load each required block. Offset equals the count for the label
before the block plus its rank before the row within the block. For bulk work,
seed per-label counters at a block boundary and sweep. Regroup after each hop:
a sequential first hop may scatter the second hop's accesses.

The current RowIdRemapper is synchronous. Introduce asynchronous batched
preparation/translation at decode boundaries and keep arithmetic over prepared
blocks synchronous. Do not hide object-store I/O inside the synchronous trait
or load all labels merely to satisfy it.

Cache immutable labels by storage identity, transition ID and block. Cache
translated index units/coverage by segment generation and target manifest
identity, incorporating both histories. The V1 FRI UUID alone is insufficient.

### Appending table rows

Append fresh fragments normally. No existing addresses move, so no rewrite
transition is needed. Existing indices do not automatically cover appended
fragments. An index built on V records that read snapshot even if it publishes
later; publication must validate that its required history remains available.

Appending a V2 transition is different: publish another immutable descriptor
reference. Do not append bytes to an old row-map file.

### Deleting rows

Evaluate against a pinned snapshot, translating indexed candidates into current
addresses first. Write deletion vectors for current fragments. Historical labels
stay immutable: they describe relocation then, not visibility now.

If A:3 mapped to C:1 and C:1 is deleted, an old index candidate A:3 translates to
C:1 and is suppressed by C's current deletion vector. If later V1 compaction
moves C to F, its survivor bitmap drops C:1 and the composed mapping reports
deletion. Entirely deleted fragments may disappear without a replacement map.

If delete commits first, a prepared rewrite fails source validation. If rewrite
commits first, a stale delete against its retired sources retries/re-evaluates.
Disjoint operations can rebase. Folding concurrent deletion vectors through a
prepared row map can be a later optimization; it is not implicit in this format.

Value-changing updates/overlays are not pure relocation. Index invalidation must
use current derived coverage. Initially withdraw affected segments and rebuild
them, preserving logical index definitions, rather than treating relocation as
proof that indexed values remain valid.

## 8. Retention and cleanup

Keep V1 and V2 logically separate on disk but coordinate retention. In particular,
the existing V1-only caught-up calculation must not trim an edge that feeds a
retained V2 edge or an old payload reachable through it.

Safe first policy: while V2 history is retained, preserve all retained V1/V2
mapping records. Make V1-only trim explicitly defer on such a dataset. This
conservatively increases metadata retention; it does not stop compaction,
append, delete, or query operations. A later coordinated drain rebuilds/remaps
all dependent segments and retires obsolete history atomically.

For concurrent index builders, record a history-retention floor when draining.
A stale builder whose required history predates the floor must retry/rebuild
before publication. Validate against the latest manifest on every commit retry.
An active builder is not automatically protected by an assumed lease.

Physical cleanup roots are all retained/tagged manifests. For each, follow its
V1 external details and V2 catalog -> descriptor -> row-map references, including
storage bases. A live descriptor must protect its row map even when they reside
outside an index UUID directory. Unreferenced files become eligible only after
the ordinary in-progress protection period.

Cloning/restoring must preserve the reference graph and ownership of referenced
bases. Namespace and out-of-band cleanup must check the same capabilities.
Unknown formats must prevent destructive cleanup, not merely dataset queries.

## 9. Compatibility and rollout

Allocate fresh reader and writer capability flags when the final format is
approved. Old binaries must reject mixed datasets initially. Keeping V1's wire
format unchanged does not make old maintenance code safe on V2 dependencies.
Flags must survive recomputation, retry, clone, and restore, and gate destructive
maintenance. Do not choose a numeric bit already reserved by another feature.

V1-only datasets keep existing behavior. Activation of mixed handling validates
V1 metadata and initializes coverage frontiers; it does not force eager remap
of existing index bytes. Malformed/ambiguous legacy coverage must fall back
conservatively rather than invent a fully indexed frontier.

Later, add order-preserving writes directly to V2's mapping discriminator and
change the compaction writer when ready. Keep the V1 reader adapter for existing
snapshots. No V2 order-preserving codec is defined or required by this draft.

## 10. Decisions to settle together

- Dedicated manifest catalog versus a system-index envelope with explicit
  structural semantics; prefer the former, but assess integration cost.
- Exact descriptor/FileRef schema, base-path rules, and validation limits.
- Coverage frontier persistence: dedicated metadata versus atomically maintained
  existing coverage, without conflating it with payload address space.
- Initial supported index kinds and where each performs asynchronous translation.
- Whether conservative V1/V2 history retention is acceptable for first activation,
  or coordinated draining must be implemented at the same time.
- Which clone/namespace paths must be supported before activation.

Before enabling mixed writes, require real-data tests for V1 -> V2 -> V1 and
V2 -> V1 -> V2, partial coverage, source deletions, later deletes, append, both
delete/rewrite commit orders, disjoint rewrite rebase, late index publication,
cache refresh, and cleanup with retained/tagged snapshots. Validate actual row
identities and output order, not only counts or a query falling back to scans.

This draft can evolve while implementation is underway. Formal schema allocation,
format review/voting, and release compatibility commitments follow only after
we are comfortable with the contract.
