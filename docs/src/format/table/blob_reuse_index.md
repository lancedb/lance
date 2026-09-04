# Blob Reuse Index

## Scope

A Blob Reuse Index (BRI) is optional metadata on a DataFile. It provides sparse
redirects from file-local Blob v2 identifiers to existing physical sidecars.
The index applies only to Packed and Dedicated descriptors. Inline payloads
remain in the data file, and External descriptors retain their URI and base-path
semantics.

BRI changes the table manifest format only. It does not change the Blob v2
descriptor layout or the sidecar file format.

## Representation

`DataFile.blob_reuse_index` contains one or more `BlobReuseSource` messages. A
source identifies an optional base path, one sidecar directory, a sequence of
output-local identifiers, and a positionally aligned sequence of physical
identifiers. `local_ids` uses `RowIdSequence`; `physical_ids` uses
`EncodedU64Array`.

An absent source `base_id` means the containing DataFile's `base_id`. An absent
DataFile `base_id` means the dataset's primary root. `blob_dir` is relative to
the effective base's data directory and is exactly one non-empty path segment.

The BRI field is absent when no redirects exist. A present BRI has at least one
source, and every source has at least one mapping.

## Validation

A reader validates BRI metadata while opening the manifest:

- Each effective `(base_id, blob_dir)` pair occurs at most once.
- Every explicit or inherited non-primary base exists in
  `Manifest.base_paths`.
- Each source's local identifiers are strictly increasing, and local
  identifiers do not overlap across sources.
- The local and physical sequences have equal lengths.
- Every local and physical identifier is in `1..=u32::MAX`.
- Physical identifiers retain positional order and need not be sorted.

BRI is an immutable part of physical DataFile identity. If a commit retains the
same `(base_id, path)`, it retains a semantically identical sparse mapping.
Adding, removing, or changing BRI metadata for that identity is invalid. A
writer changes BRI only by writing a new physical DataFile.

## Read Resolution

For a Packed or Dedicated local identifier, a reader first searches the BRI. A
hit selects the source containing that local identifier and the physical
identifier at the same sequence position. The sidecar path is the effective
base's data directory, followed by `blob_dir`, followed by the 32-bit binary
representation of the physical identifier after bit reversal and the `.blob`
suffix.

A miss uses identity resolution: the containing DataFile's effective data
directory, the DataFile stem, and the local identifier encoded with the same
sidecar filename rule. Dedicated descriptors read the complete sidecar. Packed
descriptors additionally apply their descriptor position and size.

## Write and Compaction Requirements

Each output DataFile uses one identifier allocator for reused references and
new sidecars. Before allocation, a reused reference is resolved to its terminal
physical `(base_id, blob_dir, physical_id)` tuple. Equal terminal references in
one output DataFile receive one local identifier. A reused identifier is added
to BRI and does not create a sidecar under the output stem. A new or repacked
payload uses its allocated identifier under the output stem and does not receive
a BRI entry. Sources therefore have increasing local identifiers even when
their physical identifiers are unordered.

Compaction reuses terminal Dedicated sidecars by default. Packed sidecars use a
physical-pack utilization threshold. For every terminal physical pack reachable
from the current snapshot, the implementation computes:

```text
utilization = union_bytes(visible_packed_ranges) / physical_pack_size
```

All visible Packed descriptors that resolve to the same physical object
participate, including descriptors reached through different DataFiles, BRI
sources, fragments, fields, or equivalent base-path aliases. Duplicate and
overlapping ranges count once. The calculation reads descriptors and physical
object sizes, not payload bytes.

The pack is reused when utilization is at least the configured
`blob_repack_utilization_threshold`, and every selected reference to it is
repacked under the output stem when utilization is below the threshold. The
default threshold is 0.3. The descriptor keeps its original Packed position and
size when the sidecar is reused.

A low-utilization pack makes each current fragment that visibly references it a
compaction candidate even when row-count and deletion rules would not select
that fragment. Existing source-fragment, source-row, and source-byte budgets
still bound the work admitted to one compaction plan. If only part of a shared
pack's referring fragments fit, later compactions recompute utilization from
the then-current snapshot and continue reclaiming it.

Inline values remain inline, and External descriptors remain external. An
implementation may expose a mode that disables sidecar reuse; in that mode both
Packed and Dedicated payloads are materialized like ordinary writes, and a
complete rewrite can remove the last BRI from a manifest.

## Cleanup, Rollback, and Clone

Cleanup treats every retained DataFile's own stem directory as live. It also
treats each exact BRI sidecar path in the current root as live. Other sidecars
under a source stem may be collected once no retained DataFile or BRI entry
references them. Cleanup of one root does not delete objects owned by another
base.

Rollback of an uncommitted output removes only that output's `.lance` file and
its own stem directory. It does not traverse BRI sources.

A shallow clone preserves BRI metadata. An implicit source base follows the
containing DataFile into the clone's source base, so no Blob payload copy is
required.

A deep clone copies every DataFile and its complete own stem directory. It also
copies each distinct explicit BRI physical reference. Distinct effective
`(base_id, blob_dir)` sources receive distinct target directories. The cloned
BRI uses those target directories with no source base, leaving the target
independent of every source root.

## Feature Flag

If any DataFile contains BRI metadata, the manifest sets bit 512 in both
`reader_feature_flags` and `writer_feature_flags`. Both bits are cleared after
the last BRI disappears. A manifest with BRI metadata and missing bits, with
bits but no BRI metadata, or with only one of the reader and writer bits is
invalid.
