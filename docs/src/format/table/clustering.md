# Clustering Metadata

Clustering puts rows with similar values close together. For example, grouping
rows by time helps a query for one day skip data from other days. Column
statistics tell queries which fragments, index segments, or smaller row groups
can be skipped. Writes can use the same information to reduce the data
they need to inspect.

Lance does not choose a clustering algorithm or provide a default clustering
implementation. Instead, it defines metadata that different implementations can
use. Each implementation is called a **provider**. Contributors can build and
maintain their own providers.

The provider decides how to organize data during writes. Readers do not need to
know that algorithm: they use column statistics to skip data. These statistics
can come from zone-level write seeds, a ZoneMap index, or fragment and segment
summaries built from the seeds.

## What Lance Stores

### Table Configuration

`Manifest.clustering` is a repeated list of immutable definitions. Each contains:

- `id`: a unique, non-zero table-local ID, not a position in the list.
- `columns`: the clustering columns, in order. Store their schema field IDs;
  the list must be non-empty, with no duplicate or negative IDs, and every ID
  must exist in the schema when the definition is registered. Historical
  definitions may retain IDs of subsequently dropped columns.
- `provider`: the provider's non-empty, case-sensitive name.
- `version`: a non-zero version assigned by the provider to these layout rules.
- `provider_metadata`: provider configuration as `map<string, string>`.

IDs start at 1. Changing a definition creates a new one with a larger ID.
New clustering work defaults to the largest ID, regardless of list order.
An empty list has no default. Existing fragments and segments keep their own
references, so old and new layouts can coexist.

Keep the largest-ID definition and every definition still referenced in the
snapshot. Other definitions may be removed; older snapshots retain their own
copies. The current manifest does not need the entire configuration history.

### Fragment and Index Segment Information

Each fragment and index segment can record how it was organized:

- `clustering_id`: optional reference to a definition in the same manifest.
  Unset means no clustering claim; a present 0 or an unresolved ID is invalid.
- `metadata`: generic `map<string, string>` for per-object information, including
  provider-defined bucket, range, or group IDs. Empty maps are valid.

The clustering ID identifies shared rules, not a bucket. A segment's metadata
describes its own grouping; it does not change its input fragments' metadata.
Other features can also use the generic map. Each feature owns and maintains its
keys; see [Object Metadata](index.md#object-metadata).

## What Providers Do

Providers decide which fragments to process together, do the work, and write
metadata for the results:

- **Clustering:** choose fragments, reorganize their rows, and write new
  fragments with clustering metadata.
- **Compaction:** choose fragments to compact together and write the compacted
  fragments with clustering metadata.
- **Indexing:** choose groups for each index, build one segment per group, and
  write segment clustering metadata. Different indices can group the same
  fragments differently.

Providers can reuse Lance's builders. Lance checks the inputs and conflicting
concurrent changes, then commits the new files and metadata together. The
provider API is outside this format proposal.

### When No Provider Strategy Is Available

- **Fragments without clustering metadata:** normal Lance compaction and
  indexing apply, unless a provider chooses to process them.
- **Compaction of fragments with clustering metadata:** use the provider's
  strategy if available. Otherwise, skip them by default. Users may explicitly
  request normal compaction; its new fragments have no clustering metadata.
- **Indexing of fragments with clustering metadata:** use the provider's strategy
  if available. Otherwise, normal indexing can still run; its new segments have
  no clustering metadata.

If a provider fails, return an error. Do not silently switch to normal maintenance.

## What Happens During Maintenance

- **Nothing changed:** keep definitions, references, and metadata on unchanged
  fragments and segments, including keys and values the writer does not recognize.
- **Normal data rewrite:** clear `clustering_id` and provider-owned clustering
  keys on the rewritten fragments. Column replacement, refresh, and overlays
  follow this rule even if the fragment
  ID stays the same or the changed column is not a clustering key. There is no
  need to decide whether the layout was affected.
- **Only deletion vectors changed:** keep the fragment's `clustering_id` and
  provider-owned clustering keys. Providers must account for deleted rows when
  planning later maintenance.
- **Index merge:** the provider decides how to merge and writes the new segments'
  clustering metadata. A normal merge without a provider strategy produces
  segments with no clustering metadata.
- **Index remap or rebuild of an existing group:** keep the segment's clustering
  metadata when updating row addresses or rebuilding index files for that group.
  Rebuilding with different fragment groups creates a new layout and follows the
  rule below.
- **New provider-created layout:** the provider writes its own metadata. Lance
  does not copy or combine the input maps automatically. A rebuild with different
  groups and no provider strategy produces segments with no clustering metadata.

Index maintenance does not change the data fragments' clustering metadata.
These rules apply to clustering references and clustering-owned keys, not every
entry in a generic map. Other key owners implement their own maintenance rules.

## How Readers Benefit

Providers should compare clustering values using Lance's defined Arrow Rust
ordering rules, because readers use those rules for column statistics. Sorting
values differently can make the min/max ranges wider and reduce pruning. It does
not change the reader's rules.

Clustering writes collect zone-level write seeds for clustering columns using
the same rules. Readers can use zone statistics or combine them into fragment
and segment summaries. This does not require a particular statistics index. If
statistics are missing or unusable, keep the data as a query candidate.

Readers can ignore clustering metadata. A provider's metadata is not proof that
data can be skipped: readers still need valid statistics and index coverage.
FRI address translation and deletion handling follow their own rules. Clustering
metadata must not add hidden requirements for correct reads or normal writes.

## Reader and Writer Compatibility

Set `FLAG_CLUSTERING_METADATA` (`1 << 17`) in a snapshot's `writer_feature_flags`
if it contains any of the following:

- Table clustering definitions.
- Fragment clustering information, including records in a fragment metadata tree.
- Index segment clustering information, including records in the index section.
- Nonempty generic fragment or segment metadata maps, even without clustering.

Readers need no clustering flag because they can ignore this metadata. Writers
that do not support the feature must refuse writes. Writers that do support it
follow the rules above; they do not need to understand every provider.

Clear the flag only when all the above are absent. All providers and layout
versions use this one flag. FRI and write seeds have separate feature requirements.
Reserving this flag does
not implement clustering APIs or metadata-aware writes.

### Legacy Column-Level Hints

`Field.unenforced_clustering_key_position` and `Field.unenforced_clustering_key`
are **legacy column-level hints**. Keep their existing rules. Do not remove them
or change them to match the new configuration.

When `Manifest.clustering` is nonempty, implementations supporting this
proposal use the largest-ID definition's columns instead of the legacy hints.
Otherwise, the hints keep their existing meaning. Historical object references
are interpreted through their own definitions, not the default or legacy hints.
