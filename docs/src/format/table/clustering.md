# Clustering Metadata

Clustering organizes data to improve query pruning. A **provider** is the
implementation that chooses the layout and how to maintain it. Lance records
the table's current policy separately from the layout information on fragments
and index segments. This proposal defines the format in `clustering.proto`.

## Stored Information

`Manifest.clustering` holds the current `Clustering` configuration:

- `columns`: an ordered, non-empty list of distinct, non-negative schema field
  IDs for existing clustering columns.
- `provider`: a non-empty, case-sensitive implementation name.
- `version`: a non-zero, provider-assigned layout/configuration identifier.
- `provider_metadata`: `map<string, string>` containing provider configuration.

`DataFragment.clustering` and `IndexMetadata.clustering` each hold optional
`ClusteringInfo`: `provider`, `version`, and `metadata: map<string, string>`.
Providers define keys and string encodings, such as bucket, range, or group
identifiers. Empty maps are valid. A segment's information describes its own
grouping; it does not relabel its input fragments.

`version` is not a dataset version or software release. Within a table, including
branches, `(provider, version)` must keep the same layout meaning. Layout changes
use a new version; providers manage any history they need. An unknown old version
must not be interpreted using a different current configuration.

## Main Rules

- Enabling clustering does not require every append to cluster its output or
  impose a deadline for organizing existing data.
- Changing or removing configuration does not rewrite history or erase existing
  fragment and segment information. Historical reclustering is a separate operation.
- Providers may take over other providers' fragments, recluster them, and write
  their own information on the outputs. Lance imposes no provider ownership boundary.

### Legacy Column-Level Hints

Preserve `Field.unenforced_clustering_key_position` and
`Field.unenforced_clustering_key` under their existing rules. These are **legacy
column-level hints**, not fields to clear or synchronize with the new configuration.
When `Manifest.clustering` is present, supporting implementations use its `columns`
as the clustering keys instead. Otherwise, the legacy hints retain their existing
meaning.

## Clustering, Compaction, and Indexing

Providers choose the fragments to process and return groups of fragments:

- **Clustering:** the provider chooses which fragments to recluster together,
  reorganizes their rows, and writes its information on the new fragments.
- **Compaction:** the provider compacts each group and writes clustering
  information for the new fragments.
- **Indexing:** each index has its own groups. The provider builds each segment
  and writes its clustering information; different index types may use different
  grouping sizes over the same fragment layout.

Providers may reuse Lance's builders. Lance validates inputs against concurrent
changes and commits the outputs and metadata atomically. Runtime interface
signatures are outside this format contract.

The presence of fragment clustering information determines default maintenance:

- **Without clustering information:** use normal Lance compaction and indexing,
  unless a provider selects the fragments for its own operations.
- **With clustering information:** use the provider's available maintenance
  strategy. Without a compaction strategy, default compaction skips these
  fragments. Explicit ordinary compaction may override this and produces
  fragments without clustering information. Without an indexing strategy, normal
  indexing remains available and produces segments without clustering information.

Provider failures are reported, not silently retried with a different strategy.

## Ordinary Maintenance

- Unrelated commits preserve configuration and unchanged fragments' and
  segments' information, including unknown metadata keys and values.
- Ordinary data rewrites produce fragments without clustering information.
  Column replacement and overlays clear it even when the fragment ID stays the
  same or only non-clustering columns change. No layout-impact analysis is needed.
- Deletion-vector-only changes preserve the original fragment's information.
  Providers must tolerate deleted rows and revalidate maintenance bookkeeping.
- Ordinary index merge, rebuild, and remap produce segments without clustering
  information. Data fragments' clustering information is unaffected.
- Provider operations set their outputs' information. Metadata maps are not
  automatically copied or merged from inputs.

Dropping a clustering field or changing its type removes the current configuration.
Renaming while retaining field IDs and types preserves it. Overwrite may retain
configuration only when its field IDs and types remain compatible. Clearing output
information does not otherwise disable the policy. Restore restores the selected
snapshot's information; clone or import preserves layout identity or clears
conflicting claims. Metadata paths acquire no file-retention or clone semantics.

## Statistics and Reads

Clustering writes collect zone-level write seeds for clustering columns using
Lance's defined Arrow Rust comparison semantics. Consumers may use zones directly
or aggregate them into fragment or segment statistics; no particular statistics
index is required. Missing or unusable statistics keep the data as a query candidate.

Readers may ignore clustering information. Provider metadata alone cannot prove
row order or justify pruning. Statistics, index coverage, FRI translation, and
deletions retain their own contracts. Metadata cannot impose hidden requirements
for correct queries or ordinary writes.

## Compatibility

A snapshot referencing table configuration, any fragment clustering information,
or any index segment clustering information must set `FLAG_CLUSTERING_METADATA`
(`1 << 17`) in `writer_feature_flags`, not `reader_feature_flags`. This includes
fragments in a fragment metadata tree and indices in the index section.

Older writers must refuse to write; readers need no clustering capability.
Supporting writers preserve or clear information using the rules above without
understanding every provider. Clear the flag only when all three kinds of
information are absent; disabling table clustering alone is insufficient.
Providers share one flag, independent of layout version. FRI and write seeds
keep their own feature requirements. This reservation does not enable clustering
APIs or metadata-aware writes.
