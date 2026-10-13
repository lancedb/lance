# Clustering Metadata

Clustering organizes data to improve query pruning. Like Delta, Lance separates
the table's current policy from information about individual outputs. Providers
choose the layout and maintenance logic. This proposal defines the format in
`clustering.proto`; the writer feature remains unsupported until implemented.

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
- When configuration is present, `columns` defines the clustering keys. Writers
  clear the older `Field.unenforced_clustering_key_position` and
  `Field.unenforced_clustering_key` hints to avoid conflicting definitions.

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

Without an available provider compaction strategy, default compaction skips
clustered fragments and processes unclustered ones normally. An explicit ordinary
compaction may override this; its output is unmarked. Provider failures are
reported rather than silently switching strategies. Without a provider indexing
strategy, normal index building remains available, producing unmarked segments.
Fragments without clustering information use Lance's default compaction and
indexing unless a provider selects them for its own operations.

## Ordinary Maintenance

- Unrelated commits preserve configuration and unchanged fragments' and
  segments' information, including unknown metadata keys and values.
- Ordinary data rewrites do not inherit fragment clustering information.
  Column replacement and overlays clear it even when the fragment ID remains
  unchanged or only non-clustering columns change. No layout-impact analysis is needed.
- Deletion-vector-only changes preserve the original fragment's information.
  Providers must tolerate deleted rows and revalidate maintenance bookkeeping.
- Ordinary index merge, rebuild, and remap do not inherit information on the
  rewritten segments. Data fragments' clustering information is unaffected.
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

Sources: [Delta clustered-table protocol](https://github.com/delta-io/delta/blob/ef3e91b509365cdc2b23c1afdbbab3b4d3828d63/PROTOCOL.md#clustered-table)
and [changing clustering columns](https://docs.delta.io/delta-clustering/#change-clustering-columns).
