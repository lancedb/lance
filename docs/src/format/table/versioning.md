# Format Versioning

## Feature Flags

As the table format evolves, new feature flags are added to the format.
There are two separate fields for checking for feature flags,
depending on whether you are trying to read or write the table.
Readers should check the `reader_feature_flags` to see if there are any flag it is not aware of.
Writers should check `writer_feature_flags`. If either sees a flag they don't know,
they should return an "unsupported" error on any read or write operation.

## Current Feature Flags

<style>
.feature-flags-table th:nth-child(2),
.feature-flags-table td:nth-child(2) {
  white-space: nowrap;
  min-width: 250px;
}
</style>

<div class="feature-flags-table" markdown="1">

| Flag Bit | Flag Name                       | Reader Required | Writer Required | Description                                                                                                 |
|----------|---------------------------------|-----------------|-----------------|-------------------------------------------------------------------------------------------------------------|
| 1        | `FLAG_DELETION_FILES`           | Yes             | Yes             | Fragments may contain deletion files, which record the tombstones of soft-deleted rows.                     |
| 2        | `FLAG_STABLE_ROW_IDS`           | Yes             | Yes             | Row IDs are stable for both moves and updates. Fragments contain an index mapping row IDs to row addresses. |
| 4        | `FLAG_USE_V2_FORMAT_DEPRECATED` | No              | No              | Files are written with the new v2 format. This flag is deprecated and no longer used.                       |
| 8        | `FLAG_TABLE_CONFIG`             | No              | Yes             | Table config is present in the manifest.                                                                    |
| 16       | `FLAG_BASE_PATHS`               | Yes             | Yes             | Dataset uses multiple base paths (for shallow clones or multi-base datasets).                               |
| 32       | `FLAG_DISABLE_TRANSACTION_FILE` | No              | Yes             | Transactions are recorded in the manifest rather than in a separate transaction file.                       |
| 64       | `FLAG_UNSTABLE_DATA_OVERLAY_FILES` | Yes          | Yes             | Fragments may carry data overlay files. Unstable: release builds reject it unless explicitly opted in.      |
| 128      | `FLAG_COVERED_INDEX_METADATA`   | Yes             | Yes             | Some index declares covering columns (`IndexMetadata.covering_fields`), so `fields` means keyed columns followed by carried ones. An implementation without this flag selects an index by membership of `fields` and would answer a query on a merely-carried column with an index keyed on a different one. |
| 256      | `FLAG_MIXED_DATA_FILE_VERSIONS` | Yes             | Yes             | The snapshot may reference recognized V2 data files with different exact versions. Both bits must be set and remain set on later versions. |

</div>

Flags with bit values 512 and above are unknown and will cause implementations to reject the dataset with an "unsupported" error. The paired mixed-version reader and writer bits must either both be set or both be clear; a half-set manifest is invalid.

## Mixed V2 Data File Versions

The manifest data storage version is the default for operations that do not
select an exact output version. It is a fallback, not a summary, minimum,
maximum, or profile of the data files referenced by the snapshot. Once
`FLAG_MIXED_DATA_FILE_VERSIONS` is enabled, each base data file and data overlay
file is decoded according to its own normalized version identity.

Mixed snapshots have the following invariants:

- Only exact V2.0, V2.1, V2.2, and V2.3 data file versions may be mixed.
- V1 and V2 data files may not appear in the same snapshot.
- A commit that first produces a mixed snapshot derives and sets both the reader
  and writer capability bits from its final manifest. The bits remain set on all
  later snapshots, even if a later compaction makes the files homogeneous again.
- A snapshot without the capability may only reference files matching its
  manifest fallback. The only repair exception is an unambiguous, homogeneous
  historical V2 snapshot whose legacy manifest metadata is stale.
- An operation-level `data_storage_version` selects the exact output version
  for that operation. Omitting it uses the manifest fallback. Neither case
  changes the fallback.

For example, a dataset whose fallback is V2.1 can append V2.2 files by setting
`data_storage_version="2.2"`. The same commit adds both mixed-version capability
bits. Reads then dispatch V2.1 files to the V2.1 decoder and V2.2 files to the
V2.2 decoder. Compaction can deliberately rewrite selected fragments to any
supported exact V2 target; binary copy is only valid when every selected input
file already has that exact target version.

### Compatibility Matrix

| Dataset state | Mixed-aware client | Client without bit 256 support |
| --- | --- | --- |
| Historical homogeneous V1 | Reads and writes through legacy paths | Unchanged |
| Historical homogeneous V2 | Reads and writes; legacy metadata repair remains uniform-only | Unchanged |
| New homogeneous V2 without bit 256 | Reads and writes using the manifest fallback | Unchanged |
| Mixed V2.0-V2.3 with both bits set | Reads and writes by exact per-file identity | Rejects before reading or writing |
| Mixed V2 without both bits | Rejects as a per-file capability mismatch | Not a valid dataset state |
| V1/V2 mixture | Rejects | Not a valid dataset state |

### Error Categories

Implementations distinguish these failures in their error messages so operators
can identify the violated boundary:

- unsupported reader or writer feature bit;
- half-set mixed-version capability corruption;
- unknown or malformed data file version identity;
- V1/V2 mixture;
- a non-fallback file without mixed-version capability; and
- binary-copy target mismatch, including the target, actual version, and path.

### Rollout Gate

Before the first mixed-version commit, deploy mixed-aware readers and writers
everywhere that can access the dataset. Then drain or fence writers that opened
the dataset with an older client. Only after both steps may a writer select a
different exact V2 output version. The capability bit makes clients that open
the resulting snapshot fail closed, but it cannot retroactively fence an old
writer that read an earlier manifest.
