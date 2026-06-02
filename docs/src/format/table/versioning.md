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

| Flag Bit | Flag Name                               | Reader Required | Writer Required | Description                                                                                                 |
|----------|-----------------------------------------|-----------------|-----------------|-------------------------------------------------------------------------------------------------------------|
| 1        | `FLAG_DELETION_FILES`                   | Yes             | Yes             | Fragments may contain deletion files, which record the tombstones of soft-deleted rows.                     |
| 2        | `FLAG_STABLE_ROW_IDS`                   | Yes             | Yes             | Row IDs are stable for both moves and updates. Fragments contain an index mapping row IDs to row addresses. |
| 4        | `FLAG_USE_V2_FORMAT_DEPRECATED`         | No              | No              | Files are written with the new v2 format. This flag is deprecated and no longer used.                       |
| 8        | `FLAG_TABLE_CONFIG`                     | No              | Yes             | Table config is present in the manifest.                                                                    |
| 16       | `FLAG_BASE_PATHS`                       | Yes             | Yes             | Dataset uses multiple base paths (for shallow clones or multi-base datasets).                               |
| 32       | `FLAG_DISABLE_TRANSACTION_FILE`         | No              | Yes             | Transaction files under `_transactions/` are disabled.                                                       |
| 64       | `FLAG_INDEX_SEGMENT_SEQ`                | No              | Yes             | Index metadata uses name-scoped `segment_seq`; writers must preserve physical segment assignments.          |
| 128      | `FLAG_INDEX_SEGMENT_SEQ_HIGH_WATER`     | No              | Yes             | Index sections store logical high-water marks; writers must preserve monotonic `segment_seq` assignment.    |

</div>

Flags with bit values 256 and above are unknown and will cause implementations to reject the dataset with an "unsupported" error.
