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
}
</style>

<div class="feature-flags-table" markdown="1">

| Flag Bit | Flag Name                       | Reader Required | Writer Required | Introduced          | Enabled by                                                      | Description                                                                                                 |
|----------|---------------------------------|-----------------|-----------------|---------------------|----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| 1        | `FLAG_DELETION_FILES`           | Yes             | Yes             | v0.5.0 (2023-06-23) | Automatic when rows are soft-deleted                           | Fragments may contain deletion files, which record the tombstones of soft-deleted rows.                     |
| 2        | `FLAG_STABLE_ROW_IDS`           | Yes             | Yes             | v0.12.0 (2024-06-04) | Opt-in: `enable_stable_row_ids=true` at dataset creation       | Row IDs are stable for both moves and updates. Fragments contain an index mapping row IDs to row addresses. |
| 4        | `FLAG_USE_V2_FORMAT_DEPRECATED` | No              | No              | —                   | Deprecated; no longer used                                     | Files are written with the new v2 format. This flag is deprecated and no longer used.                       |
| 8        | `FLAG_TABLE_CONFIG`             | No              | Yes             | v0.19.1 (2024-10-21) | Automatic when table config is set (`update_config`)           | Table config is present in the manifest.                                                                    |
| 16       | `FLAG_BASE_PATHS`               | Yes             | Yes             | v0.34.0 (2025-08-26) | Automatic when a dataset uses base paths (e.g. shallow clone)  | Dataset uses multiple base paths (for shallow clones or multi-base datasets).                               |
| 32       | `FLAG_DISABLE_TRANSACTION_FILE` | No              | Yes             | v1.0.0 (2025-12-12) | Reserved — not yet exposed via any public API                 | The transaction is written inline in the manifest instead of in a separate `_transactions/` file.          |

</div>

Flags with bit values 64 and above are unknown and will cause implementations to reject the dataset with an "unsupported" error.

## Feature Maturity

The **Introduced** column above records the first stable release that supports each flag,
with that release's date. Feature flags are the format's compatibility contract: a reader
or writer that does not recognize a flag rejects the dataset rather than risk
misinterpreting it. The **Introduced** version is therefore also the *downgrade floor* —
the oldest release that can open a dataset using that feature. This table serves two
purposes:

- **Users** deciding when it is safe to turn a feature on: only enable it once every reader
  and writer in your deployment is at or above the Introduced version, so a rollback stays
  readable.
- **Maintainers** deciding when to make a feature the default. The rule of thumb is to wait
  until a feature has been released and stable for roughly 3–6 months, so most of the user
  base can already read data written with it before it becomes the write-time default.

Only features that affect on-disk read/write compatibility get a feature flag. Higher-level
capabilities that do not change how existing data is interpreted (tags, branches, MemTable
& WAL, change data feed) have no flag and impose no downgrade floor. If a feature *would*
break older readers but lacks a flag, that is a bug in the feature, not an omission here.

How the flags behave:

- **Automatic** flags are set by the writer whenever the underlying data is present (a
  deletion file exists, the manifest config is non-empty, base paths are present). They are
  in effect from their Introduced version onward and clear again if that data is later
  removed (for example, after compaction materializes deletes).
- **`FLAG_STABLE_ROW_IDS`** is opt-in and off by default. It must be set at dataset creation
  and cannot be toggled on or off afterward. It landed in v0.12.0 but only became usable
  end-to-end with secondary indices and compaction in v0.16.1 (2024-08-09); the write
  parameter was named `enable_move_stable_row_ids` until v0.34.0.
- **`FLAG_BASE_PATHS`** was introduced as `FLAG_SHALLOW_CLONE` in v0.34.0 and
  renamed/generalized to multi-base datasets in v0.38.3 (2025-10-28); the bit value is
  unchanged.
- **`FLAG_DISABLE_TRANSACTION_FILE`** is reserved: the flag and the commit-path support
  exist, but no public API sets it yet. Every commit currently writes the transaction both
  inline in the manifest (`transaction_section`) *and* to a separate `_transactions/` file;
  wiring this flag on would let writers skip the now-redundant external file.

For index (e.g. full-text search) format versions, see the
[Full Text Search Index](../index/scalar/fts.md#format-versions) page.
