# Lance Test data

Most test cases can be generated from previous versions of Lance format. However
some require files written by previous versions of Lance.

The folders correspond to the versions of Lance that generated the files. Each
folder contains a `datagen.py` script that generates one or more lance datasets.

## List of datasets

* `v0.7.5/with_deletions`: This is a simple table created with deletions. It is
  written in a version of Lance that did not record the `Fragment.physical_rows`
  or `DeletionFile.num_deleted_rows`, so these values are not present in the
  file. Writers can copy this table and migrate it by filling in those new
  fields.
* `v0.8.0/migrated_from_v0.7.5`: This table was originally as above, but was
  **incorrectly** migrated from v0.7.5. The `Fragment.physical_rows` field is
  incorrect, as it was filled in with the row count (after deletions). Readers
  should know to ignore these stats. Writers should correct the statistics.
* `v0.8.14/corrupt_index`: This dataset has a vector index whose fragment
  bitmap is incorrect and cannot be trusted.  If the writer version is 0.8.14
  or older then bugs may occur when searching this kind of dataset.  There is
  no good workaround for readers.  Writers should make sure to recompute the
  fragment bitmap when updating indices that were sourced from old versions.
* `v0.10.5/corrupt_schema`: This dataset had `add_columns` and `drop_columns`
  applied to it. In earlier versions of Lance, the field ids were not handled
  correctly, so there are duplicate field ids in the schema. There aren't great
  workarounds for readers. Writers should make sure to check the field ids in
  the schema and re-compute them if necessary.