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
