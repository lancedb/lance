# Migration Guides

Lance aims to avoid breaking changes when possible.  Currently, we are refining the Rust public API so that we can move
it out of experimental status and make stronger commitments to backwards compatibility.  The python API is considered
stable and breaking changes should generally be communicated (via warnings) for 1-2 months prior to being finalized to
give users a chance to migrate.  This page documents the breaking changes between releases and gives advice on how to
migrate.

## 0.39

* The `lance` crate no longer re-exports utilities from `lance-arrow` such as `RecordBatchExt` or `SchemaExt`.  In the
short term, if you are relying on these utilities,  you can add a dependency on the `lance-arrow` crate.  However, we
do not expect `lance-arrow` to ever be stable, and you may want to consider forking these utilities.

* Previously, we exported `Error` and `Result` as both `lance::Error` and `lance::error::Error`.  We have now reduced
this to just `lance::Error`.  We have also removed some internal error utilities (such as `OptionExt`) from the public
API and do not plan on reintroducing these.

* The Python and Rust `dataset::diff_meta` API has been removed in favor of `dataset::delta`, 
which returns a `DatasetDelta` that offers both metadata diff through `list_transactions` and data diff 
through `get_inserted_rows` and `get_updated_rows`.

* Some other minor utilities which had previously been public are now private.  It is unlikely anyone was utilizing'
these.  Please open an issue if you were relying on any of these.

* The `lance-namespace` Rust crate now splits into `lance-namespace` that contains the main `LanceNamespace` trait 
and data models, and `lance-namespace-impls` that has different implementations of the namespace. 
The `DirectoryNamespace` and `RestNamespace` interfaces have been refactored to be more user friendly.
The `DirectoryNamespace` also now uses Lance ObjectStore for IO instead of directly depending on Apache OpenDAL.