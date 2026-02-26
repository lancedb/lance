## I/O cheat sheet (Python)

Use this file when the user asks how to write/read Lance datasets, manage versions, or work with object stores.

## Write a dataset

Use `lance.write_dataset(data, uri, mode=...)`.

Modes:

- `mode="create"`: create new dataset (error if exists)
- `mode="overwrite"`: create a new version that replaces the latest snapshot
- `mode="append"`: append data as a new version (or create if missing)

Inputs:

- `pyarrow.Table`
- `pyarrow.RecordBatchReader`
- pandas DataFrame
- other reader-like sources supported by the installed Lance version

## Open a dataset

Use `lance.dataset(uri, version=..., asof=..., storage_options=...)`.

Notes:

- `version` can be a number or a tag (depending on the environment/version).
- Use `storage_options` for object stores (credentials, endpoint, etc.).

## Read / scan

Use `ds.scanner(...)` for pushdowns:

- `columns=[...]` for projection
- `filter="..."` for predicate pushdown
- `limit=...` for limit pushdown
- `nearest={...}` for vector search
- `prefilter=True/False` to control filter ordering when combined with `nearest`
- `use_scalar_index=True/False` to control scalar index usage

Then materialize:

- `scanner(...).to_table()`
- `scanner(...).to_batches()`

## Hybrid query: vector + filter

Use a scalar index for the filter column when the filter is selective and you set `prefilter=True`.

Example:

```python
tbl = ds.scanner(
    nearest={"column": "vector", "q": q, "k": 10},
    filter="category = 'a'",
    prefilter=True,
).to_table()
```

## Inspect indices

Prefer:

- `ds.describe_indices()`

Use with care:

- `ds.list_indices()` can be expensive because it may load index statistics.
