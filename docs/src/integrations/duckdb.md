# DuckDB

Lance datasets can be queried in SQL with [DuckDB](https://duckdb.org/),
an in-process OLAP relational database. Using DuckDB means you can write complex SQL queries (that may not yet be supported in Lance), without needing to move your data out of Lance.

!!! note
    This integration is done via a DuckDB extension, whose source code is available
    [here](https://github.com/lance-format/lance-duckdb).
    To ensure you see the latest examples and syntax, check out the
    [DuckDB extension](https://duckdb.org/community_extensions/extensions/lance) 
    documentation page.

## Usage: Python

### Install dependencies

Install Lance, DuckDB and Pyarrow and follow the examples below.

```bash
pip install pylance duckdb pyarrow
```

### Add data to a Lance dataset

Let's add some data to a Lance dataset.

```python
import lance
import pyarrow as pa

data = [
    {"animal": "duck", "noise": "quack", "vector": [0.9, 0.7, 0.1]},
    {"animal": "horse", "noise": "neigh", "vector": [0.3, 0.1, 0.5]},
    {"animal": "dragon", "noise": "roar", "vector": [0.5, 0.2, 0.7]},
]
pa_table = pa.Table.from_pylist(data)

lance_path = "./lance_duck.lance"
ds = lance.write_dataset(pa_table, lance_path, mode="overwrite")
```

This will store the Lance dataset to the specified local path.

### Install Lance extension in DuckDB

Install the Lance extension in DuckDB as follows.

```python
import duckdb

duckdb.sql(
    """
    INSTALL lance FROM community;
    LOAD lance;
    """
)
```

### Query a `*.lance` path

You're now ready to query the Lance dataset using SQL!

```python
# Get results from Lance in DuckDB!
r1 = duckdb.sql(
    """
    SELECT *
    FROM './lance_duck.lance'
    LIMIT 5;
    """
)
print(r1)
```
This returns:
```
┌─────────┬─────────┬─────────────────┐
│ animal  │  noise  │     vector      │
│ varchar │ varchar │    double[]     │
├─────────┼─────────┼─────────────────┤
│ duck    │ quack   │ [0.9, 0.7, 0.1] │
│ horse   │ neigh   │ [0.3, 0.1, 0.5] │
│ dragon  │ roar    │ [0.5, 0.2, 0.7] │
└─────────┴─────────┴─────────────────┘
```

???+ info "Query S3 paths directly"
    You can also query `s3://` paths directly. To do this, you can use DuckDB's native secrets mechanism to provide credentials.

    ```sql
    r1 = duckdb.sql(
        """
        CREATE SECRET (TYPE S3, provider credential_chain);

        SELECT *
        FROM 's3://bucket/path/to/dataset.lance'
        LIMIT 5;
        """
    )
    ```

### Search

The extension exposes lance_search(...) as a unified entry point for:

- Vector search (KNN / ANN)
- Full-text search (FTS)
- Hybrid search (vector + FTS)

!!! warning
    DuckDB treats `column` as a keyword in some contexts. It's recommended to
    use `text_column` / `vector_column` as column names for the Lance extension.

#### Vector search

You can perform vector search on a column. This returns the `_distance`
(smaller is closer, so sort in ascending order for nearest neighbors).

```python
# Show results similar to "the duck goes quack"
q2 = [0.8, 0.7, 0.2]

r2 = duckdb.sql(
    """
    SELECT animal, noise, vector
    FROM lance_vector_search(
        './lance_duck.lance',
        'vector',
        q2::FLOAT[],
        k = 1,
        prefilter = true
    )
    ORDER BY _distance ASC;
    """
)
print(r2)
```
This returns:
```
┌─────────┬─────────┬─────────────────┐
│ animal  │  noise  │     vector      │
│ varchar │ varchar │    double[]     │
├─────────┼─────────┼─────────────────┤
│ duck    │ quack   │ [0.9, 0.7, 0.1] │
└─────────┴─────────┴─────────────────┘
```

#### Full-text search (FTS)

Run keyword-based BM25 search as shown below. This returns a `_score`, which
is sorted in descending order to get the most relevant results.

```python
# Show results for the query "the brave knight faced the dragon"
r3 = duckdb.sql(
    """
    SELECT animal, noise, vector
    FROM lance_fts(
        './lance_duck.lance',
        'animal',
        'the brave knight faced the dragon',
        k = 1,
        prefilter = true)
    ORDER BY _score DESC;
    """
)
print(r3)
```
This returns:
```
┌─────────┬─────────┬─────────────────┐
│ animal  │  noise  │     vector      │
│ varchar │ varchar │    double[]     │
├─────────┼─────────┼─────────────────┤
│ dragon  │ roar    │ [0.5, 0.2, 0.7] │
└─────────┴─────────┴─────────────────┘
```

#### Hybrid search

Hybrid search combines vector and FTS scores, returning a `_hybrid_score` in addition
to `_distance` / `_score`. To get the most relevant results, sort in descending order.

```python
# Show results similar to "the duck surprised the dragon"
q4 = [0.8, 0.7, 0.2]

r4 = duckdb.sql(
    """
    SELECT animal, noise, vector
    FROM lance_hybrid_search(
        './lance_duck.lance',
        'vector', q4::FLOAT[],
        'animal', 'the duck surprised the dragon',        
        k = 2,
        prefilter = true
    )
    ORDER BY _hybrid_score DESC;
    """
)
print(r4)
```
This should give:
```
┌─────────┬─────────┬─────────────────┐
│ animal  │  noise  │     vector      │
│ varchar │ varchar │    double[]     │
├─────────┼─────────┼─────────────────┤
│ duck    │ quack   │ [0.9, 0.7, 0.1] │
│ dragon  │ roar    │ [0.5, 0.2, 0.7] │
└─────────┴─────────┴─────────────────┘
```

## Usage: DuckDB CLI

DuckDB comes with a CLI that makes it easy to run SQL queries in the terminal.
Check out the [DuckDB extension](https://duckdb.org/community_extensions/extensions/lance)  documentation page for examples using the DuckDB CLI.
