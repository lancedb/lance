# Quick Start

We've provided Linux and MacOS wheels for Lance in PyPI. You can install Lance python binding via:

``` sh
pip install pylance
```

Thanks for its [Apache Arrow](https://arrow.apache.org/)-first APIs, `lance` can be used as a native Arrow extension. For example, it enables users to directly use [DuckDB](https://duckdb.org/)
to analyze `lance` dataset via [DuckDB's Arrow integration](https://duckdb.org/docs/guides/python/sql_on_arrow).

``` python
# pip install pylance duckdb
import lance
import duckdb
```

## Understand Label distribution of Oxford Pet Dataset

``` python
ds = lance.dataset("s3://eto-public/datasets/oxford_pet/pet.lance")
duckdb.query('select label, count(1) from ds group by label').to_arrow_table()
```