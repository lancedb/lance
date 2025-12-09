# DuckDB

In Python, Lance datasets can also be queried with [DuckDB](https://duckdb.org/),
an in-process SQL OLAP database. This means you can write complex SQL queries to analyze your data in Lance.

This integration is done via [DuckDB SQL on Apache Arrow](https://duckdb.org/docs/guides/python/sql_on_arrow),
which provides zero-copy data sharing between LanceDB and DuckDB.
DuckDB is capable of passing down column selections and basic filters to Lance,
reducing the amount of data that needs to be scanned to perform your query.
Finally, the integration allows streaming data from Lance tables,
allowing you to aggregate tables that won't fit into memory.
All of this uses the same mechanism described in DuckDB's
blog post *[DuckDB quacks Arrow](https://duckdb.org/2021/12/03/duck-arrow.html)*.

A `LanceDataset` is accessible to DuckDB through the Arrow compatibility layer directly.
To query the resulting Lance dataset in DuckDB,
all you need to do is reference the dataset by the same name in your SQL query.

```python
import duckdb # pip install duckdb
import lance

ds = lance.dataset("./my_lance_dataset.lance")

duckdb.query("SELECT * FROM ds")
# ┌─────────────┬─────────┬────────┐
# │   vector    │  item   │ price  │
# │   float[]   │ varchar │ double │
# ├─────────────┼─────────┼────────┤
# │ [3.1, 4.1]  │ foo     │   10.0 │
# │ [5.9, 26.5] │ bar     │   20.0 │
# └─────────────┴─────────┴────────┘

duckdb.query("SELECT mean(price) FROM ds")
# ┌─────────────┐
# │ mean(price) │
# │   double    │
# ├─────────────┤
# │        15.0 │
# └─────────────┘
```
