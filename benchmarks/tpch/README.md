# Compare Lance vs Parquet for TPCH Q1 and Q6 (SF1)

## Prerequisites

Install Python dependencies:

```bash
python3 -m pip install duckdb pyarrow pylance
```

## Prepare Dataset (generated locally with DuckDB)

Run from this directory (`lance/benchmarks/tpch`):

```bash
mkdir -p dataset
python3 - <<'PY'
import duckdb
import pyarrow.parquet as pq
import lance

con = duckdb.connect(database=":memory:")
con.execute("INSTALL tpch; LOAD tpch")
con.execute("CALL dbgen(sf=1)")

lineitem = con.query("SELECT * FROM lineitem").to_arrow_table()
pq.write_table(lineitem, "dataset/lineitem_sf1.parquet")
lance.write_dataset(lineitem, "dataset/lineitem.lance", mode="overwrite")
PY
```

This creates:

- `dataset/lineitem_sf1.parquet`
- `dataset/lineitem.lance`

## Run Benchmark

```bash
python3 benchmark.py q1
python3 benchmark.py q6
```
