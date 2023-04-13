***Compare lance vs parquet for TPCH Q1 and Q6 using SF1 dataset***

**Steps to run the benchmark:**

1. `cd lance/benchmarks/tpch`
2. `mkdir dataset && cd dataset`
3. download parquet file lineitem from : "https://github.com/cwida/duckdb-data/releases/download/v1.0/lineitemsf1.snappy.parquet"; then rename it to "lineitem_sf1.parquet"
4. generate lance file from the parquet file in the same directory
5. `cd ..`
6. `python3 benchmark.py q1`
