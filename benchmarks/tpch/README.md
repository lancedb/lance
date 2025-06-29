# TPC-H

Compares Lance with Parquet, using `duckdb` as the query engine. Uses the default options for both formats.

## Setting Up

```shell
cd benchmarks/tpch
python3 -m venv .venv
source .venv/bin/activate # unix
# .venv\Scripts\activate # windows
pip3 install pylance duckdb pyarrow matplotlib pandas
```

## Running

```shell
# prints the results in the terminal and generates CSV and PNG files
python3 benchmark.py ...
```

```
usage: benchmark.py [-h] [-r RUNS] [-s SCALEFACTOR] [-d DATASET] [-l LOGGING_LEVEL]

TPCH Benchmark

options:
  -h, --help            show this help message and exit
  -r RUNS, --runs RUNS  Number of runs per query
  -s SCALEFACTOR, --scalefactor SCALEFACTOR
                        Scale of the TPC-H dataset
  -d DATASET, --dataset DATASET
                        Path to the dataset
  -l LOGGING_LEVEL, --logging_level LOGGING_LEVEL
                        Logging level
```
