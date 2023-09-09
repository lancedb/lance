# Python bindings for Lance Data Format

> :warning: **Under heavy development**

<div align="center">
<p align="center">

<img width="257" alt="Lance Logo" src="https://user-images.githubusercontent.com/917119/199353423-d3e202f7-0269-411d-8ff2-e747e419e492.png">

Lance is a new columnar data format for data science and machine learning
</p></div>

Why you should use Lance
1. Is order of magnitude faster than parquet for point queries and nested data structures common to DS/ML
2. Comes with a fast vector index that delivers sub-millisecond nearest neighbors search performance
3. Is automatically versioned and supports lineage and time-travel for full reproducibility
4. Integrated with duckdb/pandas/polars already. Easily convert from/to parquet in 2 lines of code


## Quick start

**Installation**

```shell
pip install pylance
```

Make sure you have a recent version of pandas (1.5+), pyarrow (10.0+), and DuckDB (0.7.0+)

**Converting to Lance**
```python
import lance

import pandas as pd
import pyarrow as pa
import pyarrow.dataset

df = pd.DataFrame({"a": [5], "b": [10]})
uri = "/tmp/test.parquet"
tbl = pa.Table.from_pandas(df)
pa.dataset.write_dataset(tbl, uri, format='parquet')

parquet = pa.dataset.dataset(uri, format='parquet')
lance.write_dataset(parquet, "/tmp/test.lance")
```

**Reading Lance data**
```python
dataset = lance.dataset("/tmp/test.lance")
assert isinstance(dataset, pa.dataset.Dataset)
```

**Pandas**
```python
df = dataset.to_table().to_pandas()
```

**DuckDB**
```python
import duckdb

# If this segfaults, make sure you have duckdb v0.7+ installed
duckdb.query("SELECT * FROM dataset LIMIT 10").to_df()
```

**Vector search**

Download the sift1m subset

```shell
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
```

Convert it to Lance

```python
import lance
from lance.vector import vec_to_table
import numpy as np
import struct

nvecs = 1000000
ndims = 128
with open("sift/sift_base.fvecs", mode="rb") as fobj:
    buf = fobj.read()
    data = np.array(struct.unpack("<128000000f", buf[4 : 4 + 4 * nvecs * ndims])).reshape((nvecs, ndims))
    dd = dict(zip(range(nvecs), data))

table = vec_to_table(dd)
uri = "vec_data.lance"
sift1m = lance.write_dataset(table, uri, max_rows_per_group=8192, max_rows_per_file=1024*1024)
```

Build the index

```python
sift1m.create_index("vector",
                    index_type="IVF_PQ", 
                    num_partitions=256,  # IVF
                    num_sub_vectors=16)  # PQ
```

Search the dataset

```python
# Get top 10 similar vectors
import duckdb

dataset = lance.dataset(uri)

# Sample 100 query vectors. If this segfaults, make sure you have duckdb v0.7+ installed
sample = duckdb.query("SELECT vector FROM dataset USING SAMPLE 100").to_df()
query_vectors = np.array([np.array(x) for x in sample.vector])

# Get nearest neighbors for all of them
rs = [dataset.to_table(nearest={"column": "vector", "k": 10, "q": q})      
      for q in query_vectors]
```

*More distance metrics, HNSW, and distributed support is on the roadmap


## Python package details

Install from PyPI: `pip install pylance`  # >=0.3.0 is the new rust-based implementation
Install from source: `maturin develop` (under the `/python` directory)
Run unit tests: `make test`
Run integration tests: `make integtest`

Import via: `import lance`

The python integration is done via pyo3 + custom python code:

1. We make wrapper classes in Rust for Dataset/Scanner/RecordBatchReader that's exposed to python.
2. These are then used by LanceDataset / LanceScanner implementations that extend pyarrow Dataset/Scanner for duckdb compat.
3. Data is delivered via the Arrow C Data Interface

## Motivation

Why do we *need* a new format for data science and machine learning?

### 1. Reproducibility is a must-have

Versioning and experimentation support should be built into the dataset instead of requiring multiple tools.<br/>
It should also be efficient and not require expensive copying everytime you want to create a new version.<br/>
We call this "Zero copy versioning" in Lance. It makes versioning data easy without increasing storage costs.

### 2. Cloud storage is now the default

Remote object storage is the default now for data science and machine learning and the performance characteristics of cloud are fundamentally different.<br/>
Lance format is optimized to be cloud native. Common operations like filter-then-take can be order of magnitude faster
using Lance than Parquet, especially for ML data.

### 3. Vectors must be a first class citizen, not a separate thing

The majority of reasonable scale workflows should not require the added complexity and cost of a
specialized database just to compute vector similarity. Lance integrates optimized vector indices
into a columnar format so no additional infrastructure is required to get low latency top-K similarity search.

### 4. Open standards is a requirement

The DS/ML ecosystem is incredibly rich and data *must be* easily accessible across different languages, tools, and environments.
Lance makes Apache Arrow integration its primary interface, which means conversions to/from is 2 lines of code, your
code does not need to change after conversion, and nothing is locked-up to force you to pay for vendor compute.
We need open-source not fauxpen-source.

