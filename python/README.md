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

tbl = dataset.to_table()  # next release of duckdb will have pushdowns enabled
duckdb.query("SELECT * FROM tbl LIMIT 10").to_df()
```

**Vector search**

Download an indexed [sift dataset](https://eto-public.s3.us-west-2.amazonaws.com/datasets/sift/sift_ivf256_pq16.tar.gz),
and unzip it into `vec_data.lance`

```python
# Get top 10 similar vectors
import lance
import duckdb
import numpy as np

uri = "vec_data.lance"
dataset = lance.dataset(uri)

# Sample 100 query vectors
tbl = dataset.to_table()
sample = duckdb.query("SELECT vector FROM tbl USING SAMPLE 100").to_df()
query_vectors = np.array([np.array(x) for x in sample.vector])

# Get nearest neighbors for all of them
rs = [dataset.to_table(nearest={"column": "vector", 
                                "k": 10, 
                                "q": query_vectors[i, :]}) 
      for i in range(query_vectors.shape[0])]
```

For the fast indexing capability, you can 
and run the same code as above. We're working on a more convenient indexing tool via python.

*More distance metrics, supported types, and compute integration coming


## Python package details

Install from PyPI: `pip install pylance`  # >=0.3.0 is the new rust-based implementation
Install from source: `maturin develop` (under the `/python` directory)

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

