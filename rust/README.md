# Rust Implementation of Lance Data Format

> :warning: **Under heavy development**

<div align="center">
<p align="center">

<img width="257" alt="Lance Logo" src="https://user-images.githubusercontent.com/917119/199353423-d3e202f7-0269-411d-8ff2-e747e419e492.png">

**A new columnar data format for data science and machine learning**
</p></div>

## Quick start

Warning: the pyo3 package is not yet on PyPI

From under the /pylance directory, run `maturin develop` to build and install the python package.

**Converting to Lance**
```python
import lance

import pyarrow as pa
import pyarrow.dataset

uri = "/path/to/parquet"
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

```python
# Get top 10 similar vectors
import numpy as np
q = np.random.randn(128)  # query vector
query = {
    "column": "emb",  # assume `emb` column is FixedSizeList of Float32
    "q": q,
    "k": 10
}
dataset.to_table(nearest=query).to_pandas()
```

*More distance metrics, supported types, and compute integration coming

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


## Python package

Currently under development in the `pylance` directory. This will become the main python integration once ready.

Install from source: `maturin develop` (later on `pip install pylance` will be from this package)

Import via: `import lance`

The python integration is done via pyo3 + custom python code:

1. We make wrapper classes in Rust for Dataset/Scanner/RecordBatchReader that's exposed to python.
2. These are then used by LanceDataset / LanceScanner implementations that extend pyarrow Dataset/Scanner for duckdb compat.
3. Data is delivered via the Arrow C Data Interface

## Rust package

Include package "lance" in Cargo.toml as dependency.

For macos we recommend you enable the blas feature flag for hardware acceleration.

```toml
[target.'cfg(target_os = "macos")'.dependencies]
lance = { features = ["blas"]}
```