<div align="center">
<p align="center">

<img width="257" alt="Lance Logo" src="https://user-images.githubusercontent.com/917119/199353423-d3e202f7-0269-411d-8ff2-e747e419e492.png">

**Modern data lake built for AI & ML workflows. Lance combines classic data lake features (table operations, time travel, filter pushdown), with better support for multimodal data, the ability to run in embedded mode (no catalog needed), 100x faster random access, zero-cost column evolution (not just schema evolution), rich secondary indices, and more.<br/><br/>**

**Compatible with Pandas, DuckDB, Polars, Pyarrow, and Ray with more integrations on the way.**

<a href="https://lancedb.github.io/lance/">Documentation</a> •
<a href="https://blog.lancedb.com/">Blog</a> •
<a href="https://discord.gg/zMM32dvNtd">Discord</a> •
<a href="https://x.com/lancedb">X</a>

[CI]: https://github.com/lancedb/lance/actions/workflows/rust.yml
[CI Badge]: https://github.com/lancedb/lance/actions/workflows/rust.yml/badge.svg
[Docs]: https://lancedb.github.io/lance/
[Docs Badge]: https://img.shields.io/badge/docs-passing-brightgreen
[crates.io]: https://crates.io/crates/lance
[crates.io badge]: https://img.shields.io/crates/v/lance.svg
[Python versions]: https://pypi.org/project/pylance/
[Python versions badge]: https://img.shields.io/pypi/pyversions/pylance

[![CI Badge]][CI]
[![Docs Badge]][Docs]
[![crates.io badge]][crates.io]
[![Python versions badge]][Python versions]

</p>
</div>

<hr />

Lance is a modern data lake that is optimized for ML workflows and tables. Lance is perfect for:

1. Building search engines with any combination of semantic (vector) search, full
   text search, and filtering.
2. Large-scale ML training requiring high performance IO and shuffles.
3. Running experiments by quickly adding and removing features without data copies.
4. Storing, querying, and inspecting deeply nested data for robotics or large blobs like images, point clouds, and more.

The following table summarizes the key features of Lance compared with other options.

| Feature          | Lance             | WebDataset    | Iceberg / Delta Lake        | Parquet                     |
| ---------------- | ----------------- | ------------- | --------------------------- | --------------------------- |
| Random Access    | ✅ Fast           | ❌ No Support | ⚠️ Slow                     | ⚠️ Slow                     |
| Fast Scan        | ✅ Yes            | ✅ Yes        | ✅ Yes                      | ✅ Yes                      |
| Schema Evolution | ✅ Yes, zero-copy | ❌ No Support | ⚠️ Copy table to add column | ⚠️ Copy table to add column |
| Multimodal       | ✅ Yes            | ✅ Yes        | ❌ No                       | ❌ No                       |
| Search           | ✅ Yes            | ❌ No Support | ⚠️ Slow                     | ⚠️ Slow                     |
| Analytics        | ✅ Fast           | ❌ No Support | ✅ Fast                     | ✅ Fast                     |

> [!TIP]
> Lance is in active development and we welcome contributions. Please see our [contributing guide](docs/contributing.rst) for more information.

## Quick Start

**Installation**

```shell
pip install pylance
```

To install a preview release:

```shell
pip install --pre --extra-index-url https://pypi.fury.io/lancedb/ pylance
```

> [!TIP]
> Preview releases are released more often than full releases and contain the
> latest features and bug fixes. They receive the same level of testing as full releases.
> We guarantee they will remain published and available for download for at
> least 6 months. When you want to pin to a specific version, prefer a stable release.

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
df
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

## Directory structure

| Directory          | Description                               |
| ------------------ | ----------------------------------------- |
| [rust](./rust)     | Core Rust implementation                  |
| [python](./python) | Python bindings (PyO3)                    |
| [java](./java)     | Java bindings (JNI) and Spark integration |
| [docs](./docs)     | Documentation source                      |

## What makes Lance different

Here we will highlight a few aspects of Lance’s design. For more details, see the full [Lance design document](https://lancedb.github.io/lance/format.html).

**Vector index**: Lance includes a state of the art vector index implementation for
similarity search over embedding space. Supports both CPUs (`x86_64` and `arm`) and GPU (`Nvidia (cuda)` and `Apple Silicon (mps)`).

**Columnar + Random Access**: To achieve both fast columnar scan and sub-linear point queries, Lance uses a custom storage format that is optimized for both random access and sequential scans.

**Nested fields**: Lance stores each subfield as a separate column to support efficient filters like “find images where detected objects include cats”.

**Versioning**: Similar to Iceberg and Delta Lake, Lance files are immutable and changes are done by creating new files. This enables time travel, allowing the dataset to be queried at or easily restored to a previous state.

**Rich secondary indices**: Support `BTree`, `Bitmap`, `Full text search`, `Label list`,
`NGrams`, and more. These indices can speed up vector search, analytics, and filtering.
They can also speed up dataset maintenance tasks such as upserts and deletes.

## Benchmarks

### Vector search

We used the SIFT dataset to benchmark our results with 1M vectors of 128D

1. For 100 randomly sampled query vectors, we get <1ms average response time (on a 2023 m2 MacBook Air)

![avg_latency.png](docs/avg_latency.png)

2. ANNs are always a trade-off between recall and performance

![avg_latency.png](docs/recall_vs_latency.png)

### Vs. parquet

We create a Lance dataset using the Oxford Pet dataset to do some preliminary performance testing of Lance as compared to Parquet and raw image/XMLs. For analytics queries, Lance is 50-100x better than reading the raw metadata. For batched random access, Lance is 100x better than both parquet and raw files.

![](docs/lance_perf.png)

## Why are you building yet another table format?!

The machine learning development cycle involves the steps:

```mermaid
graph LR
    A[Collection] --> B[Exploration];
    B --> C[Analytics];
    C --> D[Feature Engineering];
    D --> E[Training];
    E --> F[Evaluation];
    F --> C;
    E --> G[Deployment];
    G --> H[Monitoring];
    H --> A;
```

Currently, people use different data representations at the various stages due to limited performance or tooling.

Academia mainly uses XML / JSON for annotations and zipped images/sensors data for deep learning, which is difficult to integrate into data infrastructure and slow to train over cloud storage.

Industry uses classic Parquet-based data lakes (e.g. Delta Lake, Iceberg) or data warehouses (e.g. AWS Redshift, Google BigQuery) to collect and analyze data. These have limited support for multimodal types, are difficult to experiment with, and are often slow or expensive when it comes to data exploration and search.

Multiple single-purpose data transforms, manually syncing copies between cloud storage to local training instances, and ad-hoc infrastructure has become a common practice.

While each of the existing storage choices excels at the workload it was originally designed for, we need a new choice tailored for multistage ML development cycles to reduce costs, engineering overhead, and data silos.

## Community Highlights

Lance is currently used in production by:

- [LanceDB](https://github.com/lancedb/lancedb), a serverless, low-latency vector database for ML applications
- [LanceDB Enterprise](https://docs.lancedb.com/enterprise/introduction), hyperscale LanceDB with enterprise SLA.
- Leading multimodal Gen AI companies for training over petabyte-scale multimodal data.
- Self-driving car company for large-scale storage, retrieval and processing of multi-modal data.
- E-commerce company for billion-scale+ vector personalized search.
- and more.

## Presentations, Blogs and Talks

- [Designing a Table Format for ML Workloads](https://blog.lancedb.com/designing-a-table-format-for-ml-workloads/), Feb 2025.
- [Transforming Multimodal Data Management with LanceDB, Ray Summit](https://www.youtube.com/watch?v=xmTFEzAh8ho), Oct 2024.
- [Lance v2: A columnar container format for modern data](https://blog.lancedb.com/lance-v2/), Apr 2024.
- [Lance Deep Dive](https://drive.google.com/file/d/1Orh9rK0Mpj9zN_gnQF1eJJFpAc6lStGm/view?usp=drive_link). July 2023.
- [Lance: A New Columnar Data Format](https://docs.google.com/presentation/d/1a4nAiQAkPDBtOfXFpPg7lbeDAxcNDVKgoUkw3cUs2rE/edit#slide=id.p), [Scipy 2022, Austin, TX](https://www.scipy2022.scipy.org/posters). July, 2022.
