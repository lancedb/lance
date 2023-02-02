<div align="center">
<p align="center">
 
<img width="257" alt="Lance Logo" src="https://user-images.githubusercontent.com/917119/199353423-d3e202f7-0269-411d-8ff2-e747e419e492.png">

**Blazing fast exploration and analysis of ML data using python and SQL, backed by an Apache-Arrow compatible data format**

<a href="https://eto-ai.github.io/lance/">Documentation</a> •
<a href="https://blog.eto.ai/">Blog</a> •
<a href="https://discord.gg/zMM32dvNtd">Discord</a> •
<a href="https://twitter.com/etodotai">Twitter</a>

![CI](https://github.com/eto-ai/lance/actions/workflows/rust.yml/badge.svg)
[![Docs](https://img.shields.io/badge/docs-passing-brightgreen)](https://eto-ai.github.io/lance/)
![Python versions](https://img.shields.io/pypi/pyversions/pylance)

</p>
</div>

<hr />

Lance makes machine learning workflows with ML data easy (images, videos, point clouds, audio, and of course tabular data), by allowing Developers, Analysts and Operations to:

* Use SQL to greatly simplify common operations on ML data, such as similarity search for data discovery, model inference and computing evaluation metrics.

* Search for nearest neighbors in under 1 millisecond.

* Version, compare and diff ML datasets easily.

* (Coming soon) visualize, slice and drill-into datasets to inspect embeddings, labels/annotations and metrics.

Lance is powered by Lance Format, an Apache-Arrow compatible columnar data format which is an alternative to Parquet, Iceberg and Delta. Lance has 50-100x faster query performance for ML data.


## Quick Start

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
df
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

```shell
wget https://eto-public.s3.us-west-2.amazonaws.com/datasets/sift/sift_ivf256_pq16.tar.gz
tar -xzf sift_ivf256_pq16.tar.gz
```

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

## Directory structure

| Directory          | Description              |
|--------------------|--------------------------|
| [rust](./rust)     | Core Rust implementation |
| [python](./python) | Python bindings (pyo3)   |
| [docs](./docs)     | Documentation source     |

## What makes Lance different

Here we will highlight a few aspects of Lance’s design. For more details, see the full [Lance design document](https://eto-ai.github.io/lance/format.html).

**Vector index**: Vector index for similarity search over embedding space

**Encodings**: to achieve both fast columnar scan and sub-linear point queries, Lance uses custom encodings and layouts.

**Nested fields**: Lance stores each subfield as a separate column to support efficient filters like “find images where detected objects include cats”.

**Versioning**: a Manifest can be used to record snapshots. Currently we support creating new versions automatically via appends, overwrites, and index creation 

**Fast updates** (ROADMAP): Updates will be supported via write-ahead logs.

**Rich secondary indices** (ROADMAP): 
  - Inverted index for fuzzy search over many label / annotation fields

## Benchmarks

### Vector search

We used the sift dataset to benchmark our results with 1M vectors of 128D

1. For 100 randomly sampled query vectors, we get <1ms average response time (on a 2023 m2 macbook air)

![avg_latency.png](docs/avg_latency.png)

2. ANN is always a trade-off between recall and performance

![avg_latency.png](docs/recall_vs_latency.png)

### Vs parquet

We create a Lance dataset using the Oxford Pet dataset to do some preliminary performance testing of Lance as compared to Parquet and raw image/xmls. For analytics queries, Lance is 50-100x better than reading the raw metadata. For batched random access, Lance is 100x better than both parquet and raw files.

![](docs/lance_perf.png)

## Why are you building yet another data format?!

Machine Learning development cycle involves the steps:

```mermaid
graph LR
    A[Collection] --> B[Exploration];
    B --> C[Analytics];
    C --> D[Feature Engineer];
    D --> E[Training];
    E --> F[Evaluation];
    F --> C;
    E --> G[Deployment];
    G --> H[Monitoring];
    H --> A;
```

People use different data representations to varying stages for the performance or limited by the tooling available.
The academia mainly uses XML / JSON for annotations and zipped images/sensors data for deep learning, which
is difficult to integrated into data infrastructure and slow to train over cloud storage.
While the industry uses data lake (Parquet-based techniques, i.e., Delta Lake, Iceberg) or data warehouse (AWS Redshift
or Google BigQuery) to collect and analyze data, they have to convert the data into training-friendly formats, such
as [Rikai](https://github.com/eto-ai/rikai)/[Petastorm](https://github.com/uber/petastorm)
or [Tfrecord](https://www.tensorflow.org/tutorials/load_data/tfrecord).
Multiple single-purpose data transforms, as well as syncing copies between cloud storage to local training
instances have become a common practice among ML practices.

While each of the existing data formats excel at its original designed workload, we need a new data format
to tailored for multistage ML development cycle to reduce the fraction in tools and data silos.

A comparison of different data formats in each stage of ML development cycle.

|                     | Lance | Parquet & ORC | JSON & XML | Tfrecord | Database | Warehouse |
|---------------------|-------|---------------|------------|----------|----------|-----------|
| Analytics           | Fast  | Fast          | Slow       | Slow     | Decent   | Fast      |
| Feature Engineering | Fast  | Fast          | Decent     | Slow     | Decent   | Good      |
| Training            | Fast  | Decent        | Slow       | Fast     | N/A      | N/A       |
| Exploration         | Fast  | Slow          | Fast       | Slow     | Fast     | Decent    |
| Infra Support       | Rich  | Rich          | Decent     | Limited  | Rich     | Rich      |

## Presentations and Talks

* [Lance: A New Columnar Data Format](https://docs.google.com/presentation/d/1a4nAiQAkPDBtOfXFpPg7lbeDAxcNDVKgoUkw3cUs2rE/edit#slide=id.p), [Scipy 2022, Austin, TX](https://www.scipy2022.scipy.org/posters). July, 2022.
