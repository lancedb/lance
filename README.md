# Lance: A Columnar Data Format for Computer Vision

![CI](https://github.com/eto-ai/lance/actions/workflows/cpp.yml/badge.svg)
[![Docs](https://img.shields.io/badge/docs-passing-brightgreen)](https://eto-ai.github.io/lance/)

[![PyPi](https://img.shields.io/pypi/v/pylance)](https://pypi.org/project/pylance/)
![Python versions](https://img.shields.io/pypi/pyversions/pylance)

[![](https://dcbadge.vercel.app/api/server/zMM32dvNtd?style=flat)](https://discord.gg/zMM32dvNtd)

Lance is a cloud-native columnar data format designed for managing large-scale computer vision datasets in production
environments. Lance delivers blazing fast performance for image and video data use cases from exploratory data analysis to training and evaluation.

Lance core is written in C++ and comes with python bindings to start. With first class Apache Arrow integration, Lance is queryable by tools like DuckDB out of the box and can be converted from Parquet with a single line of code.

## What problems does Lance solve?

Today, the data tooling stack for computer vision is insufficient to serve the needs of the ML Engineering and Data Science community.

### Working with computer vision data for ML is different from working with tabular data:
- Performance: images, video, audio and point clouds are large blobs that are difficult and slow to query by existing engines
- Nested semantics: data annotations are almost always deeply nested, which means metadata and nested fields are written to separate files, and stored in different locations (i.e. S3 vs CSV/XML)
- Siloed datasets across tools: exploratory data analysis, labeling, training and evaluation uses different tools requiring different formats, with no source of truth, making consistent versioning and reproducibility challenging
- Vector space: increasingly ML is making use of vector-based semantics and embeddings, but it is challenging to query across both vectors and other column types in a performant and usable way

### Lance to the rescue
To solve these pain-points, we are building Lance, an open-source columnar data format optimized for computer vision with the following goals:
- Blazing fast performance for analytical scans and random access to individual records (for visualization and annotation)
- Rich ML data types and integrations to eliminate manual data conversions
- Support for vector and search indices, versioning, and schema evolution

## Quick Start

We've provided Linux and MacOS wheels for Lance in PyPI. You can install Lance python bindings via:

```
pip install pylance
```

Thanks to its Apache Arrow-first APIs, `lance` can be used as a native `Arrow` extension.
For example, it enables users to directly use `DuckDB` to analyze lance dataset
via [DuckDB's Arrow integration](https://duckdb.org/docs/guides/python/sql_on_arrow).

```python
# pip install pylance duckdb
import lance
import duckdb

# Understand Label distribution of Oxford Pet Dataset
ds = lance.dataset("s3://eto-public/datasets/oxford_pet/oxford_pet.lance")
duckdb.query('select class, count(1) from ds group by 1').to_arrow_table()
```

## Important directories

| Directory                                  | Description                            |
|--------------------------------------------|----------------------------------------|
| [cpp](./cpp)                               | Source code for Eto services and sites |
| [python](./python)                         | Infrastructure as code                 |
| [notebooks](./python/notebooks)            | Build scripts for supporting services  |
| [duckdb extension](./integrations/duckdb)  | Lance Duckdb extension                 |


## What makes Lance different

Here we will highlight a few aspects of Lance’s design. For more details, see the full [Lance design document](https://docs.google.com/document/d/1kknVcqRK65YqGkKASuQ40apr2A2DyK0Qtx5nhCPCdqQ/edit).

**Encodings**: to achieve both fast columnar scan and sub-linear point queries, Lance uses custom encodings and layouts.

**Nested fields**: Lance stores each subfield as a separate column to support efficient filters like “find images where detected objects include cats”.

**Versioning / updates** (ROADMAP): a Manifest can be used to record snapshots. Updates are supported via write-ahead logs.

**Secondary Indices** (ROADMAP):
  - Vector index for similarity search over embedding space
  - Inverted index for fuzzy search over many label / annotation fields

## Benchmarks

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

* [Lance: A New Columnar Data Format](https://docs.google.com/presentation/d/1a4nAiQAkPDBtOfXFpPg7lbeDAxcNDVKgoUkw3cUs2rE/edit#slide=id.p)
  .
  [Scipy 2022, Austin, TX](https://www.scipy2022.scipy.org/posters). July, 2022.
