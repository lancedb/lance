# Java bindings and SDK for Lance Data Format

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

Introduce the Lance SDK for Java Maven dependency(It is recommended to choose the latest version.):

```shell
<dependency>
    <groupId>com.lancedb</groupId>
    <artifactId>lance-core</artifactId>
    <version>0.18.0</version>
</dependency>
```

### Basic I/Os

* create and write a Lance dataset
* read dataset
* drop dataset

### Random Access

### Indexing and Searching

### Schema evolution

* add columns

* alter columns

* drop columns

## Integrations

### Spark connector

## Contributing

### Environment(IDE) setup