# Rust Implementation of Lance Data Format

<div align="center">
<p align="center">

<img width="257" alt="Lance Logo" src="https://user-images.githubusercontent.com/917119/199353423-d3e202f7-0269-411d-8ff2-e747e419e492.png">

**A new columnar data format for data science and machine learning**
</p></div>

## Installation

Install using cargo:

```shell
cargo install lance
```

> [!TIP]
> A transitive dependency of `lance` is `lzma-sys`, which uses dynamic linking
> by default. If you want to statically link `lzma-sys`, you should activate it's
> `static` feature by adding the following to your dependencies:
>
> ```toml
> lzma-sys = { version = "*", features = ["static"] }
> ```

## Examples

### Create dataset

Suppose `batches` is an Arrow `Vec<RecordBatch>` and schema is Arrow `SchemaRef`:

```rust
use lance::{dataset::WriteParams, Dataset};

let write_params = WriteParams::default();
let mut reader = RecordBatchIterator::new(
    batches.into_iter().map(Ok),
    schema
);
Dataset::write(reader, &uri, Some(write_params)).await.unwrap();
```

### Read

```rust
let dataset = Dataset::open(path).await.unwrap();
let mut scanner = dataset.scan();
let batches: Vec<RecordBatch> = scanner
    .try_into_stream()
    .await
    .unwrap()
    .map(|b| b.unwrap())
    .collect::<Vec<RecordBatch>>()
    .await;
```

### Take

```rust
let values: Result<RecordBatch> = dataset.take(&[200, 199, 39, 40, 100], &projection).await;
```

### Vector index

Assume "embeddings" is a FixedSizeListArray<f32>
```rust
use ::lance::index::vector::VectorIndexParams;

let params = VectorIndexParams::default();
params.num_partitions = 256;
params.num_sub_vectors = 16;

// this will Err if list_size(embeddings) / num_sub_vectors does not meet simd alignment
dataset.create_index(&["embeddings"], IndexType::Vector, None, &params, true).await;
```

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
