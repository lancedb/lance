---
hide: toc
---

# Welcome to Lance Open Source Documentation! 

<img src="./logo/wide.png" alt="Lance Logo" width="400">

*Lance is a modern columnar data format optimized for machine learning and AI applications. It efficiently handles diverse multimodal data types while providing high-performance querying and versioning capabilities.*

[Quickstart Locally With Python](quickstart/index.md){ .md-button .md-button--primary } [Read the Format Specification](format/index.md){ .md-button .md-button } [Train Your LLM on a Lance Dataset](examples/python/llm_training.md){ .md-button .md-button--primary } 

## 🎯 How Does Lance Work?

Lance is designed to be used with images, videos, 3D point clouds, audio and tabular data. It supports any POSIX file systems, and cloud storage like AWS S3 and Google Cloud Storage.

This file format is particularly suited for [**vector search**](quickstart/vector-search.md), full-text search and [**LLM training**](examples/python/llm_training.md) on multimodal data. To learn more about how Lance works, [**read the format specification**](format/index.md). 

!!! info "Looking for LanceDB?"
    **This is the Lance table format project** - the open source core that powers LanceDB.
    If you want the complete vector database and multimodal lakehouse built on Lance, visit [lancedb.com](https://lancedb.com)

## ⚡ Key Features of Lance Format

| Feature | Description |
|---------|-------------|
| 🚀 **[High-Performance Random Access](guide/performance.md)** | 100x faster than Parquet for random access patterns |
| 🔄 **[Zero-Copy Data Evolution](guide/data_evolution.md)** | Add, drop or update column data without rewriting the entire dataset |
| 🎨 **[Multimodal Data](guide/blob.md)** | Natively store large text, images, videos, documents and embeddings |
| 🔍 **[Vector Search](quickstart/vector-search.md)** | Find nearest neighbors in under 1 millisecond with IVF-PQ, IVF-SQ, HNSW |
| 📝 **[Full-Text Search](guide/tokenizer.md)** | Fast search over text with inverted index, Ngram index plus tokenizers |
| 💾 **[Row Level Transaction](format/index.md#conflict-resolution)** | Fully ACID transaction with row level conflict resolution |




