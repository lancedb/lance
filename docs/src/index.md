# Introduction

![Lance Logo](./logo/wide.png)

## Modern Columnar Data Format for ML and LLMs

Lance is an open table and file format that is easy and fast to version, query and train on.
It's designed to be used with images, videos, 3D point clouds, audio and of course tabular data.
It supports any POSIX file systems, and cloud storage like AWS S3 and Google Cloud Storage.
The key features of Lance include:

* **High-Performance Random Access:** 100x faster than Parquet.

* **Zero-Copy Data Evolution:** add, drop or update column data in existing rows without the need to rewrite the entire dataset.

* **Multimodal Data:** natively store large text, images, videos, documents and embeddings all together, and efficiently access them using the Blob API.

* **Vector Search:** find nearest neighbors in under 1 millisecond with IVF-PQ, IVF-SQ, HNSW and more.

* **Full-Text Search:** fast search over text with inverted index, Ngram index plus different tokenizers.

* **Hybrid Search:** perform hybrid vector, full-text and OLAP queries all using the same format.

* **Row Level Transaction:** fully ACID transaction with row level conflict resolution.

* **Ecosystem Integrations:** PyTorch, Tensorflow, Ray, HuggingFace, Apache Arrow, Apache DataFusion, Apache Spark, Trino, DuckDB, PostgreSQL, and more on the way.
