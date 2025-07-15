---
title: Full Text Search
description: High-performance full-text search with inverted indexes and FTS capabilities in Lance datasets
---

# Full Text Search with Lance Datasets

Lance provides powerful full-text search capabilities using inverted indexes and FTS (Full Text Search) technology. This tutorial will guide you through building and using full-text search indexes to dramatically speed up text search operations while maintaining high accuracy.

By the end of this tutorial, you'll be able to build and use full-text search indexes, understand performance differences between indexed and non-indexed searches, and learn how to tune search parameters for optimal performance.

## Install the Python SDK

First, install the required dependencies:

```python
pip install pylance pandas
```

## Set Up Your Environment

Import the necessary libraries for working with Lance datasets:

```python
import pandas as pd
import lance
```

## Prepare Your Text Data

For this tutorial, we'll create a simple dataset with text documents to demonstrate full-text search capabilities:

```python
df = pd.DataFrame({
    "id": [1, 2, 3],
    "text": [
        "The quick brown fox jumps over the lazy dog",
        "The fast red fox leaps over the sleepy cat",
        "A completely unrelated sentence"
    ]
})

# Write to a new Lance dataset
lance.write_dataset(df, "/tmp/docs.lance", mode="create")
```

This creates a Lance dataset with three text documents. Each document contains different content that we'll use to demonstrate various search scenarios.

## Explore Your Dataset Schema

Let's examine the structure of our dataset:

```python
ds = lance.dataset("/tmp/docs.lance")
print(ds.schema)
```

This will show you the schema of your dataset, including the data types and column information. Understanding your schema is crucial for building effective indexes.

## Build the Full-Text Search Index

To enable fast full-text search operations, you need to create a scalar index on your text column. Lance supports two types of full-text search indexes:

```python
ds.create_scalar_index(
    column="text",
    index_type="INVERTED",  # or "FTS"
)
```

!!! note "Index Types"
    - **INVERTED**: Traditional inverted index that maps terms to document IDs
    - **FTS**: Full Text Search index with additional linguistic processing capabilities

The index creation process analyzes your text data and builds an efficient lookup structure that maps words to the documents containing them. This enables sub-second search performance even on large datasets.

!!! warning "Index Creation Time"
    Index creation time depends on the size of your text data. For large datasets, this process may take several minutes, but the performance benefits are substantial.

## Advanced Index Configuration

You can customize the index creation with various parameters to optimize for your specific use case:

```python
ds.create_scalar_index(
    column="text",
    index_type="INVERTED",
    with_position=True,           # Store word positions for phrase queries
    base_tokenizer="simple",      # "simple", "whitespace", or "raw"
    language="English",           # For stemming and stop words
    max_token_length=40,         # Maximum token length
    lower_case=True,             # Convert to lowercase
    stem=False,                  # Enable stemming
    remove_stop_words=False,     # Remove stop words
    ascii_folding=False          # Convert non-ASCII to ASCII
)
```

### Tokenizer Options

- **simple**: Splits tokens on whitespace and punctuation
- **whitespace**: Splits tokens only on whitespace
- **raw**: No tokenization (useful for exact matching)

### Language Processing Features

- **stemming**: Reduces words to their root form (e.g., "running" → "run")
- **stop words**: Removes common words like "the", "and", "is"
- **ascii folding**: Converts accented characters to ASCII (e.g., "é" → "e")

## Perform Full-Text Search Queries

Now you can perform fast full-text search operations using your newly created index:

```python
import lance

ds = lance.dataset("/tmp/docs.lance")

table = ds.to_table(
    full_text_query="fox sleepy"
)

print(table)
```

This query will return documents that contain either "fox" or "sleepy" (or both). The search is case-insensitive and uses the inverted index for fast retrieval.

## Advanced Search Features

### Boolean Search Operators

You can use boolean operators in your search queries:

```python
# Search for documents containing both "fox" AND "dog"
table = ds.to_table(full_text_query="fox AND dog")

# Search for documents containing "fox" OR "cat"
table = ds.to_table(full_text_query="fox OR cat")

# Search for documents containing "fox" but NOT "lazy"
table = ds.to_table(full_text_query="fox NOT lazy")
```

### Phrase Search

For exact phrase matching (requires `with_position=True` in index creation):

```python
# Search for the exact phrase "quick brown fox"
table = ds.to_table(full_text_query='"quick brown fox"')
```

### Wildcard Search

Use wildcards for pattern matching:

```python
# Search for words starting with "fox"
table = ds.to_table(full_text_query="fox*")

# Search for words ending with "ing"
table = ds.to_table(full_text_query="*ing")
```

## Tuning Search Parameters

### Fast Search Mode

For maximum performance, you can enable fast search mode which only searches indexed data:

```python
table = ds.to_table(
    full_text_query="fox",
    fast_search=True  # Only search indexed data for faster results
)
```

### BM25 Ranking

Lance uses BM25 ranking algorithm for relevance scoring. Results are automatically ranked by relevance, with higher scores indicating better matches.

## Combining Full-Text Search with Metadata

One of the powerful features of Lance is the ability to combine full-text search with metadata filtering in a single query:

```python
# Add metadata columns to your dataset
df_with_metadata = df.copy()
df_with_metadata['category'] = ['animals', 'animals', 'general']
df_with_metadata['date'] = ['2024-01-01', '2024-01-02', '2024-01-03']

# Write updated dataset
lance.write_dataset(df_with_metadata, "/tmp/docs_with_metadata.lance", mode="create")

# Create index on text column
ds_with_metadata = lance.dataset("/tmp/docs_with_metadata.lance")
ds_with_metadata.create_scalar_index(column="text", index_type="INVERTED")

# Combine full-text search with metadata filtering
table = ds_with_metadata.to_table(
    full_text_query="fox",
    filter="category = 'animals'"
)
```

This allows you to perform complex queries that combine text search with structured data filtering, all in a single efficient operation.

## Performance Optimization Tips

### Index Maintenance

As your data evolves over time, it's important to periodically rebuild your indexes to maintain optimal search performance. When you add, modify, or delete documents, the existing index may become less efficient. Consider scheduling regular index rebuilds, especially for datasets with frequent updates.

Large indexes consume more storage space but provide significantly faster search performance. Monitor your index sizes and balance storage costs against performance requirements. For production systems, you may want to set up monitoring to track index growth and performance metrics.

Choose between INVERTED and FTS index types based on your specific use case. INVERTED indexes are more memory-efficient and suitable for most applications, while FTS indexes provide additional linguistic processing capabilities that may be beneficial for certain text analysis tasks.

### Query Optimization

Using specific, targeted search terms often yields better performance than broad, generic queries. More specific terms reduce the number of potential matches and allow the index to work more efficiently. Consider analyzing your most common search patterns and optimizing your index configuration accordingly.

For maximum performance when you only need results from indexed data, enable the `fast_search=True` parameter. This bypasses any non-indexed data and can dramatically improve query speed, though it may miss some results if your index doesn't cover all your data.

Combining full-text search with metadata filters can significantly reduce the search space and improve performance. Use structured data filters to narrow down results before applying text search, or vice versa. This approach is particularly effective for large datasets where you can eliminate many irrelevant documents early in the query process.

### Index Configuration Best Practices

The `with_position` parameter should be enabled when you need phrase queries, as it stores word positions within documents. However, for simple term searches, disabling this feature can save considerable storage space without impacting performance.

Keep `lower_case=True` enabled for most applications to ensure case-insensitive search behavior. This provides a better user experience and matches common search expectations, though you can disable it if case sensitivity is important for your use case.

Enable stemming (`stem=True`) when you want better recall by matching word variations (e.g., "running" matches "run"). However, disable stemming if you need exact term matching or if your domain requires precise terminology.

Consider enabling `remove_stop_words=True` for cleaner search results, especially in content-heavy applications. This removes common words like "the", "and", "is" from the index, reducing noise and improving relevance. However, keep stop words if they carry important meaning in your domain.

## Next Steps

You should check out **[Vector Search with Lance](./vector-search.md)** to learn how to combine full-text search with vector search for powerful hybrid search capabilities. This combination enables semantic search alongside traditional keyword-based search.

For more advanced features, explore **[Versioning Your Datasets with Lance](./versioning.md)** to learn how to track changes in your search indexes over time.
