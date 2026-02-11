---
title: Full-Text Search
description: Full-text search (FTS) with inverted BM25 indexes and N-gram search in Lance
---

# Full-Text Search in Lance

Lance provides powerful full-text search (FTS) capabilities using an inverted index. This tutorial will guide you through building and using FTS indexes to dramatically speed up text search operations while maintaining high accuracy.

By the end of this tutorial, you'll be able to build and use an FTS index, understand performance differences between indexed and non-indexed searches, and learn how to tune search parameters for optimal performance.

## Install the Python SDK

First, install the required dependencies:

```bash
pip install pylance pyarrow
```

## Set Up Your Environment

Import the necessary libraries for working with Lance datasets:

```python
import lance
import pyarrow as pa
```

## Prepare Your Text Data

For this quickstart page, we'll create a simple dataset with text documents:

```python
table = pa.table(
    {
        "id": [1, 2, 3],
        "text": [
            "I left my umbrella on the morning train to Boston",
            "This ramen recipe simmers the broth for three hours with dried mushrooms.",
            "This train is scheduled to leave for Edinburgh at 9:30 AM",
        ],
    }
)

# Write to a new Lance dataset
lance.write_dataset(table, "/tmp/fts.lance", mode="overwrite")
```

This creates a Lance dataset with three text documents containing somewhat overlapping keywords that we'll use to demonstrate various search scenarios.

## Explore Your Dataset Schema

Let's examine the structure of our dataset:

```python
ds = lance.dataset("/tmp/fts.lance")
print(ds.schema)
```

This prints the PyArrow schema of the dataset:
```
id: int64
text: large_string
```

## Build the Full-Text Search Index

Full-text search is created with an inverted scalar index on your text column. You can choose the `INVERTED` index type when calling the `create_scalar_index` method on your Lance dataset. Lance uses the BM25 ranking algorithm for relevance scoring. Results are automatically ranked by relevance, with higher scores indicating better matches.

```python
ds.create_scalar_index(
    column="text",
    index_type="INVERTED"
)
```

The index creation process builds an efficient lookup structure that maps words to the documents containing them. This enables high-performance keyword-based search, even on large datasets.

!!! warning "Index Creation Time"
    Index creation time depends on the size of your text data. For large datasets, this process may take several minutes, but the performance benefits at query time are substantial.

## Advanced Index Configuration

You can customize the index creation with various parameters to optimize for your specific use case:

```python
ds.create_scalar_index(
    column="text",
    index_type="INVERTED",
    name="text_idx",              # Optional index name (if omitted, default is "text_idx")
    with_position=False,          # Set True to enable phrase queries (stores token positions)
    base_tokenizer="simple",      # Tokenizer: "simple" (whitespace+punct), "whitespace", or "raw" (no tokenization)
    language="English",           # Language used for stemming + stop words (only used if `stem` or `remove_stop_words` is True)
    max_token_length=40,          # Drop tokens longer than this length
    lower_case=True,              # Lowercase text before tokenization
    stem=True,                    # Stem tokens (language-dependent)
    remove_stop_words=True,       # Remove stop words (language-dependent)
    custom_stop_words=None,       # Optional additional stop words (only used if remove_stop_words=True)
    ascii_folding=True,           # Fold accents to ASCII when possible (e.g., "é" -> "e")
)
```

### Tokenizer Options

- **simple**: Splits tokens on whitespace and punctuation
- **whitespace**: Splits tokens only on whitespace
- **raw**: No tokenization (useful for exact matching)

Lance also supports multilingual tokenization:

- **jieba/default**: Chinese text tokenization using Jieba
- **lindera/ipadic**: Japanese text tokenization using Lindera with IPAdic dictionary
- **lindera/ko-dic**: Korean text tokenization using Lindera with Ko-dic dictionary
- **lindera/unidic**: Japanese text tokenization using Lindera with UniDic dictionary

### Language Processing Features

- **stemming**: Reduces words to their root form (e.g., "running" → "run")
- **stop words**: Removes common words like "the", "and", "is"
- **ascii folding**: Converts accented characters to ASCII (e.g., "é" → "e")

## Search With FTS Queries

Now you can run FTS queries using your inverted index:

```python
import lance

# Open dataset
ds = lance.dataset("/tmp/fts.lance")

# Specify keyword phrases when calling the `to_table` method
query_result = ds.to_table(
    full_text_query="umbrella train"
)
print(query_result)
```

This query returns documents that contain either "umbrella" or "train" (or both). The search is case-insensitive and uses the inverted index for fast retrieval.

```
id: [[1, 3]]
text: [["I left my umbrella on the evening train to Boston", "This train is scheduled to leave for Edinburgh at 9:30 in the morning"]]
_score: [[..., ...]]
```

## Advanced Search Features

### Boolean Search Operators

You can query via boolean search operators by constructing a structured query object.

#### All terms: `AND`
```python
from lance.query import FullTextOperator, MatchQuery

# Require the terms 'umbrella AND train AND boston' to be present
and_query = MatchQuery("umbrella train boston", "text", operator=FullTextOperator.AND)
query_result = ds.to_table(full_text_query=and_query)

# Returns
# text: [["I left my umbrella on the evening train to Boston"]]
```

#### Any terms: `OR`
```python
from lance.query import FullTextOperator, MatchQuery

# Require the terms 'morning OR evening' to be present
or_query = MatchQuery("morning evening", "text", operator=FullTextOperator.OR)
query_result = ds.to_table(full_text_query=or_query)

# Returns the Boston document that mentions 'evening', and the Edinburgh document that mentions 'morning'
# text: [["This train is scheduled to leave for Edinburgh at 9:30 in the morning", "I left my umbrella on the evening train to Boston"]]
```

#### Mix `AND`/`OR` queries via operators

You can mix `AND`/`OR` queries using operators and syntactic sugar in Python:

```python
from lance.query import FullTextOperator, MatchQuery

# Combine AND and OR semantics
# Require 'train' AND ('morning' OR 'evening')
q1 = MatchQuery("morning evening", "text", operator=FullTextOperator.OR)
q2 = MatchQuery("train", "text")
query_result = ds.to_table(full_text_query=(q1 & q2))

# Returns both the Boston and Edinburgh documents that mention 'train'
# text: [["I left my umbrella on the evening train to Boston", "This train is scheduled to leave for Edinburgh at 9:30 in the morning"]]
```

To combine `OR` queries via operators, use the pattern `q1 | q2`.

#### Exclude terms: `NOT`

Queries that exclude specific keywords are explicitly written using `BooleanQuery`/`Occur`
as shown below.

```python
from lance.query import MatchQuery, BooleanQuery, Occur

# Require that 'umbrella' be present, but 'train' NOT be present
q = BooleanQuery(
    [
        (Occur.MUST, MatchQuery("umbrella", "text")),
        (Occur.MUST_NOT, MatchQuery("train", "text")),
    ]
)
query_result = ds.to_table(full_text_query=q)

# Returns empty result, as no document matches this condition
# text: []
```

### Phrase Search

For exact phrase matching, ensure you enable `with_position=True` during index creation, which is disabled by default.

```python
# Rebuild the index with positions enabled (required for phrase queries)
ds.create_scalar_index(
    "text",
    "INVERTED",
    with_position=True,
    remove_stop_words=False,
)
# Search for the exact phrase "train to boston"
table = ds.to_table(full_text_query="'train to boston'")

# If stopwords are removed, this phrase query would return an empty result
# text: [["I left my umbrella on the evening train to Boston"]]
```

!!! warning "Stopwords are removed by default"
    Common words like "_to_", "_the_", etc. are categorized as stopwords, and are removed by default when creating the index. If you want to search exact phrases that include stopwords, you need to set `remove_stop_words=False` when creating the index.

### Substring matches with N-gram indexing

`NGRAM` is a type of scalar index for **substring / pattern-style** searches over text. It's a good alternative to "wildcard" search queries like `term*` / `*term` (which are not parsed by `full_text_query` in Lance).

The N-gram index creates a bitmap for each N-gram in the string. By default, Lance uses trigrams. This index can be used to speed up queries using the `contains` function in filters.

```python
import lance

ds = lance.dataset("/tmp/fts.lance")

# Build an NGRAM index for substring search (speeds up `contains(...)` filters)
# Give the index a distinct name so it won't replace your FTS index
ds.create_scalar_index(column="text", index_type="NGRAM", name="text_ngram")

# Substring search
q1 = ds.to_table(filter="contains(text, 'ramen')")

# Returns the document about ramen
# text: [["This ramen recipe simmers the broth for three hours with dried mushrooms."]]
```

You can explain the query plan to confirm the N-gram index's usage as shown below:

```python
# Inspect the query plan to confirm index usage
print(ds.scanner(filter="contains(text, 'train')").explain_plan())
```

## Combining Full-Text Search with Metadata

It can be useful to combine FTS with metadata filtering in a single query to find more relevant results.
This is done by passing in the column name and its value to the `filter` parameter when querying.

```python
import lance
import pyarrow as pa

table = pa.table(
    {
        "id": [1, 2, 3],
        "text": [
            "I left my umbrella on the morning train to Boston",
            "This ramen recipe simmers the broth for three hours with dried mushrooms.",
            "This train is scheduled to leave for Edinburgh at 9:30 AM",
        ],
        "category": ["travel", "food", "travel"],
    }
)

# Temp write dataset
lance.write_dataset(table, "./fts_test_with_metadata.lance", mode="overwrite")

ds = lance.dataset("./fts_test_with_metadata.lance")

# Create FTS index
ds.create_scalar_index(
    column="text",
    index_type="INVERTED",
)

# Run FTS query with metadata filter
query_result = ds.to_table(
    full_text_query="three",
    filter='category = "food"',
)

# Returns
# id: [[2]]
# text: [["This ramen recipe simmers the broth for three hours with dried mushrooms."]]
# category: [["food"]]
```

## Index Maintenance

When you append new rows after creating an `INVERTED` index, Lance still returns those rows in `full_text_query` results. It searches indexed fragments using the FTS index and scans unindexed fragments with flat search, then merges the results.

To keep FTS latency low as new data arrives, you can periodically add unindexed fragments into the existing FTS index to keep the index up-to-date by calling `ds.optimize.optimize_indices()` as follows.

```python
# Append new data
new_rows = pa.table(
    {
        "id": [4],
        "text": ["The next train leaves at noon"],
    }
)
ds.insert(new_rows)

# Incrementally update existing indices (including "text_idx")
ds.optimize.optimize_indices(index_names=["text_idx"])

# Optional: monitor index coverage
stats = ds.stats.index_stats("text_idx")
print(stats["num_unindexed_rows"], stats["num_indexed_rows"])
```

!!! info
    If you used a custom index name, replace `"text_idx"` with your index name.
    If you did not set `name=...` when creating the FTS index on column `"text"`, the default index name is `"text_idx"`.

If you changed tokenizer settings (such as `with_position`, `base_tokenizer`, stop words, or stemming), rebuild the index with `create_scalar_index(..., replace=True)` so the full dataset is indexed with the new configuration.

## Index Configuration Best Practices

- The `with_position` parameter should be enabled when you need phrase queries, as it stores word positions within documents. However, for simple term searches, disabling this feature can save considerable storage space without impacting performance.

- Keep `lower_case=True` enabled for most applications to ensure case-insensitive search behavior. This provides a better user experience and matches common search expectations, though you can disable it if case sensitivity is important for your use case.

- Enable stemming (`stem=True`) when you want better recall by matching word variations (e.g., "running" matches "run"). However, disable stemming if you need exact term matching or if your domain requires precise terminology.

- Consider enabling `remove_stop_words=True` for cleaner search results, especially in content-heavy applications. This removes common words like "the", "and", "is" from the index, reducing noise and improving relevance. However, keep stop words if they carry important meaning in your domain.


## Query Optimization

Using specific, targeted search terms often yields better performance than broad, generic queries. More specific terms reduce the number of potential matches and allow the index to work more efficiently. Consider analyzing your most common search patterns and optimizing your index configuration accordingly.

Combining full-text search with metadata filters can significantly reduce the search space and improve performance. Use structured data filters to narrow down results before applying text search, or vice versa. This approach is particularly effective for large datasets where you can eliminate many irrelevant documents early in the query process.

## Next Steps

Check out the **[User Guide](../guide/read_and_write/)** and explore the Lance API in more detail.
