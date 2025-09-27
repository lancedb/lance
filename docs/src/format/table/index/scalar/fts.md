# Full Text Search Index

The full text search (FTS) index (a.k.a. inverted index) provides efficient text search by mapping terms to the documents containing them.
It's designed for high-performance text search with support for various scoring algorithms and phrase queries.

## Index Details

```protobuf
%%% proto.message.InvertedIndexDetails %%%
```

## Storage Layout

The FTS index consists of multiple files storing the token dictionary, document information, and posting lists:

1. `tokens.lance` - Token dictionary mapping tokens to token IDs
2. `docs.lance` - Document metadata including token counts
3. `invert.lance` - Compressed posting lists for each token
4. `metadata.lance` - Index metadata and configuration

### Token Dictionary File Schema

| Column      | Type   | Nullable | Description                     |
|-------------|--------|----------|---------------------------------|
| `_token`    | Utf8   | false    | The token string                |
| `_token_id` | UInt32 | false    | Unique identifier for the token |

### Document File Schema

| Column        | Type   | Nullable | Description                      |
|---------------|--------|----------|----------------------------------|
| `_rowid`      | UInt64 | false    | Document row ID                  |
| `_num_tokens` | UInt32 | false    | Number of tokens in the document |

### FTS List File Schema

| Column                 | Type                    | Nullable | Description                                                      |
|------------------------|-------------------------|----------|------------------------------------------------------------------|
| `_posting`             | List<LargeBinary>       | false    | Compressed posting lists (delta-encoded row IDs and frequencies) |
| `_max_score`           | Float32                 | false    | Maximum score for the token (for query optimization)             |
| `_length`              | UInt32                  | false    | Number of documents containing the token                         |
| `_compressed_position` | List<List<LargeBinary>> | true     | Optional compressed position lists for phrase queries            |

### Metadata File Schema

The metadata file contains JSON-serialized configuration and partition information:

| Key          | Type          | Description                                              |
|--------------|---------------|----------------------------------------------------------|
| `partitions` | Array<UInt64> | List of partition IDs for distributed index organization |
| `params`     | JSON Object   | Serialized InvertedIndexParams with tokenizer config     |

#### InvertedIndexParams Structure

| Field               | Type    | Default   | Description                                                    |
|---------------------|---------|-----------|----------------------------------------------------------------|
| `base_tokenizer`    | String  | "simple"  | Base tokenizer type (see Tokenizers section)                   |
| `language`          | String  | "English" | Language for stemming and stop words                           |
| `with_position`     | Boolean | false     | Store term positions for phrase queries (increases index size) |
| `max_token_length`  | UInt32? | None      | Maximum token length (tokens longer than this are removed)     |
| `lower_case`        | Boolean | true      | Convert tokens to lowercase                                    |
| `stem`              | Boolean | false     | Apply language-specific stemming                               |
| `remove_stop_words` | Boolean | false     | Remove common stop words for the specified language            |
| `ascii_folding`     | Boolean | true      | Convert accented characters to ASCII equivalents               |
| `min_gram`          | UInt32  | 2         | Minimum n-gram length (only for ngram tokenizer)               |
| `max_gram`          | UInt32  | 15        | Maximum n-gram length (only for ngram tokenizer)               |
| `prefix_only`       | Boolean | false     | Generate only prefix n-grams (only for ngram tokenizer)        |

## Tokenizers

The full text search index supports multiple tokenizer types for different text processing needs:

### Base Tokenizers

| Tokenizer      | Description                                                               | Use Case               |
|----------------|---------------------------------------------------------------------------|------------------------|
| **simple**     | Splits on whitespace and punctuation, removes non-alphanumeric characters | General text (default) |
| **whitespace** | Splits only on whitespace characters                                      | Preserve punctuation   |
| **raw**        | No tokenization, treats entire text as single token                       | Exact matching         |
| **ngram**      | Breaks text into overlapping character sequences                          | Substring/fuzzy search |
| **jieba/***    | Chinese text tokenizer with word segmentation                             | Chinese text           |
| **lindera/***  | Japanese text tokenizer with morphological analysis                       | Japanese text          |

#### Jieba Tokenizer (Chinese)

Jieba is a popular Chinese text segmentation library that uses a dictionary-based approach with statistical methods for word segmentation.

- **Configuration**: Uses a `config.json` file in the model directory
- **Models**: Must be downloaded and placed in the Lance home directory under `jieba/`
- **Usage**: Specify as `jieba/<model_name>` or just `jieba` for the default model
- **Config Structure**:
  ```json
  {
    "main": "path/to/main/dictionary",
    "users": ["path/to/user/dict1", "path/to/user/dict2"]
  }
  ```
- **Features**:
  - Accurate word segmentation for Simplified and Traditional Chinese
  - Support for custom user dictionaries
  - Multiple segmentation modes (precise, full, search engine)

#### Lindera Tokenizer (Japanese)

Lindera is a morphological analysis tokenizer specifically designed for Japanese text. It provides proper word segmentation for Japanese, which doesn't use spaces between words.

- **Configuration**: Uses a `config.yml` file in the model directory
- **Models**: Must be downloaded and placed in the Lance home directory under `lindera/`
- **Usage**: Specify as `lindera/<model_name>` where `<model_name>` is the subdirectory containing the model files
- **Features**:
  - Morphological analysis with part-of-speech tagging
  - Dictionary-based tokenization
  - Support for custom user dictionaries

### Token Filters

Token filters are applied in sequence after the base tokenizer:

| Filter           | Description                                 | Configuration                   |
|------------------|---------------------------------------------|---------------------------------|
| **RemoveLong**   | Removes tokens exceeding max_token_length   | `max_token_length`              |
| **LowerCase**    | Converts tokens to lowercase                | `lower_case` (default: true)    |
| **Stemmer**      | Reduces words to their root form            | `stem`, `language`              |
| **StopWords**    | Removes common words like "the", "is", "at" | `remove_stop_words`, `language` |
| **AsciiFolding** | Converts accented characters to ASCII       | `ascii_folding` (default: true) |

### Supported Languages

For stemming and stop word removal, the following languages are supported:
Arabic, Danish, Dutch, English, Finnish, French, German, Greek, Hungarian, Italian, Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish, Tamil, Turkish

## Accelerated Queries

Lance SDKs provide dedicated full text search APIs to leverage the FTS index capabilities. 
These APIs support complex query types beyond simple token matching, 
enabling sophisticated text search operations.
Here are the query types enabled by the FTS index:

| Query Type          | Description                                                                              | Example Usage                                        | Result Type |
|---------------------|------------------------------------------------------------------------------------------|------------------------------------------------------|-------------|
| **contains_tokens** | Basic token-based search (UDF) with BM25 scoring and automatic result ranking            | SQL: `contains_tokens(column, 'search terms')`       | AtMost      |
| **match**           | Match query with configurable AND/OR operators and relevance scoring                     | `{"match": {"query": "text", "operator": "and/or"}}` | AtMost      |
| **phrase**          | Exact phrase matching with position information (requires `with_position: true`)         | `{"phrase": {"query": "exact phrase"}}`              | AtMost      |
| **boolean**         | Complex boolean queries with must/should/must_not clauses for sophisticated search logic | `{"boolean": {"must": [...], "should": [...]}}`      | AtMost      |
| **multi_match**     | Search across multiple fields simultaneously with unified scoring                        | `{"multi_match": [{"field1": "query"}, ...]}`        | AtMost      |
| **boost**           | Boost relevance scores for specific terms or queries by a configurable factor            | `{"boost": {"query": {...}, "factor": 2.0}}`         | AtMost      |