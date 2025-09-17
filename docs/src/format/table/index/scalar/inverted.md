# Inverted Index

Inverted indices map terms to the documents containing them, enabling efficient full-text search.

## Index Details

```protobuf
%%% proto.message.InvertedIndexDetails %%%
```

## Storage Layout

The inverted index uses a partitioned structure with multiple file types:

### Directory Structure

```
_indices/{UUID}/
├── metadata.lance                  # Global metadata
├── tokens_{partition_id}.lance     # Token dictionary per partition
├── docs_{partition_id}.lance       # Document list per partition
├── postings_{partition_id}.lance   # Posting lists per partition
└── part_{partition_id}_metadata.lance  # Partition metadata
```

### File Formats

#### Metadata File
```
metadata.lance
├── Schema: Empty
└── Data: Serialized metadata including:
    - Tokenizer configuration
    - Number of partitions
    - Document count
    - Token statistics
```

#### Token Dictionary
```
tokens_{partition_id}.lance
├── Schema
│   ├── token: Utf8           # Token string
│   ├── frequency: UInt64     # Document frequency
│   └── posting_offset: UInt64 # Offset in posting file
└── Data: Sorted token entries
```

#### Document List
```
docs_{partition_id}.lance
├── Schema
│   ├── doc_id: UInt64        # Document ID
│   ├── length: UInt32        # Document length (tokens)
│   └── norm: Float32         # Document norm for scoring
└── Data: Document metadata
```

#### Posting Lists
```
postings_{partition_id}.lance
├── Schema
│   ├── doc_id: UInt64        # Document containing token
│   ├── frequency: UInt32     # Term frequency in document
│   └── positions: List<UInt32> # Token positions (optional)
└── Data: Compressed posting lists
```

## Implementation Details

- **Partitioning**: Index is split into partitions for parallel processing
- **Tokenization**: Configurable tokenizer with language-specific support
- **Compression**: Posting lists use delta encoding and compression
- **Scoring**: Supports TF-IDF and BM25 scoring
- **Position Storage**: Optional positional information for phrase queries