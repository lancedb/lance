# Inverted Index

Inverted indices map terms to the documents containing them, enabling efficient full-text search and token-based queries.

## Overview

An inverted index reverses the traditional document-to-terms mapping:
- Forward index: Document → [Terms]
- Inverted index: Term → [Documents]

This structure enables fast searches for documents containing specific terms.

## Use Cases

Inverted indices are essential for:
- **Full-text search**: Find documents containing words/phrases
- **Keyword queries**: `WHERE description CONTAINS 'machine learning'`
- **Boolean text queries**: AND, OR, NOT operations on terms
- **Phrase searches**: Exact phrase matching with positional information
- **Faceted search**: Aggregate by terms
- **Auto-completion**: Prefix-based term suggestions

## Structure

### Basic Components

```
Document Collection:
  Doc1: "The quick brown fox"
  Doc2: "The lazy brown dog"
  Doc3: "Quick brown rabbits run"

Inverted Index:
  "the"     → [Doc1, Doc2]
  "quick"   → [Doc1, Doc3]
  "brown"   → [Doc1, Doc2, Doc3]
  "fox"     → [Doc1]
  "lazy"    → [Doc2]
  "dog"     → [Doc2]
  "rabbits" → [Doc3]
  "run"     → [Doc3]
```

### Storage Format

```
Inverted Index File:
  Header:
    - Index version
    - Tokenizer configuration
    - Total documents
    - Total unique terms

  Dictionary:
    - Term → Term ID mapping
    - Term frequency statistics
    - Document frequency

  Postings Lists:
    - Term ID → Document IDs
    - Positional information (optional)
    - Term frequency per document
    - Field information

  Auxiliary:
    - Stop words list
    - Synonym mappings
    - Stemming rules
```

## Tokenization

### Text Processing Pipeline

1. **Normalization**: Convert to lowercase, remove accents
2. **Tokenization**: Split text into tokens
3. **Filtering**: Remove stop words
4. **Stemming/Lemmatization**: Reduce to root form
5. **Indexing**: Add to inverted index

### Tokenizer Types

```python
dataset.create_index(
    column="description",
    index_type="inverted",
    config={
        "tokenizer": "standard",      # or "whitespace", "ngram", "custom"
        "lowercase": True,
        "stop_words": "english",      # or custom list
        "stemmer": "porter",          # or "snowball", "none"
        "min_token_length": 2,
        "max_token_length": 20
    }
)
```

## Posting List Formats

### Simple Posting List
```
Term: "brown"
Postings: [1, 5, 12, 45, 78]  # Document IDs only
```

### Posting List with Frequencies
```
Term: "brown"
Postings: [(1, 2), (5, 1), (12, 3)]  # (DocID, Term Frequency)
```

### Posting List with Positions
```
Term: "brown"
Postings: [
    (1, 2, [3, 7]),      # Doc 1, appears 2 times at positions 3 and 7
    (5, 1, [2]),         # Doc 5, appears 1 time at position 2
    (12, 3, [1, 5, 9])   # Doc 12, appears 3 times
]
```

## Compression Techniques

### Delta Encoding
Store differences between consecutive document IDs:
```
Original: [5, 8, 12, 13, 20]
Delta:    [5, 3, 4, 1, 7]
```

### Variable Byte Encoding
Encode small integers using fewer bytes:
```
Number 127:  [0x7F]           # 1 byte
Number 128:  [0x80, 0x01]     # 2 bytes
Number 300:  [0xAC, 0x02]     # 2 bytes
```

### Bitmap Compression
For frequent terms, use compressed bitmaps:
```
Term appears in 80% of documents → Use bitmap
Term appears in 1% of documents → Use posting list
```

## Query Processing

### Single Term Query
```sql
-- Find documents containing "machine"
1. Look up "machine" in dictionary → Term ID
2. Retrieve posting list for Term ID
3. Return document IDs
```

### Boolean Queries
```sql
-- Query: "machine" AND "learning" NOT "deep"
1. Get posting list for "machine" → L1
2. Get posting list for "learning" → L2
3. Get posting list for "deep" → L3
4. Result = (L1 ∩ L2) - L3
```

### Phrase Queries
```sql
-- Query: "machine learning"
1. Get positional postings for "machine" and "learning"
2. Find documents where "learning" appears at position(machine) + 1
3. Return matching documents
```

## Scoring and Ranking

### TF-IDF Scoring
```python
def tf_idf_score(term, document, corpus):
    tf = term_frequency(term, document)
    idf = log(total_documents / documents_containing(term))
    return tf * idf
```

### BM25 Scoring
```python
def bm25_score(term, document, k1=1.2, b=0.75):
    tf = term_frequency(term, document)
    idf = compute_idf(term)
    dl = document_length
    avgdl = average_document_length

    score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
    return score
```

## Configuration Options

```python
dataset.create_index(
    column="content",
    index_type="inverted",
    config={
        # Tokenization
        "tokenizer": "standard",
        "analyzer": "english",

        # Storage
        "store_positions": True,      # Enable phrase queries
        "store_frequencies": True,    # Enable TF-IDF scoring
        "store_offsets": True,        # Enable highlighting

        # Optimization
        "min_doc_frequency": 2,       # Skip rare terms
        "max_doc_frequency": 0.95,    # Skip too common terms
        "compression": "vbyte",       # Compression algorithm
    }
)
```

## Performance Characteristics

### Time Complexity

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| Index term | O(1) | Amortized, hash table insertion |
| Single term query | O(k) | k = posting list size |
| AND query | O(min(k1, k2)) | Merge sorted lists |
| OR query | O(k1 + k2) | Union of lists |
| Phrase query | O(k * p) | p = average positions per doc |

### Space Complexity

- **Dictionary**: O(unique_terms)
- **Postings**: O(total_term_occurrences)
- **With positions**: 2-4x larger than without
- **Compressed**: 20-40% of uncompressed size

## Optimizations

### Skip Lists
Accelerate intersection of long posting lists:
```
Posting List: [1, 3, 5, 8, 12, 15, 20, 25, 30, 35, 40]
Skip Pointers: 1 → 12 → 30
```

### Early Termination
For top-k queries, stop processing after finding enough results:
```python
def top_k_search(query, k=10):
    results = []
    for posting in sorted_postings:  # Sorted by score
        if len(results) >= k and posting.max_score < results[-1].score:
            break  # Can't improve top-k
        results.append(process(posting))
    return results[:k]
```

### Index Sharding
Distribute large indices across multiple shards:
```python
shard_id = hash(term) % num_shards
posting_list = shards[shard_id].get_postings(term)
```

## Best Practices

### Text Preprocessing
```python
def preprocess_text(text):
    # Normalize
    text = text.lower()
    text = remove_accents(text)

    # Tokenize
    tokens = word_tokenize(text)

    # Filter
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [t for t in tokens if len(t) >= min_length]

    # Stem
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens
```

### Index Maintenance

1. **Incremental updates**: Batch small updates, rebuild periodically
2. **Merge optimization**: Combine small index segments
3. **Cache warming**: Pre-load frequently accessed terms
4. **Statistics updates**: Refresh TF-IDF values periodically

## Example: Document Search System

```python
# Create inverted index for document search
dataset.create_index(
    column="content",
    index_type="inverted",
    config={
        "tokenizer": "standard",
        "analyzer": "english",
        "store_positions": True,
        "scoring": "bm25"
    }
)

# Search for documents
def search_documents(query, limit=10):
    # Parse query
    terms = parse_query(query)

    # Get posting lists
    postings = [index.get_postings(term) for term in terms]

    # Merge and score
    results = merge_postings(postings)
    scores = compute_scores(results, terms)

    # Return top results
    return sorted(scores, reverse=True)[:limit]

# Example query
results = search_documents("machine learning algorithms")
```

## Limitations

1. **Space overhead**: Can be larger than original text
2. **Update complexity**: Requires reindexing for modifications
3. **Not suitable for**: Numeric ranges, fuzzy matching (without extensions)
4. **Language dependency**: Tokenization rules vary by language
5. **Synonym handling**: Requires additional mapping structures