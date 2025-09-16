# N-gram Index

N-gram indices break text into overlapping character or word sequences, enabling efficient substring matching and fuzzy search capabilities.

## Overview

An n-gram is a contiguous sequence of n items from a text:
- **Character n-grams**: "hello" → ["he", "el", "ll", "lo"] (bigrams)
- **Word n-grams**: "the quick brown" → ["the quick", "quick brown"] (bigrams)

N-gram indices map these sequences to their source documents or positions.

## Use Cases

N-gram indices excel at:
- **Substring search**: Find all strings containing a substring
- **Fuzzy matching**: Approximate string matching with typos
- **Wildcard queries**: `LIKE '%substring%'` patterns
- **Similarity search**: Find similar strings based on shared n-grams
- **Spell correction**: Suggest corrections for misspelled words
- **Language detection**: Identify language from n-gram patterns

## Structure

### N-gram Generation

```
Text: "database"
Bigrams (n=2):  ["da", "at", "ta", "ab", "ba", "as", "se"]
Trigrams (n=3): ["dat", "ata", "tab", "aba", "bas", "ase"]

With padding:
Text: "database"
Padded: "_database_"
Bigrams: ["_d", "da", "at", "ta", "ab", "ba", "as", "se", "e_"]
```

### Storage Format

```
N-gram Index File:
  Header:
    - Index version
    - N-gram size (n)
    - Padding configuration
    - Total documents
    - Total unique n-grams

  N-gram Dictionary:
    - N-gram → N-gram ID
    - Frequency statistics
    - Document frequency

  Inverted Lists:
    - N-gram ID → Document IDs
    - Position information (optional)
    - N-gram count per document

  Auxiliary:
    - Character set configuration
    - Case sensitivity settings
    - Special character handling
```

## Types of N-gram Indices

### Character N-grams

```python
dataset.create_index(
    column="product_name",
    index_type="ngram",
    config={
        "type": "character",
        "n": 3,                    # Trigrams
        "min_n": 2,               # Minimum n-gram size
        "max_n": 4,               # Maximum n-gram size
        "padding": True,          # Add boundary markers
        "case_sensitive": False
    }
)
```

### Word N-grams

```python
dataset.create_index(
    column="description",
    index_type="ngram",
    config={
        "type": "word",
        "n": 2,                    # Bigrams
        "tokenizer": "whitespace",
        "stop_words": None        # Include all words
    }
)
```

### Skip-grams

Allow gaps between elements:

```
Text: "the quick brown fox"
2-skip-bigrams: ["the quick", "the brown", "the fox", "quick brown", "quick fox", "brown fox"]
```

## Query Processing

### Substring Search

```sql
-- Query: Find all products containing "phone"

1. Generate n-grams for "phone" (e.g., "pho", "hon", "one")
2. Find documents containing ALL these n-grams
3. Verify actual substring presence (eliminate false positives)
```

### Fuzzy Search

```python
def fuzzy_search(query, threshold=0.7):
    query_ngrams = generate_ngrams(query)

    results = []
    for doc_id, doc_ngrams in index:
        similarity = jaccard_similarity(query_ngrams, doc_ngrams)
        if similarity >= threshold:
            results.append((doc_id, similarity))

    return sorted(results, key=lambda x: x[1], reverse=True)
```

### Wildcard Queries

```sql
-- Query: LIKE '%data%base%'

1. Split pattern into fixed parts: ["data", "base"]
2. Find documents containing n-grams from both parts
3. Verify pattern match with positions
```

## Similarity Metrics

### Jaccard Similarity

```python
def jaccard_similarity(ngrams1, ngrams2):
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    return intersection / union if union > 0 else 0
```

### Dice Coefficient

```python
def dice_coefficient(ngrams1, ngrams2):
    intersection = len(ngrams1 & ngrams2)
    return 2 * intersection / (len(ngrams1) + len(ngrams2))
```

### Cosine Similarity

```python
def cosine_similarity(ngram_vector1, ngram_vector2):
    dot_product = sum(v1 * v2 for v1, v2 in zip(ngram_vector1, ngram_vector2))
    magnitude1 = sqrt(sum(v ** 2 for v in ngram_vector1))
    magnitude2 = sqrt(sum(v ** 2 for v in ngram_vector2))
    return dot_product / (magnitude1 * magnitude2)
```

## Optimization Strategies

### N-gram Selection

```python
def optimal_n_value(avg_string_length, query_patterns):
    if avg_string_length < 10:
        return 2  # Bigrams for short strings

    if substring_queries_dominant:
        return 3  # Trigrams balance precision and recall

    if fuzzy_matching_required:
        return [2, 3, 4]  # Multiple n-gram sizes

    return 3  # Default to trigrams
```

### Filtering and Pruning

```python
config={
    "min_frequency": 2,       # Skip rare n-grams
    "max_frequency": 0.8,     # Skip too common n-grams
    "min_document_frequency": 2,
    "positional": False       # Save space if positions not needed
}
```

### Index Compression

- **Dictionary encoding**: Map frequent n-grams to short codes
- **Delta encoding**: For sorted n-gram lists
- **Huffman coding**: Variable-length codes based on frequency

## Performance Characteristics

### Time Complexity

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| Generate n-grams | O(s * n) | s = string length |
| Index lookup | O(1) | Hash table lookup |
| Substring search | O(q * k) | q = query n-grams, k = posting size |
| Fuzzy search | O(n * m) | n = query n-grams, m = doc n-grams |

### Space Complexity

- **Index size**: O(D * S * n) where D = documents, S = avg string length
- **With positions**: 2-3x larger
- **Compressed**: 30-50% of uncompressed size

### False Positive Rate

N-gram indices may return false positives:

```
Query: "cat"
N-grams: ["ca", "at"]
False positive: "scattered" (contains both "ca" and "at")
```

Mitigation:
- Use larger n values (reduces false positives)
- Post-filter results
- Combine with positional information

## Advanced Techniques

### Positional N-grams

Store position information for exact matching:

```python
"database" with positional trigrams:
{
    "dat": [0],
    "ata": [1],
    "tab": [2],
    "aba": [3],
    "bas": [4],
    "ase": [5]
}
```

### Weighted N-grams

Assign weights based on position or frequency:

```python
def weighted_ngram_score(ngrams):
    scores = {}
    for pos, ngram in enumerate(ngrams):
        # Higher weight for beginning/end n-grams
        weight = 2.0 if pos < 2 or pos > len(ngrams) - 3 else 1.0
        scores[ngram] = weight
    return scores
```

### Hierarchical N-grams

Build indices with multiple n values:

```python
index = {
    "bigrams": ngram_index(n=2),
    "trigrams": ngram_index(n=3),
    "quadgrams": ngram_index(n=4)
}

def search(query):
    if len(query) < 3:
        return index["bigrams"].search(query)
    elif len(query) < 5:
        return index["trigrams"].search(query)
    else:
        return index["quadgrams"].search(query)
```

## Best Practices

### Configuration Guidelines

```python
def configure_ngram_index(column_stats):
    config = {
        "type": "character",
        "case_sensitive": False,
        "padding": True
    }

    avg_length = column_stats.avg_string_length

    if avg_length < 20:
        config["n"] = 2  # Short strings
    elif avg_length < 100:
        config["n"] = 3  # Medium strings
    else:
        config["n"] = 4  # Long strings

    # Multi-gram for better accuracy
    if column_stats.requires_high_precision:
        config["min_n"] = 2
        config["max_n"] = 4

    return config
```

### Query Optimization

1. **Use appropriate n-gram size**: Match query length to n-gram size
2. **Combine with other indices**: Use n-grams for filtering, then exact match
3. **Implement caching**: Cache frequent n-gram queries
4. **Batch processing**: Process multiple queries together

## Example: Product Search

```python
# Create n-gram index for product search
dataset.create_index(
    column="product_name",
    index_type="ngram",
    config={
        "type": "character",
        "n": 3,
        "padding": True,
        "case_sensitive": False,
        "min_frequency": 2
    }
)

# Fuzzy product search
def find_similar_products(query, threshold=0.6):
    # Generate query n-grams
    query_ngrams = set(generate_trigrams(query.lower()))

    results = []
    for product_id, product_name in products:
        product_ngrams = set(generate_trigrams(product_name.lower()))

        # Calculate similarity
        similarity = len(query_ngrams & product_ngrams) / len(query_ngrams | product_ngrams)

        if similarity >= threshold:
            results.append({
                'id': product_id,
                'name': product_name,
                'similarity': similarity
            })

    return sorted(results, key=lambda x: x['similarity'], reverse=True)

# Example: Find products similar to "iPhone 13"
# Returns: ["iPhone 13 Pro", "iPhone 13 Mini", "iPhone 12", ...]
```

## Limitations

1. **Space overhead**: Can be several times larger than original data
2. **False positives**: Requires post-filtering for exact matches
3. **Update complexity**: Changes require re-indexing affected n-grams
4. **Not suitable for**: Very long strings (exponential n-gram growth)
5. **Language specific**: Some languages (e.g., Chinese) need special handling