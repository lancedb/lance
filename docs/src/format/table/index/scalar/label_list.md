# Label List Index

Label list indices are specialized structures optimized for columns containing multiple labels or tags per row, common in multi-label classification and tagging systems.

## Overview

A label list index efficiently handles columns where each row contains a set of labels:
- Supports fast membership queries across label sets
- Optimizes storage for repeated label combinations
- Enables efficient label-based filtering and aggregation

## Use Cases

Label list indices are ideal for:
- **Multi-label classification**: Documents with multiple categories
- **Tagging systems**: Items with multiple tags
- **Feature sets**: Entities with multiple attributes
- **Permission systems**: Users with multiple roles
- **Recommendation systems**: Items with multiple genres/categories
- **Graph properties**: Nodes with multiple labels

## Structure

### Data Representation

```
Example Data:
  Row 1: ["sports", "news", "international"]
  Row 2: ["technology", "news"]
  Row 3: ["sports", "local"]
  Row 4: ["technology", "science", "research"]

Label List Index:
  Label Dictionary:
    "sports" → 0
    "news" → 1
    "international" → 2
    "technology" → 3
    "local" → 4
    "science" → 5
    "research" → 6

  Inverted Index:
    0 (sports) → [1, 3]
    1 (news) → [1, 2]
    2 (international) → [1]
    3 (technology) → [2, 4]
    4 (local) → [3]
    5 (science) → [4]
    6 (research) → [4]

  Forward Index:
    Row 1 → [0, 1, 2]
    Row 2 → [1, 3]
    Row 3 → [0, 4]
    Row 4 → [3, 5, 6]
```

### Storage Format

```
Label List Index File:
  Header:
    - Index version
    - Total rows
    - Unique labels count
    - Encoding type

  Label Dictionary:
    - Label string → Label ID
    - Label frequency
    - Label statistics

  Forward Index:
    - Row ID → [Label IDs]
    - Compressed using delta/varint encoding

  Inverted Index:
    - Label ID → [Row IDs]
    - Posting lists (compressed)

  Label Combinations (optional):
    - Frequent combinations cache
    - Co-occurrence matrix
```

## Query Operations

### Contains Any Labels (OR)

```sql
-- Find rows with "sports" OR "technology"
WHERE labels @> ANY(['sports', 'technology'])
```

```python
def contains_any(query_labels):
    result_rows = set()
    for label in query_labels:
        label_id = label_dict[label]
        result_rows.update(inverted_index[label_id])
    return result_rows
```

### Contains All Labels (AND)

```sql
-- Find rows with both "technology" AND "research"
WHERE labels @> ALL(['technology', 'research'])
```

```python
def contains_all(query_labels):
    if not query_labels:
        return all_rows

    label_ids = [label_dict[label] for label in query_labels]
    result_rows = set(inverted_index[label_ids[0]])

    for label_id in label_ids[1:]:
        result_rows &= set(inverted_index[label_id])

    return result_rows
```

### Exact Match

```sql
-- Find rows with exactly these labels
WHERE labels = ['sports', 'news']
```

```python
def exact_match(query_labels):
    query_set = set(query_labels)
    results = []

    for row_id, row_labels in forward_index.items():
        if set(row_labels) == query_set:
            results.append(row_id)

    return results
```

## Optimization Techniques

### Bitmap Encoding

For high-frequency labels, use bitmaps:

```python
class BitmapLabelIndex:
    def __init__(self, threshold=0.1):
        self.bitmap_labels = {}  # Labels appearing in > 10% of rows
        self.posting_labels = {}  # Labels appearing in < 10% of rows

    def query(self, label):
        if label in self.bitmap_labels:
            return self.bitmap_labels[label].to_row_ids()
        else:
            return self.posting_labels[label]
```

### Frequent Itemset Mining

Cache frequent label combinations:

```python
frequent_combinations = {
    frozenset(["technology", "news"]): [2, 15, 27, ...],
    frozenset(["sports", "local"]): [3, 8, 19, ...],
    frozenset(["science", "research"]): [4, 12, 31, ...]
}

def query_combination(labels):
    key = frozenset(labels)
    if key in frequent_combinations:
        return frequent_combinations[key]  # O(1) lookup
    else:
        return compute_intersection(labels)  # Fall back to computation
```

### Hierarchical Labels

Support label hierarchies:

```python
label_hierarchy = {
    "sports": ["football", "basketball", "tennis"],
    "technology": ["software", "hardware", "AI"],
    "news": ["local", "international", "breaking"]
}

def expand_query(label):
    """Expand parent label to include children"""
    if label in label_hierarchy:
        return [label] + label_hierarchy[label]
    return [label]
```

## Configuration

```python
dataset.create_index(
    column="tags",
    index_type="label_list",
    config={
        "encoding": "bitmap",           # or "posting", "hybrid"
        "min_frequency": 2,             # Minimum label frequency to index
        "max_labels_per_row": 100,      # Maximum labels per row
        "cache_combinations": True,      # Cache frequent combinations
        "combination_threshold": 0.01,   # Min support for combinations
        "compression": "roaring"        # Compression for bitmaps
    }
)
```

## Performance Characteristics

### Time Complexity

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| Contains any (OR) | O(k * m) | k = query labels, m = avg posting size |
| Contains all (AND) | O(k * min(m)) | Intersection of k lists |
| Exact match | O(n) worst case | Can optimize with hashing |
| Add label | O(1) amortized | Update inverted index |
| Remove label | O(m) | Update posting list |

### Space Complexity

- **Dictionary**: O(unique_labels)
- **Forward index**: O(total_labels_assigned)
- **Inverted index**: O(total_labels_assigned)
- **With combinations**: Additional O(frequent_combinations)

## Advanced Features

### Label Co-occurrence Analysis

```python
def build_cooccurrence_matrix(forward_index):
    cooccurrence = defaultdict(int)

    for row_labels in forward_index.values():
        for i, label1 in enumerate(row_labels):
            for label2 in row_labels[i+1:]:
                pair = tuple(sorted([label1, label2]))
                cooccurrence[pair] += 1

    return cooccurrence

# Use for query optimization
def optimize_query_order(labels, cooccurrence):
    """Order labels by selectivity for faster intersection"""
    return sorted(labels, key=lambda l: label_frequency[l])
```

### Similarity Search

```python
def find_similar_rows(row_id, threshold=0.5):
    """Find rows with similar label sets"""
    target_labels = set(forward_index[row_id])
    similar = []

    for other_id, other_labels in forward_index.items():
        if other_id == row_id:
            continue

        other_set = set(other_labels)
        similarity = len(target_labels & other_set) / len(target_labels | other_set)

        if similarity >= threshold:
            similar.append((other_id, similarity))

    return sorted(similar, key=lambda x: x[1], reverse=True)
```

### Label Statistics

```python
class LabelStatistics:
    def __init__(self, index):
        self.index = index

    def label_distribution(self):
        """Get frequency distribution of labels"""
        return {
            label: len(rows)
            for label, rows in self.index.inverted_index.items()
        }

    def avg_labels_per_row(self):
        """Calculate average number of labels per row"""
        total_labels = sum(len(labels) for labels in self.index.forward_index.values())
        return total_labels / len(self.index.forward_index)

    def label_correlation(self, label1, label2):
        """Calculate correlation between two labels"""
        rows1 = set(self.index.inverted_index[label1])
        rows2 = set(self.index.inverted_index[label2])
        intersection = len(rows1 & rows2)
        union = len(rows1 | rows2)
        return intersection / union if union > 0 else 0
```

## Integration with Other Systems

### Machine Learning Pipeline

```python
# Use label index for multi-label classification
def prepare_training_data(label_index):
    X = []  # Feature vectors
    y = []  # Label sets

    for row_id, labels in label_index.forward_index.items():
        X.append(get_features(row_id))
        y.append(labels)

    return X, y

# Fast label-based sampling
def stratified_sample(label_index, label, sample_size):
    rows_with_label = label_index.inverted_index[label]
    return random.sample(rows_with_label, min(sample_size, len(rows_with_label)))
```

### Faceted Search

```python
def get_facets(query_results, label_index):
    """Get label counts for search results"""
    facets = defaultdict(int)

    for row_id in query_results:
        for label in label_index.forward_index[row_id]:
            facets[label] += 1

    return dict(facets)
```

## Best Practices

### Index Design

1. **Choose appropriate encoding**:
   - Bitmap for labels in >1% of rows
   - Posting lists for rare labels
   - Hybrid for mixed distributions

2. **Optimize for query patterns**:
   - Cache frequent combinations
   - Pre-compute label hierarchies
   - Index label pairs for correlation queries

3. **Handle updates efficiently**:
   - Batch label updates
   - Maintain incremental statistics
   - Periodic index reorganization

### Query Optimization

```python
def optimize_label_query(query_type, labels):
    if query_type == "any":
        # Process most selective labels first
        return sorted(labels, key=lambda l: label_frequency[l])

    elif query_type == "all":
        # Start with least frequent label
        return sorted(labels, key=lambda l: label_frequency[l])

    elif query_type == "exact":
        # Check combination cache first
        if frozenset(labels) in frequent_combinations:
            return "use_cache"
        return "compute"
```

## Example: Document Tagging System

```python
# Create label list index for document tags
dataset.create_index(
    column="document_tags",
    index_type="label_list",
    config={
        "encoding": "hybrid",
        "cache_combinations": True,
        "combination_threshold": 0.005
    }
)

# Query documents by tags
class DocumentSearch:
    def __init__(self, label_index):
        self.index = label_index

    def find_by_tags(self, must_have=[], should_have=[], must_not_have=[]):
        # Start with all documents
        results = set(range(len(self.index.forward_index)))

        # Apply MUST filters (AND)
        if must_have:
            results &= self.index.contains_all(must_have)

        # Apply SHOULD filters (OR)
        if should_have:
            should_results = self.index.contains_any(should_have)
            results &= should_results

        # Apply MUST NOT filters
        if must_not_have:
            exclude = self.index.contains_any(must_not_have)
            results -= exclude

        return results

# Example usage
searcher = DocumentSearch(label_index)
results = searcher.find_by_tags(
    must_have=["technology", "published"],
    should_have=["AI", "machine-learning"],
    must_not_have=["draft", "archived"]
)
```

## Limitations

1. **Memory usage**: Forward index can be large for many labels
2. **Update complexity**: Adding/removing labels requires index updates
3. **Not suitable for**: Continuous values, unique labels per row
4. **Cardinality limits**: Performance degrades with too many unique labels
5. **Order dependency**: Doesn't preserve label order (use array index if needed)