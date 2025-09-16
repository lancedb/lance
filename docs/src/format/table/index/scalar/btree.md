# BTree Index

The BTree (Balanced Tree) index is a fundamental index structure in Lance that provides efficient range queries, sorted access, and exact match lookups.

## Overview

A BTree index maintains sorted keys in a tree structure where:
- All leaf nodes are at the same depth
- Each node contains multiple keys and child pointers
- Keys are kept sorted within each node
- Supports O(log n) search, insert, and delete operations

## Use Cases

BTree indices are ideal for:
- **Range queries**: `WHERE price BETWEEN 100 AND 500`
- **Sorted scans**: `ORDER BY timestamp`
- **Prefix matching**: `WHERE name LIKE 'John%'`
- **Inequality predicates**: `WHERE age > 25`
- **Min/max queries**: `SELECT MAX(salary)`

## Structure

### Node Types

1. **Internal Nodes**
   - Contains keys and pointers to child nodes
   - Keys act as separators for routing searches
   - Typically stores hundreds of keys per node

2. **Leaf Nodes**
   - Contains keys and row IDs or row addresses
   - Linked to siblings for efficient range scans
   - May store additional statistics

### Storage Layout

```
BTree Index File:
  Header:
    - Index version
    - Root node offset
    - Node size
    - Key type and size
    - Total entries
    - Tree height

  Nodes:
    - Internal nodes (keys + child pointers)
    - Leaf nodes (keys + row IDs)

  Auxiliary:
    - Free space map
    - Node allocation table
```

## Operations

### Search
1. Start at root node
2. Binary search within node to find appropriate child
3. Repeat until reaching leaf node
4. Return matching row IDs

### Range Scan
1. Find starting leaf node using search
2. Scan leaf nodes sequentially using sibling pointers
3. Stop when reaching end condition

### Insert
1. Find appropriate leaf node
2. Insert key maintaining sort order
3. Split node if full, propagating splits upward
4. Update root if necessary

## Configuration

BTree indices can be configured with:

```python
dataset.create_index(
    column="timestamp",
    index_type="btree",
    config={
        "node_size": 4096,      # Bytes per node
        "fill_factor": 0.9,     # Max fill before split
        "unique": False,        # Enforce uniqueness
        "null_position": "last" # Where to sort nulls
    }
)
```

## Optimizations

### Prefix Compression
- Common key prefixes are stored once per node
- Reduces storage and improves cache efficiency

### Bulk Loading
- Bottom-up construction for initial index creation
- Produces optimally packed nodes
- Faster than incremental insertion

### Adaptive Node Sizes
- Larger nodes for sequential access patterns
- Smaller nodes for random access patterns

## Performance Characteristics

| Operation | Time Complexity | I/O Complexity |
|-----------|----------------|----------------|
| Point Query | O(log n) | O(log_B n) |
| Range Query | O(log n + k) | O(log_B n + k/B) |
| Insert | O(log n) | O(log_B n) |
| Delete | O(log n) | O(log_B n) |

Where:
- n = number of entries
- k = number of results
- B = entries per node

## Memory Usage

- **Index Size**: ~(n * (key_size + 8)) / fill_factor
- **Cache Requirements**: Working set of frequently accessed nodes
- **Build Memory**: 2-3x final index size for bulk loading

## Limitations

- Not suitable for high-cardinality columns with random inserts
- Range scans may require multiple I/O operations
- Rebalancing overhead for heavily updated datasets

## Best Practices

1. **Choose appropriate node size**
   - Larger nodes (8-16KB) for sequential workloads
   - Smaller nodes (4KB) for random access

2. **Consider compound indices**
   - Order columns by selectivity (most to least)
   - Include frequently queried columns together

3. **Monitor fill factor**
   - Lower fill factor reduces splits but increases size
   - Higher fill factor saves space but causes more splits

4. **Periodic maintenance**
   - Rebuild indices after significant updates
   - Analyze statistics for query optimizer