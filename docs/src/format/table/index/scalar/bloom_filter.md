# Bloom Filter Index

Bloom filters are space-efficient probabilistic data structures that test whether an element is a member of a set, with possible false positives but no false negatives.

## Overview

A Bloom filter uses multiple hash functions to map elements to a bit array:
- Can definitively say an element is NOT in the set
- May have false positives (element might be in the set)
- Extremely space-efficient compared to exact methods
- No false negatives guarantee

## Use Cases

Bloom filters excel at:
- **Existence checks**: `WHERE user_id = 'abc123'`
- **Join optimization**: Pre-filter before expensive joins
- **Partition pruning**: Skip partitions that don't contain values
- **Negative lookups**: Quickly eliminate non-existent keys
- **Cache filtering**: Avoid unnecessary database lookups

## Structure

### Basic Operation

```
Insert "hello":
1. Hash1("hello") = 42  -> Set bit 42
2. Hash2("hello") = 137 -> Set bit 137
3. Hash3("hello") = 255 -> Set bit 255

Check "hello":
1. Check bit 42: Set? Yes
2. Check bit 137: Set? Yes
3. Check bit 255: Set? Yes
Result: "hello" might be in set

Check "world":
1. Check bit 38: Set? No
Result: "world" definitely not in set
```

### Storage Format

```
Bloom Filter Index:
  Header:
    - Index version
    - Filter size (bits)
    - Number of hash functions
    - Number of elements
    - False positive probability

  Filter Data:
    - Bit array (compressed)
    - Hash function seeds

  Metadata:
    - Creation timestamp
    - Column statistics
    - Filter parameters
```

## Mathematical Foundation

### Optimal Parameters

Given:
- n = number of elements
- p = desired false positive probability

Calculate:
- m = optimal bit array size = -(n * ln(p)) / (ln(2)^2)
- k = optimal number of hash functions = (m/n) * ln(2)

### False Positive Rate

Actual false positive probability:
```
p_actual = (1 - e^(-kn/m))^k
```

## Configuration

```python
dataset.create_index(
    column="product_id",
    index_type="bloom_filter",
    config={
        "false_positive_rate": 0.01,  # 1% FPR
        "expected_elements": 1000000,  # Expected number of distinct values
        "hash_functions": "murmur3",   # Hash algorithm
        "compression": True             # Compress bit array
    }
)
```

### Parameter Trade-offs

| False Positive Rate | Bits per Element | Hash Functions |
|--------------------|------------------|----------------|
| 0.1 (10%) | 4.8 bits | 3 |
| 0.01 (1%) | 9.6 bits | 7 |
| 0.001 (0.1%) | 14.4 bits | 10 |
| 0.0001 (0.01%) | 19.2 bits | 13 |

## Implementation Details

### Hash Functions

Lance uses non-cryptographic hash functions for speed:

```python
def hash_element(value, seed):
    # MurmurHash3 for speed and good distribution
    hash1 = murmur3_hash(value, seed)
    hash2 = murmur3_hash(value, seed + 1)

    # Double hashing to generate k hash values
    for i in range(k):
        yield (hash1 + i * hash2) % m
```

### Bit Array Optimization

- **Blocked Bloom Filters**: Improve cache locality
- **Compression**: Run-length encoding for sparse filters
- **SIMD Operations**: Vectorized bit checking

## Query Execution

### Single Value Lookup

```sql
-- Query: SELECT * FROM orders WHERE order_id = 'ABC123'

Execution:
1. Check Bloom filter for 'ABC123'
2. If not present -> return empty (skip disk I/O)
3. If maybe present -> perform actual lookup
```

### Join Optimization

```sql
-- Query: SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id

Execution:
1. Build Bloom filter from smaller table (customers.id)
2. Pre-filter larger table (orders) using Bloom filter
3. Perform join only on remaining rows
```

## Performance Characteristics

### Time Complexity

| Operation | Time Complexity |
|-----------|----------------|
| Insert | O(k) |
| Lookup | O(k) |
| Build | O(n*k) |
| Merge | O(m) |

Where k = number of hash functions, n = elements, m = bits

### Space Complexity

- **Bloom filter size**: ~1.44 * n * log2(1/p) bits
- **Example**: 1M elements, 1% FPR = ~1.2 MB

## Advanced Variants

### Counting Bloom Filter

Supports deletions by using counters instead of bits:

```python
config={
    "type": "counting",
    "counter_bits": 4  # 4-bit counters
}
```

### Scalable Bloom Filter

Grows dynamically as elements are added:

```python
config={
    "type": "scalable",
    "initial_capacity": 1000,
    "growth_factor": 2,
    "tightening_ratio": 0.9
}
```

### Locality-Sensitive Bloom Filter

Optimized for similar items:

```python
config={
    "type": "lsh",
    "similarity_threshold": 0.8
}
```

## Integration with Other Indices

Bloom filters work well with:

1. **Zone Maps**: Check Bloom filter after zone pruning
2. **BTree**: Use Bloom filter for negative lookups before BTree search
3. **Inverted Index**: Pre-filter documents before text search

## Best Practices

### Sizing Guidelines

```python
def calculate_bloom_size(expected_elements, false_positive_rate):
    if expected_elements < 10000:
        # Small datasets: prioritize accuracy
        return false_positive_rate * 0.1
    elif expected_elements < 1000000:
        # Medium datasets: balance
        return false_positive_rate
    else:
        # Large datasets: prioritize space
        return min(false_positive_rate * 10, 0.1)
```

### Use Case Selection

| Scenario | Use Bloom Filter? | Alternative |
|----------|------------------|-------------|
| Primary key lookup | Yes | - |
| Foreign key validation | Yes | - |
| Range queries | No | Zone Map |
| Full-text search | Maybe | Inverted Index |
| Low cardinality | No | Bitmap Index |

### Maintenance

1. **Monitor false positive rate**: Track actual vs. expected
2. **Rebuild periodically**: After significant data changes
3. **Size appropriately**: Under-sizing increases false positives
4. **Consider workload**: More effective for read-heavy workloads

## Example: E-commerce Application

```python
# Product existence check
dataset.create_index(
    column="sku",
    index_type="bloom_filter",
    config={
        "false_positive_rate": 0.001,  # 0.1% FPR
        "expected_elements": 500000     # 500K products
    }
)

# Query optimization
def check_product_exists(sku):
    # Bloom filter check (microseconds)
    if not bloom_filter.might_contain(sku):
        return False  # Definitely doesn't exist

    # Actual database lookup (milliseconds)
    return database.find_product(sku) is not None

# Result: 99.9% of negative lookups avoid database access
```

## Limitations

1. **No deletions**: Standard Bloom filters don't support removal
2. **No exact counts**: Can't determine exact membership
3. **Fixed size**: Must estimate element count in advance
4. **False positives**: Requires handling of false positive cases
5. **No value retrieval**: Only tests membership, doesn't store values