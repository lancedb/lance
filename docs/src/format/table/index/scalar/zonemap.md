# Zone Map Index

Zone maps (also known as min-max indices or block range indices) store statistical metadata about data zones to enable efficient scan pruning.

## Overview

A zone map divides data into zones (blocks or pages) and maintains statistics for each zone:
- Minimum and maximum values
- Null count
- Distinct count (optional)
- Other aggregates (sum, mean, etc.)

These statistics allow the query engine to skip entire zones that cannot contain matching data.

## Use Cases

Zone maps are effective for:
- **Range queries**: Skip zones outside the query range
- **Time-series data**: Natural ordering enables efficient pruning
- **Sorted/clustered data**: Maximum pruning efficiency
- **Large scans**: Reduce I/O by skipping irrelevant zones
- **Aggregation queries**: Use pre-computed statistics

## Structure

### Zone Organization

```
Data File:
  Zone 0: [Rows 0-999]     -> Min: 100, Max: 500
  Zone 1: [Rows 1000-1999] -> Min: 450, Max: 900
  Zone 2: [Rows 2000-2999] -> Min: 50,  Max: 150
  Zone 3: [Rows 3000-3999] -> Min: 800, Max: 950

Query: WHERE value BETWEEN 200 AND 400
Result: Only scan Zone 0 (Zones 1, 2, 3 can be skipped)
```

### Storage Format

```
Zone Map Index:
  Header:
    - Index version
    - Zone size (rows per zone)
    - Number of zones
    - Statistics types

  Zone Statistics:
    - Zone ID
    - Row range [start, end]
    - Min value
    - Max value
    - Null count
    - Additional stats (optional)

  Auxiliary:
    - Bloom filters per zone (optional)
    - Histograms (optional)
```

## Statistics Types

### Basic Statistics
- **Min/Max**: Bounds for numeric and string values
- **Null count**: Number of null values in zone
- **Row count**: Total rows in zone

### Extended Statistics
- **Distinct count**: Cardinality estimation
- **Sum/Mean**: For aggregation pushdown
- **Bloom filter**: Probabilistic membership test
- **Histogram**: Value distribution

## Query Optimization

### Range Query Pruning

```sql
-- Query: SELECT * FROM table WHERE timestamp BETWEEN '2024-01-01' AND '2024-01-31'

Zone pruning logic:
1. For each zone:
   if (zone.max < '2024-01-01' OR zone.min > '2024-01-31'):
     skip zone
   else:
     scan zone
```

### Predicate Combinations

```sql
-- Query: WHERE price > 100 AND category = 'electronics'

Combined pruning:
1. Skip zones where price.max <= 100
2. Skip zones where category bloom filter doesn't contain 'electronics'
3. Scan remaining zones
```

## Configuration

```python
dataset.create_index(
    column="timestamp",
    index_type="zonemap",
    config={
        "zone_size": 10000,        # Rows per zone
        "statistics": ["min", "max", "null_count", "bloom"],
        "bloom_fpp": 0.01,         # Bloom filter false positive probability
        "clustering_hint": "timestamp"  # Optimize zones for this column
    }
)
```

## Performance Characteristics

### Effectiveness Factors

| Factor | Impact on Pruning |
|--------|-------------------|
| Data clustering | High clustering = better pruning |
| Zone size | Smaller zones = finer granularity |
| Query selectivity | More selective = more pruning |
| Column correlation | Correlated columns enhance pruning |

### Space Overhead

- **Basic stats**: ~16 bytes per zone per column
- **With bloom filter**: +~1KB per zone
- **Total overhead**: Typically < 0.1% of data size

## Optimizations

### Dynamic Zone Sizing

Adjust zone size based on data characteristics:

```python
if high_selectivity_queries:
    zone_size = 1000  # Smaller zones for fine-grained pruning
elif mostly_sequential_scans:
    zone_size = 100000  # Larger zones for less overhead
```

### Multi-Column Zone Maps

Maintain correlated statistics for column groups:

```
Zone 0:
  (timestamp, user_id) -> Min: (2024-01-01, 1), Max: (2024-01-02, 1000)
  Enables pruning on compound predicates
```

### Adaptive Statistics

Choose statistics based on column type and query patterns:

| Column Type | Recommended Statistics |
|-------------|----------------------|
| Timestamp | Min, Max, Null count |
| ID columns | Min, Max, Bloom filter |
| Categories | Bloom filter, Distinct count |
| Metrics | Min, Max, Sum, Mean |

## Integration with Other Indices

Zone maps complement other index types:

1. **With BTree**: Zone maps for initial pruning, BTree for precise lookup
2. **With Bitmap**: Zone maps for range pruning, Bitmap for equality
3. **With Inverted**: Zone maps for document filtering, Inverted for text search

## Best Practices

### Data Organization

1. **Sort by query columns**: Maximize zone pruning efficiency
2. **Cluster related data**: Keep correlated values in same zones
3. **Regular reorganization**: Maintain clustering over time

### Zone Size Selection

```python
# Guidelines for zone size
def recommend_zone_size(table_stats):
    if table_stats.rows < 1_000_000:
        return 10_000  # 10K rows per zone
    elif table_stats.rows < 100_000_000:
        return 50_000  # 50K rows per zone
    else:
        return 100_000  # 100K rows per zone
```

### Monitoring

Track zone map effectiveness:
- Zones scanned vs. zones skipped ratio
- I/O reduction percentage
- Query latency improvement

## Example: Time-Series Data

```python
# Create zone map for time-series data
dataset.create_index(
    column="timestamp",
    index_type="zonemap",
    config={
        "zone_size": 3600,  # 1 hour of data per zone (assuming 1 row/second)
        "statistics": ["min", "max", "null_count"]
    }
)

# Query benefits from zone pruning
query = """
    SELECT avg(temperature)
    FROM sensor_data
    WHERE timestamp BETWEEN '2024-01-15 00:00:00' AND '2024-01-15 01:00:00'
"""
# Only scans 1-2 zones instead of entire dataset
```

## Limitations

1. **Ineffective on random data**: No pruning benefit without clustering
2. **Update overhead**: Statistics need recalculation
3. **Not suitable for**: Point queries, highly selective filters
4. **Space overhead**: For many columns with extended statistics