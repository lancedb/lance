# Vector Indices

Lance provides a sophisticated multi-layer vector indexing system optimized for approximate nearest neighbor (ANN) search at scale.

## Overview

Vector indices in Lance use a hierarchical architecture with three main layers:

1. **Partitioning Layer (IVF)** - Divides the vector space into regions
2. **Quantization Layer** - Compresses vectors to reduce memory and improve speed
3. **Sub-index Layer** - Provides fine-grained search within partitions

This layered approach enables efficient search across billions of vectors while maintaining high recall and low latency.

## Architecture

### Three-Layer Design

```
Query Vector
     ↓
[IVF Partitioning]
     ↓
Select top-k partitions
     ↓
[Quantization]
     ↓
Approximate distances
     ↓
[Sub-index Search]
     ↓
Final candidates
     ↓
[Re-ranking]
     ↓
Results
```

### Components

1. **[IVF (Inverted File Index)](ivf.md)** - Coarse quantization for partitioning
2. **Quantization Methods**:
   - **[Product Quantization (PQ)](pq.md)** - Decompose vectors into subspaces
   - **[Scalar Quantization (SQ)](sq.md)** - Reduce precision of vector components
3. **Sub-index Types**:
   - **[HNSW (Hierarchical Navigable Small World)](hnsw.md)** - Graph-based search
   - **[Flat Index](flat.md)** - Exhaustive search within partitions

## Index Types and Configurations

### IVF_FLAT
Simple IVF with no compression:

```python
dataset.create_index(
    column="embedding",
    index_type="IVF_FLAT",
    metric="L2",
    num_partitions=256,
    num_sub_vectors=16
)
```

### IVF_PQ
IVF with product quantization:

```python
dataset.create_index(
    column="embedding",
    index_type="IVF_PQ",
    metric="cosine",
    num_partitions=1024,
    num_sub_vectors=32,
    num_bits=8
)
```

### IVF_SQ
IVF with scalar quantization:

```python
dataset.create_index(
    column="embedding",
    index_type="IVF_SQ",
    metric="L2",
    num_partitions=512,
    num_bits=8  # 8-bit scalar quantization
)
```

### IVF_HNSW
IVF with HNSW sub-indices:

```python
dataset.create_index(
    column="embedding",
    index_type="IVF_HNSW",
    metric="cosine",
    num_partitions=256,
    max_edges=32,
    ef_construction=200
)
```

### IVF_PQ_HNSW
Full three-layer configuration:

```python
dataset.create_index(
    column="embedding",
    index_type="IVF_PQ_HNSW",
    metric="L2",
    num_partitions=1024,
    num_sub_vectors=16,
    num_bits=8,
    max_edges=32,
    ef_construction=200
)
```

## Distance Metrics

Lance supports multiple distance metrics:

| Metric | Formula | Use Case |
|--------|---------|----------|
| L2 (Euclidean) | √Σ(x_i - y_i)² | General similarity |
| Cosine | 1 - (x·y)/(‖x‖‖y‖) | Text embeddings |
| Dot Product | -Σ(x_i * y_i) | Maximum inner product search |

## Query Processing

### Search Parameters

```python
results = dataset.search(
    query_vector,
    k=10,                    # Number of results
    nprobes=20,             # Partitions to search
    refine_factor=10,       # Candidates for re-ranking
    ef_search=100          # HNSW search parameter
)
```

### Multi-stage Search Process

1. **Partition Selection**
   - Find nearest IVF centroids
   - Select top nprobes partitions

2. **Candidate Generation**
   - Search within selected partitions
   - Use quantized distances for speed

3. **Re-ranking**
   - Compute exact distances for candidates
   - Return top-k results

## Performance Optimization

### Index Size vs. Recall Trade-offs

| Configuration | Index Size | Build Time | Query Speed | Recall@10 |
|--------------|------------|------------|-------------|-----------|
| IVF_FLAT | 100% | Fast | Medium | 95-99% |
| IVF_SQ8 | 25% | Fast | Fast | 90-95% |
| IVF_PQ16 | 6% | Medium | Fast | 85-95% |
| IVF_HNSW | 110% | Slow | Very Fast | 97-99% |
| IVF_PQ_HNSW | 15% | Slow | Fast | 90-97% |

### Tuning Guidelines

```python
def recommend_index_config(num_vectors, vector_dim, memory_budget):
    if num_vectors < 1_000_000:
        # Small dataset: prioritize recall
        return "IVF_FLAT"

    elif memory_budget == "unlimited":
        # Best recall with fast search
        return "IVF_HNSW"

    elif memory_budget == "limited":
        # Balance compression and recall
        if vector_dim > 256:
            return "IVF_PQ"  # Better for high dimensions
        else:
            return "IVF_SQ"  # Better for low dimensions

    else:  # memory_budget == "minimal"
        # Maximum compression
        return "IVF_PQ_HNSW"
```

## Storage Format

```
Vector Index Directory:
  metadata.json           -- Index configuration
  ivf/
    centroids.lance      -- IVF cluster centers
    assignments.lance    -- Vector to partition mapping
  quantizer/
    codebook.lance      -- Quantization codebook
    codes.lance         -- Quantized vectors
  subindex/
    partition_0/        -- Sub-index for partition 0
    partition_1/        -- Sub-index for partition 1
    ...
```

## Building Indices

### Training Phase

```python
# Sample vectors for training
training_vectors = dataset.sample(n=100000)

# Train index components
index = VectorIndex()
index.train_ivf(training_vectors, num_partitions=1024)
index.train_quantizer(training_vectors, num_sub_vectors=16)
```

### Indexing Phase

```python
# Add vectors to index
for batch in dataset.iter_batches():
    index.add(batch.vectors, batch.ids)

# Build sub-indices
index.build_subindices(index_type="hnsw")
```

## Incremental Updates

Lance supports incremental index updates:

```python
# Append new vectors
new_vectors = load_new_data()
dataset.append(new_vectors)

# Update index incrementally
dataset.update_index(
    column="embedding",
    mode="append"  # or "rebuild" for full reconstruction
)
```

## GPU Acceleration

Lance can leverage GPUs for index operations:

```python
dataset.create_index(
    column="embedding",
    index_type="IVF_PQ",
    accelerator="cuda",
    gpu_device=0
)

# GPU-accelerated search
results = dataset.search(
    query_vector,
    use_gpu=True
)
```

## Monitoring and Diagnostics

### Index Statistics

```python
stats = dataset.index_stats("embedding")
print(f"Partitions: {stats.num_partitions}")
print(f"Vectors: {stats.num_vectors}")
print(f"Index size: {stats.size_bytes / 1e9:.2f} GB")
print(f"Compression ratio: {stats.compression_ratio:.2f}")
```

### Search Quality Metrics

```python
def evaluate_index(dataset, test_queries, ground_truth):
    recalls = []
    latencies = []

    for query, true_neighbors in zip(test_queries, ground_truth):
        start = time.time()
        results = dataset.search(query, k=10)
        latency = time.time() - start

        recall = len(set(results) & set(true_neighbors[:10])) / 10
        recalls.append(recall)
        latencies.append(latency)

    return {
        "recall@10": np.mean(recalls),
        "p50_latency": np.percentile(latencies, 50),
        "p99_latency": np.percentile(latencies, 99)
    }
```

## Best Practices

### Index Selection

1. **Dataset Size**
   - < 1M vectors: IVF_FLAT or HNSW
   - 1M-10M vectors: IVF_SQ or IVF_PQ
   - > 10M vectors: IVF_PQ_HNSW

2. **Vector Dimensions**
   - < 128 dims: Scalar quantization works well
   - > 256 dims: Product quantization more effective
   - > 1024 dims: Consider dimensionality reduction first

3. **Query Requirements**
   - High recall (>95%): IVF_FLAT or IVF_HNSW
   - Low latency (<10ms): IVF_SQ with pre-computed distances
   - Memory constrained: IVF_PQ with aggressive compression

### Training Data Selection

```python
def select_training_data(dataset, target_size=100000):
    if len(dataset) <= target_size:
        return dataset

    # Stratified sampling for representative training
    return dataset.sample(
        n=target_size,
        strategy="stratified",  # Maintain data distribution
        random_state=42
    )
```

### Parameter Tuning

```python
# Grid search for optimal parameters
param_grid = {
    "num_partitions": [256, 512, 1024],
    "nprobes": [10, 20, 40],
    "num_sub_vectors": [8, 16, 32]
}

best_config = optimize_index_params(
    dataset,
    param_grid,
    metric="recall@10",
    constraint="latency < 20ms"
)
```

## Future Enhancements

- **Learned Indices**: Neural network-based index structures
- **Hybrid Indices**: Combining multiple index types dynamically
- **Streaming Indices**: Real-time index updates without rebuilding
- **Multi-vector Search**: Searching with multiple query vectors simultaneously
- **Filtered Search**: Efficient nearest neighbor search with metadata filters