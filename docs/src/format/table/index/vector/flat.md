# Flat Sub-index

The Flat sub-index provides exhaustive search within IVF partitions, guaranteeing the highest recall at the cost of computational complexity.

## Overview

A Flat sub-index performs brute-force search by:
- Computing exact distances to all vectors in a partition
- No approximation or compression within the partition
- Providing perfect recall within searched partitions
- Trading computation time for accuracy

## Architecture

### Structure

```
IVF Partition
    ↓
[Flat Sub-index]
    ↓
Store all vectors
    ↓
Exhaustive search
    ↓
Exact distances
```

### Storage Layout

```
Flat Sub-index:
  vectors.lance       # Raw vector storage
  metadata.json      # Index configuration
  ids.lance         # Vector IDs mapping
```

## Implementation

### Basic Flat Search

```python
class FlatIndex:
    def __init__(self, metric="L2"):
        self.vectors = []
        self.ids = []
        self.metric = metric

    def add(self, vectors, ids):
        """Add vectors to the flat index"""
        self.vectors.extend(vectors)
        self.ids.extend(ids)

    def search(self, query, k=10):
        """Exhaustive search for k nearest neighbors"""
        if self.metric == "L2":
            distances = np.linalg.norm(self.vectors - query, axis=1)
        elif self.metric == "IP":
            distances = -np.dot(self.vectors, query)
        elif self.metric == "cosine":
            normalized_vectors = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)
            normalized_query = query / np.linalg.norm(query)
            distances = 1 - np.dot(normalized_vectors, normalized_query)

        # Find k nearest
        indices = np.argsort(distances)[:k]
        return [(self.ids[i], distances[i]) for i in indices]
```

### Optimized Distance Computation

```python
def compute_distances_optimized(query, vectors, metric="L2"):
    """
    Optimized distance computation using BLAS
    """
    if metric == "L2":
        # Expand: ||x - y||² = ||x||² + ||y||² - 2⟨x,y⟩
        query_norm = np.sum(query ** 2)
        vector_norms = np.sum(vectors ** 2, axis=1)
        dot_products = vectors @ query
        distances = query_norm + vector_norms - 2 * dot_products
        return np.sqrt(np.maximum(distances, 0))  # Avoid numerical errors

    elif metric == "IP":
        # Inner product (negative for distance)
        return -vectors @ query

    elif metric == "cosine":
        # Cosine similarity
        query_normalized = query / np.linalg.norm(query)
        vectors_normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return 1 - vectors_normalized @ query_normalized
```

## Batch Processing

### Batched Search

```python
class BatchedFlatIndex:
    def __init__(self, batch_size=1000):
        self.batch_size = batch_size
        self.vectors = None
        self.ids = None

    def search_batch(self, queries, k=10):
        """
        Search multiple queries efficiently
        """
        results = []

        # Process queries in batches
        for i in range(0, len(queries), self.batch_size):
            batch_queries = queries[i:i + self.batch_size]

            # Compute all distances at once
            # Shape: (batch_size, num_vectors)
            distances = cdist(batch_queries, self.vectors, metric='euclidean')

            # Find top-k for each query
            for j, dist_row in enumerate(distances):
                indices = np.argsort(dist_row)[:k]
                query_results = [(self.ids[idx], dist_row[idx]) for idx in indices]
                results.append(query_results)

        return results
```

### Memory-Efficient Processing

```python
def search_memory_efficient(query, vectors_file, k=10, chunk_size=10000):
    """
    Process large vector sets without loading all into memory
    """
    min_heap = []

    # Process vectors in chunks
    with open(vectors_file, 'rb') as f:
        offset = 0
        while True:
            # Load chunk
            chunk = load_chunk(f, chunk_size)
            if chunk is None:
                break

            # Compute distances for chunk
            distances = compute_distances(query, chunk)

            # Update top-k
            for i, dist in enumerate(distances):
                if len(min_heap) < k:
                    heapq.heappush(min_heap, (-dist, offset + i))
                elif -dist > min_heap[0][0]:
                    heapq.heapreplace(min_heap, (-dist, offset + i))

            offset += len(chunk)

    # Return sorted results
    return sorted([(-dist, idx) for dist, idx in min_heap])
```

## SIMD Optimization

### AVX2/AVX512 Distance Computation

```python
def simd_l2_distance(query, vectors):
    """
    SIMD-optimized L2 distance using NumPy (which uses BLAS)
    """
    # NumPy automatically uses SIMD when available
    diff = vectors - query
    distances = np.einsum('ij,ij->i', diff, diff)
    return np.sqrt(distances)

# For explicit SIMD control, use libraries like numba
from numba import njit, prange

@njit(parallel=True)
def simd_dot_product(query, vectors):
    n = len(vectors)
    distances = np.empty(n)

    for i in prange(n):
        distances[i] = np.dot(query, vectors[i])

    return distances
```

## GPU Acceleration

### CUDA Implementation

```python
def gpu_flat_search(query, vectors, k=10):
    """
    GPU-accelerated flat search using CuPy
    """
    import cupy as cp

    # Transfer to GPU
    query_gpu = cp.asarray(query)
    vectors_gpu = cp.asarray(vectors)

    # Compute distances on GPU
    if len(query.shape) == 1:
        distances = cp.linalg.norm(vectors_gpu - query_gpu, axis=1)
    else:
        # Batch query
        distances = cp.linalg.norm(
            vectors_gpu[None, :, :] - query_gpu[:, None, :],
            axis=2
        )

    # Find top-k (still on GPU)
    if k < 1024:  # Small k: partial sort
        indices = cp.argpartition(distances, k, axis=-1)[..., :k]
        indices = cp.take_along_axis(
            indices,
            cp.argsort(cp.take_along_axis(distances, indices, axis=-1), axis=-1),
            axis=-1
        )
    else:  # Large k: full sort
        indices = cp.argsort(distances, axis=-1)[..., :k]

    # Transfer results back to CPU
    return indices.get(), distances.get()
```

### Multi-GPU Search

```python
def multi_gpu_search(query, vectors, k=10, num_gpus=4):
    """
    Distribute search across multiple GPUs
    """
    import cupy as cp
    from concurrent.futures import ThreadPoolExecutor

    chunk_size = len(vectors) // num_gpus
    chunks = [vectors[i:i+chunk_size] for i in range(0, len(vectors), chunk_size)]

    def search_on_gpu(gpu_id, chunk):
        with cp.cuda.Device(gpu_id):
            return gpu_flat_search(query, chunk, k * 2)

    # Parallel GPU execution
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [
            executor.submit(search_on_gpu, i, chunk)
            for i, chunk in enumerate(chunks)
        ]
        results = [f.result() for f in futures]

    # Merge results
    all_indices = np.concatenate([r[0] for r in results])
    all_distances = np.concatenate([r[1] for r in results])

    # Final top-k
    top_k_indices = np.argsort(all_distances)[:k]
    return all_indices[top_k_indices], all_distances[top_k_indices]
```

## Integration with IVF

### IVF_FLAT Configuration

```python
class IVF_FLAT:
    def __init__(self, num_partitions=256, metric="L2"):
        self.ivf = IVF(num_partitions)
        self.flat_indices = [FlatIndex(metric) for _ in range(num_partitions)]
        self.metric = metric

    def build(self, vectors, ids):
        # Train IVF
        self.ivf.train(vectors)

        # Assign vectors to partitions
        assignments = self.ivf.assign(vectors)

        # Build flat indices for each partition
        for partition_id in range(self.ivf.num_partitions):
            mask = assignments == partition_id
            partition_vectors = vectors[mask]
            partition_ids = ids[mask]
            self.flat_indices[partition_id].add(partition_vectors, partition_ids)

    def search(self, query, k=10, nprobes=10):
        # Find nearest partitions
        nearest_partitions = self.ivf.find_nearest_partitions(query, nprobes)

        # Search within partitions
        candidates = []
        for partition_id in nearest_partitions:
            partition_results = self.flat_indices[partition_id].search(query, k)
            candidates.extend(partition_results)

        # Sort and return top-k
        candidates.sort(key=lambda x: x[1])
        return candidates[:k]
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Build | O(n) | Just storing vectors |
| Search (single) | O(n*d) | n vectors, d dimensions |
| Search (batch) | O(b*n*d) | b queries |
| Memory | O(n*d) | Direct storage |

### Optimization Strategies

```python
def optimize_flat_search(vectors, query_patterns):
    """
    Optimize flat index based on usage patterns
    """
    optimizations = {}

    # Check vector dimensions
    n, d = vectors.shape

    if d < 128:
        optimizations["use_simd"] = True
        optimizations["batch_size"] = 1000
    else:
        optimizations["use_blas"] = True
        optimizations["batch_size"] = 100

    # Check available hardware
    if cuda_available():
        optimizations["use_gpu"] = True
        optimizations["gpu_batch_size"] = 10000

    # Memory considerations
    memory_required = n * d * 4  # float32
    if memory_required > available_memory() * 0.5:
        optimizations["use_mmap"] = True
        optimizations["chunk_size"] = 10000

    return optimizations
```

## Caching Strategies

### Distance Cache

```python
class CachedFlatIndex:
    def __init__(self, cache_size=10000):
        self.flat_index = FlatIndex()
        self.cache = LRUCache(cache_size)

    def search(self, query, k=10):
        # Check cache
        query_hash = hash(query.tobytes())
        if query_hash in self.cache:
            return self.cache[query_hash][:k]

        # Compute and cache
        results = self.flat_index.search(query, k * 2)
        self.cache[query_hash] = results

        return results[:k]
```

### Pre-computed Tables

```python
class PrecomputedFlatIndex:
    def __init__(self, vectors):
        self.vectors = vectors
        # Pre-compute norms for L2 distance
        self.vector_norms = np.linalg.norm(vectors, axis=1) ** 2

    def search_l2(self, query, k=10):
        # Use pre-computed norms
        query_norm = np.linalg.norm(query) ** 2
        dot_products = self.vectors @ query
        distances = self.vector_norms + query_norm - 2 * dot_products
        distances = np.sqrt(np.maximum(distances, 0))

        indices = np.argsort(distances)[:k]
        return indices, distances[indices]
```

## Best Practices

### When to Use Flat Sub-index

```python
def should_use_flat(partition_size, dimension, recall_requirement):
    """
    Determine if flat index is appropriate
    """
    if recall_requirement >= 0.99:
        return True  # Need perfect recall within partitions

    if partition_size < 1000:
        return True  # Small enough for brute force

    if dimension > 1024 and partition_size < 10000:
        return True  # High dimension benefits from exact search

    return False
```

### Optimization Checklist

1. **Use BLAS**: Leverage optimized linear algebra libraries
2. **Batch queries**: Process multiple queries together
3. **Memory layout**: Ensure contiguous memory access
4. **Parallelization**: Use multiple cores/GPUs
5. **Caching**: Cache frequent queries or pre-compute distances

## Example: Production Configuration

```python
class ProductionFlatIndex:
    def __init__(self, config):
        self.config = config
        self.setup_optimization()

    def setup_optimization(self):
        # Choose backend based on hardware
        if self.config.get("gpu_available"):
            self.backend = "gpu"
            self.init_gpu()
        elif self.config.get("vector_dim") < 128:
            self.backend = "simd"
            self.init_simd()
        else:
            self.backend = "blas"
            self.init_blas()

    def search(self, query, k=10):
        if self.backend == "gpu":
            return self.gpu_search(query, k)
        elif self.backend == "simd":
            return self.simd_search(query, k)
        else:
            return self.blas_search(query, k)

# Usage
config = {
    "gpu_available": torch.cuda.is_available(),
    "vector_dim": 768,
    "partition_size": 50000
}

index = ProductionFlatIndex(config)
results = index.search(query_vector, k=10)
```

## Limitations

1. **Computational Cost**: O(n) complexity per query
2. **No Compression**: Full vector storage required
3. **Memory Bandwidth**: Limited by memory transfer speed
4. **Scalability**: Not suitable for very large partitions
5. **No Incremental Updates**: Append-only structure