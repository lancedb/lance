# IVF (Inverted File Index)

IVF is the partitioning layer in Lance's vector index architecture that divides the vector space into regions using clustering, enabling efficient approximate nearest neighbor search.

## Overview

IVF works by:
1. Clustering the vector dataset into partitions (Voronoi cells)
2. Assigning each vector to its nearest cluster centroid
3. During search, examining only the most promising partitions

This reduces the search space from millions of vectors to just thousands, providing orders of magnitude speedup.

## Architecture

### Training Phase

```
Training Vectors (sample)
         ↓
    [K-means Clustering]
         ↓
    Cluster Centroids
         ↓
    IVF Structure
```

### Indexing Phase

```
Input Vector → Find Nearest Centroid → Assign to Partition → Store in Inverted List
```

### Query Phase

```
Query Vector → Find Top-k Centroids → Search Partitions → Merge Results → Return Top-k
```

## Clustering Algorithms

### K-means Clustering

Standard IVF uses k-means to create partitions:

```python
def train_ivf_kmeans(vectors, num_partitions, max_iters=20):
    # Initialize centroids using k-means++
    centroids = kmeans_plusplus_init(vectors, num_partitions)

    for iteration in range(max_iters):
        # Assign vectors to nearest centroids
        assignments = assign_to_nearest(vectors, centroids)

        # Update centroids
        new_centroids = compute_centroids(vectors, assignments)

        # Check convergence
        if converged(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids
```

### Hierarchical K-means

For large numbers of partitions:

```python
def train_hierarchical_kmeans(vectors, num_partitions):
    # Two-level clustering
    level1_partitions = int(sqrt(num_partitions))
    level2_partitions = num_partitions // level1_partitions

    # First level clustering
    level1_centroids = train_kmeans(vectors, level1_partitions)

    # Second level clustering within each partition
    level2_centroids = []
    for partition in level1_partitions:
        partition_vectors = get_partition_vectors(vectors, partition)
        centroids = train_kmeans(partition_vectors, level2_partitions)
        level2_centroids.extend(centroids)

    return level1_centroids, level2_centroids
```

## Configuration Parameters

### Number of Partitions

```python
def recommend_num_partitions(num_vectors):
    """
    Common heuristic: sqrt(n) to 4*sqrt(n) partitions
    """
    if num_vectors < 10_000:
        return 32  # Minimum partitions

    elif num_vectors < 1_000_000:
        return int(sqrt(num_vectors))

    elif num_vectors < 100_000_000:
        return int(2 * sqrt(num_vectors))

    else:
        return min(65536, int(4 * sqrt(num_vectors)))
```

### Partition Size Distribution

Monitoring partition balance is crucial:

```python
def analyze_partition_distribution(ivf_index):
    partition_sizes = ivf_index.get_partition_sizes()

    stats = {
        "mean": np.mean(partition_sizes),
        "std": np.std(partition_sizes),
        "min": np.min(partition_sizes),
        "max": np.max(partition_sizes),
        "imbalance_ratio": np.max(partition_sizes) / np.mean(partition_sizes)
    }

    # Warn if partitions are highly imbalanced
    if stats["imbalance_ratio"] > 3.0:
        print("Warning: Partition imbalance detected")

    return stats
```

## Search Strategies

### Basic Multi-probe

Search multiple partitions to improve recall:

```python
def ivf_search(query, centroids, inverted_lists, nprobes=10, k=10):
    # Find nearest centroids
    centroid_distances = compute_distances(query, centroids)
    nearest_partitions = np.argsort(centroid_distances)[:nprobes]

    # Search within selected partitions
    candidates = []
    for partition_id in nearest_partitions:
        partition_vectors = inverted_lists[partition_id]
        distances = compute_distances(query, partition_vectors)
        candidates.extend(zip(distances, partition_vectors))

    # Return top-k results
    candidates.sort(key=lambda x: x[0])
    return candidates[:k]
```

### Adaptive Probing

Dynamically adjust nprobes based on partition distances:

```python
def adaptive_probe(query, centroids, distance_threshold):
    """
    Probe partitions within a distance threshold
    """
    centroid_distances = compute_distances(query, centroids)

    # Include all partitions within threshold
    selected = centroid_distances <= distance_threshold

    # Ensure minimum probes
    if selected.sum() < MIN_PROBES:
        selected = np.argsort(centroid_distances)[:MIN_PROBES]

    return selected
```

## Optimizations

### Centroid Refinement

Periodically refine centroids for better clustering:

```python
def refine_centroids(ivf_index, sample_rate=0.1):
    # Sample vectors from each partition
    samples = []
    for partition in ivf_index.partitions:
        partition_samples = partition.sample(sample_rate)
        samples.extend(partition_samples)

    # Re-cluster with current assignments as initialization
    new_centroids = train_kmeans(
        samples,
        num_partitions=len(ivf_index.centroids),
        init=ivf_index.centroids  # Warm start
    )

    return new_centroids
```

### GPU Acceleration

Leverage GPU for centroid computation:

```python
def gpu_find_nearest_centroids(queries, centroids, nprobes):
    """
    Batch computation of nearest centroids on GPU
    """
    import cupy as cp

    # Transfer to GPU
    queries_gpu = cp.asarray(queries)
    centroids_gpu = cp.asarray(centroids)

    # Compute distances on GPU
    distances = cp.linalg.norm(
        queries_gpu[:, None, :] - centroids_gpu[None, :, :],
        axis=2
    )

    # Find top-k partitions for each query
    nearest = cp.argsort(distances, axis=1)[:, :nprobes]

    return nearest.get()  # Transfer back to CPU
```

### Pre-computed Tables

Store pre-computed distance tables for common metrics:

```python
class PrecomputedIVF:
    def __init__(self, centroids, metric="L2"):
        self.centroids = centroids
        self.metric = metric

        if metric == "IP":  # Inner product
            # Pre-compute centroid norms for optimization
            self.centroid_norms = np.linalg.norm(centroids, axis=1) ** 2

    def find_nearest_partitions(self, query, nprobes):
        if self.metric == "IP":
            # Use pre-computed norms
            distances = -2 * query @ self.centroids.T
            distances += self.centroid_norms
        else:
            distances = compute_distances(query, self.centroids)

        return np.argsort(distances)[:nprobes]
```

## Storage Format

### IVF Metadata

```json
{
    "index_type": "IVF",
    "num_partitions": 1024,
    "vector_dim": 768,
    "metric": "L2",
    "training_samples": 100000,
    "centroid_file": "centroids.lance",
    "inverted_lists": "partitions/"
}
```

### Centroid Storage

```
centroids.lance:
  Schema:
    - partition_id: int32
    - centroid_vector: fixed_size_list<float32>[768]
    - partition_size: int32
    - radius: float32  # Max distance from centroid
```

### Inverted Lists

```
partitions/
  partition_0000.lance
  partition_0001.lance
  ...

Each partition file:
  Schema:
    - vector_id: int64
    - vector: fixed_size_list<float32>[768]
    - metadata: struct<...>  # Optional metadata
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Training | O(n*k*d*i) | i iterations, k partitions, d dimensions |
| Indexing | O(n*k*d) | Assign n vectors to k partitions |
| Search | O(nprobes*m*d + k*log(k)) | m vectors per partition |

### Space Complexity

- **Centroids**: O(k*d) where k = partitions, d = dimensions
- **Inverted lists**: O(n) pointers + vector storage
- **Metadata**: O(k) partition statistics

## Best Practices

### Training Data Selection

```python
def select_training_data(vectors, target_size=100000):
    """
    Select representative training data for IVF
    """
    if len(vectors) <= target_size:
        return vectors

    # Random sampling with seed for reproducibility
    np.random.seed(42)
    indices = np.random.choice(len(vectors), target_size, replace=False)
    return vectors[indices]
```

### Partition Balance

Monitor and maintain balanced partitions:

```python
def rebalance_partitions(ivf_index, max_imbalance=3.0):
    """
    Rebalance partitions if they become too imbalanced
    """
    sizes = ivf_index.get_partition_sizes()
    mean_size = np.mean(sizes)
    max_size = np.max(sizes)

    if max_size / mean_size > max_imbalance:
        # Split large partitions
        for partition_id, size in enumerate(sizes):
            if size > max_imbalance * mean_size:
                ivf_index.split_partition(partition_id)

        # Merge small partitions
        for partition_id, size in enumerate(sizes):
            if size < mean_size / max_imbalance:
                ivf_index.merge_partition(partition_id)
```

### Query Optimization

```python
def optimize_nprobes(ivf_index, target_recall=0.95):
    """
    Find optimal nprobes for target recall
    """
    test_queries = ivf_index.get_test_queries()
    ground_truth = ivf_index.get_ground_truth()

    for nprobes in [1, 5, 10, 20, 40, 80, 160]:
        recall = evaluate_recall(ivf_index, test_queries, ground_truth, nprobes)

        if recall >= target_recall:
            return nprobes

    return min(160, ivf_index.num_partitions)
```

## Integration with Quantization

IVF commonly pairs with quantization methods:

### IVF + PQ

```python
class IVF_PQ:
    def __init__(self, num_partitions, num_sub_vectors):
        self.ivf = IVF(num_partitions)
        self.pq = ProductQuantizer(num_sub_vectors)

    def train(self, vectors):
        # Train IVF centroids
        self.ivf.train(vectors)

        # Train PQ within each partition
        for partition_id in range(self.ivf.num_partitions):
            partition_vectors = self.ivf.get_partition_vectors(partition_id)
            self.pq.train(partition_vectors)
```

### IVF + SQ

```python
class IVF_SQ:
    def __init__(self, num_partitions, num_bits=8):
        self.ivf = IVF(num_partitions)
        self.sq = ScalarQuantizer(num_bits)

    def encode(self, vectors):
        # Assign to partitions
        assignments = self.ivf.assign(vectors)

        # Quantize within partitions
        quantized = []
        for partition_id, partition_vectors in group_by_partition(vectors, assignments):
            quantized.extend(self.sq.encode(partition_vectors))

        return quantized
```

## Limitations and Considerations

1. **Training Quality**: Poor clustering leads to imbalanced partitions
2. **Curse of Dimensionality**: Less effective in very high dimensions
3. **Update Cost**: Adding vectors may require rebalancing
4. **Memory Overhead**: Stores centroids and inverted lists
5. **Query Latency**: Increases linearly with nprobes