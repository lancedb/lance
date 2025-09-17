# Product Quantization (PQ)

Product Quantization is a vector compression technique in Lance that decomposes high-dimensional vectors into subspaces and quantizes each independently, achieving significant memory reduction while maintaining search quality.

## Overview

PQ works by:
1. Splitting vectors into m subvectors
2. Learning a codebook for each subspace
3. Encoding each subvector with its nearest codeword
4. Storing only the code indices instead of full vectors

This reduces storage from d*32 bits to m*log2(k) bits per vector, where k is the codebook size.

## Algorithm

### Training Phase

```python
def train_product_quantizer(vectors, num_subvectors=8, codebook_size=256):
    """
    Train PQ codebooks using k-means clustering
    """
    n, d = vectors.shape
    subvector_dim = d // num_subvectors
    codebooks = []

    for m in range(num_subvectors):
        # Extract subvectors for this subspace
        start_idx = m * subvector_dim
        end_idx = (m + 1) * subvector_dim
        subvectors = vectors[:, start_idx:end_idx]

        # Train k-means for this subspace
        kmeans = KMeans(n_clusters=codebook_size)
        kmeans.fit(subvectors)

        # Store codebook (cluster centers)
        codebooks.append(kmeans.cluster_centers_)

    return codebooks
```

### Encoding Phase

```python
def encode_vectors(vectors, codebooks):
    """
    Encode vectors using trained codebooks
    """
    n, d = vectors.shape
    num_subvectors = len(codebooks)
    subvector_dim = d // num_subvectors
    codes = np.zeros((n, num_subvectors), dtype=np.uint8)

    for m in range(num_subvectors):
        # Extract subvectors
        start_idx = m * subvector_dim
        end_idx = (m + 1) * subvector_dim
        subvectors = vectors[:, start_idx:end_idx]

        # Find nearest codeword for each subvector
        distances = cdist(subvectors, codebooks[m])
        codes[:, m] = np.argmin(distances, axis=1)

    return codes
```

### Query Processing

```python
def compute_pq_distances(query, codes, codebooks):
    """
    Compute approximate distances using PQ codes
    """
    num_subvectors = len(codebooks)
    subvector_dim = len(query) // num_subvectors

    # Pre-compute distance tables
    distance_tables = []
    for m in range(num_subvectors):
        start_idx = m * subvector_dim
        end_idx = (m + 1) * subvector_dim
        query_subvector = query[start_idx:end_idx]

        # Distance from query subvector to all codewords
        table = np.linalg.norm(
            codebooks[m] - query_subvector,
            axis=1
        ) ** 2
        distance_tables.append(table)

    # Look up distances using codes
    n = len(codes)
    distances = np.zeros(n)
    for i in range(n):
        for m in range(num_subvectors):
            distances[i] += distance_tables[m][codes[i, m]]

    return np.sqrt(distances)
```

## Optimizations

### Optimized Product Quantization (OPQ)

Rotate data to minimize quantization error:

```python
class OptimizedPQ:
    def __init__(self, num_subvectors=8, codebook_size=256):
        self.num_subvectors = num_subvectors
        self.codebook_size = codebook_size
        self.rotation_matrix = None
        self.codebooks = None

    def train(self, vectors, num_iterations=10):
        """
        Alternating optimization of rotation and codebooks
        """
        n, d = vectors.shape

        # Initialize with PCA rotation
        self.rotation_matrix = self.init_rotation_pca(vectors)

        for iteration in range(num_iterations):
            # Rotate vectors
            rotated_vectors = vectors @ self.rotation_matrix

            # Update codebooks
            self.codebooks = train_product_quantizer(
                rotated_vectors,
                self.num_subvectors,
                self.codebook_size
            )

            # Update rotation
            self.rotation_matrix = self.optimize_rotation(
                vectors,
                self.codebooks
            )

            # Compute reconstruction error
            error = self.compute_error(vectors)
            print(f"Iteration {iteration}: Error = {error}")

    def optimize_rotation(self, vectors, codebooks):
        """
        Find rotation that minimizes quantization error
        """
        # Encode with current codebooks
        codes = self.encode(vectors @ self.rotation_matrix)
        reconstructed = self.decode(codes)

        # Procrustes problem: find R minimizing ||X*R - Y||
        u, _, vt = np.linalg.svd(vectors.T @ reconstructed)
        return u @ vt
```

### Polysemous Codes

Use codes with multiple interpretations for faster search:

```python
class PolysemousPQ:
    def __init__(self, base_pq, hamming_threshold=2):
        self.pq = base_pq
        self.hamming_threshold = hamming_threshold

    def search_approximate(self, query, codes, k):
        """
        Fast search using Hamming distance on codes
        """
        # Encode query
        query_code = self.pq.encode(query.reshape(1, -1))[0]

        # Compute Hamming distances
        hamming_distances = np.sum(codes != query_code, axis=1)

        # Filter by Hamming threshold
        candidates = np.where(hamming_distances <= self.hamming_threshold)[0]

        # Compute exact PQ distances for candidates
        if len(candidates) > 0:
            pq_distances = self.pq.compute_distances(query, codes[candidates])
            top_k = candidates[np.argsort(pq_distances)[:k]]
            return top_k
        else:
            return np.array([])
```

## Configuration

### PQ Parameters

```python
dataset.create_index(
    column="embedding",
    index_type="IVF_PQ",
    config={
        "num_subvectors": 16,      # Number of subspaces
        "codebook_size": 256,       # Codes per subspace (8-bit)
        "num_training_samples": 100000,  # Samples for training
        "opq": True,                # Use Optimized PQ
        "polysemous": False         # Use Polysemous codes
    }
)
```

### Parameter Selection

```python
def select_pq_parameters(vector_dim, memory_budget_ratio=0.1):
    """
    Select PQ parameters based on vector dimension and memory budget
    """
    # Memory budget as ratio of original size
    bits_per_vector = vector_dim * 32 * memory_budget_ratio

    if bits_per_vector >= 128:
        # Can afford 16 bytes per vector
        num_subvectors = 16
        codebook_bits = 8
    elif bits_per_vector >= 64:
        # 8 bytes per vector
        num_subvectors = 8
        codebook_bits = 8
    elif bits_per_vector >= 32:
        # 4 bytes per vector
        num_subvectors = 8
        codebook_bits = 4
    else:
        # Extreme compression
        num_subvectors = 4
        codebook_bits = 8

    codebook_size = 2 ** codebook_bits

    return {
        "num_subvectors": num_subvectors,
        "codebook_size": codebook_size,
        "compression_ratio": (num_subvectors * codebook_bits) / (vector_dim * 32)
    }
```

## Fast Distance Computation

### SIMD Lookup Tables

```python
def compute_distances_simd(query, codes, codebooks):
    """
    SIMD-optimized distance computation
    """
    import numpy as np

    num_vectors = len(codes)
    num_subvectors = len(codebooks)

    # Pre-compute all distance tables
    tables = np.zeros((num_subvectors, 256))
    for m in range(num_subvectors):
        subvector_dim = len(query) // num_subvectors
        start = m * subvector_dim
        end = (m + 1) * subvector_dim
        query_sub = query[start:end]

        # Vectorized distance computation
        tables[m] = np.sum((codebooks[m] - query_sub) ** 2, axis=1)

    # Vectorized lookup
    distances = np.zeros(num_vectors)
    for m in range(num_subvectors):
        distances += tables[m, codes[:, m]]

    return np.sqrt(distances)
```

### Cache-Friendly Access

```python
class CacheOptimizedPQ:
    def __init__(self, num_subvectors, codebook_size):
        self.num_subvectors = num_subvectors
        self.codebook_size = codebook_size

    def reorder_codes(self, codes):
        """
        Reorder codes for better cache locality
        """
        # Store codes in column-major order for sequential access
        return np.ascontiguousarray(codes.T)

    def compute_distances_cached(self, query, codes_transposed):
        """
        Cache-friendly distance computation
        """
        distances = np.zeros(codes_transposed.shape[1])

        # Process one subvector at a time (better cache usage)
        for m in range(self.num_subvectors):
            subvector_codes = codes_transposed[m]
            table = self.distance_tables[m]
            distances += table[subvector_codes]

        return distances
```

## Asymmetric Distance Computation (ADC)

More accurate distance computation for queries:

```python
class AsymmetricPQ:
    def __init__(self, pq):
        self.pq = pq

    def compute_adc(self, query, codes):
        """
        Asymmetric distance: query remains uncompressed
        """
        num_subvectors = self.pq.num_subvectors
        subvector_dim = len(query) // num_subvectors

        # Pre-compute exact distances from query to all codewords
        distance_tables = []
        for m in range(num_subvectors):
            start = m * subvector_dim
            end = (m + 1) * subvector_dim
            query_subvector = query[start:end]

            # Exact distances to codebook entries
            distances = np.linalg.norm(
                self.pq.codebooks[m] - query_subvector,
                axis=1
            )
            distance_tables.append(distances ** 2)

        # Lookup and sum
        total_distances = np.zeros(len(codes))
        for m in range(num_subvectors):
            total_distances += distance_tables[m][codes[:, m]]

        return np.sqrt(total_distances)
```

## Storage Format

### PQ Index Structure

```
PQ Index:
  metadata.json
    - num_subvectors
    - codebook_size
    - vector_dimension
    - training_size

  codebooks/
    codebook_0.npy    # Shape: (256, subvector_dim)
    codebook_1.npy
    ...

  codes.lance         # Compressed codes
    - vector_id: int64
    - codes: fixed_size_list<uint8>[num_subvectors]

  rotation.npy        # OPQ rotation matrix (optional)
```

## Performance Characteristics

### Compression Ratios

| Configuration | Bits/Vector | Compression Ratio | Typical Recall@10 |
|--------------|-------------|-------------------|-------------------|
| PQ8x8 | 64 | 16:1 | 85-90% |
| PQ16x8 | 128 | 8:1 | 90-95% |
| PQ32x8 | 256 | 4:1 | 95-98% |
| PQ16x4 | 64 | 16:1 | 80-85% |

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Training | O(n*k*m*iter) | k-means iterations |
| Encoding | O(n*m*k) | Finding nearest codewords |
| Query | O(n*m) | Table lookups |
| Storage | O(n*m*log(k)) | Compressed codes |

## Best Practices

### Training Set Selection

```python
def select_training_set(vectors, target_size=100000):
    """
    Select representative vectors for PQ training
    """
    if len(vectors) <= target_size:
        return vectors

    # Use stratified sampling to maintain distribution
    # First, cluster to find representative regions
    n_clusters = 100
    kmeans = MiniBatchKMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(vectors)

    # Sample proportionally from each cluster
    samples = []
    samples_per_cluster = target_size // n_clusters

    for i in range(n_clusters):
        cluster_vectors = vectors[labels == i]
        n_samples = min(len(cluster_vectors), samples_per_cluster)
        indices = np.random.choice(len(cluster_vectors), n_samples, replace=False)
        samples.append(cluster_vectors[indices])

    return np.vstack(samples)
```

### Error Analysis

```python
def analyze_pq_error(pq, test_vectors):
    """
    Analyze quantization error
    """
    # Encode and decode
    codes = pq.encode(test_vectors)
    reconstructed = pq.decode(codes)

    # Compute errors
    mse = np.mean((test_vectors - reconstructed) ** 2)
    max_error = np.max(np.linalg.norm(test_vectors - reconstructed, axis=1))

    # Per-subvector analysis
    subvector_errors = []
    for m in range(pq.num_subvectors):
        start = m * pq.subvector_dim
        end = (m + 1) * pq.subvector_dim
        subvector_mse = np.mean(
            (test_vectors[:, start:end] - reconstructed[:, start:end]) ** 2
        )
        subvector_errors.append(subvector_mse)

    return {
        "mse": mse,
        "max_error": max_error,
        "subvector_errors": subvector_errors,
        "worst_subvector": np.argmax(subvector_errors)
    }
```

## Example: Large-Scale Deployment

```python
class ProductionPQ:
    def __init__(self, config):
        self.config = config
        self.pq = None
        self.setup()

    def setup(self):
        # Determine parameters based on requirements
        if self.config["priority"] == "memory":
            num_subvectors = 8
            codebook_size = 256
        elif self.config["priority"] == "accuracy":
            num_subvectors = 32
            codebook_size = 256
        else:  # balanced
            num_subvectors = 16
            codebook_size = 256

        self.pq = OptimizedPQ(num_subvectors, codebook_size)

    def build_index(self, vectors):
        # Train on sample
        training_vectors = self.select_training_set(vectors)
        self.pq.train(training_vectors)

        # Encode all vectors
        codes = []
        batch_size = 10000
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            batch_codes = self.pq.encode(batch)
            codes.append(batch_codes)

        return np.vstack(codes)

    def search(self, query, codes, k=10):
        # Use asymmetric distance for better accuracy
        distances = self.pq.compute_adc(query, codes)
        indices = np.argsort(distances)[:k]
        return indices, distances[indices]
```

## Limitations

1. **Training Required**: Needs representative training data
2. **Fixed Compression**: Cannot adjust compression after training
3. **Subspace Independence**: Assumes subvectors are independent
4. **Reconstruction Error**: Lossy compression affects recall
5. **Update Complexity**: Retraining needed for distribution shifts