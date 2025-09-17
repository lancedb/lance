# Scalar Quantization (SQ)

Scalar Quantization reduces the precision of vector components to achieve compression while maintaining relatively high accuracy for similarity search.

## Overview

SQ works by:
1. Mapping floating-point values to a smaller range (e.g., 8-bit integers)
2. Storing min/max values for reconstruction
3. Using quantized values for approximate distance computation
4. Optionally re-ranking with original precision

This typically achieves 4x compression (32-bit float → 8-bit int) with minimal accuracy loss.

## Quantization Methods

### Uniform Scalar Quantization

```python
class UniformSQ:
    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.num_levels = 2 ** num_bits
        self.min_vals = None
        self.max_vals = None

    def train(self, vectors):
        """
        Compute min/max values for each dimension
        """
        self.min_vals = np.min(vectors, axis=0)
        self.max_vals = np.max(vectors, axis=0)

        # Add small epsilon to avoid division by zero
        ranges = self.max_vals - self.min_vals
        ranges[ranges == 0] = 1e-6
        self.ranges = ranges

    def encode(self, vectors):
        """
        Quantize vectors to integers
        """
        # Normalize to [0, 1]
        normalized = (vectors - self.min_vals) / self.ranges

        # Quantize to integers
        quantized = np.round(normalized * (self.num_levels - 1))
        quantized = np.clip(quantized, 0, self.num_levels - 1)

        if self.num_bits == 8:
            return quantized.astype(np.uint8)
        elif self.num_bits == 4:
            return self.pack_4bit(quantized)
        else:
            return quantized.astype(np.uint16)

    def decode(self, codes):
        """
        Reconstruct vectors from codes
        """
        if self.num_bits == 4:
            codes = self.unpack_4bit(codes)

        # Map back to original range
        normalized = codes.astype(np.float32) / (self.num_levels - 1)
        reconstructed = normalized * self.ranges + self.min_vals

        return reconstructed
```

### Non-uniform Scalar Quantization

Use Lloyd-Max quantizer for optimal quantization levels:

```python
class NonUniformSQ:
    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.num_levels = 2 ** num_bits
        self.quantization_levels = None
        self.boundaries = None

    def train(self, vectors):
        """
        Learn optimal quantization levels using Lloyd-Max algorithm
        """
        d = vectors.shape[1]
        self.quantization_levels = np.zeros((d, self.num_levels))
        self.boundaries = np.zeros((d, self.num_levels - 1))

        for dim in range(d):
            dim_values = vectors[:, dim]

            # Initialize with uniform quantiles
            percentiles = np.linspace(0, 100, self.num_levels + 1)
            levels = np.percentile(dim_values, percentiles[1:-1])
            centroids = np.percentile(dim_values, percentiles[:-1] + np.diff(percentiles)/2)

            # Lloyd-Max iterations
            for _ in range(10):
                # Update boundaries
                self.boundaries[dim] = (centroids[:-1] + centroids[1:]) / 2

                # Update centroids
                for i in range(self.num_levels):
                    if i == 0:
                        mask = dim_values <= self.boundaries[dim, 0]
                    elif i == self.num_levels - 1:
                        mask = dim_values > self.boundaries[dim, -1]
                    else:
                        mask = (dim_values > self.boundaries[dim, i-1]) & \
                               (dim_values <= self.boundaries[dim, i])

                    if np.any(mask):
                        centroids[i] = np.mean(dim_values[mask])

            self.quantization_levels[dim] = centroids

    def encode(self, vectors):
        """
        Encode using learned quantization levels
        """
        n, d = vectors.shape
        codes = np.zeros((n, d), dtype=np.uint8)

        for dim in range(d):
            dim_values = vectors[:, dim]
            # Find nearest quantization level
            for i in range(n):
                distances = np.abs(self.quantization_levels[dim] - dim_values[i])
                codes[i, dim] = np.argmin(distances)

        return codes
```

## 4-bit and 8-bit Variants

### 4-bit Packing

```python
def pack_4bit(values):
    """
    Pack two 4-bit values into one byte
    """
    n, d = values.shape
    packed_d = (d + 1) // 2  # Round up for odd dimensions

    packed = np.zeros((n, packed_d), dtype=np.uint8)

    for i in range(0, d, 2):
        if i + 1 < d:
            # Pack two values
            packed[:, i // 2] = (values[:, i] << 4) | values[:, i + 1]
        else:
            # Last value if odd dimension
            packed[:, i // 2] = values[:, i] << 4

    return packed

def unpack_4bit(packed):
    """
    Unpack 4-bit values from bytes
    """
    n, packed_d = packed.shape
    d = packed_d * 2  # Assuming even dimensions

    unpacked = np.zeros((n, d), dtype=np.uint8)

    for i in range(packed_d):
        unpacked[:, i * 2] = (packed[:, i] >> 4) & 0x0F
        unpacked[:, i * 2 + 1] = packed[:, i] & 0x0F

    return unpacked
```

### Multi-level Quantization

```python
class MultiLevelSQ:
    def __init__(self, bits_per_dim=None):
        """
        Different quantization levels for different dimensions
        """
        self.bits_per_dim = bits_per_dim
        self.quantizers = []

    def train(self, vectors):
        """
        Train with different bits for each dimension based on variance
        """
        d = vectors.shape[1]

        if self.bits_per_dim is None:
            # Assign bits based on dimension variance
            variances = np.var(vectors, axis=0)
            variance_ranks = np.argsort(variances)

            self.bits_per_dim = np.zeros(d, dtype=int)
            # Top 25% dimensions get 8 bits
            self.bits_per_dim[variance_ranks[-d//4:]] = 8
            # Next 25% get 6 bits
            self.bits_per_dim[variance_ranks[-d//2:-d//4]] = 6
            # Rest get 4 bits
            self.bits_per_dim[variance_ranks[:-d//2]] = 4

        # Train individual quantizers
        for dim in range(d):
            bits = self.bits_per_dim[dim]
            quantizer = UniformSQ(num_bits=bits)
            quantizer.train(vectors[:, dim:dim+1])
            self.quantizers.append(quantizer)
```

## Distance Computation

### Approximate Distance with SQ

```python
def compute_sq_distances(query, sq_codes, sq_params):
    """
    Compute distances using quantized values
    """
    # Quantize query
    query_quantized = sq_params.encode(query.reshape(1, -1))[0]

    # Compute distances in quantized space
    if sq_params.num_bits == 8:
        # Can use integer arithmetic for speed
        diff = sq_codes.astype(np.int16) - query_quantized.astype(np.int16)
        distances = np.sum(diff ** 2, axis=1)
    else:
        # Decode and compute in float space
        decoded_codes = sq_params.decode(sq_codes)
        decoded_query = sq_params.decode(query_quantized.reshape(1, -1))[0]
        distances = np.linalg.norm(decoded_codes - decoded_query, axis=1)

    return distances
```

### Look-up Table Acceleration

```python
class SQWithLUT:
    def __init__(self, sq_params):
        self.sq = sq_params

    def build_lut(self, query):
        """
        Build look-up tables for fast distance computation
        """
        d = len(query)
        num_levels = self.sq.num_levels

        # Pre-compute squared differences for all possible values
        self.luts = []
        for dim in range(d):
            # Quantize query dimension
            query_val = self.sq.encode(query[dim:dim+1].reshape(1, -1))[0, 0]

            # Compute distances to all possible quantized values
            lut = np.zeros(num_levels)
            for level in range(num_levels):
                diff = level - query_val
                lut[level] = diff ** 2

            self.luts.append(lut)

    def compute_distances_lut(self, codes):
        """
        Fast distance computation using look-up tables
        """
        n, d = codes.shape
        distances = np.zeros(n)

        for dim in range(d):
            distances += self.luts[dim][codes[:, dim]]

        return np.sqrt(distances)
```

## Integration with IVF

### IVF_SQ Configuration

```python
class IVF_SQ:
    def __init__(self, num_partitions=256, num_bits=8):
        self.ivf = IVF(num_partitions)
        self.sq = UniformSQ(num_bits)

    def build(self, vectors):
        # Train IVF
        self.ivf.train(vectors)

        # Train SQ on all vectors
        self.sq.train(vectors)

        # Assign and encode vectors
        assignments = self.ivf.assign(vectors)
        codes = self.sq.encode(vectors)

        # Store in partitions
        self.partitions = []
        for p in range(self.num_partitions):
            mask = assignments == p
            partition_codes = codes[mask]
            partition_ids = np.where(mask)[0]
            self.partitions.append({
                'codes': partition_codes,
                'ids': partition_ids
            })

    def search(self, query, k=10, nprobes=10):
        # Find nearest partitions
        nearest_partitions = self.ivf.find_nearest_partitions(query, nprobes)

        # Build LUT for query
        sq_lut = SQWithLUT(self.sq)
        sq_lut.build_lut(query)

        # Search within partitions
        candidates = []
        for p in nearest_partitions:
            partition = self.partitions[p]
            distances = sq_lut.compute_distances_lut(partition['codes'])

            for idx, dist in enumerate(distances):
                candidates.append((partition['ids'][idx], dist))

        # Sort and return top-k
        candidates.sort(key=lambda x: x[1])
        return candidates[:k]
```

## Performance Optimization

### SIMD Acceleration

```python
def compute_sq_distances_simd(query_code, database_codes):
    """
    SIMD-accelerated distance computation for 8-bit SQ
    """
    import numpy as np

    # Convert to int16 to avoid overflow
    query_int = query_code.astype(np.int16)
    db_int = database_codes.astype(np.int16)

    # Vectorized subtraction and squaring
    diff = db_int - query_int
    squared_diff = diff * diff

    # Sum along dimension axis
    distances = np.sum(squared_diff, axis=1)

    return distances.astype(np.float32)
```

### GPU Acceleration

```python
def gpu_sq_search(query, database_codes, sq_params, k=10):
    """
    GPU-accelerated SQ search
    """
    import cupy as cp

    # Transfer to GPU
    query_gpu = cp.asarray(query)
    codes_gpu = cp.asarray(database_codes)

    # Quantize query on GPU
    query_normalized = (query_gpu - sq_params.min_vals) / sq_params.ranges
    query_quantized = cp.round(query_normalized * (sq_params.num_levels - 1))
    query_quantized = cp.clip(query_quantized, 0, sq_params.num_levels - 1)

    # Compute distances on GPU
    diff = codes_gpu.astype(cp.int16) - query_quantized.astype(cp.int16)
    distances = cp.sum(diff ** 2, axis=1)

    # Find top-k on GPU
    if k < 1024:
        indices = cp.argpartition(distances, k)[:k]
        indices = indices[cp.argsort(distances[indices])]
    else:
        indices = cp.argsort(distances)[:k]

    return indices.get(), distances[indices].get()
```

## Configuration Guidelines

### Bit Selection

```python
def select_sq_bits(compression_ratio_target, vector_dim):
    """
    Select number of bits based on compression target
    """
    original_bits = vector_dim * 32  # float32

    if compression_ratio_target >= 8:
        return 4  # 8x compression
    elif compression_ratio_target >= 4:
        return 8  # 4x compression
    elif compression_ratio_target >= 2:
        return 16  # 2x compression
    else:
        return 32  # No compression
```

### Quality vs Compression Trade-off

| Bits | Compression | Memory | Typical Error | Use Case |
|------|-------------|--------|---------------|----------|
| 4 | 8x | 12.5% | 5-10% | Memory critical |
| 6 | 5.3x | 18.75% | 2-5% | Balanced |
| 8 | 4x | 25% | 1-2% | Quality priority |
| 16 | 2x | 50% | <0.5% | High precision |

## Best Practices

### Training Data Selection

```python
def select_training_data_sq(vectors, sample_size=100000):
    """
    Select representative data for SQ training
    """
    if len(vectors) <= sample_size:
        return vectors

    # Include extremes to capture full range
    dim_samples = []
    for dim in range(vectors.shape[1]):
        dim_values = vectors[:, dim]
        # Include min, max, and percentiles
        percentiles = np.percentile(dim_values, [0, 25, 50, 75, 100])
        extreme_indices = []
        for p in percentiles:
            idx = np.argmin(np.abs(dim_values - p))
            extreme_indices.append(idx)
        dim_samples.extend(extreme_indices)

    # Add random samples
    remaining = sample_size - len(set(dim_samples))
    random_indices = np.random.choice(len(vectors), remaining, replace=False)

    all_indices = list(set(dim_samples)) + list(random_indices)
    return vectors[all_indices]
```

### Error Monitoring

```python
def analyze_sq_error(sq, test_vectors):
    """
    Analyze quantization error by dimension
    """
    encoded = sq.encode(test_vectors)
    decoded = sq.decode(encoded)

    errors = {
        'mse': np.mean((test_vectors - decoded) ** 2),
        'max_error': np.max(np.abs(test_vectors - decoded)),
        'relative_error': np.mean(np.abs(test_vectors - decoded) / (np.abs(test_vectors) + 1e-8))
    }

    # Per-dimension analysis
    dim_errors = np.mean((test_vectors - decoded) ** 2, axis=0)
    errors['worst_dims'] = np.argsort(dim_errors)[-10:]
    errors['best_dims'] = np.argsort(dim_errors)[:10]

    return errors
```

## Example: Production Deployment

```python
class ProductionSQ:
    def __init__(self, config):
        self.config = config
        self.sq = None

    def train(self, vectors):
        # Select quantization based on requirements
        if self.config['mode'] == 'quality':
            self.sq = NonUniformSQ(num_bits=8)
        elif self.config['mode'] == 'balanced':
            self.sq = UniformSQ(num_bits=6)
        else:  # memory
            self.sq = UniformSQ(num_bits=4)

        # Train on representative sample
        training_data = self.select_training_data_sq(vectors)
        self.sq.train(training_data)

        # Validate quality
        error = self.validate_quality(vectors[:1000])
        if error > self.config['max_error']:
            print(f"Warning: Quantization error {error} exceeds threshold")

    def encode_batch(self, vectors, batch_size=10000):
        """
        Encode vectors in batches for memory efficiency
        """
        codes = []
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            batch_codes = self.sq.encode(batch)
            codes.append(batch_codes)
        return np.vstack(codes)

    def search(self, query, codes, k=10, rerank=True):
        """
        Search with optional re-ranking
        """
        # Initial search with SQ
        distances = self.compute_sq_distances(query, codes)
        candidates = np.argsort(distances)[:k * 2 if rerank else k]

        if rerank and hasattr(self, 'original_vectors'):
            # Re-rank with original precision
            exact_distances = np.linalg.norm(
                self.original_vectors[candidates] - query,
                axis=1
            )
            reranked = candidates[np.argsort(exact_distances)[:k]]
            return reranked

        return candidates[:k]
```

## Limitations

1. **Range Dependency**: Performance depends on value distribution
2. **Dimension Independence**: Treats each dimension separately
3. **Fixed Precision**: Cannot adjust precision after training
4. **Outlier Sensitivity**: Extreme values affect quantization quality
5. **No Correlation**: Doesn't exploit correlations between dimensions