# HNSW (Hierarchical Navigable Small World)

HNSW is a graph-based sub-index in Lance that provides extremely fast approximate nearest neighbor search through a hierarchical graph structure.

## Overview

HNSW builds a multi-layer graph where:
- Each layer is a proximity graph with different connectivity
- Higher layers have fewer nodes and longer-range connections
- Search starts from the top layer and descends to find neighbors
- Combines the small-world property with hierarchical structure

## Architecture

### Hierarchical Structure

```
Layer 2 (sparse):    ○───────────○
                     │           │
Layer 1 (medium):    ○───○───○───○
                     │   │   │   │
Layer 0 (dense):     ○─○─○─○─○─○─○─○
```

Each node exists in layer 0, with probability of appearing in higher layers decreasing exponentially.

### Graph Properties

1. **Small World**: Short paths between any two nodes
2. **Navigability**: Greedy search finds near-optimal paths
3. **Hierarchy**: Express lanes for long-range traversal

## Construction Algorithm

### Node Insertion

```python
def insert_node(hnsw, new_node, M=16, M_max=32, ef_construction=200):
    """
    Insert a new node into HNSW graph
    M: number of connections for new nodes
    M_max: maximum connections for layer 0
    ef_construction: size of dynamic candidate list
    """
    # Assign layer for new node (exponential decay probability)
    node_layer = select_layer(ml=1.0/log(2))

    # Find nearest neighbors at all layers
    entry_point = hnsw.entry_point
    nearest = []

    # Search from top to target layer
    for level in range(hnsw.max_layer, node_layer, -1):
        nearest = search_layer(new_node, entry_point, 1, level)
        entry_point = nearest[0]

    # Insert from target layer to bottom
    for level in range(node_layer, -1, -1):
        candidates = search_layer(new_node, entry_point, ef_construction, level)

        # Select M neighbors
        if level == 0:
            M_cur = M_max
        else:
            M_cur = M

        neighbors = select_neighbors_heuristic(candidates, M_cur)

        # Add bidirectional links
        for neighbor in neighbors:
            hnsw.add_connection(new_node, neighbor, level)
            hnsw.add_connection(neighbor, new_node, level)

            # Prune neighbor's connections if needed
            prune_connections(neighbor, M_max, level)

    # Update entry point if necessary
    if node_layer > hnsw.max_layer:
        hnsw.entry_point = new_node
        hnsw.max_layer = node_layer

    return hnsw
```

### Neighbor Selection Heuristic

```python
def select_neighbors_heuristic(candidates, M):
    """
    Select diverse neighbors to maintain connectivity
    """
    selected = []
    candidates = sorted(candidates, key=lambda x: x.distance)

    for candidate in candidates:
        if len(selected) >= M:
            break

        # Check if candidate increases connectivity
        if is_closer_to_selected(candidate, selected):
            continue

        selected.append(candidate)

    return selected
```

## Search Algorithm

### K-NN Search

```python
def search_knn(hnsw, query, k=10, ef=100):
    """
    Find k nearest neighbors
    ef: size of dynamic candidate list (ef >= k)
    """
    # Start from entry point
    entry_point = hnsw.entry_point
    curr_nearest = entry_point
    curr_dist = distance(query, entry_point)

    # Search from top layer to layer 1
    for level in range(hnsw.max_layer, 0, -1):
        curr_nearest = search_layer_greedy(query, curr_nearest, 1, level)

    # Search at layer 0 with ef candidates
    candidates = search_layer(query, curr_nearest, ef, level=0)

    # Return top k
    return sorted(candidates, key=lambda x: x.distance)[:k]
```

### Layer Search

```python
def search_layer(query, entry_points, num_to_return, level):
    """
    Search within a single layer
    """
    visited = set()
    candidates = BinaryHeap()  # Min-heap
    nearest = BinaryHeap()     # Max-heap

    # Initialize with entry points
    for point in entry_points:
        dist = distance(query, point)
        candidates.push((-dist, point))
        nearest.push((dist, point))
        visited.add(point)

    while candidates:
        curr_dist, current = candidates.pop()

        # Check stopping condition
        if -curr_dist > nearest.top()[0]:
            break

        # Check neighbors
        for neighbor in hnsw.get_connections(current, level):
            if neighbor in visited:
                continue

            visited.add(neighbor)
            dist = distance(query, neighbor)

            if dist < nearest.top()[0] or len(nearest) < num_to_return:
                candidates.push((-dist, neighbor))
                nearest.push((dist, neighbor))

                # Maintain size limit
                if len(nearest) > num_to_return:
                    nearest.pop()

    return list(nearest)
```

## Configuration Parameters

### Construction Parameters

```python
dataset.create_index(
    column="embedding",
    index_type="HNSW",
    config={
        "M": 16,                    # Connections per node
        "ef_construction": 200,     # Construction search width
        "max_elements": 1000000,    # Maximum graph size
        "seed": 42,                 # Random seed for layer assignment
        "metric": "L2"              # Distance metric
    }
)
```

### Parameter Guidelines

| Parameter | Default | Effect | Trade-off |
|-----------|---------|--------|-----------|
| M | 16 | Graph connectivity | Higher = better recall, more memory |
| ef_construction | 200 | Construction quality | Higher = better graph, slower build |
| ef | 100 | Search accuracy | Higher = better recall, slower search |

### Optimization Strategies

```python
def optimize_parameters(dataset_size, recall_target, latency_target):
    if dataset_size < 100_000:
        # Small dataset: prioritize quality
        M = 32
        ef_construction = 400
        ef = 200

    elif dataset_size < 10_000_000:
        # Medium dataset: balance
        M = 16
        ef_construction = 200
        ef = 100

    else:
        # Large dataset: prioritize efficiency
        M = 12
        ef_construction = 100
        ef = 50

    # Adjust for recall requirements
    if recall_target > 0.95:
        ef *= 2
        ef_construction *= 2

    # Adjust for latency requirements
    if latency_target < 1:  # < 1ms
        ef = min(ef, 50)
        M = min(M, 16)

    return M, ef_construction, ef
```

## Storage Format

### Graph Structure

```
HNSW Index:
  metadata.json
    - M, ef_construction, max_layer
    - entry_point_id
    - num_elements

  levels/
    level_0.graph    # Dense connections
    level_1.graph    # Sparser connections
    ...

  nodes.lance        # Node data and layer assignments
```

### Node Storage

```python
class HNSWNode:
    def __init__(self, id, vector, level):
        self.id = id
        self.vector = vector
        self.level = level
        self.connections = [[] for _ in range(level + 1)]

    def serialize(self):
        return {
            "id": self.id,
            "vector": self.vector.tobytes(),
            "level": self.level,
            "connections": [
                [conn_id for conn_id in level_conns]
                for level_conns in self.connections
            ]
        }
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Insert | O(log n) | Expected, due to layer selection |
| Delete | O(M²) | Requires connection repair |
| Search | O(log n) | With optimal parameters |
| Build | O(n log n) | Sequential insertion |

### Space Complexity

- **Nodes**: O(n) for vector storage
- **Edges**: O(n*M) for connections
- **Total**: O(n*(M + d)) where d = vector dimension

### Recall vs Speed Trade-off

```python
def benchmark_parameters(hnsw, test_queries, ground_truth):
    results = []

    for ef in [10, 50, 100, 200, 500]:
        recalls = []
        latencies = []

        for query, true_neighbors in zip(test_queries, ground_truth):
            start = time.time()
            found = hnsw.search(query, k=10, ef=ef)
            latency = time.time() - start

            recall = len(set(found) & set(true_neighbors[:10])) / 10
            recalls.append(recall)
            latencies.append(latency)

        results.append({
            "ef": ef,
            "recall@10": np.mean(recalls),
            "latency_ms": np.mean(latencies) * 1000
        })

    return results
```

## Optimizations

### Prefetching

```python
class OptimizedHNSW:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefetch_size = 4  # Prefetch next 4 nodes

    def search_with_prefetch(self, query, k, ef):
        # Prefetch likely candidates
        for candidate in candidates[:self.prefetch_size]:
            self.prefetch_node(candidate)

        # Continue normal search
        return self.search(query, k, ef)
```

### SIMD Optimization

```python
def batch_distance_computation(query, nodes):
    """
    Compute distances using SIMD instructions
    """
    import numpy as np

    # Vectorized distance computation
    node_vectors = np.array([node.vector for node in nodes])
    distances = np.linalg.norm(node_vectors - query, axis=1)

    return distances
```

### Cache-Friendly Layout

```python
class CacheOptimizedHNSW:
    def __init__(self):
        # Store frequently accessed data together
        self.hot_nodes = []  # Top layer nodes
        self.cold_nodes = []  # Bottom layer only nodes

    def organize_by_access_pattern(self):
        for node in self.nodes:
            if node.level > 0:
                self.hot_nodes.append(node)
            else:
                self.cold_nodes.append(node)
```

## Integration with IVF

HNSW as sub-index in IVF:

```python
class IVF_HNSW:
    def __init__(self, num_partitions, M=16, ef_construction=200):
        self.ivf = IVF(num_partitions)
        self.hnsw_indices = []

    def build(self, vectors):
        # Train IVF
        self.ivf.train(vectors)
        assignments = self.ivf.assign(vectors)

        # Build HNSW for each partition
        for partition_id in range(self.ivf.num_partitions):
            partition_vectors = vectors[assignments == partition_id]

            hnsw = HNSW(M=self.M, ef_construction=self.ef_construction)
            hnsw.build(partition_vectors)
            self.hnsw_indices.append(hnsw)

    def search(self, query, k=10, nprobes=10, ef=100):
        # Find nearest partitions
        nearest_partitions = self.ivf.find_nearest_partitions(query, nprobes)

        # Search within partitions using HNSW
        candidates = []
        for partition_id in nearest_partitions:
            hnsw = self.hnsw_indices[partition_id]
            partition_results = hnsw.search(query, k * 2, ef)
            candidates.extend(partition_results)

        # Merge and return top-k
        candidates.sort(key=lambda x: x.distance)
        return candidates[:k]
```

## Best Practices

### Building Large Graphs

```python
def build_hnsw_parallel(vectors, num_threads=8):
    """
    Parallel HNSW construction for large datasets
    """
    import concurrent.futures

    # Split vectors into chunks
    chunk_size = len(vectors) // num_threads
    chunks = [vectors[i:i+chunk_size] for i in range(0, len(vectors), chunk_size)]

    # Build sub-graphs in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        sub_graphs = list(executor.map(build_hnsw_chunk, chunks))

    # Merge sub-graphs
    merged_graph = merge_hnsw_graphs(sub_graphs)
    return merged_graph
```

### Dynamic Updates

```python
class DynamicHNSW:
    def __init__(self, rebuild_threshold=0.2):
        self.rebuild_threshold = rebuild_threshold
        self.num_updates = 0

    def insert(self, vector):
        super().insert(vector)
        self.num_updates += 1

        # Rebuild if too many updates
        if self.num_updates / len(self.nodes) > self.rebuild_threshold:
            self.rebuild()
            self.num_updates = 0

    def delete(self, node_id):
        # Mark as deleted (lazy deletion)
        self.deleted_nodes.add(node_id)
        self.num_updates += 1
```

### Memory Management

```python
def estimate_memory_usage(num_vectors, dim, M=16):
    """
    Estimate HNSW memory requirements
    """
    # Vector storage
    vector_memory = num_vectors * dim * 4  # float32

    # Graph connections (average M connections per node)
    avg_connections = M * (1 + 1/log(2))  # Account for all layers
    connection_memory = num_vectors * avg_connections * 8  # int64 ids

    # Metadata
    metadata_memory = num_vectors * 16  # Layer info, etc.

    total_memory = vector_memory + connection_memory + metadata_memory
    return total_memory / (1024**3)  # Convert to GB
```

## Limitations

1. **Memory Intensive**: Requires storing full graph in memory
2. **Build Time**: Sequential insertion can be slow for large datasets
3. **No Exact Search**: Always approximate, cannot guarantee finding true nearest neighbor
4. **Update Complexity**: Deletions and updates require graph repair
5. **Parameter Sensitivity**: Performance heavily depends on M and ef settings