# Vector Indices

Lance provides a powerful and extensible secondary index system for efficient vector similarity search. 
All vector indices are stored as regular Lance files, making them portable and easy to manage.
It is designed for efficient similarity search across large-scale vector datasets.

## Concepts

Lance splits each vector index into 3 parts - clustering, sub-index and quantization.

### Clustering

Clustering divides all the vectors into different disjoint clusters (a.k.a. partitions).
Lance currently supports using Inverted File (IVF) as the primary clustering mechanism.
IVF partitions the vectors into clusters using the k-means clustering algorithm. 
Each cluster contains vectors that are similar to the cluster centroid.
During search, only the most relevant clusters are examined, dramatically reducing search time.
IVF can be combined with any sub-index type and quantization method.

### Sub-Index

The sub-index determines how vectors are organized for search. Lance currently supports:

- **FLAT**: Exact search with no approximation - scans all vectors
- **HNSW**: Hierarchical Navigable Small World graphs for fast approximate search

### Quantization

The quantization method determines how vectors are stored and compressed. Lance currently supports:

- **Product Quantization (PQ)**: Compresses vectors by splitting them into smaller sub-vectors and quantizing each independently
- **Scalar Quantization (SQ)**: Applies scalar quantization to each dimension of the vector independently  
- **FLAT**: No quantization, keeps original vectors for exact search

### Common Combinations

When we refer to an index type, it is typically `{clustering}_{sub_index}_{quantization}`.
If sub-index is just `FLAT`, we usually omit it and just refer to it by `{clustering}_{quantization}`.
Here are the commonly used combinations:

| Index Type      | Name                                            | Description                                                                              |
|-----------------|-------------------------------------------------|------------------------------------------------------------------------------------------|
| **IVF_PQ**      | Inverted File with Product Quantization         | Combines IVF clustering with PQ compression for efficient storage and search             |
| **IVF_HNSW_SQ** | Inverted File with HNSW and Scalar Quantization | Uses IVF for coarse clustering and HNSW for fine-grained search with scalar quantization |
| **IVF_SQ**      | Inverted File with Scalar Quantization          | Combines IVF clustering with scalar quantization for balanced compression                |
| **IVF_FLAT**    | Inverted File without quantization              | Uses IVF clustering with exact vector storage for precise search within clusters         |

### Versioning

The Lance vector index format has gone through 3 versions so far.
This document currently only records version 3 which is the latest version.
The specific version of the vector index is recorded in the `index_version` field of the generic [index metadata](../index.md#index-metadata).

## Storage Layout (V3)

Each vector index is stored as 2 regular Lance files - index file and auxiliary file.

### Index File

The index structure file containing the search graph/structure with index-specific schema.
It is stored as a Lance file with name `index.idx` within the index directory.

#### Arrow Schema

The index file stores the search structure with graph or flat organization.
The Arrow schema of the Lance file varies depending on the sub-index type used.

!!! note
    All partitions are stored in the same file, and partitions must be written in order.

##### FLAT

FLAT indices perform exact search with no approximation. This is essentially an empty file with a minimal schema:

| Column          | Type   | Nullable | Description                                  |
|-----------------|--------|----------|----------------------------------------------|
| `__flat_marker` | uint64 | false    | Marker field for FLAT index (no actual data) |

##### HNSW

HNSW (Hierarchical Navigable Small World) indices provide fast approximate search through a multi-level graph structure. This stores the HNSW graph with the following schema:

| Column        | Type          | Nullable | Description            |
|---------------|---------------|----------|------------------------|
| `__vector_id` | uint64        | false    | Vector identifier      |
| `__neighbors` | list<uint32>  | false    | Neighbor node IDs      |
| `_distance`   | list<float32> | false    | Distances to neighbors |

!!! note
    HNSW consists of multiple levels, and all levels must be written in order starting from level 0.

#### Arrow Schema Metadata

The index file contains metadata in its Arrow schema metadata to describe the index configuration and structure.
Here are the metadata keys and their corresponding values:

##### "lance:index"

Contains basic index configuration information in JSON:

| JSON Key        | Type   | Expected Values                                 |
|-----------------|--------|-------------------------------------------------|
| `type`          | String | Index type (e.g., "IVF_PQ", "IVF_HNSW", "FLAT") |
| `distance_type` | String | Distance metric (e.g., "l2", "cosine", "dot")   |

##### "lance:ivf"

References the IVF metadata stored in the Lance file global buffer.
This value records the global buffer index, currently this is always "1".

!!! note
    Global buffer indices in Lance files are 1-based, 
    so you need to subtract 1 when accessing them through code.

##### "lance:flat"

Contains partition-specific metadata for the `FLAT` sub-index structure.
This is an empty string since FLAT indices don't require additional metadata at this moment.

##### "lance:hnsw"

Contains the HNSW-specific JSON metadata for each partition, including graph structure information:

| JSON Key        | Type         | Expected Values                    |
|-----------------|--------------|------------------------------------|
| `entry_point`   | u32          | Starting node for graph traversal  |
| `params`        | Object       | HNSW construction parameters (see below) |
| `level_offsets` | Array<usize> | Offset for each level in the graph |

The `params` object contains the following HNSW construction parameters:

| JSON Key            | Type          | Description                                                    | Default |
|---------------------|---------------|----------------------------------------------------------------|---------|
| `max_level`         | u16           | Maximum level of the HNSW graph                                | 7       |
| `m`                 | usize         | Number of connections to establish while inserting new element | 20      |
| `ef_construction`   | usize         | Size of the dynamic list for candidates                        | 150     |
| `prefetch_distance` | Option<usize> | Number of vectors ahead to prefetch while building             | Some(2) |

#### Lance File Global Buffer

##### IVF Metadata

For efficiency, Lance serializes IVF metadata to protobuf format and stores it in the Lance file global buffer:

```protobuf
%%% proto.message.IVF %%%
```

### Auxiliary File

The auxiliary file is a vector storage for quantized vectors.
It is stored as a Lance file named `auxiliary.idx` within the index directory.

#### Arrow Schema

Since the auxiliary file stores the actual (quantized) vectors,
the Arrow schema of the Lance file varies depending on the quantization method used.

!!! note
    All partitions are stored in the same file, and partitions must be written in order.

##### FLAT

No quantization applied - stores original vectors in their full precision:

| Column   | Type                            | Nullable | Description                                           |
|----------|---------------------------------|----------|-------------------------------------------------------|
| `_rowid` | uint64                          | false    | Row identifier                                        |
| `flat`   | list<float32>[dimension]        | false    | Original vector values (list_size = vector dimension) |

##### PQ

Compresses vectors using product quantization for significant memory savings:

| Column      | Type           | Nullable | Description                                 |
|-------------|----------------|----------|---------------------------------------------|
| `_rowid`    | uint64         | false    | Row identifier                              |
| `__pq_code` | list<uint8>[m] | false    | PQ codes (list_size = number of subvectors) |

##### SQ

Compresses vectors using scalar quantization for moderate memory savings:

| Column      | Type                   | Nullable | Description                             |
|-------------|------------------------|----------|-----------------------------------------|
| `_rowid`    | uint64                 | false    | Row identifier                          |
| `__sq_code` | list<uint8>[dimension] | false    | SQ codes (list_size = vector dimension) |

#### Arrow Schema Metadata

The auxiliary file also contains metadata in its Arrow schema metadata for vector storage configuration.
Here are the metadata keys and their corresponding values:

##### "distance_type"
The distance metric used to compute similarity between vectors (e.g., "l2", "cosine", "dot").

##### "lance:ivf"

Similar to the index file's "lance:ivf" but focused on vector storage layout. 
This doesn't contain the partitions' centroids.
It's only used for tracking each partition's offset and length in the auxiliary file.

##### "storage_metadata"

Contains quantizer-specific metadata as a list of JSON strings.
Currently, the list always contains exactly 1 element with the quantizer metadata.

For **Product Quantization (PQ)**:

| JSON Key            | Type  | Description                                                      |
|---------------------|-------|------------------------------------------------------------------|
| `codebook_position` | usize | Position of the codebook in the global buffer                    |
| `nbits`             | u32   | Number of bits per subvector code (e.g., 8 bits = 256 codewords) |
| `num_sub_vectors`   | usize | Number of subvectors (m)                                         |
| `dimension`         | usize | Original vector dimension                                        |
| `transposed`        | bool  | Whether the codebook is stored in transposed layout              |

For **Scalar Quantization (SQ)**:

| JSON Key   | Type       | Description                            |
|------------|------------|----------------------------------------|
| `dim`      | usize      | Vector dimension                       |
| `num_bits` | u16        | Number of bits for quantization        |
| `bounds`   | Range<f64> | Min/max bounds for scalar quantization |

#### Lance File Global Buffer

##### Quantization Codebook

For product quantization, the codebook is stored in `Tensor` format 
in the auxiliary file's global buffer for efficient access:

```protobuf
%%% proto.message.Tensor %%%
```

## Appendices

### Appendix 1: Example IVF_PQ Format

This example shows how an `IVF_PQ` index is physically laid out. Assume vectors have dimension 128,
PQ uses 16 num_sub_vectors (m=16) with 8 num_bits per subvector, and distance type is "l2".

#### Index File

- Arrow Schema Metadata:
    - `"lance:index"` → `{ "type": "IVF_PQ", "distance_type": "l2" }`
    - `"lance:ivf"` → "1" (references IVF metadata in the global buffer)
    - `"lance:flat"` → `["", "", ...]` (one empty string per partition; IVF_PQ uses a FLAT sub-index inside each partition)

- Lance File Global buffer (Protobuf):
    - `Ivf` message containing:
        - `centroids_tensor`: shape `[num_partitions, 128]` (float32)
        - `offsets`: start offset (row) of each partition in `auxiliary.idx`
        - `lengths`: number of vectors in each partition
        - `loss`: k-means loss (optional)

#### Auxiliary File

- Arrow Schema Metadata:
    - `"distance_type"` → `"l2"`
    - `"lance:ivf"` → tracks per-partition `offsets` and `lengths` (no centroids here)
    - `"storage_metadata"` → `[ "{"pq":{"num_sub_vectors":16,"nbits":8,"dimension":128,"transposed":true}}" ]`
- Lance File Global buffer:
    - `Tensor` codebook with shape `[256, num_sub_vectors, dim/num_sub_vectors]` = `[256, 16, 8]` (float32)
- Rows with Arrow schema: 

```python
pa.schema([
    pa.field("_rowid", pa.uint64()),
    pa.field("__pq_code", pa.list(pa.uint8(), list_size=16)), # m subvector codes
])
```

### Appendix 2: Accessing Index File with Python

The following example demonstrates how to read and parse different components in the Lance index files using Python:

```python
import pyarrow as pa
import lance

# Open the index file
index_reader = lance.LanceFileReader.read_file("path/to/index.idx")

# Access schema metadata
schema_metadata = index_reader.metadata().schema.metadata

# Get the IVF metadata reference from schema
ivf_ref = schema_metadata.get(b"lance:ivf")  # Returns b"1" for global buffer index

# Read the global buffer containing IVF metadata
if ivf_ref:
    buffer_index = int(ivf_ref) - 1  # Global buffer indices are 1-based
    ivf_buffer = index_reader.global_buffer(buffer_index)

    # Parse the protobuf message (requires lance protobuf definitions)
    # ivf_metadata = parse_ivf_protobuf(ivf_buffer)

# For auxiliary file with PQ codebook
aux_reader = lance.LanceFileReader.read_file("path/to/auxiliary.idx")

# Get storage metadata
storage_metadata = aux_reader.metadata().schema.metadata.get(b"storage_metadata")
if storage_metadata:
    import json
    pq_metadata = json.loads(storage_metadata.decode())[0]  # First element of the list
    pq_params = json.loads(pq_metadata)

    # Access the codebook from global buffer
    codebook_position = pq_params.get("codebook_position", 1)
    if codebook_position > 0:
        codebook_buffer = aux_reader.global_buffer(codebook_position - 1)
        # Parse the tensor protobuf
        # codebook_tensor = parse_tensor_protobuf(codebook_buffer)
```
