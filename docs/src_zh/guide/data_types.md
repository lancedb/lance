# 数据类型

Lance 使用 [Apache Arrow](https://arrow.apache.org/) 作为其内存数据格式。本指南介绍支持的数据类型，重点关注数组类型，这些类型对于向量嵌入（Vector Embeddings）和机器学习应用至关重要。

## Arrow 类型系统

Lance 支持完整的 Apache Arrow 类型系统。当通过 Python（PyArrow）或 Rust（arrow-rs）写入数据时，Arrow 类型会自动映射到 Lance 的内部表示。

### 基本类型

| Arrow 类型 | 描述 | 示例用途 |
|------------|------|---------|
| `Boolean` | 真/假值 | 标志、过滤器 |
| `Int8`, `Int16`, `Int32`, `Int64` | 有符号整数 | ID、计数 |
| `UInt8`, `UInt16`, `UInt32`, `UInt64` | 无符号整数 | ID、索引 |
| `Float16`, `Float32`, `Float64` | 浮点数 | 测量值、评分 |
| `Decimal128`, `Decimal256` | 固定精度小数 | 金融数据 |
| `Date32`, `Date64` | 日期值 | 出生日期、事件日期 |
| `Time32`, `Time64` | 时间值 | 时刻 |
| `Timestamp` | 带时区的日期和时间 | 事件时间戳 |
| `Duration` | 时间持续 | 经过时间 |

### 字符串和二进制类型

| Arrow 类型 | 描述 | 示例用途 |
|------------|------|---------|
| `Utf8` | 变长 UTF-8 字符串 | 文本、名称 |
| `LargeUtf8` | 大型 UTF-8 字符串（64 位偏移量） | 大型文档 |
| `Binary` | 变长二进制数据 | 原始字节 |
| `LargeBinary` | 大型二进制数据（64 位偏移量） | 大型 blob |
| `FixedSizeBinary(n)` | 固定长度二进制数据 | UUID、哈希值 |

### 大型二进制对象的 Blob 类型

Lance 提供了专门的 **Blob** 类型，用于高效存储和检索非常大的二进制对象，如视频、图像、音频文件或其他多媒体内容。与常规二进制列不同，blob 支持延迟加载（Lazy Loading），这意味着你可以读取部分数据而无需将所有内容加载到内存中。

对于新数据集，通过 `blob_field` 和 `blob_array` 使用 blob v2（`lance.blob.v2`）。

Blob 版本遵循数据集文件格式规则：

- `data_storage_version` 是数据集的 Lance 文件格式版本。
- 数据集的 `data_storage_version` 一旦创建就固定不变。
- 对于 `data_storage_version >= 2.2`，写入时会拒绝旧版 blob 元数据（`lance-encoding:blob`）。
- 旧版基于元数据的 blob 写入在 `0.1`、`2.0` 和 `2.1` 版本中仍然可用。

```python
import lance
import pyarrow as pa
from lance import blob_array, blob_field

schema = pa.schema([
    pa.field("id", pa.int64()),
    blob_field("video"),
])

table = pa.table(
    {
        "id": [1],
        "video": blob_array([b"sample-video-bytes"]),
    },
    schema=schema,
)

ds = lance.write_dataset(table, "./videos_v22.lance", data_storage_version="2.2")
blob = ds.take_blobs("video", indices=[0])[0]
with blob as f:
    payload = f.read()
```

对于旧版兼容性（`data_storage_version <= 2.1`），你仍然可以使用带有 `lance-encoding:blob=true` 的 `LargeBinary` 来写入 blob 列。

要使用旧版路径创建 blob 列，请将 `lance-encoding:blob` 元数据添加到 `LargeBinary` 字段：

```python
import pyarrow as pa
import lance

# Define schema with a blob column for videos
schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("filename", pa.utf8()),
    pa.field("video", pa.large_binary(), metadata={"lance-encoding:blob": "true"}),
])

# Read video file
with open("sample_video.mp4", "rb") as f:
    video_data = f.read()

# Create and write dataset
table = pa.table({
    "id": [1],
    "filename": ["sample_video.mp4"],
    "video": [video_data],
}, schema=schema)

ds = lance.write_dataset(
    table,
    "./videos_legacy.lance",
    schema=schema,
    data_storage_version="2.1",
)
```

要读取 blob 数据，使用 `take_blobs()`，它返回文件类对象用于延迟读取：

```python
# Retrieve blob as a file-like object (lazy loading)
blobs = ds.take_blobs("video", ids=[0])

# Use with libraries that accept file-like objects
import av  # pip install av
with av.open(blobs[0]) as container:
    for frame in container.decode(video=0):
        # Process video frames without loading entire video into memory
        pass
```

更多详情请参阅 [Blob API 指南](blob.md)。

## 向量嵌入的数组类型

Lance 为数组类型提供了出色的支持，这对于在 AI/ML 应用中存储向量嵌入至关重要。

### FixedSizeList - 向量嵌入的首选类型

`FixedSizeList` 是存储固定维度向量嵌入的推荐类型。每个向量具有相同的维度数，使其在存储和计算方面非常高效。

=== "Python"

    ```python
    import lance
    import pyarrow as pa
    import numpy as np

    # Create a schema with a vector embedding column
    # This defines a 128-dimensional float32 vector
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("text", pa.utf8()),
        pa.field("vector", pa.list_(pa.float32(), 128)),  # FixedSizeList of 128 floats
    ])

    # Create sample data with embeddings
    num_rows = 1000
    vectors = np.random.rand(num_rows, 128).astype(np.float32)

    table = pa.Table.from_pydict({
        "id": list(range(num_rows)),
        "text": [f"document_{i}" for i in range(num_rows)],
        "vector": [v.tolist() for v in vectors],
    }, schema=schema)

    # Write to Lance format
    ds = lance.write_dataset(table, "./embeddings.lance")
    print(f"Created dataset with {ds.count_rows()} rows")
    ```

=== "Rust"

    ```rust
    use arrow_array::{
        ArrayRef, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, StringArray,
    };
    use arrow_schema::{DataType, Field, Schema};
    use lance::dataset::WriteParams;
    use lance::Dataset;
    use std::sync::Arc;

    #[tokio::main]
    async fn main() -> lance::Result<()> {
        // Define schema with a 128-dimensional vector column
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("text", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    128,
                ),
                false,
            ),
        ]));

        // Create sample data
        let ids = Int64Array::from(vec![0, 1, 2]);
        let texts = StringArray::from(vec!["doc_0", "doc_1", "doc_2"]);
        
        // Create vector embeddings (128-dimensional)
        let values: Vec<f32> = (0..384).map(|i| i as f32 / 100.0).collect();
        let values_array = Float32Array::from(values);
        let vectors = FixedSizeListArray::try_new_from_values(values_array, 128)?;

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(ids) as ArrayRef,
                Arc::new(texts) as ArrayRef,
                Arc::new(vectors) as ArrayRef,
            ],
        )?;

        // Write to Lance
        let dataset = Dataset::write(
            vec![batch].into_iter().map(Ok),
            "embeddings.lance",
            WriteParams::default(),
        )
        .await?;

        println!("Created dataset with {} rows", dataset.count_rows().await?);
        Ok(())
    }
    ```

### 使用嵌入向量进行向量搜索

将向量嵌入存储在 Lance 中后，你可以进行高效的向量相似性搜索：

```python
import lance
import numpy as np

# Open the dataset
ds = lance.dataset("./embeddings.lance")

# Create a query vector (same dimension as stored vectors)
query_vector = np.random.rand(128).astype(np.float32).tolist()

# Perform vector search - find 10 nearest neighbors
results = ds.to_table(
    nearest={
        "column": "vector",
        "q": query_vector,
        "k": 10,
    }
)
print(results.to_pandas())
```

对于大数据集的生产工作负载，创建向量索引可以实现更快的搜索：

```python
# Create an IVF-PQ index for fast approximate nearest neighbor search
ds.create_index(
    "vector",
    index_type="IVF_PQ",
    num_partitions=256,  # Number of IVF partitions
    num_sub_vectors=16,  # Number of PQ sub-vectors
)

# Search with the index (automatically used)
results = ds.to_table(
    nearest={
        "column": "vector",
        "q": query_vector,
        "k": 10,
        "nprobes": 20,  # Number of partitions to search
    }
)
```

### List 和 LargeList - 变长数组

对于每行可能具有不同元素数量的变长数组，使用 `List` 或 `LargeList`：

```python
import lance
import pyarrow as pa

# Schema with variable-length arrays
schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("tags", pa.list_(pa.utf8())),      # Variable number of string tags
    pa.field("scores", pa.list_(pa.float32())), # Variable number of float scores
])

table = pa.Table.from_pydict({
    "id": [1, 2, 3],
    "tags": [["python", "ml"], ["rust"], ["data", "analytics", "ai"]],
    "scores": [[0.9, 0.8], [0.95], [0.7, 0.85, 0.9]],
}, schema=schema)

ds = lance.write_dataset(table, "./variable_arrays.lance")
```

## 嵌套和复杂类型

### 结构体类型（Struct）

存储具有多个命名字段的结构化数据：

```python
import lance
import pyarrow as pa

# Schema with nested struct
schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("metadata", pa.struct([
        pa.field("source", pa.utf8()),
        pa.field("timestamp", pa.timestamp("us")),
        pa.field("embedding_model", pa.utf8()),
    ])),
    pa.field("vector", pa.list_(pa.float32(), 384)),  # 384-dim embedding
])

table = pa.Table.from_pydict({
    "id": [1, 2],
    "metadata": [
        {"source": "web", "timestamp": "2024-01-15T10:30:00", "embedding_model": "text-embedding-3-small"},
        {"source": "api", "timestamp": "2024-01-15T11:45:00", "embedding_model": "text-embedding-3-small"},
    ],
    "vector": [
        [0.1] * 384,
        [0.2] * 384,
    ],
}, schema=schema)

ds = lance.write_dataset(table, "./with_metadata.lance")
```

### Map 类型

存储具有动态键的键值对：
Map 写入需要 Lance 文件格式 2.2 或更高版本。

```python
import lance
import pyarrow as pa

schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("attributes", pa.map_(pa.utf8(), pa.utf8())),
])

table = pa.Table.from_pydict({
    "id": [1, 2],
    "attributes": [
        [("color", "red"), ("size", "large")],
        [("color", "blue"), ("material", "cotton")],
    ],
}, schema=schema)

ds = lance.write_dataset(table, "./with_maps.lance", data_storage_version="2.2")
```

## 集成的数据类型映射

当将 Lance 与其他系统（如 Apache Flink、Spark 或 Presto）集成时，适用以下类型映射：

| 外部类型 | Lance/Arrow 类型 | 说明 |
|---------|-----------------|------|
| `BOOLEAN` | `Boolean` | |
| `TINYINT` | `Int8` | |
| `SMALLINT` | `Int16` | |
| `INT` / `INTEGER` | `Int32` | |
| `BIGINT` | `Int64` | |
| `FLOAT` | `Float32` | |
| `DOUBLE` | `Float64` | |
| `DECIMAL(p,s)` | `Decimal128(p,s)` | |
| `STRING` / `VARCHAR` | `Utf8` | |
| `CHAR(n)` | `Utf8` | 在源系统中为固定宽度；存储为变长 Utf8 |
| `DATE` | `Date32` | |
| `TIME` | `Time64` | 微秒精度 |
| `TIMESTAMP` | `Timestamp` | |
| `TIMESTAMP WITH LOCAL TIMEZONE` | `Timestamp` | 带时区信息 |
| `BINARY` / `VARBINARY` | `Binary` | |
| `BYTES` | `Binary` | |
| `BLOB` | Blob v2 扩展类型（`lance.blob.v2`） | 新数据集使用 `blob_field` / `blob_array`；旧版元数据路径适用于 `data_storage_version <= 2.1` |
| `ARRAY<T>` | `List(T)` | 变长数组 |
| `ARRAY<T>(n)` | `FixedSizeList(T, n)` | 固定长度数组（向量） |
| `ROW` / `STRUCT` | `Struct` | 嵌套结构 |
| `MAP<K,V>` | `Map(K, V)` | 键值对 |

### 集成中的向量嵌入

对于向量嵌入列，使用 `ARRAY<FLOAT>(n)` 或 `ARRAY<DOUBLE>(n)`，其中 `n` 是嵌入维度：

```sql
-- Example: Creating a table with vector embeddings in SQL-compatible systems
CREATE TABLE embeddings (
    id BIGINT,
    text STRING,
    vector ARRAY<FLOAT>(384)  -- 384-dimensional vector
);
```

这映射到 Lance 的 `FixedSizeList(Float32, 384)` 类型，该类型针对以下方面进行了优化：

- 高效的列式存储
- SIMD 加速的距离计算
- 向量索引创建和搜索

## 向量数据最佳实践

1. **使用 FixedSizeList 存储嵌入向量**：始终使用 `FixedSizeList`（而非变长 `List`）存储向量嵌入，以实现高效的存储和索引。

2. **选择合适的精度**：
   - `Float32` 是标准选择，在精度和存储之间取得平衡
   - `Float16` 或 `BFloat16` 可以减少 50% 的存储空间，精度损失极小
   - `Int8` 用于量化嵌入

3. **对齐维度以优化 SIMD**：可被 8 整除的向量维度能启用最佳的 SIMD 加速。常见维度：128、256、384、512、768、1024、1536。

4. **为大数据集创建索引**：对于超过约 10,000 个向量的数据集，创建 ANN 索引以实现快速搜索：

    ```python
    # IVF_PQ is recommended for most use cases
    ds.create_index("vector", index_type="IVF_PQ", num_partitions=256, num_sub_vectors=16)
    
    # IVF_HNSW_SQ offers better recall at the cost of more memory
    ds.create_index("vector", index_type="IVF_HNSW_SQ", num_partitions=256)
    ```

5. **将元数据与向量一起存储**：Lance 能高效处理向量和标量数据混合的工作负载：

    ```python
    # Combine vector search with metadata filtering
    results = ds.to_table(
        filter="category = 'electronics'",
        nearest={"column": "vector", "q": query, "k": 10}
    )
    ```

## 另请参阅

- [向量搜索教程](../quickstart/vector-search.md) - Lance 向量搜索完整指南
- [Blob API 指南](blob.md) - 存储和检索大型二进制对象（视频、图像）
- [扩展数组](arrays.md) - ML 专用的特殊数组类型（BFloat16、图像）
- [性能指南](performance.md) - 大规模部署的优化技巧
