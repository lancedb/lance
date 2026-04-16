# 向量索引（Vector Indices）

Lance 提供了一个强大且可扩展的二级索引系统，用于高效的向量相似性搜索。所有向量索引都以普通 Lance 文件的形式存储，使其具有可移植性且易于管理。该系统专为大规模向量数据集上的高效相似性搜索而设计。

## 概念

Lance 将每个向量索引分为 3 个部分：聚类（Clustering）、子索引（Sub-Index）和量化（Quantization）。

### 聚类

聚类将所有向量划分为不同的不相交簇（也称为分区，Partitions）。Lance 目前支持使用倒排文件（IVF，Inverted File）作为主要的聚类机制。IVF 使用 k-means 聚类算法将向量划分为簇。每个簇包含与簇中心相似的向量。在搜索时，只检查最相关的簇，从而大幅减少搜索时间。IVF 可以与任何子索引类型和量化方法组合使用。

### 子索引

子索引决定了向量的搜索组织方式。Lance 目前支持：

- **FLAT**：精确搜索，无近似——扫描所有向量
- **HNSW**：分层可导航小世界（Hierarchical Navigable Small World）图，用于快速近似搜索

### 量化

量化方法决定了向量的存储和压缩方式。Lance 目前支持：

- **乘积量化（PQ，Product Quantization）**：通过将向量拆分为更小的子向量并独立量化每个子向量来压缩向量
- **标量量化（SQ，Scalar Quantization）**：对向量的每个维度独立应用标量量化
- **RabitQ（RQ）**：使用随机旋转和二值量化实现极致压缩
- **FLAT**：不进行量化，保留原始向量用于精确搜索

### 常用组合

当我们提到索引类型时，通常使用 `{聚类}_{子索引}_{量化}` 的格式。如果子索引只是 `FLAT`，通常省略它，仅用 `{聚类}_{量化}` 来表示。以下是常用的组合：

| 索引类型         | 名称                                            | 描述                                                                              |
| --------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **IVF_PQ**      | Inverted File with Product Quantization         | 结合 IVF 聚类和 PQ 压缩，实现高效存储和搜索             |
| **IVF_HNSW_SQ** | Inverted File with HNSW and Scalar Quantization | 使用 IVF 进行粗聚类，HNSW 进行细粒度搜索，配合标量量化 |
| **IVF_SQ**      | Inverted File with Scalar Quantization          | 结合 IVF 聚类和标量量化，实现平衡的压缩                |
| **IVF_RQ**      | Inverted File with RabitQ                       | 结合 IVF 聚类和 RabitQ，使用二值量化实现极致压缩    |
| **IVF_FLAT**    | Inverted File without quantization              | 使用 IVF 聚类配合精确向量存储，在簇内进行精确搜索         |

### 版本控制

Lance 向量索引格式目前经历了 3 个版本。本文档目前仅记录最新的版本 3。向量索引的具体版本记录在通用[索引元数据](../index.md#loading-an-index)的 `index_version` 字段中。

## 存储布局（V3）

每个向量索引以 2 个普通 Lance 文件的形式存储——索引文件和辅助文件。

### 索引文件

包含搜索图/结构的索引结构文件，具有索引特定的 schema。它以名为 `index.idx` 的 Lance 文件存储在索引目录中。

#### Arrow Schema

索引文件以图或扁平组织形式存储搜索结构。Lance 文件的 Arrow schema 根据所使用的子索引类型而有所不同。

!!! note
所有分区存储在同一文件中，且分区必须按顺序写入。

##### FLAT

FLAT 索引执行无近似的精确搜索。本质上是一个具有最小 schema 的空文件：

| 列               | 类型   | 可空   | 描述                                  |
| --------------- | ------ | -------- | -------------------------------------------- |
| `__flat_marker` | uint64 | false    | FLAT 索引的标记字段（无实际数据） |

##### HNSW

HNSW（分层可导航小世界）索引通过多层图结构提供快速近似搜索。它以如下 schema 存储 HNSW 图：

| 列             | 类型          | 可空   | 描述            |
| ------------- | ------------- | -------- | ---------------------- |
| `__vector_id` | uint64        | false    | 向量标识符      |
| `__neighbors` | list<uint32>  | false    | 邻居节点 ID      |
| `_distance`   | list<float32> | false    | 到邻居的距离 |

!!! note
HNSW 包含多个层级，所有层级必须从第 0 层开始按顺序写入。

#### Arrow Schema 元数据

索引文件在其 Arrow schema 元数据中包含描述索引配置和结构的元数据。以下是元数据键及其对应的值：

##### "lance:index"

包含 JSON 格式的基本索引配置信息：

| JSON 键          | 类型   | 期望值                                           |
| --------------- | ------ | --------------------------------------------------------- |
| `type`          | String | 索引类型（如 "IVF_PQ"、"IVF_RQ"、"IVF_HNSW"、"FLAT"） |
| `distance_type` | String | 距离度量（如 "l2"、"cosine"、"dot"）             |

##### "lance:ivf"

引用存储在 Lance 文件全局缓冲区中的 IVF 元数据。此值记录全局缓冲区索引，目前始终为 "1"。

!!! note
Lance 文件中的全局缓冲区索引从 1 开始，因此通过代码访问时需要减 1。

##### "lance:flat"

包含 `FLAT` 子索引结构的分区特定元数据。目前为空字符串，因为 FLAT 索引此时不需要额外的元数据。

##### "lance:hnsw"

包含每个分区的 HNSW 特定 JSON 元数据，包括图结构信息：

| JSON 键          | 类型         | 期望值                          |
| --------------- | ------------ | ---------------------------------------- |
| `entry_point`   | u32          | 图遍历的起始节点        |
| `params`        | Object       | HNSW 构建参数（见下文） |
| `level_offsets` | Array<usize> | 图中每个层级的偏移量       |

`params` 对象包含以下 HNSW 构建参数：

| JSON 键              | 类型          | 描述                                                    | 默认值  |
| ------------------- | ------------- | -------------------------------------------------------------- | ------- |
| `max_level`         | u16           | HNSW 图的最大层数                                | 7       |
| `m`                 | usize         | 插入新元素时建立的连接数 | 20      |
| `ef_construction`   | usize         | 候选动态列表的大小                        | 150     |
| `prefetch_distance` | Option<usize> | 构建时提前预取的向量数             | Some(2) |

#### Lance 文件全局缓冲区

##### IVF 元数据

为了效率，Lance 将 IVF 元数据序列化为 protobuf 格式并存储在 Lance 文件全局缓冲区中：

```protobuf
%%% proto.message.IVF %%%
```

### 辅助文件

辅助文件是量化向量的向量存储。它以名为 `auxiliary.idx` 的 Lance 文件存储在索引目录中。

#### Arrow Schema

由于辅助文件存储实际的（量化后的）向量，Lance 文件的 Arrow schema 根据所使用的量化方法而有所不同。

!!! note
所有分区存储在同一文件中，且分区必须按顺序写入。

##### FLAT

未应用量化——以完整精度存储原始向量：

| 列       | 类型                     | 可空   | 描述                                           |
| -------- | ------------------------ | -------- | ----------------------------------------------------- |
| `_rowid` | uint64                   | false    | 行标识符                                        |
| `flat`   | list<float32>[dimension] | false    | 原始向量值（list_size = 向量维度） |

##### PQ

使用乘积量化压缩向量以显著节省内存：

| 列           | 类型           | 可空   | 描述                                 |
| ----------- | -------------- | -------- | ------------------------------------------- |
| `_rowid`    | uint64         | false    | 行标识符                              |
| `__pq_code` | list<uint8>[m] | false    | PQ 编码（list_size = 子向量数量） |

##### SQ

使用标量量化压缩向量以适度节省内存：

| 列           | 类型                   | 可空   | 描述                             |
| ----------- | ---------------------- | -------- | --------------------------------------- |
| `_rowid`    | uint64                 | false    | 行标识符                          |
| `__sq_code` | list<uint8>[dimension] | false    | SQ 编码（list_size = 向量维度） |

##### RQ

使用 RabitQ 通过随机旋转和二值量化压缩向量以实现极致压缩：

| 列                 | 类型                       | 可空   | 描述                                                     |
| ----------------- | -------------------------- | -------- | --------------------------------------------------------------- |
| `_rowid`          | uint64                     | false    | 行标识符                                                  |
| `_rabit_codes`    | list<uint8>[dimension / 8] | false    | 二值量化编码（每维度 1 位，打包为字节） |
| `__add_factors`   | float32                    | false    | 距离计算的加性修正因子            |
| `__scale_factors` | float32                    | false    | 距离计算的缩放修正因子               |

#### Arrow Schema 元数据

辅助文件也在其 Arrow schema 元数据中包含向量存储配置的元数据。以下是元数据键及其对应的值：

##### "distance_type"

用于计算向量间相似度的距离度量（如 "l2"、"cosine"、"dot"）。

##### "lance:ivf"

与索引文件的 "lance:ivf" 类似，但侧重于向量存储布局。不包含分区的质心（Centroids）。仅用于跟踪辅助文件中每个分区的偏移量和长度。

##### "lance:rabit"

包含 JSON 格式的 RabitQ 特定元数据（仅在 RQ 量化时存在）。包括旋转矩阵位置、位数和打包信息。详见下方 "storage_metadata" 部分中的 RQ 元数据规范。

##### "storage_metadata"

包含量化器特定元数据的 JSON 字符串列表。目前，该列表始终只包含 1 个元素，即量化器元数据。

对于**乘积量化（PQ）**：

| JSON 键              | 类型  | 描述                                                      |
| ------------------- | ----- | ---------------------------------------------------------------- |
| `codebook_position` | usize | 码本在全局缓冲区中的位置                    |
| `nbits`             | u32   | 每个子向量编码的位数（如 8 位 = 256 个码字） |
| `num_sub_vectors`   | usize | 子向量数量（m）                                         |
| `dimension`         | usize | 原始向量维度                                        |
| `transposed`        | bool  | 码本是否以转置布局存储              |

对于**标量量化（SQ）**：

| JSON 键     | 类型       | 描述                            |
| ---------- | ---------- | -------------------------------------- |
| `dim`      | usize      | 向量维度                       |
| `num_bits` | u16        | 量化位数        |
| `bounds`   | Range<f64> | 标量量化的最小/最大边界 |

对于**RabitQ（RQ）**：

| JSON 键                | 类型 | 描述                                          |
| --------------------- | ---- | ---------------------------------------------------- |
| `rotate_mat_position` | u32  | 旋转矩阵在全局缓冲区中的位置 |
| `num_bits`            | u8   | 每维度的位数（目前始终为 1）    |
| `packed`              | bool | 编码是否打包以优化计算   |

#### Lance 文件全局缓冲区

##### 量化码本

对于乘积量化，码本以 `Tensor` 格式存储在辅助文件的全局缓冲区中以实现高效访问：

```protobuf
%%% proto.message.Tensor %%%
```

##### 旋转矩阵

对于 RabitQ，旋转矩阵以 `Tensor` 格式存储在辅助文件的全局缓冲区中。旋转矩阵是一个正交矩阵，用于在二值量化之前旋转向量：

```protobuf
%%% proto.message.Tensor %%%
```

旋转矩阵的形状为 `[code_dim, code_dim]`，其中 `code_dim = dimension * num_bits`。

## 附录

### 附录 1：IVF_PQ 格式示例

本示例展示了 `IVF_PQ` 索引的物理布局。假设向量维度为 128，PQ 使用 16 个子向量（m=16），每个子向量 8 位（num_bits=8），距离类型为 "l2"。

#### 索引文件

- Arrow Schema 元数据：
  - `"lance:index"` -> `{ "type": "IVF_PQ", "distance_type": "l2" }`
  - `"lance:ivf"` -> "1"（引用全局缓冲区中的 IVF 元数据）
  - `"lance:flat"` -> `["", "", ...]`（每个分区一个空字符串；IVF_PQ 在每个分区内部使用 FLAT 子索引）

- Lance 文件全局缓冲区（Protobuf）：
  - `Ivf` 消息包含：
    - `centroids_tensor`：形状 `[num_partitions, 128]`（float32）
    - `offsets`：每个分区在 `auxiliary.idx` 中的起始偏移量（行）
    - `lengths`：每个分区的向量数量
    - `loss`：k-means 损失（可选）

#### 辅助文件

- Arrow Schema 元数据：
  - `"distance_type"` -> `"l2"`
  - `"lance:ivf"` -> 跟踪每个分区的 `offsets` 和 `lengths`（此处无质心）
  - `"storage_metadata"` -> `[ "{"pq":{"num_sub_vectors":16,"nbits":8,"dimension":128,"transposed":true}}" ]`
- Lance 文件全局缓冲区：
  - `Tensor` 码本，形状 `[256, num_sub_vectors, dim/num_sub_vectors]` = `[256, 16, 8]`（float32）
- 行数据的 Arrow schema：

```python
pa.schema([
    pa.field("_rowid", pa.uint64()),
    pa.field("__pq_code", pa.list(pa.uint8(), list_size=16)), # m subvector codes
])
```

### 附录 2：IVF_RQ 格式示例

本示例展示了 `IVF_RQ` 索引的物理布局。假设向量维度为 128，RQ 每维度使用 1 位（num_bits=1），距离类型为 "l2"。

#### 索引文件

- Arrow Schema 元数据：
  - `"lance:index"` -> `{ "type": "IVF_RQ", "distance_type": "l2" }`
  - `"lance:ivf"` -> "1"（引用全局缓冲区中的 IVF 元数据）
  - `"lance:flat"` -> `["", "", ...]`（每个分区一个空字符串；IVF_RQ 在每个分区内部使用 FLAT 子索引）

- Lance 文件全局缓冲区（Protobuf）：
  - `Ivf` 消息包含：
    - `centroids_tensor`：形状 `[num_partitions, 128]`（float32）
    - `offsets`：每个分区在 `auxiliary.idx` 中的起始偏移量（行）
    - `lengths`：每个分区的向量数量
    - `loss`：k-means 损失（可选）

#### 辅助文件

- Arrow Schema 元数据：
  - `"distance_type"` -> `"l2"`
  - `"lance:ivf"` -> 跟踪每个分区的 `offsets` 和 `lengths`（此处无质心）
  - `"lance:rabit"` -> `"{"rotate_mat_position":1,"num_bits":1,"packed":true}"`
- Lance 文件全局缓冲区：
  - `Tensor` 旋转矩阵，形状 `[code_dim, code_dim]` = `[128, 128]`（float32）
- 行数据的 Arrow schema：

```python
pa.schema([
    pa.field("_rowid", pa.uint64()),
    pa.field("_rabit_codes", pa.list(pa.uint8(), list_size=16)), # dimension/8 = 128/8 = 16 bytes
    pa.field("__add_factors", pa.float32()),
    pa.field("__scale_factors", pa.float32()),
])
```

### 附录 3：使用 Python 访问索引文件

以下示例演示如何使用 Python 读取和解析 Lance 索引文件中的不同组件：

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
