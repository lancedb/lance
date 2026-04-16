# 存储布局规范（Storage Layout Specification）

## 概述

本规范定义了 Lance 数据集在对象存储上的组织方式。
布局设计强调可移植性，允许数据集通过最小的元数据更改在多个存储系统之间重新定位或引用。

## 数据集根目录（Dataset Root）

数据集根目录是数据集最初创建的位置。
每个 Lance 数据集恰好有一个数据集根目录，作为数据集文件的主要存储位置。
数据集根目录包含标准子目录结构（`data/`、`_versions/`、`_deletions/`、`_indices/`、`_refs/`、`tree/`），用于组织数据集的文件。

## 基本布局

Lance 数据集在其基本形式中将所有文件存储在数据集根目录结构中：

```
{dataset_root}/
    data/
        *.lance           -- 包含列数据的数据文件
    _versions/
        *.manifest        -- 清单文件（每个版本一个）
    _transactions/
        *.txn             -- 用于提交协调的事务文件
    _deletions/
        *.arrow           -- 删除向量文件（Arrow 格式）
        *.bin             -- 删除向量文件（位图格式）
    _indices/
        {UUID}/
            ...           -- 索引内容（不同索引类型不同）
    _refs/
        tags/
            *.json        -- 标签元数据
        branches/
            *.json        -- 分支元数据
    tree/
        {branch_name}/
            ...           -- 分支数据集
            
```

## 基础路径系统（Base Path System） { #base-path-system }

### BasePath 消息

清单的 `base_paths` 字段包含一个 `BasePath` 条目数组，定义数据集文件的替代存储位置。
每个基础路径条目有一个唯一的数字标识符，文件元数据可以引用它来指示文件的存储位置。
`path` 字段指定对象存储可解释的绝对路径。
`is_dataset_root` 字段决定路径的解释方式：当为 true 时，路径指向具有标准子目录（`data/`、`_deletions/`、`_indices/`）的数据集根目录；当为 false 时，路径直接指向文件目录，不含子目录。
可选的 `name` 字段提供人类可读的别名，在浅克隆中引用标签时特别有用。

<details>
<summary>BasePath protobuf 消息</summary>

```protobuf
message BasePath {
  uint32 id = 1;
  optional string name = 2;
  bool is_dataset_root = 3;
  string path = 4;
}
```

</details>

### 文件元数据基础引用

三种类型的文件可以指定替代基础路径：数据文件、删除文件和索引元数据。
这些文件类型中的每一种都在其元数据中包含一个可选的 `base_id` 字段，通过数字标识符引用基础路径条目。
当文件的 `base_id` 不存在时，文件相对于数据集根目录定位。
当文件的 `base_id` 存在时，读取器必须在清单的 `base_paths` 数组中查找对应的基础路径条目以确定文件的存储位置。

在读取时，路径解析遵循两步过程。
首先，读取器确定基础路径：如果 `base_id` 不存在，基础路径为数据集根目录；否则，读取器使用 `base_id` 查找基础路径条目以获取路径及其 `is_dataset_root` 标志。
其次，读取器根据基础路径是否表示数据集根目录来构建完整的文件路径。
对于数据集根目录（`is_dataset_root` 为 true 时），完整路径包含标准子目录：数据文件位于 `data/` 下，删除文件位于 `_deletions/` 下，索引位于 `_indices/` 下。
对于非根基础路径（`is_dataset_root` 为 false 时），基础路径直接指向文件目录，文件路径直接追加而不带子目录前缀。

### 复杂布局场景示例

#### 冷热分层（Hot/Cold Tiering）

```
Manifest base_paths:
[
  { id: 0, is_dataset_root: true, path: "s3://hot-bucket/dataset" },
  { id: 1, is_dataset_root: true, path: "s3://cold-bucket/dataset-archive" }
]

Fragment 0 (近期数据):
  DataFile { path: "fragment-0.lance", base_id: 0 }
  → 解析为: s3://hot-bucket/dataset/data/fragment-0.lance

Fragment 100 (历史数据):
  DataFile { path: "fragment-100.lance", base_id: 1 }
  → 解析为: s3://cold-bucket/dataset-archive/data/fragment-100.lance
```

这允许无需数据移动即可跨存储层进行无缝查询。

#### 多区域分布（Multi-Region Distribution）

```
Manifest base_paths:
[
  { id: 0, is_dataset_root: true, path: "s3://us-east-bucket/dataset" },
  { id: 1, is_dataset_root: true, path: "s3://eu-west-bucket/dataset" },
  { id: 2, is_dataset_root: true, path: "s3://ap-south-bucket/dataset" }
]

按数据局部性分布的片段：
  Fragment 0 (美国用户): base_id: 0
  Fragment 1 (欧洲用户): base_id: 1
  Fragment 2 (亚洲用户): base_id: 2
```

计算任务可以从最近的区域读取数据而无需数据传输。

#### 浅克隆（Shallow Clone） { #shallow-clone }

浅克隆创建一个引用源数据集数据文件而不复制的新数据集：

**示例：浅克隆**

```
源数据集:   s3://production/main-dataset
克隆数据集: s3://experiments/test-variant

克隆清单 base_paths:
[
  { id: 0, is_dataset_root: true, path: "s3://experiments/test-variant" },
  { id: 1, is_dataset_root: true, path: "s3://production/main-dataset",
    name: "v1.0" }
]

原始片段（继承的）：
  DataFile { path: "fragment-0.lance", base_id: 1 }
  → 解析为: s3://production/main-dataset/data/fragment-0.lance

新片段（克隆特有的）：
  DataFile { path: "fragment-new.lance", base_id: 0 }
  → 解析为: s3://experiments/test-variant/data/fragment-new.lance
```

克隆可以追加新数据、修改模式或删除行，而不影响源数据集。
只有清单和新数据文件存储在克隆位置。

**工作流程：**

1. [Clone 事务](transaction.md#clone)在目标位置创建新清单
2. 清单包含指向源数据集的基础路径
3. 原始片段通过 `base_id: 1` 引用源数据集
4. 后续写入通过 `base_id: 0` 引用克隆位置
5. 源数据集保持不可变，可以独立进行垃圾回收

## 数据集可移植性

基础路径系统结合相对文件引用为 Lance 数据集提供了强大的可移植性保证。
Lance 文件中的所有文件路径都相对于其所在目录存储，使得数据集可以在不修改文件的情况下重新定位。

要将数据集移植到新位置，只需复制数据集根目录中的所有内容。
复制的数据集将在新位置立即可用，无需任何清单更新，因为数据集根目录中的所有文件引用都通过相对路径解析。

当数据集使用多个基础路径时（例如浅克隆或多桶配置），用户可以灵活选择如何移植数据集。
最简单的方法是只复制数据集根目录，这将保留对原始基础路径位置的引用。
或者，用户可以将额外的基础路径复制到新位置，并更新清单的 `base_paths` 数组以反映新的基础路径。
由于只需修改清单中的 `base_paths` 字段，这仍然是一个轻量级的元数据操作，不需要重写额外的元数据或数据文件。

## 文件命名约定

### 数据文件

模式：`data/{uuid-based-filename}.lance`

数据文件使用基于 UUID 的文件名，为 S3 吞吐量优化。
文件名由 UUID（16 字节）生成，将前 3 字节转换为 24 字符的二进制字符串，其余 13 字节转换为 26 字符的十六进制字符串，产生 50 字符的文件名。
二进制前缀（而非十六进制）提供了每字符最大的熵，允许 S3 的内部分区快速识别访问模式并适当扩展，最大限度减少限流。

示例：`data/101100101101010011010110a1b2c3d4e5f6g7h8i9j0.lance`

### 删除文件

模式：`_deletions/{fragment_id}-{read_version}-{id}.{extension}`

删除文件使用两种扩展名：`.arrow` 用于 Arrow IPC 格式（稀疏删除），`.bin` 用于 Roaring 位图格式（密集删除）。

示例：`_deletions/42-10-a1b2c3d4.arrow`

### 事务文件

模式：`_transactions/{read_version}-{uuid}.txn`

其中 `read_version` 是构建事务时所基于的表版本。

示例：`_transactions/5-550e8400-e29b-41d4-a716-446655440000.txn`

### 清单文件

清单文件存储在 `_versions/` 目录中，命名方案支持原子提交。

有关 V1 和 V2 模式及其对版本发现的影响的详细信息，请参阅[清单命名方案](transaction.md#manifest-naming-schemes)。
