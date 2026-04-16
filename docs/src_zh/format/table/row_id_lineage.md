# 行 ID 与血统规范（Row ID and Lineage Specification）

## 概述

Lance 提供行标识和血统跟踪功能。
行寻址（Row Addressing）通过物理位置编码实现对表中行的高效随机访问。
稳定行 ID（Stable Row ID）提供在行的整个生命周期内保持不变的持久标识符，即使其物理位置发生变化。
行版本跟踪（Row Version Tracking）记录行的创建和最后修改时间，支持增量处理、变更数据捕获（Change Data Capture）和时间旅行查询。

## 行标识符形式

Lance 中的行有两种形式的行标识符：

- **行地址（Row Address）**- 行在数据集中的当前物理位置。
- **行 ID（Row ID）**- 行的逻辑标识符。当启用稳定行 ID 时，该标识符在逻辑行的整个生命周期内保持稳定。当禁用时（默认模式），它等同于行地址。


### 行地址（Row Address）

行地址是行在表中的物理位置，表示为由两个 32 位值组成的 64 位标识符：

```
row_address = (fragment_id << 32) | local_row_offset
```

这种寻址方案实现了高效的随机访问：给定行地址，通过位操作即可提取片段和偏移量。
当数据通过压缩或更新进行重组时，行地址会发生变化。

行地址目前是用于索引目的的主要标识符形式。
二级索引（向量索引、标量索引、全文搜索索引）通过行地址引用行。

!!! note
      支持在索引中使用稳定行 ID 的工作正在进行中。

### 行 ID（Row ID）

行 ID 是行的逻辑标识符。

#### 稳定行 ID（Stable Row ID）

当数据集创建时启用了稳定行 ID，每行将被分配一个唯一的自增 `u64` 标识符，该标识符在行的整个生命周期内保持不变，即使行的物理位置（行地址）发生变化。
`_rowid` 系统列将此逻辑标识符暴露给用户。
有关分配和更新语义的更多详情，请参阅下一节。

#### 历史/不稳定用法

从历史上看，"行 ID"这个术语经常用来指代物理行地址（`_rowaddr`），该地址在压缩或更新后并不稳定。

!!! warning
      随着稳定行 ID 的引入，代码和文档中可能仍然存在混淆"行 ID"和"行地址"或"行 ID"和"稳定行 ID"的地方。
      如果您发现任何不正确或令人困惑的地方，请提交 PR。

## 稳定行 ID（Stable Row ID）

### 行 ID 分配

行 ID 使用存储在清单中的单调递增 `next_row_id` 计数器进行分配。

**分配协议：**

1. 写入者从读取版本的清单中读取当前 `next_row_id`
2. 写入者为新行从 `next_row_id` 开始顺序分配行 ID
3. 写入者在新清单中将 `next_row_id` 更新为 `next_row_id + num_new_rows`
4. 如果由于冲突导致提交失败，写入者进行变基：
   - 从最新版本重新读取新的 `next_row_id`
   - 使用更新的计数器为新行重新分配行 ID
   - 重试提交

此协议与片段 ID 分配类似，确保行 ID 在所有表版本中唯一。

### 启用稳定行 ID

稳定行 ID 是一个数据集级别的功能，记录在表清单中。

- 稳定行 ID **必须在数据集首次创建时启用**。
- 目前，它们**不能在现有数据集上后续开启**。尝试对未启用稳定行 ID 的数据集使用 `enable_stable_row_ids = true` 进行写入不会更改数据集的配置。
- 当稳定行 ID 被禁用时，`_rowid` 列（如果请求的话）不是稳定的，不应用作持久标识符。

行级版本跟踪（`_row_created_at_version`、`_row_last_updated_at_version`）和下面描述的行 ID 索引仅在启用稳定行 ID 时可用。

### 更新时的行 ID 行为

当启用稳定行 ID 时，更新操作会保留逻辑行 ID 并将其重新映射到新的物理地址，而不是分配新的 ID。

**更新工作流程：**

1. 具有 `_rowid = R` 的原始行存在于地址 `(F1, O1)`。
2. 更新操作将具有更新值的新物理行写入地址 `(F2, O2)`。
3. 新物理行被分配相同的 `_rowid = R`，因此逻辑标识符得以保留。
4. 位于 `(F1, O1)` 的原始物理行使用片段 `F1` 的删除向量标记为已删除。
5. 新数据集版本的行 ID 索引将 `_rowid = R` 映射到 `(F2, O2)`，并使用删除向量和片段位图避免返回 `(F1, O1)` 处的墓碑行。

这种设计使 `_rowid` 在逻辑行的整个生命周期内保持稳定，同时允许物理存储和二级索引独立维护。

### 行 ID 序列

#### 存储格式

行 ID 序列使用 `RowIdSequence` protobuf 消息存储。
序列被分成多个段，每个段根据数据模式进行最优编码。

<details>
<summary>RowIdSequence protobuf 消息</summary>

```protobuf
%%% proto.message.RowIdSequence %%%
```

</details>

#### 段编码

每个段使用五种编码之一，针对不同的数据模式进行优化：

##### 范围（Range，连续值）

用于已排序的、无间隙的连续值。
示例：行 ID `[100, 101, 102, 103, 104]` → `Range{start: 100, end: 105}`。
用于按顺序分配行 ID 的新片段。

<details>
<summary>Range protobuf 消息</summary>

```protobuf
%%% proto.message.Range %%%
```

</details>

##### 带空洞的范围（Range with Holes，稀疏删除）

用于具有少量间隙的已排序值。
示例：行 ID `[100, 101, 103, 104]`（缺少 102）→ `RangeWithHoles{start: 100, end: 105, holes: [102]}`。
用于具有稀疏删除的片段，此时维护范围更高效。

<details>
<summary>RangeWithHoles protobuf 消息</summary>

```protobuf
%%% proto.message.RangeWithHoles %%%
```

</details>

##### 带位图的范围（Range with Bitmap，密集删除）

用于具有大量间隙的已排序值。
位图每字节编码 8 个值，最高有效位表示第一个值。
用于具有密集删除模式的片段。

<details>
<summary>RangeWithBitmap protobuf 消息</summary>

```protobuf
%%% proto.message.RangeWithBitmap %%%
```

</details>

##### 排序数组（Sorted Array，稀疏值）

用于已排序但不连续的值，以 `EncodedU64Array` 形式存储。
用于合并后的片段或压缩后的片段。

##### 未排序数组（Unsorted Array，通用情况）

用于未排序的值，以 `EncodedU64Array` 形式存储。
较少见；大多数操作保持排序顺序。

#### 编码的 U64 数组

`EncodedU64Array` 消息支持位打包（Bitpacked）编码以最小化存储。
实现根据值范围选择最紧凑的编码，在基址 + 16 位偏移、基址 + 32 位偏移或完整 64 位值之间进行选择。

<details>
<summary>EncodedU64Array protobuf 消息</summary>

```protobuf
%%% proto.message.EncodedU64Array %%%
```

</details>

#### 内联与外部存储

行 ID 序列存储为内联（Inline）在片段元数据中或存储在外部文件中。
小于约 200KB 的序列以内联方式存储以避免额外的 I/O，而较大的序列写入由路径和偏移量引用的外部文件。
此阈值在清单大小和单独文件读取开销之间取得平衡。

<details>
<summary>DataFragment row_id_sequence 字段</summary>

```protobuf
message DataFragment {
  oneof row_id_sequence {
    bytes inline_row_ids = 5;
    ExternalFile external_row_ids = 6;
  }
}
```

</details>

### 行 ID 索引

#### 构建

行 ID 索引在表加载时通过聚合所有片段的行 ID 序列来构建：

```
对于每个片段 F（ID 为 f）：
  对于 F.row_id_sequence 中的每个 (position p, row_id r)：
    index[r] = (f, p)
```

这创建了从行 ID 到当前行地址的映射。

#### 更新时的索引失效

当行被更新且启用了稳定行 ID 时，给定数据集版本的行 ID 索引仅包含活跃物理行的映射。墓碑行通过删除向量排除，内容已更改的逻辑行 ID 仅映射到新的行地址。

**示例场景：**

1. 初始状态（版本 V）：片段 1 包含 ID 为 `[1, 2, 3]` 的行，偏移量为 `[0, 1, 2]`。
2. 更新操作修改了 `_rowid = 2` 的行：
    - 创建新的片段 2，其中包含 `_rowid = 2` 的行，偏移量为 `0`。
    - 在片段 1 中，偏移量 `1` 处的原始物理行在删除向量中标记为已删除。
3. 版本 V+1 的行 ID 索引：
    - `1 → (1, 0)` ✓ 有效
    - `2 → (2, 0)` ✓ 有效（片段 2 中更新后的行）
    - `3 → (1, 2)` ✓ 有效

地址 `(1, 1)` 不再可通过行 ID 索引访问，因为在构建索引时它已被删除向量过滤掉。

#### 用于索引掩码的片段位图

二级索引使用片段位图（Fragment Bitmap）来跟踪哪些行 ID 仍然有效：

**无行更新时：**

```
字符串索引，列 "str"：
  Fragment Bitmap: {1, 2}  (覆盖片段 1 和 2)
  所有已索引的行地址都有效
```

**有行更新时：**

```
向量索引，列 "vec"：
  Fragment Bitmap: {1}  (仅片段 1)
  _rowid = 2 的行已被更新，因此指向其旧物理地址的索引条目已过时
  索引查询使用删除向量过滤掉过时的地址，同时返回新地址处的行
```

这种基于位图的方法允许索引保持不可变，同时考虑行的修改。

## 行版本跟踪

行版本跟踪适用于使用稳定行 ID 的数据集。版本序列与每个片段内稳定的 `_rowid` 排序对齐。

### 创建版本（Created At Version）

每行跟踪其创建时的版本。
对于后来被更新的行，此创建版本保持为行首次出现的版本；更新不会改变它。
序列使用游程编码（Run-Length Encoding）进行高效存储，每个游程指定连续行的跨度及其创建时的版本。

示例：在版本 5 中创建的包含 1000 行的片段：
```
RowDatasetVersionSequence {
  runs: [
    RowDatasetVersionRun { span: Range{start: 0, end: 1000}, version: 5 }
  ]
}
```

<details>
<summary>DataFragment created_at_version_sequence 字段</summary>

```protobuf
message DataFragment {
  oneof created_at_version_sequence {
    bytes inline_created_at_versions = 9;
    ExternalFile external_created_at_versions = 10;
  }
}
```

</details>

<details>
<summary>RowDatasetVersionSequence protobuf 消息</summary>

```protobuf
%%% proto.message.RowDatasetVersionSequence %%%
```

</details>

### 最后更新版本（Last Updated At Version）

每行跟踪其最后修改时的版本。
当行创建时，`last_updated_at_version` 等于 `created_at_version`。

当启用稳定行 ID 并更新行时，Lance 为相同的逻辑 `_rowid` 写入新的物理行，同时将旧物理行标记为墓碑。该逻辑行的 `created_at_version` 从原始行保留，`last_updated_at_version` 设置为更新时的当前数据集版本。

示例：在版本 3 创建、在版本 7 更新的行：
```
旧物理行（已墓碑标记）：
  _rowid: R
  created_at_version: 3
  last_updated_at_version: 3

新物理行（当前）：
  _rowid: R
  created_at_version: 3
  last_updated_at_version: 7
```

<details>
<summary>DataFragment last_updated_at_version_sequence 字段</summary>

```protobuf
message DataFragment {
  oneof last_updated_at_version_sequence {
    bytes inline_last_updated_at_versions = 7;
    ExternalFile external_last_updated_at_versions = 8;
  }
}
```

</details>

## 变更数据馈送（Change Data Feed）

Lance 支持通过版本跟踪列查询在版本之间发生变更的行。
这些查询可以表示为 `_row_created_at_version` 和 `_row_last_updated_at_version` 列上的标准 SQL 谓词。

### 插入的行

通过过滤 `_row_created_at_version` 可以检索在两个版本之间创建的行：

```sql
SELECT * FROM dataset
WHERE _row_created_at_version > {begin_version}
  AND _row_created_at_version <= {end_version}
```

此查询返回在指定版本范围内插入的所有行，包括版本元数据列 `_row_created_at_version`、`_row_last_updated_at_version` 和 `_rowid`。

### 更新的行

通过组合两个版本列上的过滤器，可以检索在两个版本之间被修改（但非新创建）的行：

```sql
SELECT * FROM dataset
WHERE _row_created_at_version <= {begin_version}
  AND _row_last_updated_at_version > {begin_version}
  AND _row_last_updated_at_version <= {end_version}
```

此查询通过要求 `_row_created_at_version <= {begin_version}` 来排除新插入的行，确保只返回预先存在的、随后被更新的行。
