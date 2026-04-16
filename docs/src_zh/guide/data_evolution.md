# 数据演进（Data Evolution）

Lance 支持传统的模式演进（Schema Evolution）：添加、删除和修改数据集中的列。大多数这些操作可以在*不重写*数据集中数据文件的情况下执行，使其成为非常高效的操作。此外，Lance 支持**数据演进（Data Evolution）**，允许你在不重写数据文件的情况下回填现有行的新列数据，非常适合 ML 特征工程等场景。

一般来说，模式变更会与大多数其他并发写入操作冲突。例如，如果你在其他人向数据集追加数据时更改模式，你的模式变更或追加操作之一会失败，具体取决于操作的顺序。因此，建议在没有其他写入操作时执行模式变更。

## 添加新列

### 仅模式（Schema Only）

我们在生产中看到的一个常见用例是向数据集添加新列但不填充数据。这对于稍后运行大型分布式作业来延迟填充列很有用。为此，你可以使用 `lance.LanceDataset.add_columns` 方法通过 `pyarrow.Field` 或 `pyarrow.Schema` 添加列。

```python
table = pa.table({"id": pa.array([1, 2, 3])})
dataset = lance.write_dataset(table, "null_columns")

# With pyarrow Field
dataset.add_columns(pa.field("embedding", pa.list_(pa.float32(), 128)))
assert dataset.schema == pa.schema([
    ("id", pa.int64()),
    ("embedding", pa.list_(pa.float32(), 128)),
])

# With pyarrow Schema
dataset.add_columns(pa.schema([
    ("label", pa.string()),
    ("score", pa.float32()),
]))
assert dataset.schema == pa.schema([
    ("id", pa.int64()),
    ("embedding", pa.list_(pa.float32(), 128)),
    ("label", pa.string()),
    ("score", pa.float32()),
])
```

此操作非常快，因为它只更新数据集的元数据。

对于 Lance 文件格式 `<= 2.1`，不支持在现有 `struct` 下添加子列。从 Lance 文件格式 `2.2` 开始，仅模式的添加也可以扩展嵌套的 `struct` 字段（包括嵌套在 list 类型内的 `struct` 字段），例如在 `list<struct<...>>` 下添加 `people.item.location`。

### 带数据回填

可以使用 `lance.LanceDataset.add_columns` 方法在单个操作中添加并填充新列。有两种方式指定如何填充新列：第一种是为每个新列提供 SQL 表达式，第二种是提供生成新列数据的函数。

SQL 表达式可以是独立表达式或引用现有列。SQL 字面值可用于为所有现有行设置单个值。

```python
table = pa.table({"name": pa.array(["Alice", "Bob", "Carla"])})
dataset = lance.write_dataset(table, "names")
dataset.add_columns({
    "hash": "sha256(name)",
    "status": "'active'",
})
print(dataset.to_table().to_pandas())
#     name                                               hash  status
# 0  Alice  b';\xc5\x10b\x97<E\x8dZo-\x8dd\xa0#$cT\xad~\x0...  active
# 1    Bob  b'\xcd\x9f\xb1\xe1H\xcc\xd8D.Z\xa7I\x04\xccs\x...  active
# 2  Carla  b'\xad\x8d\x83\xff\xd8+Z\x8e\xd4)\xe8Y+\\\xb3\...  active
```

你也可以提供 Python 函数来生成新列数据。例如，可以用它来计算新的嵌入列。此函数应接受一个 PyArrow RecordBatch 并返回 PyArrow RecordBatch 或 Pandas DataFrame。该函数将为数据集中的每个批次调用一次。

如果函数计算成本较高且可能失败，建议在 UDF 中设置检查点文件。此检查点文件在每次调用后保存 UDF 的状态，这样如果 UDF 失败，可以从最后一个检查点重新开始。请注意，此文件可能会变得很大，因为它需要存储最多整个数据文件的未保存结果。

```python
import lance
import pyarrow as pa
import numpy as np

table = pa.table({"id": pa.array([1, 2, 3])})
dataset = lance.write_dataset(table, "ids")

@lance.batch_udf(checkpoint_file="embedding_checkpoint.sqlite")
def add_random_vector(batch):
    embeddings = np.random.rand(batch.num_rows, 128).astype("float32")
    return pa.RecordBatch.from_arrays(
        [pa.FixedSizeListArray.from_arrays(embeddings.flatten(), 128)],
        names=["embedding"]
    )
dataset.add_columns(add_random_vector)
```

### 使用 merge

如果你已经预计算了一个或多个新列，可以使用 `lance.LanceDataset.merge` 方法将它们添加到现有数据集中。这允许填充额外的列而无需重写整个数据集。

要使用 `merge` 方法，提供一个包含要添加列的新数据集，以及用于将新数据与现有数据集连接的列名。

例如，假设我们有一个嵌入向量和 ID 的数据集：

```python
table = pa.table({
   "id": pa.array([1, 2, 3]),
   "embedding": pa.array([np.array([1, 2, 3]), np.array([4, 5, 6]),
                          np.array([7, 8, 9])])
})
dataset = lance.write_dataset(table, "embeddings", mode="overwrite")
```

现在如果我们想添加已生成的标签列，可以通过合并新表来实现：

```python
new_data = pa.table({
   "id": pa.array([1, 2, 3]),
   "label": pa.array(["horse", "rabbit", "cat"])
})
dataset.merge(new_data, "id")
print(dataset.to_table().to_pandas())
#    id  embedding   label
# 0   1  [1, 2, 3]   horse
# 1   2  [4, 5, 6]  rabbit
# 2   3  [7, 8, 9]     cat
```

## 删除列

最后，你可以使用 `lance.LanceDataset.drop_columns` 方法从数据集中删除列。这是仅元数据操作，不会删除磁盘上的数据，因此非常快。

```python
table = pa.table({"id": pa.array([1, 2, 3]),
                 "name": pa.array(["Alice", "Bob", "Carla"])})
dataset = lance.write_dataset(table, "names", mode="overwrite")
dataset.drop_columns(["name"])
print(dataset.schema)
# id: int64
```

从 Lance 文件格式 `2.2` 开始，支持嵌套子列的删除（例如 `list<struct<...>>` 上的 `people.item.city`），而不仅限于 `struct`。

要实际从磁盘删除数据，必须重写文件以移除列，然后删除旧文件。这可以使用 `lance.dataset.DatasetOptimizer.compact_files()` 接着 `lance.LanceDataset.cleanup_old_versions()` 来完成。

!!! warning

    `drop_columns` 是仅元数据操作，只要保留旧版本就可以撤销。
    在 `compact_files()` 重写数据文件和 `cleanup_old_versions()` 移除旧的清单/文件之后，
    移除的数据可能会永久不可恢复。

    对于生产工作流，请使用回滚窗口：
    - 在嵌套列删除之前创建标签（或快照/备份）
    - 延迟清理直到回滚窗口过去
    - 仅在回滚验证后执行积极清理

## 重命名列

可以使用 `lance.LanceDataset.alter_columns` 方法重命名列。

```python
table = pa.table({"id": pa.array([1, 2, 3])})
dataset = lance.write_dataset(table, "ids")
dataset.alter_columns({"path": "id", "name": "new_id"})
print(dataset.to_table().to_pandas())
#    new_id
# 0       1
# 1       2
# 2       3
```

这也适用于嵌套列。要定位嵌套列，使用点（`.`）分隔嵌套层级。例如：

```python
data = [
  {"meta": {"id": 1, "name": "Alice"}},
  {"meta": {"id": 2, "name": "Bob"}},
]
schema = pa.schema([
    ("meta", pa.struct([
        ("id", pa.int32()),
        ("name", pa.string()),
    ]))
])
dataset = lance.write_dataset(data, "nested_rename")
dataset.alter_columns({"path": "meta.id", "name": "new_id"})
print(dataset.to_table().to_pandas())
#                                  meta
# 0  {'new_id': 1, 'name': 'Alice'}
# 1    {'new_id': 2, 'name': 'Bob'}
```

## 转换列数据类型

除了更改列名外，你还可以使用 `lance.LanceDataset.alter_columns` 方法更改列的数据类型。这需要将该列重写到新的数据文件中，但不需要重写其他列。

!!! note

    如果列有索引，在更改列类型时索引将被删除。

此方法可用于更改列的向量类型。例如，我们可以将 float32 嵌入列更改为 float16 列，以牺牲精度换取磁盘空间：

```python
table = pa.table({
   "id": pa.array([1, 2, 3]),
   "embedding": pa.FixedShapeTensorArray.from_numpy_ndarray(
       np.random.rand(3, 128).astype("float32"))
})
dataset = lance.write_dataset(table, "embeddings")
dataset.alter_columns({"path": "embedding",
                       "data_type": pa.list_(pa.float16(), 128)})
print(dataset.schema)
# id: int64
# embedding: fixed_size_list<item: halffloat>[128]
#   child 0, item: halffloat
```
